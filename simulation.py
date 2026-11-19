"""Apply Firefly arm joint targets and measure the TCP in the MuJoCo workcell."""
from pathlib import Path
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET
import sys
from threading import Lock

import mujoco as mj

import numpy as np
from hardware import MAX_OVERLOAD_S, RATED_TORQUE_NM, model_paths


SIM_ROOT = Path(__file__).resolve().parent / "firefly-sim-ik"
sys.path.insert(0, str(SIM_ROOT))

from firefly_sim.build import SIDES
from firefly_sim.ik import joint_vector, sim_fk
from firefly_sim.plant import Firefly, IK_ORIENTATION_TOL_RAD, IK_POSITION_TOL_M


def normalized_pose(pose):
    xyz, quat = (np.asarray(vector, dtype=float) for vector in pose)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("Expected three finite workcell coordinates in metres")
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("Expected four finite quaternion components in w x y z order")
    scale = np.max(np.abs(quat))
    if scale == 0:
        raise ValueError("Expected a nonzero quaternion")
    quat = quat / scale
    return xyz, quat / np.linalg.norm(quat)


def pose_error(actual, target):
    """Euclidean position error in metres and shortest rotation angle in radians."""
    xyz, quat = normalized_pose(actual)
    goal_xyz, goal_quat = normalized_pose(target)
    chord = min(np.linalg.norm(quat - goal_quat), np.linalg.norm(quat + goal_quat))
    return float(np.linalg.norm(xyz - goal_xyz)), float(4 * np.arcsin(np.clip(chord / 2, 0, 1)))


def within_tolerance(errors):
    distance, angle = errors
    return distance < IK_POSITION_TOL_M and angle < IK_ORIENTATION_TOL_RAD


class GuardedFirefly(Firefly):
    """Integrate one motor budget per drive, shared across successive commands."""

    def __init__(self, scene):
        super().__init__(scene=scene)
        self.drive_joints = self.model.actuator_trnid[:, 0].copy()
        self.rated = np.empty(self.model.nu)
        for side in SIDES:
            self.rated[self.arm_drives[side]] = RATED_TORQUE_NM[:6]
            self.rated[self.grip_drive[side]] = RATED_TORQUE_NM[6]
        # Leave a 2% command-speed margin for the compliant finger linkage.
        # The measured J7 speed is still checked against the packaged 1.5 rad/s.
        self.command_step = self.max_step.copy()
        self.command_step[list(self.grip_drive.values())] *= .98
        self.hard_range = self.model.jnt_actfrcrange[self.drive_joints].copy()
        self.burst_steps = int(np.floor(MAX_OVERLOAD_S / self.model.opt.timestep))
        self._step_lock = Lock()

    def reset(self):
        super().reset()
        self.overload = np.zeros(self.model.nu, dtype=np.int64)

    def step_control(self, data, command, overload):
        """Take one 2 ms step on live or scratch data with its own overload counters.

        After a full burst, use joint-level force ranges so gravity compensation
        also counts. Re-arm only when the *unclipped* torque demand subsides.
        """
        if not np.all(np.isfinite(command)):
            raise ValueError("Drive command must be finite")
        data.ctrl += np.clip(np.clip(command, *self.limits.T) - data.ctrl, -self.command_step, self.command_step)
        locked = overload >= self.burst_steps
        self._integrate_or_forward(mj.mj_step, data, locked)
        if locked.any():
            # All drives have unit gear, actuatorgravcomp and one actuator per joint.
            raw = data.actuator_force + data.qfrc_gravcomp[self.dadr]
            locked &= np.abs(raw) > self.rated
        above = np.abs(data.qfrc_actuator[self.dadr]) > self.rated
        overload[:] = np.where(locked, self.burst_steps, (overload + 1) * above)

    def _integrate_or_forward(self, operation, data, locked):
        with self._step_lock:
            if not locked.any():
                operation(self.model, data)
                return
            ids = self.drive_joints[locked]
            self.model.jnt_actfrcrange[ids, 0] = -self.rated[locked]
            self.model.jnt_actfrcrange[ids, 1] = self.rated[locked]
            try:
                operation(self.model, data)
            finally:
                self.model.jnt_actfrcrange[ids] = self.hard_range[locked]

    def refresh(self, data, overload):
        """Refresh a readback without advancing or re-arming a derated drive."""
        self._integrate_or_forward(mj.mj_forward, data, overload >= self.burst_steps)

    def advance(self, target):
        target = np.asarray(target, dtype=float)
        if (target.shape != (self.model.nu,) or not np.isfinite(target).all()
                or (target < self.limits[:, 0]).any() or (target > self.limits[:, 1]).any()):
            raise ValueError("Drive target is outside the model limits")
        self.target = target.copy()
        for _ in range(self.substeps):
            self.step_control(self.data, target, self.overload)
        self.refresh(self.data, self.overload)
        self.periods += 1


class Simulation:
    def __init__(self, *, payload_kg=None):
        scene, _ = model_paths(SIM_ROOT)
        payloads = dict(payload_kg or {})
        if payloads.keys() - set(SIDES) or any(not np.isfinite(m) or m < 0 for m in payloads.values()):
            raise ValueError("Supply a finite nonnegative TCP payload in kg per arm")
        if not any(payloads.values()):
            self.plant = GuardedFirefly(scene)
            return
        # A compact known load at the calibrated TCP contributes to both inertia
        # and compensated gravity; the latter is included in the motor limit.
        tree = ET.parse(scene)
        for s, mass in payloads.items():
            if mass == 0:
                continue
            wrist = tree.find(f".//body[@name='{s}_link_6']")
            site = wrist.find(f"site[@name='{s}_tcp']")
            load = ET.SubElement(wrist, "body", name=f"{s}_payload", pos=site.attrib["pos"], gravcomp="1")
            inertia = .4 * mass * .001**2  # solid sphere of radius 1 mm
            ET.SubElement(load, "inertial", pos="0 0 0", mass=str(mass), diaginertia=" ".join([str(inertia)] * 3))
        with NamedTemporaryFile(dir=scene.parent, suffix=".xml") as stream:
            tree.write(stream, encoding="utf-8")
            stream.flush()
            self.plant = GuardedFirefly(stream.name)

    def unsustainable_hold(self, scratch):
        """Static joint effort at this configuration; the input must be scratch data."""
        p = self.plant
        scratch.qvel[:] = 0
        mj.mj_forward(p.model, scratch)
        # Full compensated gravity enters the actuator force on these unit-gear hinges.
        # Account for externally applied joint torques and body wrenches, if supplied.
        applied = scratch.qfrc_applied.copy()
        for bid in np.flatnonzero(np.any(scratch.xfrc_applied, axis=1)):
            wrench = scratch.xfrc_applied[bid]
            mj.mj_applyFT(p.model, scratch, wrench[:3], wrench[3:], scratch.xipos[bid], bid, applied)
        demand = np.abs((scratch.qfrc_bias - applied)[p.dadr])
        over = np.flatnonzero(demand > p.rated + 1e-9)
        return {p.model.actuator(i).name: float(demand[i]) for i in over}

    def close(self):
        self.plant.close()

    def joints(self, side):
        """Measured J1..J6 in radians."""
        drives = self.plant.arm_drives[side]
        return self.plant.data.qpos[self.plant.qadr[drives]].copy()

    def tcp(self, side):
        """Measured world-frame TCP position and wxyz quaternion."""
        return self.plant.tcp(side)

    def fk(self, side, joints):
        """Compute TCP pose without changing the live simulation."""
        return sim_fk(self.plant, side, joint_vector(self.plant, side, joints))

    def execute(self, joints_by_arm, targets_by_arm, *, max_periods=80):
        """Plan and execute a requested joint branch after checking its TCP goals."""
        from motion import move_to
        return move_to(self, {side: [q] for side, q in joints_by_arm.items()},
                       targets_by_arm, max_periods=max_periods)

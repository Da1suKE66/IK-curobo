"""Apply Firefly arm joint targets and measure the TCP in the MuJoCo workcell."""
from pathlib import Path
import sys

import numpy as np
from hardware import model_paths


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


class Simulation:
    def __init__(self):
        scene_path, _ = model_paths(SIM_ROOT)
        self.plant = Firefly(scene=scene_path)

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
        """Validate all candidates in scratch state, then drive the chosen arms together."""
        sides = set(joints_by_arm)
        if not sides or sides != set(targets_by_arm) or sides.difference(SIDES):
            raise ValueError("Supply matching joints and TCP targets for left, right, or both arms")
        if not isinstance(max_periods, int) or max_periods < 1:
            raise ValueError("max_periods must be a positive integer")

        goals = {side: normalized_pose(pose) for side, pose in targets_by_arm.items()}
        command = self.plant.target.copy()
        candidates_ok = True
        candidate_errors = {}
        for side, joints in joints_by_arm.items():
            errors = pose_error(self.fk(side, joints), goals[side])
            candidate_errors[side] = dict(position_error_m=errors[0], orientation_error_rad=errors[1])
            candidates_ok = candidates_ok and within_tolerance(errors)
            command[self.plant.arm_drives[side]] = joints

        periods = 0
        status = "candidate_fk_failed"
        if candidates_ok:
            status = "tracking_timeout"
            stable_periods = 0
            for periods in range(1, max_periods + 1):
                self.plant.advance(command)
                at_goal = all(within_tolerance(pose_error(self.tcp(side), goal))
                              for side, goal in goals.items())
                stable_periods = stable_periods + 1 if at_goal else 0
                if stable_periods >= 3:
                    status = "reached"
                    break

        measured = {}
        for side, goal in goals.items():
            position, orientation = self.tcp(side)
            distance, angle = pose_error((position, orientation), goal)
            measured[side] = dict(joints_rad=self.joints(side).tolist(), xyz_m=position.tolist(),
                                  quat_wxyz=orientation.tolist(), position_error_m=distance,
                                  orientation_error_rad=angle)
        return dict(status=status, reached=status == "reached", periods=periods,
                    sim_time_s=float(self.plant.data.time), arms=measured, candidate_errors=candidate_errors)

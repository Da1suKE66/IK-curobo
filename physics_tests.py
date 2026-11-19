"""Inspect every physics step of the Firefly controller in the MuJoCo workcell."""
from unittest.mock import patch

import mujoco as mj
import numpy as np

from collision_tests import contact_kind
from hardware import PEAK_TORQUE_NM, RATED_TORQUE_NM
from simulation import SIDES, pose_error, within_tolerance


class StepTrace:
    """Observe the production stepper without reimplementing its actuator updates."""

    def __init__(self, sim, held=None, load_kg=0., keep_samples=False):
        self.sim = sim
        self.plant = sim.plant
        self.model = self.plant.model
        self.data = self.plant.data
        self.dt = float(self.model.opt.timestep)
        self.start = float(self.data.time)
        self.drives = trace_drive_order(self.plant)
        self.qadr = self.plant.qadr[self.drives]
        self.dadr = self.plant.dadr[self.drives]
        self.held = held
        self.hold_pose = sim.tcp(held) if held else None
        self.hold_error = np.zeros(2)
        self.initial_ctrl = self.data.ctrl[self.drives].copy()
        self.last_ctrl = self.initial_ctrl.copy()
        self.last_q = self.data.qpos[self.qadr].copy()
        self.qstep = np.zeros(len(self.drives))
        self.ctrlstep = self.qstep.copy()
        self.ctrl_excursion = self.qstep.copy()
        self.max_speed = self.qstep.copy()
        self.speed_excess_steps = np.zeros(len(self.drives), dtype=int)
        self.rate = self.plant.max_step[self.drives] / self.dt
        self.torque = []
        self.positions = [] if keep_samples else None
        self.setpoints = [] if keep_samples else None
        self.show_contacts = {}
        self.contact_steps = 0
        self.load_kg = load_kg
        self.real_step = mj.mj_step

    def apply_tcp_load(self):
        # xfrc_applied expects world-frame force and moment about the body COM.
        mj.mj_kinematics(self.model, self.data)
        force = self.load_kg * self.model.opt.gravity
        for side in SIDES:
            site = self.model.site(f"{side}_tcp").id
            body = self.model.site_bodyid[site]
            lever = self.data.site_xpos[site] - self.data.xipos[body]
            self.data.xfrc_applied[body, :3] = force
            self.data.xfrc_applied[body, 3:] = np.cross(lever, force)

    def capture_contacts(self, t):
        pairs = {}
        for contact in self.data.contact:
            if contact.dist >= -1.e-6:
                continue
            kind, body1, body2 = contact_kind(self.model, contact)
            if kind == "fixed":
                continue
            labels = sorted((self.model.geom(contact.geom1).name or body1,
                             self.model.geom(contact.geom2).name or body2))
            key = (kind, *labels)
            pairs[key] = max(pairs.get(key, 0.), -float(contact.dist))
        self.contact_steps += bool(pairs)
        for pair, depth in pairs.items():
            entry = self.show_contacts.setdefault(pair, dict(kind=pair[0], pair=list(pair[1:]),
                                                             steps=0, first_s=t, max_depth_m=0.))
            entry["steps"] += 1
            entry["max_depth_m"] = max(entry["max_depth_m"], depth)

    def step(self, model, data):
        if data is not self.data or model is not self.model:
            return self.real_step(model, data)
        t = float(data.time)
        if self.load_kg:
            self.apply_tcp_load()
        self.real_step(model, data)
        # implicitfast leaves forces and contacts from the starting pose of this
        # interval in data, while qpos and qvel have advanced to its end.
        self.capture_contacts(t)
        self.torque.append(data.qfrc_actuator[self.dadr].copy())
        ctrl = data.ctrl[self.drives]
        q = data.qpos[self.qadr]
        dq = np.abs(q - self.last_q)
        self.qstep = np.maximum(self.qstep, dq)
        self.ctrlstep = np.maximum(self.ctrlstep, np.abs(ctrl - self.last_ctrl))
        self.ctrl_excursion = np.maximum(self.ctrl_excursion, np.abs(ctrl - self.initial_ctrl))
        speed = np.abs(data.qvel[self.dadr])
        self.max_speed = np.maximum(self.max_speed, speed)
        self.speed_excess_steps += speed > self.rate + 1.e-8
        self.last_q, self.last_ctrl = q, ctrl
        if self.positions is not None:
            self.positions.append(q)
            self.setpoints.append(ctrl)
        if self.held:
            self.hold_error = np.maximum(self.hold_error, pose_error(self.sim.tcp(self.held), self.hold_pose))

    def summarize(self):
        rated, peak = np.tile(RATED_TORQUE_NM, 2), np.tile(PEAK_TORQUE_NM, 2)
        torque = np.asarray(self.torque).reshape(-1, len(self.drives))
        above = np.abs(torque) > rated + 1.e-8
        longest = np.zeros(len(self.drives), dtype=int)
        streak = longest.copy()
        for row in above:
            streak = (streak + 1) * row
            longest = np.maximum(longest, streak)
        abs_peak = np.max(np.abs(torque), axis=0, initial=0.)
        peaks_at = np.argmax(np.abs(torque), axis=0)*self.dt if len(torque) else None
        arm = np.tile([True] * 6 + [False], 2)
        over_step_budget = self.qstep > self.rate*self.dt + 1.e-8
        report = dict(steps=len(torque), sampled_s=len(torque)*self.dt, start_sim_s=self.start,
                      absolute_peak_Nm=abs_peak.tolist(),
                      peak_time_from_start_s=None if peaks_at is None else peaks_at.tolist(),
                      total_above_rated_s=(above.sum(axis=0)*self.dt).tolist(),
                      longest_above_rated_s=(longest*self.dt).tolist(),
                      rated_exceeded=bool(np.any(above)),
                      peak_exceeded=bool(np.any(abs_peak > peak + 1.e-8)),
                      steps_above_peak=(np.abs(torque) > peak + 1.e-8).sum(axis=0).tolist(),
                      time_at_peak_s=((np.abs(torque) >= peak - 1.e-8).sum(axis=0)*self.dt).tolist(),
                      measured_max_step_rad=self.qstep.tolist(),
                      command_max_step_rad=self.ctrlstep.tolist(),
                      measured_max_velocity_rad_s=self.max_speed.tolist(),
                      velocity_excess_steps=self.speed_excess_steps.tolist(),
                      velocity_exceeded=bool(np.any(self.speed_excess_steps[arm])),
                      gripper_speed_above_model_slew=bool(np.any(self.speed_excess_steps[~arm])),
                      command_slew_exceeded=bool(np.any(self.ctrlstep > self.rate*self.dt + 1.e-8)),
                      qstep_exceeds_speed_budget=bool(np.any(over_step_budget[arm])),
                      collision_steps=self.contact_steps, contacts=list(self.show_contacts.values()))
        if self.held:
            self.hold_error = np.maximum(self.hold_error, pose_error(self.sim.tcp(self.held), self.hold_pose))
            mask = slice(0,7) if self.held == "left" else slice(7,14)
            command_change = float(max(self.ctrl_excursion[mask]))
            report["hold"] = dict(arm=self.held, max_position_error_m=self.hold_error[0],
                                  max_orientation_error_rad=self.hold_error[1],
                                  max_setpoint_change_rad=command_change,
                                  passed=bool(command_change == 0 and within_tolerance(self.hold_error)))
        return report


def offset_goals(sim, offsets):
    goals = {}
    for side, offset in offsets.items():
        xyz, quat = sim.tcp(side)
        goals[side] = xyz + offset, quat
    return goals


def run_physics_tests(sim, controller, *, traces=None):
    plant, model, data = sim.plant, sim.plant.model, sim.plant.data
    cases = []

    def observe(name, category, action, *, held=None, load_kg=0.):
        trace = StepTrace(sim, held, load_kg, keep_samples=traces is not None)
        with patch.object(mj, "mj_step", new=trace.step):
            result = action()
        telemetry = trace.summarize()
        if traces is not None:
            traces[f"{name}_time_s"] = trace.start + np.arange(len(trace.torque))*trace.dt
            for suffix, samples in (("torque_nm", trace.torque), ("q_end_rad", trace.positions),
                                    ("ctrl_rad", trace.setpoints)):
                traces[f"{name}_{suffix}"] = np.asarray(samples).reshape(-1, len(trace.drives))
        reached = result["status"] == "reached"
        within = (telemetry["steps"] > 0 and reached and not telemetry["collision_steps"]
                  and not telemetry["rated_exceeded"] and not telemetry["peak_exceeded"]
                  and not telemetry["velocity_exceeded"] and not telemetry["command_slew_exceeded"]
                  and not telemetry["qstep_exceeds_speed_budget"] and telemetry.get("hold", {}).get("passed", True))
        cases.append(dict(name=name, category=category, result=result, telemetry=telemetry,
                          all_monitored_limits_met=bool(within)))

    for side, held in (("left", "right"), ("right", "left")):
        plant.reset()
        goals = offset_goals(sim, {side: [0.018, 0., 0.008]})
        observe(f"{side}_moves_{held}_holds", "nominal", lambda: controller.run(goals, move=True), held=held)

    plant.reset()
    goals = offset_goals(sim, {"left": [0.018, 0.003, 0.008], "right": [-0.010, -0.004, 0.014]})
    observe("asymmetric_dual_move", "nominal", lambda: controller.run(goals, move=True))

    plant.reset()
    goals = {}
    for side, sign in (("left", 1), ("right", -1)):
        q = sim.joints(side)
        q[0] += sign * 1.2
        goals[side] = sim.fk(side, q)
    observe("outward_shoulder_swing", "stress", lambda: controller.run(goals, move=True))

    plant.reset()
    home = {side: sim.tcp(side) for side in SIDES}
    goals = {}
    for side, sign in (("left", 1), ("right", -1)):
        q = sim.joints(side) + np.array([0.80*sign, 0.70, -0.45, -0.50, 0.85*sign, 1.2*sign])
        goals[side] = sim.fk(side, q)
    observe("large_dual_move_outward", "stress", lambda: controller.run(goals, move=True))
    observe("large_dual_move_return", "stress", lambda: controller.run(home, move=True))

    plant.reset()
    goals = {}
    for side, sign in (("left",-1),("right",1)):
        q = sim.joints(side)
        q[0] += sign*0.38
        goals[side] = sim.fk(side, q)
    observe("both_shoulders_turn_inward_0.38_rad", "collision_challenge",
            lambda: controller.run(goals, move=True))

    plant.reset()

    def gripper_cycle():
        phases = []
        for endpoint in (0.50, 0.):
            target = plant.target.copy()
            for drive in plant.grip_drive.values():
                target[drive] = endpoint
            for _ in range(12):
                plant.advance(target)
            actual = [float(data.joint(f"{s}_grip_joint_{i}").qpos[0]) for s in SIDES for i in (1,2)]
            phases.append(dict(target_rad=endpoint, measured_rad=actual,
                               passed=bool(np.max(np.abs(np.array(actual)-endpoint)) < 0.005)))
        return dict(status="reached" if all(p["passed"] for p in phases) else "tracking_failed", phases=phases)

    observe("grippers_close_and_open", "nominal", gripper_cycle)

    extended_q = [0., 1.15, 0.25, 0.10, 0., 0.]

    def static_hold(goals, load):
        for _ in range(15):
            plant.advance(plant.target)
        errors = {side: pose_error(sim.tcp(side), goals[side]) for side in SIDES}
        return dict(status="reached" if all(within_tolerance(v) for v in errors.values()) else "hold_drift",
                    per_tcp_mass_equivalent_kg=load, held_joint_setpoint_rad=extended_q,
                    tcp_final_error_m_rad=errors)

    for name, load in (("extended_hold_unloaded", 0.), ("extended_hold_at_1.5kg_per_tcp", 1.5)):
        plant.reset()
        for side in SIDES:
            drives = plant.arm_drives[side]
            data.qpos[plant.qadr[drives]] = extended_q
            data.ctrl[drives] = extended_q
        plant.target = data.ctrl.copy()
        mj.mj_forward(model, data)
        goals = {side: sim.tcp(side) for side in SIDES}
        observe(name, "load_stress", lambda: static_hold(goals, load), load_kg=load)
    plant.reset()
    nominal = [c for c in cases if c["category"] == "nominal"]
    completed = all(c["telemetry"]["steps"] > 0 for c in cases)
    return dict(completed=completed, nominal_passed=all(c["all_monitored_limits_met"] for c in nominal),
                sample_period_s=float(model.opt.timestep),
                joint_order=[f"{side}_J{i}" for side in SIDES for i in range(1,8)],
                rated_torque_Nm=np.tile(RATED_TORQUE_NM,2).tolist(),
                peak_torque_Nm=np.tile(PEAK_TORQUE_NM,2).tolist(),
                velocity_budget_rad_s=(plant.max_step / model.opt.timestep)[trace_drive_order(plant)].tolist(),
                penetration_threshold_m=1.e-6, cases=cases)


def trace_drive_order(plant):
    return np.concatenate([np.r_[plant.arm_drives[s], plant.grip_drive[s]] for s in SIDES])

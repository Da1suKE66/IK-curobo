"""Check both arm branches and a complete MuJoCo rollout before a point move."""
from collections import Counter
from copy import copy
from itertools import product

import mujoco as mj
import numpy as np

from simulation import SIDES, normalized_pose, pose_error, within_tolerance
from firefly_sim.plant import quaternion


# An unchanged geometric IK solution can lie exactly on a mechanical stop.
# Keep the position servo's command a few microradians inside that stop.
SERVO_ENDSTOP_MARGIN_RAD = 1e-5


def _contact(model, data):
    for c in data.contact:
        if c.dist <= 0:
            return dict(bodies=[model.body(model.geom_bodyid[g]).name for g in c.geom],
                        penetration_m=float(-c.dist))
    return None


def _goal_state(plant, data, goals):
    arms = {}
    for side, goal in goals.items():
        site = data.site(f"{side}_tcp")
        pos, quat = site.xpos, quaternion(site.xmat)
        distance, angle = pose_error((pos, quat), goal)
        arms[side] = dict(joints_rad=data.qpos[plant.qadr[plant.arm_drives[side]]].tolist(),
                          xyz_m=pos.tolist(), quat_wxyz=quat.tolist(),
                          position_error_m=distance, orientation_error_rad=angle)
    return arms


def _routes(plant, start, end, sides):
    yield "synchronous", [end]
    moving = [side for side in sides if np.max(np.abs((end - start)[plant.arm_drives[side]])) > 1e-6]
    if len(moving) == 2:
        for first in moving:
            via = start.copy()
            via[plant.arm_drives[first]] = end[plant.arm_drives[first]]
            yield first + "_first", [via, end]


def _commands(plant, start, waypoints, stretch):
    # For s(u)=10u³−15u⁴+6u⁵, max s' = 1.875, max |s''| < 5.78.
    # Duration bounds keep arm speed and acceleration moderate before dynamics screening.
    period = plant.model.opt.timestep * plant.substeps
    commands = []
    durations = []
    for end in waypoints:
        delta = end - start
        speed = np.minimum(plant.model.actuator_user[:, 0], 1.8)
        duration = max(0.4, np.max(1.875 * np.abs(delta) / speed), np.sqrt(5.78 * np.max(np.abs(delta)) / 4.0))
        periods = max(1, int(np.ceil(duration * stretch / period)))
        n = periods * plant.substeps
        u = (np.arange(1, n + 1) / n)[:, None]
        blend = u ** 3 * (10.0 + u * (-15.0 + 6.0 * u))
        segment = np.clip(start + delta * blend, *plant.limits.T)
        segment[-1] = end
        commands.append(segment)
        durations.append(periods * period)
        start = end
    return np.concatenate(commands), durations


class MotionChecks:
    def __init__(self, plant):
        self.plant = plant
        self.model = plant.model
        self.drives = np.concatenate([plant.arm_drives[s] for s in SIDES])
        self.qadr = plant.qadr[self.drives]
        self.dadr = plant.dadr[self.drives]
        self.lo, self.hi = plant.limits[self.drives].T
        self.speed = self.model.actuator_user[self.drives, 0]
        self.warning_counts = plant.data.warning.number.copy()

    def violation(self, data):
        if data.ncon:
            hit = _contact(self.model, data)
            if hit:
                return dict(reason="collision", **hit)
        q, v = data.qpos[self.qadr], data.qvel[self.dadr]
        if not np.isfinite(q).all() or not np.isfinite(v).all() or np.any(data.warning.number > self.warning_counts):
            return dict(reason="dynamics_warning")
        excess = np.maximum(self.lo - q, q - self.hi)
        if np.any(excess > 0):
            j = int(np.argmax(excess))
            return dict(reason="joint_limit", joint=self.model.actuator(self.drives[j]).name,
                        excursion_rad=float(excess[j]))
        if np.any(np.abs(v) > self.speed + 1e-8):
            return dict(reason="joint_speed")
        return None

    def settled(self, data, goals):
        if np.max(np.abs(data.qvel[self.dadr])) >= 0.01:
            return False
        return all(within_tolerance(pose_error((data.site(f"{s}_tcp").xpos,
                                                quaternion(data.site(f"{s}_tcp").xmat)), goal))
                   for s, goal in goals.items())

    def preview(self, commands, goals, max_steps):
        plant = self.plant
        data, overload = copy(plant.data), plant.overload.copy()
        consecutive = reached_at = 0
        # Include an extra hold long enough to observe the rated-torque timeout.
        # Only the motion and settling portion is committed to the live simulation.
        extra = int(np.ceil(2.4 / self.model.opt.timestep))
        for index in range(max_steps + extra):
            command = commands[min(index, len(commands) - 1)]
            plant.step_control(data, command, overload)
            problem = self.violation(data)
            if problem:
                return 0, {**problem, "time_s": (index + 1) * self.model.opt.timestep}
            elapsed = index + 1
            if elapsed % plant.substeps:
                continue
            plant.refresh(data, overload)
            if elapsed < len(commands):
                continue
            good = self.settled(data, goals)
            consecutive = consecutive + 1 if good else 0
            if not reached_at and consecutive >= 3:
                reached_at = elapsed
            if reached_at and not good:
                return 0, dict(reason="unstable_hold")
            if reached_at and elapsed >= reached_at + extra:
                return reached_at, None
            if elapsed >= max_steps and not reached_at:
                break
        return 0, dict(reason="tracking_timeout")

    def commit(self, commands, steps):
        plant = self.plant
        for i in range(steps):
            command = commands[min(i, len(commands) - 1)]
            plant.step_control(plant.data, command, plant.overload)
            problem = self.violation(plant.data)
            if problem:
                plant.target = np.clip(plant.data.qpos[plant.qadr], *plant.limits.T)
                plant.refresh(plant.data, plant.overload)
                plant.periods += (i + 1) // plant.substeps
                return i + 1, problem
            if (i + 1) % plant.substeps == 0:
                plant.refresh(plant.data, plant.overload)
        plant.target = commands[-1].copy()
        plant.periods += steps // plant.substeps
        return steps, None


def move_to(sim, candidates_by_arm, targets_by_arm, *, max_periods=80):
    """Try IK branches, arm order and timing in a private copy of the live state."""
    plant = sim.plant
    sides = [s for s in SIDES if s in targets_by_arm]
    if not sides or set(sides) != set(candidates_by_arm) or set(sides) != set(targets_by_arm):
        raise ValueError("Supply matching candidates and TCP targets for one or both arms")
    if not isinstance(max_periods, int) or max_periods < 1:
        raise ValueError("max_periods must be a positive integer")
    goals = {s: normalized_pose(targets_by_arm[s]) for s in sides}
    options = {}
    for s in sides:
        options[s] = []
        for q in candidates_by_arm[s]:
            q = np.asarray(q, dtype=float)
            errors = pose_error(sim.fk(s, q), goals[s])
            if within_tolerance(errors):
                options[s].append((q, dict(position_error_m=errors[0], orientation_error_rad=errors[1])))
        if not options[s]:
            return dict(status="candidate_fk_failed", arm=s, reached=False, periods=0,
                        sim_time_s=float(plant.data.time), arms=_goal_state(plant, plant.data, goals))

    start = plant.data.ctrl.copy()
    choices = list(product(*(options[s] for s in sides)))
    choices.sort(key=lambda group: sum(np.sum((opt[0] - sim.joints(s)) ** 2) for s, opt in zip(sides, group)))
    rollout = MotionChecks(plant)
    static = copy(plant.data)
    failures = Counter()
    first_failure = None
    max_steps = max_periods * plant.substeps
    for group in choices:
        end = plant.target.copy()
        for side, (q, _) in zip(sides, group):
            end[plant.arm_drives[side]] = q
            static.qpos[plant.qadr[plant.arm_drives[side]]] = q
        mj.mj_fwdPosition(plant.model, static)
        hit = _contact(plant.model, static)
        if hit:
            failures["collision"] += 1
            first_failure = first_failure or dict(reason="collision", phase="endpoint", **hit)
            continue
        overload = sim.unsustainable_hold(static)
        if overload:
            failures["static_torque"] += 1
            first_failure = first_failure or dict(reason="static_torque", phase="endpoint", **overload)
            continue
        for side in sides:
            drives = plant.arm_drives[side]
            lo, hi = plant.limits[drives].T
            end[drives] = np.clip(end[drives], lo + SERVO_ENDSTOP_MARGIN_RAD, hi - SERVO_ENDSTOP_MARGIN_RAD)
        for route, waypoints in _routes(plant, start, end, sides):
            for stretch in (1., 1.7, 3., 3.7, 5.):
                commands, durations = _commands(plant, start, waypoints, stretch)
                if len(commands) + 2 * plant.substeps > max_steps:
                    failures["time_budget"] += 1
                    break
                steps, issue = rollout.preview(commands, goals, max_steps)
                if issue:
                    failures[issue["reason"]] += 1
                    first_failure = first_failure or dict(phase="rollout", **issue)
                    if issue["reason"] == "collision":
                        break
                    continue
                steps, issue = rollout.commit(commands, steps)
                return dict(status="motion_interrupted" if issue else "reached", reached=issue is None,
                            periods=steps / plant.substeps, sim_time_s=float(plant.data.time),
                            arms=_goal_state(plant, plant.data, goals), route=route, ramp_s=durations,
                            candidate_errors={s: opt[1] for s, opt in zip(sides, group)},
                            selected_joints_rad={s: opt[0].tolist() for s, opt in zip(sides, group)},
                            command_joints_rad={s: end[plant.arm_drives[s]].tolist() for s in sides},
                            rejected_trials=dict(failures), **({"detail": issue} if issue else {}))
    return dict(status="motion_rejected", reached=False, periods=0, sim_time_s=float(plant.data.time),
                reason=failures.most_common(1)[0][0], rejected_trials=dict(failures),
                first_failure=first_failure, arms=_goal_state(plant, plant.data, goals))

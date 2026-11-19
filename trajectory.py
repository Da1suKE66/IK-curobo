"""Rest-to-rest Cartesian waypoint tracking on the Firefly simulation clock."""
from copy import copy

import mujoco as mj
import numpy as np
from scipy.interpolate import CubicSpline

from hardware import MAX_OVERLOAD_S
from simulation import SIDES, SIM_ROOT, normalized_pose, pose_error, within_tolerance
from firefly_sim.plant import quaternion


def _spline(times, values):
    # The user supplies rest-to-rest paths. Zero endpoint velocities also join the hold smoothly.
    return CubicSpline(times, values, axis=0, bc_type="clamped")


def _waypoints(controller, times, targets):
    from solver import Solver

    sim = controller.sim
    joints = np.tile(sim.plant.target, (len(times), 1))
    fk = {}
    for side, poses in targets.items():
        seed = sim.joints(side)
        if not within_tolerance(pose_error(sim.tcp(side), poses[0])):
            return None, fk, dict(reason="start_pose_mismatch", arm=side, waypoint=0)
        if side not in controller.solvers:
            controller.solvers[side] = Solver(SIM_ROOT, side)
        max_distance = max_angle = max_step = 0.
        for i, goal in enumerate(poses):
            solution = controller.solvers[side].solve(*goal, seed)
            if solution is None:
                return None, fk, dict(reason="ik_failed", arm=side, waypoint=i)
            errors = pose_error(sim.fk(side, solution), goal)
            if not within_tolerance(errors):
                return None, fk, dict(reason="candidate_fk_failed", arm=side, waypoint=i)
            max_distance = max(max_distance, errors[0])
            max_angle = max(max_angle, errors[1])
            max_step = max(max_step, float(np.max(np.abs(solution - seed))))
            joints[i, sim.plant.arm_drives[side]] = solution
            seed = solution
        fk[side] = dict(max_position_error_m=max_distance, max_orientation_error_rad=max_angle,
                        max_joint_step_rad=max_step)
    return joints, fk, None


def _servo_commands(plant, times, joints, grid, sides):
    model = plant.model
    spline = _spline(times, joints)
    # The integrator consumes an actuator command at the *start* of each step.
    position, velocity, acceleration = (spline(grid[:-1], order) for order in range(3))
    drives = np.concatenate([plant.arm_drives[s] for s in sides])
    lo, hi = plant.limits[drives].T
    if np.any((position[:, drives] < lo - 1e-10) | (position[:, drives] > hi + 1e-10)):
        return None, dict(reason="interpolated_joint_limit")
    if np.any(np.abs(velocity[:, drives]) > model.actuator_user[drives, 0] + 1e-8):
        return None, dict(reason="interpolated_joint_speed")

    ref = copy(plant.data)
    dofs = plant.dadr[drives]
    kp = model.actuator_gainprm[drives, 0]
    kv = -model.actuator_biasprm[drives, 2]
    commands = position.copy()
    for i, q in enumerate(position):
        ref.qpos[plant.qadr] = q
        ref.qvel[plant.dadr] = velocity[i]
        ref.qacc[plant.dadr] = acceleration[i]
        mj.mj_inverse(model, ref)
        # The position servo already provides gravity compensation. Undo the constraint
        # term in inverse dynamics so contact never produces a feed-forward command.
        torque = ref.qfrc_inverse[dofs] + ref.qfrc_constraint[dofs] - ref.qfrc_gravcomp[dofs]
        commands[i, drives] += (kv * velocity[i, drives] + torque) / kp
    return commands, None


def _references(times, targets, grid):
    states = {}
    for side, poses in targets.items():
        xyz, quat = (np.stack(column) for column in zip(*poses))
        # q and -q denote the same rotation; use a continuous representative for interpolation.
        for i in range(1, len(quat)):
            if np.dot(quat[i], quat[i - 1]) < 0:
                quat[i] *= -1
        orientations = _spline(times, quat)(grid)
        norm = np.linalg.norm(orientations, axis=1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError("Waypoint orientation interpolation is ambiguous")
        states[side] = (_spline(times, xyz)(grid), orientations / norm)
    return states


def _replay(plant, checks, data, budget, commands, hold_command, references, hold_steps, *,
            required_hold_steps):
    sites = {s: data.site(f"{s}_tcp") for s in references}
    peak_m = peak_rad = 0.
    tail = {}
    initial_time = data.time
    goals = {side: (xyz[-1], quat[-1]) for side, (xyz, quat) in references.items()}
    for i in range(len(commands) + hold_steps):
        moving = i < len(commands)
        plant.step_control(data, commands[i] if moving else hold_command, budget)
        problem = checks.violation(data)
        if problem:
            return {}, dict(**problem, time_s=float(data.time - initial_time))
        # mj_step leaves site poses at the start of the integration interval.
        mj.mj_kinematics(plant.model, data)
        index = i if moving else -1
        for side, site in sites.items():
            xyz, quat = references[side]
            distance = float(np.linalg.norm(site.xpos - xyz[index]))
            actual = quaternion(site.xmat)
            chord = min(np.linalg.norm(actual - quat[index]), np.linalg.norm(actual + quat[index]))
            angle = float(4 * np.arcsin(min(chord / 2, 1.)))
            if moving:
                peak_m, peak_rad = max(peak_m, distance), max(peak_rad, angle)
                if distance >= .001 or angle >= .004:
                    return {}, dict(reason="tracking_error", arm=side,
                                    position_error_m=distance, orientation_error_rad=angle,
                                    time_s=float(data.time - initial_time))
            tail[side] = dict(position_error_m=distance, orientation_error_rad=angle)
        elapsed_hold = i + 1 - len(commands)
        if elapsed_hold >= required_hold_steps and (elapsed_hold - required_hold_steps) % plant.substeps == 0:
            if not checks.settled(data, goals):
                return {}, dict(reason="endpoint_not_settled", time_s=float(data.time - initial_time))
    plant.refresh(data, budget)
    problem = checks.violation(data)
    return dict(moving_peak_position_m=peak_m, moving_peak_orientation_rad=peak_rad, arms=tail), problem


def track(controller, times_s, targets_by_arm, *, hold_s=.4):
    """Validate and follow synchronized world-frame poses, starting at measured TCPs.

    Poses are (xyz_m, quat_wxyz); times start at zero and end on the MuJoCo clock.
    The full reference, including stationary endpoints, is a clamped cubic spline in
    position and sign-continuous normalized quaternion components.
    """
    from motion import MotionChecks

    plant = controller.sim.plant
    dt = plant.model.opt.timestep
    times = np.asarray(times_s, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[0] != 0 or np.any(np.diff(times) < dt - 1e-12)):
        raise ValueError("Supply at least two increasing times starting at zero and spaced by a simulation step")
    steps = round(times[-1] / dt)
    if not np.isclose(steps * dt, times[-1], atol=1e-10, rtol=0):
        raise ValueError("Final waypoint must lie on a simulation step")
    if not np.isfinite(hold_s) or hold_s < 0:
        raise ValueError("hold_s must be finite and nonnegative")
    if not targets_by_arm or not set(targets_by_arm).issubset(SIDES):
        raise ValueError("Supply waypoints for left, right, or both arms")
    targets = {s: [normalized_pose(p) for p in poses] for s, poses in targets_by_arm.items()}
    if any(len(poses) != len(times) for poses in targets.values()):
        raise ValueError("Supply one TCP pose per timestamp for each selected arm")

    base = dict(waypoints=len(times), duration_s=float(times[-1]))
    joints, base["waypoint_fk"], problem = _waypoints(controller, times, targets)
    if not problem:
        endpoint = copy(plant.data)
        endpoint.qpos[plant.qadr] = joints[-1]
        demand = controller.sim.unsustainable_hold(endpoint)
        if demand:
            problem = dict(reason="static_torque", joints_nm=demand)
    if not problem:
        grid = np.arange(steps + 1) * dt
        commands, problem = _servo_commands(plant, times, joints, grid, targets)
    if not problem:
        references = _references(times, targets, grid[1:])
        checks = MotionChecks(plant)
        hold = int(np.ceil(hold_s / dt))
        # A torque-limited endpoint may drift only after a continuous overload burst.
        preview_hold = max(hold, int(np.ceil((MAX_OVERLOAD_S + .4) / dt)))
        _, problem = _replay(plant, checks, copy(plant.data), plant.overload.copy(),
                             commands, joints[-1], references, preview_hold, required_hold_steps=hold)
    if problem:
        return dict(**base, status="trajectory_rejected", reached=False, detail=problem,
                    sim_time_s=float(plant.data.time))
    start = plant.data.time
    metrics, problem = _replay(plant, checks, plant.data, plant.overload,
                               commands, joints[-1], references, hold, required_hold_steps=hold)
    plant.periods += round((plant.data.time - start) / dt) // plant.substeps
    if problem:
        plant.target = np.clip(plant.data.qpos[plant.qadr], *plant.limits.T)
        plant.refresh(plant.data, plant.overload)
        return dict(**base, status="motion_interrupted", reached=False, detail=problem,
                    sim_time_s=float(plant.data.time))
    plant.target = joints[-1].copy()
    return dict(**base, **metrics, status="reached", reached=True, sim_time_s=float(plant.data.time))

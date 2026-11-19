"""Probe geometric IK coverage and continuity with independent MuJoCo FK witnesses."""
import mujoco as mj
import numpy as np

from simulation import SIDES, pose_error, within_tolerance


SAMPLE_SEED = 20261119
JUMP_RAD = np.deg2rad(10.0)  # Audit threshold for neighbouring targets, in bounded joint coordinates.
AMPLIFIED_STEP_RAD = np.deg2rad(2.)
AMPLIFICATION_FACTOR = 5.


def witness_geometry(sim, side, joints):
    goal = sim.fk(side, joints)
    plant = sim.plant
    data = plant.scratch
    # sim.fk writes scratch joint positions and computes kinematics. Refresh dynamics
    # for this same posture before reading contacts or forming a geometric Jacobian.
    mj.mj_fwdPosition(plant.model, data)
    jac = np.empty((6, plant.model.nv))
    mj.mj_jacSite(plant.model, data, jac[:3], jac[3:], data.site(f"{side}_tcp").id)
    jac = jac[:, plant.dadr[plant.arm_drives[side]]]
    singular_values = np.linalg.svd(jac, compute_uv=False)

    penetrating = []
    for contact in data.contact:
        if contact.dist >= -1e-6:
            continue
        bodies = [plant.model.body(plant.model.geom_bodyid[g]).name for g in contact.geom]
        if any(name.startswith(f"{side}_") for name in bodies):
            penetrating.append((float(-contact.dist), bodies))
    contact = None
    if penetrating:
        depth, pair = max(penetrating, key=lambda item: item[0])
        contact = dict(depth_m=depth, bodies=pair)
    return goal, float(singular_values[-1]), contact


def singular_posture(home):
    # With these shoulder and wrist angles the Y6 Jacobian loses rank close
    # to J3=0.6065. Record its actual smallest singular value on each arm.
    return np.array([home[0], -0.30, 0.6065, 0.50, 0., 0.])


def witnesses(home, limits):
    lower, upper = limits.T
    yield "home", home
    for i in range(6):
        for label, stop, direction in (("min", lower[i], 1.), ("max", upper[i], -1.)):
            for margin in (0., 0.25):
                q = home.copy()
                q[i] = stop + direction * np.deg2rad(margin)
                yield f"J{i+1}_{label}_{margin:g}deg_inset", q
    yield "near_singularity", singular_posture(home)

    # Independently permute strata in each joint. Repeat the same strata on both arms.
    rng = np.random.default_rng(SAMPLE_SEED)
    count = 16
    fractions = np.column_stack([(rng.permutation(count) + rng.random(count)) / count
                                 for _ in range(6)])
    for i, q in enumerate(lower + (0.02 + 0.96 * fractions) * (upper - lower)):
        yield f"interior_{i+1:02d}", q


def evaluate(sim, side, solver, goal, seed, limits):
    result = solver.solve(*goal, seed)
    if result is None:
        return None, {"status": "no_solution"}
    q = np.asarray(result, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        return None, {"status": "invalid_joint_vector"}
    lower, upper = limits.T
    excess = float(np.maximum(lower - q, q - upper).max())
    # The simulated actuator accepts strict bounds. Report even float32 rounding
    # past an exact limit separately from a missing solver result.
    if excess > 0:
        return None, dict(status="outside_joint_limits", excess_rad=excess, joints_rad=q.tolist())
    distance, angle = pose_error(sim.fk(side, q), goal)
    ok = within_tolerance((distance, angle))
    info = dict(status="solved" if ok else "fk_error", joints_rad=q.tolist(), position_error_m=distance,
                orientation_error_rad=angle, max_seed_to_solution_rad=float(np.abs(q - seed).max()))
    return (q if ok else None), info


def reachability(sim, side, solver, home, limits):
    results = []
    counts = {name: dict(solved=0, no_solution=0, invalid_joint_vector=0,
                         outside_joint_limits=0, fk_error=0)
              for name in ("home", "nearby")}
    missed_free = {name: 0 for name in counts}
    for label, reference in witnesses(home, limits):
        goal, sigma_min, contact = witness_geometry(sim, side, reference)
        nearby = np.clip(reference + np.array([.023, -.029, .037, -.031, .019, -.027]), *limits.T)
        trials = {}
        for name, seed in (("home", home), ("nearby", nearby)):
            _, info = evaluate(sim, side, solver, goal, seed, limits)
            counts[name][info["status"]] += 1
            if contact is None and info["status"] != "solved":
                missed_free[name] += 1
            trials[name] = info
        results.append(dict(case=label, witness_joints_rad=reference.tolist(),
                            xyz_m=goal[0].tolist(), quat_wxyz=goal[1].tolist(),
                            jacobian_sigma_min=sigma_min, witness_penetration=contact, seeds=trials))
    neither = sum(all(trial["status"] != "solved" for trial in row["seeds"].values()) for row in results)
    return dict(witnesses=len(results), missed_from_both_seeds=neither,
                clear_witnesses=sum(r["witness_penetration"] is None for r in results),
                counts=counts, failed_from_clear_witness=missed_free, trials=results)


def continuity(sim, side, solver, name, references, limits):
    # Mimic the next control cycle receiving the previous solution as its measured
    # angle, without conflating IK branch selection with physical servo tracking.
    seed = references[0].copy()
    previous_index = 0
    previous_goal = sim.fk(side, references[0])
    events = []
    missed, jumps, amplified = 0, 0, 0
    max_ref, max_output, max_xyz, max_angle = 0., 0., 0., 0.
    worst = None
    for index in range(1, len(references)):
        q = references[index]
        goal = sim.fk(side, q)
        distance, angle = pose_error(goal, previous_goal)
        max_xyz, max_angle = max(max_xyz, distance), max(max_angle, angle)
        max_ref = max(max_ref, float(np.abs(q - references[index-1]).max()))
        previous_goal = goal
        solution, trial = evaluate(sim, side, solver, goal, seed, limits)
        if solution is None:
            missed += 1
            events.append(dict(index=index, witness_joints_rad=q.tolist(), **trial))
            continue
        step = np.abs(solution - seed)
        output_step = float(step.max())
        witness_step = float(np.abs(q - references[previous_index]).max())
        is_jump = output_step > JUMP_RAD
        is_amplified = output_step > AMPLIFIED_STEP_RAD and output_step > AMPLIFICATION_FACTOR * witness_step
        jumps += int(is_jump)
        amplified += int(is_amplified)
        if is_jump or is_amplified or worst is None or output_step > max_output:
            _, sigma_min, witness_contact = witness_geometry(sim, side, q)
            _, _, solution_contact = witness_geometry(sim, side, solution)
            shift_m, shift_rad = pose_error(goal, sim.fk(side, references[previous_index]))
            detail = dict(index=index, since_index=previous_index, worst_joint=int(step.argmax())+1,
                          previous_solution_rad=seed.tolist(), solution_rad=solution.tolist(),
                          witness_joints_rad=q.tolist(), witness_penetration=witness_contact,
                          solution_penetration=solution_contact, witness_jacobian_sigma_min=sigma_min,
                          joint_step_rad=output_step, witness_step_rad=witness_step,
                          tcp_step_m=shift_m, tcp_step_rad=shift_rad)
            if is_jump or is_amplified:
                events.append(dict(status="joint_jump" if is_jump else "amplified_step", **detail))
            if worst is None or output_step > max_output:
                worst, max_output = detail, output_step
        seed, previous_index = solution, index
    return dict(path=name, steps=len(references)-1, missed=missed, jumps=jumps, amplified_steps=amplified,
                max_witness_joint_step_rad=max_ref, max_tcp_step_m=max_xyz, max_tcp_step_rad=max_angle,
                max_output_joint_step_rad=max_output, worst_step=worst, events=events)


def run_ik_tests(sim, controller):
    arms = {}
    failed, path_misses, jumps, amplified = 0, 0, 0, 0
    for side in SIDES:
        if side not in controller.solvers:
            controller.run({side: sim.tcp(side)})
        solver = controller.solvers[side]
        home = sim.plant.model.key("home").qpos[sim.plant.qadr[sim.plant.arm_drives[side]]].copy()
        limits = sim.plant.limits[sim.plant.arm_drives[side]]
        coverage = reachability(sim, side, solver, home, limits)
        regular = home + np.sin(np.linspace(0., 2*np.pi, 25))[:, None] * [.035, -.020, .030, -.015, .022, .027]
        singular = np.tile(singular_posture(home), (25, 1))
        singular[:, 2] += np.linspace(-.06, .06, len(singular))
        paths = [continuity(sim, side, solver, "home_loop", regular, limits),
                 continuity(sim, side, solver, "near_singularity_crossing", singular, limits)]
        # Approach and retreat from a coupled base/wrist singular configuration.
        # Every FK witness remains strictly inside the manufacturer's wrist stop.
        inward = np.geomspace(1e-2, 1e-8, 13)
        offsets = np.concatenate((inward, inward[-2::-1]))
        for sign, name in ((-1, "coupled_wrist_min"), (1, "coupled_wrist_max")):
            wrist_path = np.tile([1.031, -.555115793677274, 1.05, .74, 0., .437], (len(offsets), 1))
            wrist_path[:, 1] += sign * offsets
            wrist_path[:, 4] = sign * (np.pi / 2 - offsets)
            paths.append(continuity(sim, side, solver, name, wrist_path, limits))
        failed += sum(coverage["witnesses"] - item["solved"] for item in coverage["counts"].values())
        path_misses += sum(path["missed"] for path in paths)
        jumps += sum(path["jumps"] for path in paths)
        amplified += sum(path["amplified_steps"] for path in paths)
        arms[side] = dict(reachability=coverage, continuity=paths)
    return dict(passed=failed == path_misses == jumps == amplified == 0, failed_seeded_trials=failed,
                missed_path_steps=path_misses, path_jumps=jumps, amplified_path_steps=amplified,
                jump_threshold_deg=float(np.rad2deg(JUMP_RAD)),
                amplified_step_threshold_deg=float(np.rad2deg(AMPLIFIED_STEP_RAD)),
                amplification_factor=AMPLIFICATION_FACTOR,
                interior_sample_seed=SAMPLE_SEED, arms=arms)

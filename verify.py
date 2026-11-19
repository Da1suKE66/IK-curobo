"""Run the packaged Firefly IK fixtures through cuRobo and verify them in MuJoCo."""
import json

import numpy as np

from simulation import SIDES, SIM_ROOT, Simulation, pose_error, within_tolerance


def verify_fixture(sim, controller, fixture):
    side = fixture["arm"]
    target = fixture["target"]
    goal = target["xyz_m"], target["quat_wxyz"]
    result = controller.run({side: goal})
    case = dict(arm=side, case=fixture["case"], status=result["status"])
    if fixture.get("expected") == "no solution":
        return dict(case, passed=result["status"] == "ik_failed")
    if result["status"] != "solved":
        return dict(case, passed=False)
    joints = result["arms"][side]["joints_rad"]
    distance, angle = pose_error(sim.fk(side, joints), goal)
    return dict(case, passed=within_tolerance((distance, angle)),
                position_error_m=distance, orientation_error_rad=angle)


def verify_execution(sim, controller):
    goals = {}
    for side in SIDES:
        position, quat = sim.tcp(side)
        goals[side] = position + [0.018, 0, 0.008], quat

    # A reachable left target together with an unreachable right target
    # must leave both live arms and their actuator commands at home.
    data = sim.plant.data
    qpos, ctrl, target, time = data.qpos.copy(), data.ctrl.copy(), sim.plant.target.copy(), data.time
    invalid = dict(goals, right=([5., 5., 5.], goals["right"][1]))
    rejected = controller.run(invalid, move=True)
    held = (rejected["status"] == "ik_failed" and data.time == time
            and np.array_equal(data.qpos, qpos) and np.array_equal(data.ctrl, ctrl)
            and np.array_equal(sim.plant.target, target))

    move = controller.run(goals, move=True)
    return [dict(case="atomic_rejection", passed=held, status=rejected["status"]),
            {**move, "case": "dual_move", "passed": move["status"] == "reached"}]


def verify(sim: Simulation, controller):
    fixtures = json.loads((SIM_ROOT / "validation" / "ik_checks.json").read_text())["cases"]
    cases = [verify_fixture(sim, controller, row) for row in fixtures]
    cases.extend(verify_execution(sim, controller))
    passed = sum(case["passed"] for case in cases)
    return dict(passed=passed == len(cases), passed_count=passed, case_count=len(cases), cases=cases)

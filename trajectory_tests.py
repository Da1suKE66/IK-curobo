"""Check moving TCPs against analytic curves between the supplied waypoints."""
import argparse
from contextlib import closing
import json
from pathlib import Path
from unittest.mock import patch

import mujoco as mj
import numpy as np

from control import Controller
from regression import Monitor
from simulation import SIDES, Simulation


def check(sim, controller, name, shape, seconds, scale, selected=SIDES, *, reachable=True):
    plant = sim.plant
    plant.reset()
    start = {s: sim.tcp(s) for s in SIDES}

    def desired(side, t):
        xyz, quat = start[side]
        if side not in selected:
            return xyz, quat
        u = np.clip(t / seconds, 0., 1.)
        phase = 2 * np.pi * u**3 * (10 - 15*u + 6*u*u)
        if shape == 'eight':
            sign = 1 if side == 'left' else -1
            offset = np.array([0., sign * .025*np.sin(phase), .015*np.sin(2*phase)])
            half = sign * np.deg2rad(12.) * np.sin(phase) / 2
            turn = np.empty(4)
            mj.mju_mulQuat(turn, quat, np.array([np.cos(half), 0., 0., np.sin(half)]))
            quat = turn
        else:
            sign = 1 if shape == 'up' else -1
            offset = np.array([.02*np.sin(phase), 0., sign*.02*(1-np.cos(phase))])
        return xyz + scale*offset, quat

    times = np.linspace(0., seconds, round(seconds/.1) + 1)
    targets = {s: [desired(s, t) for t in times] for s in selected}
    monitor = Monitor(sim, {s: desired(s, seconds) for s in SIDES}, desired, seconds)
    with patch.object(mj, 'mj_step', monitor.step):
        result = controller.track(times, targets, hold_s=4.)
    tolerances = dict(servo_tail_mm=.4, servo_tail_rad=.004, servo_tail_joint_speed_rad_s=.01,
                      max_speed_to_spec_ratio=1.01, max_joint_excursion_rad=1e-5)
    measured = monitor.score(tolerances)
    if reachable:
        fk = result['waypoint_fk']
        passed = (result['status'] == 'reached' and measured.get('passed')
                  and measured['moving_samples'] == 2 * round(seconds/.01)
                  and measured['moving_peak_mm'] < 1. and set(fk) == set(selected)
                  and all(r['max_position_error_m'] <= 1.1e-8 and r['max_orientation_error_rad'] <= 1.1e-7
                          for r in fk.values()))
    else:
        passed = (result['status'] == 'trajectory_rejected' and result['detail']['reason'] == 'ik_failed'
                  and measured['steps'] == 0 and plant.data.time == 0)
    return dict(name=name, seconds=seconds, translation_scale=scale, sides=list(selected),
                expected='reached' if reachable else 'rejected_before_motion', passed=bool(passed),
                control=result, physics=measured)


def run(sim, controller):
    return [
        check(sim, controller, 'larger_up_circle', 'up', 5.2, 1.025),
        check(sim, controller, 'smaller_eight', 'eight', 5.4, .91),
        check(sim, controller, 'right_only_down_circle', 'down', 6.4, 1.11, ('right',)),
        # With the same TCP orientation, a second up-circle radius (21.6 mm) puts
        # the two elbow links beyond full extension at t=2.4 s on the left.
        check(sim, controller, 'outside_up_circle', 'up', 5.2, 1.08, reachable=False),
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with closing(Simulation()) as sim:
        results = run(sim, Controller(sim))
    report = json.dumps(results, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end='')
    raise SystemExit(0 if all(row['passed'] for row in results) else 1)

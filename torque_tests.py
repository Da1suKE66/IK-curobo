"""Exercise both burst limits and per-motor recovery in a high-damping motor fixture."""
from contextlib import closing
import json
from unittest.mock import patch

import mujoco as mj
import numpy as np

from hardware import MAX_OVERLOAD_S
from physics_tests import StepTrace
from simulation import SIDES, Simulation


def locked_rotor_test():
    with closing(Simulation()) as sim:
        p = sim.plant
        p.model.dof_damping[:] = 1e5
        home = p.target.copy()
        trace = StepTrace(sim)
        command = home.copy()
        windows = []
        with patch.object(mj, 'mj_step', trace.step):
            for repeat in range(2):
                start = len(trace.torque)
                for side, positive in zip(SIDES, (True, False)):
                    arm = p.arm_drives[side]
                    command[arm] = p.limits[arm, int(positive)]
                    command[p.grip_drive[side]] = .6
                    # The arms enter overload at different times, across distinct API calls.
                    for _ in range(3 if positive else 30):
                        p.advance(command)
                torque = np.abs(np.asarray(trace.torque[start:]))
                above = torque > p.rated[trace.drives] + 1e-8
                windows.append(dict(above_steps=above.sum(axis=0).tolist(),
                                    last_sample_nm=torque[-1].tolist(),
                                    max_nm=torque.max(axis=0).tolist(),
                                    locked_counters=p.overload.tolist()))
                command[:] = home
                for _ in range(12):
                    p.advance(command)
                windows[-1]['rearmed'] = bool(np.all(p.overload == 0))
        summary = trace.summarize()
        n = p.burst_steps
        passed = (all(all(v == n for v in w['above_steps']) and all(v == n for v in w['locked_counters'])
                      and w['rearmed'] for w in windows)
                  and max(summary['longest_above_rated_s']) <= MAX_OVERLOAD_S
                  and not summary['peak_exceeded'])
        return dict(passed=passed, step_s=trace.dt, max_continuous_s=max(summary['longest_above_rated_s']),
                    rated_Nm=p.rated[trace.drives].tolist(), limits_Nm=p.hard_range[trace.drives, 1].tolist(),
                    windows=windows)


if __name__ == '__main__':
    result = locked_rotor_test()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)

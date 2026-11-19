"""Perturb joint-space witnesses, then submit only TCP poses from home for each trial."""
import argparse
from contextlib import closing
from copy import copy
import json
from pathlib import Path
from zipfile import ZipFile

import mujoco as mj
import numpy as np

from control import Controller
from regression import point
from simulation import SIDES, Simulation


def run(sim, controller, archive):
    p = sim.plant
    with ZipFile(archive) as z:
        prefix = 'ik_regression_25/'
        tol = json.loads(z.read(prefix+'manifest.json'))['tolerances']
        cases = {name: json.loads(z.read(prefix+f'cases/{name}.json')) for name in ('M01', 'H05', 'C04', 'C05')}
    results = {}
    for name, offsets in (('M01', (-.009, .016)), ('H05', (-.004, .004)), ('C04', (.003,)), ('C05', (-.003,))):
        for delta in offsets:
            p.reset()
            ref = np.array(cases[name]['reference']['q_rad'])
            if name == 'M01':
                ref = np.r_[ref, sim.joints('right')]
                ref[[0, 3]] += [delta, .1*delta]
            else:
                ref[[0, 6]] += [delta, -delta]
            goals = {s: sim.fk(s, ref[i*6:i*6+6]) for i, s in enumerate(SIDES)}
            # Recompute collision at the new witness, independently of the IK branch choice.
            scratch = copy(p.data)
            for i, s in enumerate(SIDES):
                scratch.qpos[p.qadr[p.arm_drives[s]]] = ref[i*6:i*6+6]
            mj.mj_fwdPosition(p.model, scratch)
            pairs = {tuple(sorted(p.model.body(p.model.geom_bodyid[g]).name for g in c.geom))
                     for c in scratch.contact if c.dist < 0}
            row = point(sim, controller, goals, tol)
            if name in ('C04', 'C05'):
                expected = {tuple(sorted(pair)) for pair in cases[name]['reference']['body_pairs']}
                safe_rejection = (row['control']['status'] == 'motion_rejected' and row['control']['reason'] == 'collision'
                                  and row['unchanged'] and row['physics']['steps'] == 0)
                row['passed'] = expected.issubset(pairs) and (safe_rejection or row['passed'])
            row['witness_contacts'] = sorted(pairs)
            results[f'{name}_{delta:+.3f}'] = row
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with closing(Simulation()) as sim:
        cases = run(sim, Controller(sim), args.archive)
    args.output.write_text(json.dumps(cases, indent=2, allow_nan=False) + '\n')
    raise SystemExit(0 if all(r['passed'] for r in cases.values()) else 1)

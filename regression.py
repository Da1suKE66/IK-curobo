"""Replay the frozen 25-case archive against the public cuRobo control paths."""
import argparse
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zipfile import ZipFile

import mujoco as mj
import numpy as np

from control import Controller
from hardware import MAX_OVERLOAD_S
from physics_tests import StepTrace
from simulation import SIDES, SIM_ROOT, Simulation, pose_error
from solver import Solver
from firefly_sim.plant import quaternion


def native(item):
    return np.array(item['xyz_m']), np.roll(item['quat_xyzw'], 1)


def fk_check(sim, side, q, goal, tol):
    if q is None:
        return dict(passed=False)
    m, rad = pose_error(sim.fk(side, q), goal)
    return dict(passed=m * 1000 <= tol['branch_fk_mm'] and rad <= tol['branch_fk_rad'],
                mm=1000 * m, rad=rad)


class Monitor(StepTrace):
    def __init__(self, sim, goals, desired=None, duration=0):
        super().__init__(sim)
        self.goals = goals
        self.desired = desired
        self.move_steps = round(duration / self.dt)
        self.probe = mj.MjData(self.model)
        self.excursion = self.peak_mm = self.peak_rad = 0.
        self.last_three = []
        self.moving_samples = 0

    def step(self, model, data):
        super().step(model, data)
        if data is not self.data:
            return
        n = len(self.torque)
        q = data.qpos[self.qadr]
        lo, hi = self.plant.limits[self.drives].T
        arm = np.tile([True]*6 + [False], 2)
        self.excursion = max(self.excursion, float(np.maximum(lo - q, q - hi)[arm].max()))
        moving = self.desired is not None and n <= self.move_steps and n % 5 == 0
        if not moving and n % self.plant.substeps:
            return
        self.probe.qpos[:] = data.qpos
        mj.mj_kinematics(model, self.probe)
        errors = []
        for side, final in self.goals.items():
            site = self.probe.site(side + '_tcp')
            target = self.desired(side, n * self.dt) if moving else final
            errors.append(pose_error((site.xpos, quaternion(site.xmat)), target))
        m, rad = np.max(errors, axis=0)
        if moving:
            self.moving_samples += len(errors)
            self.peak_mm = max(self.peak_mm, 1000*m)
            self.peak_rad = max(self.peak_rad, rad)
        else:
            self.last_three = (self.last_three + [[1000*m, rad, np.abs(data.qvel[self.dadr][arm]).max()]])[-3:]

    def score(self, tol):
        trace = self.summarize()
        if not trace['steps']:
            return dict(steps=0)
        tail = np.max(self.last_three, axis=0)
        ratio = (self.max_speed / self.rate).reshape(2, 7)[:, :6].max()
        peak = np.asarray(trace['absolute_peak_Nm']).reshape(2, 7)
        streak = max(trace['longest_above_rated_s'])
        # Check exact contacts in addition to StepTrace's penetration summaries.
        ok = (len(self.last_three) == 3 and tail[0] < tol['servo_tail_mm'] and tail[1] < tol['servo_tail_rad']
              and tail[2] < tol['servo_tail_joint_speed_rad_s'] and self.contact_steps == 0
              and not sum(self.data.warning.number) and ratio <= tol['max_speed_to_spec_ratio']
              and self.excursion <= tol['max_joint_excursion_rad'] and not trace['peak_exceeded']
              and not trace['command_slew_exceeded'] and streak <= MAX_OVERLOAD_S + 1e-12)
        return dict(passed=bool(ok), steps=trace['steps'], contacts=list(self.show_contacts.values()),
                    speed_ratio=ratio, excursion_rad=self.excursion, tail=tail.tolist(),
                    burst_s=streak, max_large_Nm=peak[:, :4].max(), max_small_Nm=peak[:, 4:].max(),
                    moving_peak_mm=self.peak_mm, moving_peak_rad=self.peak_rad, moving_samples=self.moving_samples)

    def capture_contacts(self, t):
        super().capture_contacts(t)
        if self.data.ncon and any(-1e-6 <= c.dist <= 0 for c in self.data.contact):
            self.contact_steps += 1


def point(sim, controller, goals, tol, *, fixed_joints=None):
    monitor = Monitor(sim, goals)
    p = sim.plant
    before = (p.data.qpos.copy(), p.data.ctrl.copy(), p.target.copy(), p.overload.copy(), p.data.time)
    with patch.object(mj, 'mj_step', monitor.step):
        if fixed_joints is None:
            result = controller.run(goals, move=True)
        else:
            result = sim.execute(fixed_joints, goals)
        if result['status'] == 'reached':
            for _ in range(round(4 / (p.substeps * p.model.opt.timestep))):
                p.advance(p.target)
    metrics = monitor.score(tol)
    unchanged = all(np.array_equal(a, b) for a, b in zip(before, (p.data.qpos, p.data.ctrl, p.target,
                                                                    p.overload, p.data.time)))
    return dict(control=result, physics=metrics, unchanged=unchanged,
                passed=result['status'] == 'reached' and metrics.get('passed', False))


def mesh_pairs(sim, ref, reference):
    p = sim.plant
    model, data = p.model, mj.MjData(p.model)
    mj.mj_resetDataKeyframe(model, data, model.key('home').id)
    data.qpos[p.qadr[np.concatenate([p.arm_drives[s] for s in SIDES])]] = ref['q_rad']
    mj.mj_forward(model, data)
    active = {tuple(sorted(model.body(model.geom_bodyid[g]).name for g in c.geom))
              for c in data.contact if c.dist < 0}
    objects = {}
    for name, frame in ref['source_meshes'].items():
        local = np.empty(9)
        mj.mju_quat2Mat(local, np.roll(frame['body_quat_xyzw'], 1))
        body = data.body(name)
        rot = body.xmat.reshape(3, 3)
        objects[name] = dict(mesh=str(SIM_ROOT.parent / frame['mesh']), scale=frame['scale'],
                             world_xyz_m=(body.xpos + rot @ frame['body_xyz_m']).tolist(),
                             world_rotation=(rot @ local.reshape(3, 3)).tolist())
    pairs = [dict(bodies=b, mujoco_contact=tuple(sorted(b)) in active, objects=[objects[n] for n in b])
             for b in ref['body_pairs']]
    proc = subprocess.run([reference/'.venv/bin/python', reference/'examples/check_sim_collision_meshes.py', '--stl'],
                          input=json.dumps(dict(pairs=pairs)), capture_output=True, text=True, check=True)
    return [{k: v for k, v in row.items() if k != 'objects'} for row in json.loads(proc.stdout)['pairs']]


def run_case(sim, controller, case, tol, reference):
    from evaluate_ik_motion import curve_pose
    p = sim.plant
    p.reset()
    inp, ref = case['input'], case['reference']
    kind = case['kind']
    if kind == 'sequence':
        side = inp['side']
        previous = np.array(inp['start_q_rad'])
        checks = [fk_check(sim, side, previous, native(inp['start_target_world']), tol)]
        steps = []
        for goal in map(native, inp['targets_world']):
            q = controller.solvers[side].solve(*goal, previous)
            checks.append(fk_check(sim, side, q, goal, tol))
            if q is None:
                break
            steps.append(float(np.rad2deg(np.abs(q-previous).max())))
            previous = q
        return dict(passed=all(c['passed'] for c in checks) and len(steps) == len(inp['targets_world'])
                    and max(steps) < case['expected']['max_joint_step_deg'], fk=checks, joint_steps_deg=steps)
    sides = (inp['side'],) if kind == 'single_pose' else SIDES
    if not np.allclose(np.concatenate([sim.joints(s) for s in sides]), inp['start_q_rad'], rtol=0, atol=1e-12):
        raise RuntimeError('Fixture start differs from live home')
    if kind == 'single_pose':
        side = inp['side']
        goal = native(inp['target_world'])
        q = controller.solvers[side].solve(*goal, sim.joints(side))
        check = fk_check(sim, side, q, goal, tol)
        out = dict(passed=check['passed'] and fk_check(sim, side, ref['q_rad'], goal, tol)['passed'], fk=check)
        if case['expected'].get('verify_numeric_trajectory'):
            other = next(s for s in SIDES if s != side)
            goals = {side: goal, other: native(inp['other_arm_at_home_world'])}
            out['motion'] = point(sim, controller, goals, tol)
            out['passed'] &= out['motion']['passed']
            # Replay the actual numerical branch as a fixed branch, independently of planner choice.
            if q is not None:
                p.reset()
                branch = point(sim, controller, goals, tol, fixed_joints={side: q, other: sim.joints(other)})
                out['numeric_branch_motion'] = branch
                out['passed'] &= branch['passed']
        return out
    if kind in ('point_to_point', 'contact_at_reference'):
        goals = {s: native(inp['target_world'][s]) for s in SIDES}
        out = point(sim, controller, goals, tol)
        witness = all(fk_check(sim, s, ref['q_rad'][6*i:6*i+6], goals[s], tol)['passed'] for i,s in enumerate(SIDES))
        if kind == 'contact_at_reference':
            rows = out['meshes'] = mesh_pairs(sim, ref, reference)
            hit = case['expected']['stl_intersection']
            agree = all(r['mujoco_contact'] == r['stl_surfaces_intersect'] == hit for r in rows)
            collision_rejected = (out['control']['status'] == 'motion_rejected'
                                  and out['control']['reason'] == 'collision' and out['unchanged']
                                  and out['physics']['steps'] == 0)
            out['passed'] = agree and (out['passed'] or (hit and collision_rejected))
        out['passed'] &= witness
        return out
    times, sec = inp['waypoint_times_s'], inp['seconds']
    paths = {s: list(map(native, inp['targets_world'][s])) for s in SIDES}
    bases = {s: dict(xyz_m=poses[0][0], quat_wxyz=poses[0][1]) for s, poses in paths.items()}

    def desired(side, t):
        pose = curve_pose(bases[side], side, inp['curve'], t/sec)
        return pose['xyz_m'], pose['quat_wxyz']

    for side, poses in paths.items():
        if any(max(pose_error(goal, desired(side, t))) > 1e-12 for goal, t in zip(poses, times, strict=True)):
            raise RuntimeError('Analytic tracking reference does not match frozen waypoints')
    monitor = Monitor(sim, {s: goals[-1] for s, goals in paths.items()}, desired, sec)
    with patch.object(mj, 'mj_step', monitor.step):
        result = controller.track(times, paths, hold_s=inp['hold_s'])
    measured = monitor.score(tol)
    fk = result['waypoint_fk']
    strict = (set(fk) == set(paths) and all(row['max_position_error_m']*1000 <= tol['branch_fk_mm']
              and row['max_orientation_error_rad'] <= tol['branch_fk_rad'] for row in fk.values()))
    ok = (strict and result['status'] == 'reached' and measured.get('passed')
          and measured['moving_samples'] == 2 * round(sec/.01)
          and measured['moving_peak_mm'] < case['expected']['moving_peak_position_mm_below'])
    return dict(passed=bool(ok), control=result, physics=measured)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--reference', type=Path, required=True, help='RoboControl-IK with original curve and Coal checker')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append', help='optional case ID, repeatable')
    args = parser.parse_args()
    reference = args.reference.resolve()
    sys.path.insert(0, str(reference/'examples'))
    with ZipFile(args.archive) as z:
        prefix = 'ik_regression_25/'
        for line in z.read(prefix+'SHA256SUMS').decode().splitlines():
            digest, name = line.split(maxsplit=1)
            if sha256(z.read(prefix+name)).hexdigest() != digest:
                raise ValueError(f'Checksum failed for {name}')
        manifest = json.loads(z.read(prefix+'manifest.json'))
        cases = [json.loads(z.read(prefix+row['path'])) for row in manifest['cases']]
    report = dict(archive_sha256=sha256(args.archive.read_bytes()).hexdigest(), cases={})
    with closing(Simulation()) as sim:
        controller = Controller(sim)
        controller.solvers.update({s: Solver(SIM_ROOT, s) for s in SIDES})
        for case in cases:
            name = case['id']
            if args.case and name not in args.case:
                continue
            result = run_case(sim, controller, case, manifest['tolerances'], reference)
            report['cases'][name] = result
            print(name, result['passed'], flush=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return 0 if report['cases'] and all(r['passed'] for r in report['cases'].values()) else 1


if __name__ == '__main__':
    sys.exit(main())

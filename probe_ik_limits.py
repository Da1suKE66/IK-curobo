"""Reproduce the cuRobo LM limit-margin experiment on the left Firefly arm.

Run in this directory: .venv/bin/python probe_ik_limits.py > audit_results/ik_limit_probe.json
"""
from contextlib import closing, redirect_stdout
from importlib.metadata import version
import json
import sys

import numpy as np
import torch

from ik_tests import SAMPLE_SEED, reachability
from simulation import SIM_ROOT, Simulation, pose_error, within_tolerance
from solver import Solver


class RawGpuSolver(Solver):
    """Return the unrefined cuRobo branch for the historical LM A/B experiment."""

    def solve(self, xyz_m, quat_wxyz, seed_rad):
        candidates = self._gpu_seeds(xyz_m, quat_wxyz / np.linalg.norm(quat_wxyz), seed_rad)
        if len(candidates) == 0:
            return None
        return min(candidates, key=lambda q: np.linalg.norm(q - seed_rad))


def run_group(sim, *, full_interval):
    # Recreate the solver with identical RNG initialization and target/seed order.
    torch.manual_seed(SAMPLE_SEED)
    solver = RawGpuSolver(SIM_ROOT, "left")
    lm = solver._ik.seed_ik_solver
    if not full_interval:
        # Restore upstream's soft inset just for the control group.
        lower, upper = lm.joint_limits.position
        margin = (upper - lower) * lm.config.joint_limit_margin
        lm.action_min.copy_(lower + margin)
        lm.action_max.copy_(upper - margin)
    inset = (lm.action_min - lm.joint_limits.position[0]).tolist()
    limits = sim.plant.limits[sim.plant.arm_drives["left"]]
    results = reachability(sim, "left", solver, sim.joints("left"), limits)
    roundoff = []
    for row in results["trials"]:
        for name, trial in row["seeds"].items():
            if trial["status"] != "outside_joint_limits" or trial["excess_rad"] > 1e-7:
                continue
            clamped = np.clip(trial["joints_rad"], *limits.T)
            goal = row["xyz_m"], row["quat_wxyz"]
            errors = pose_error(sim.fk("left", clamped), goal)
            roundoff.append(dict(case=row["case"], seed=name, excess_rad=trial["excess_rad"],
                                 position_error_m=errors[0], orientation_error_rad=errors[1],
                                 within_fk_tolerance=within_tolerance(errors)))
    return dict(lower_inset_rad=inset, lm_sampler_seed=lm.config.sampler_seed,
                witnesses=results["witnesses"], strict_results_by_seed=results["counts"],
                rounding_clamp_fk_checks=roundoff)


if __name__ == "__main__":
    with redirect_stdout(sys.stderr), closing(Simulation()) as sim:
        result = dict(arm="left", torch_seed=SAMPLE_SEED, roundoff_threshold_rad=1e-7,
                      curobo_version=version("nvidia-curobo"), device=torch.cuda.get_device_name(),
                      upstream_default_interval=run_group(sim, full_interval=False),
                      full_physical_interval=run_group(sim, full_interval=True))
    print(json.dumps(result, indent=2, allow_nan=False))

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


def run_group(sim, *, full_interval):
    # Recreate the solver with identical RNG initialization and target/seed order.
    torch.manual_seed(SAMPLE_SEED)
    solver = Solver(SIM_ROOT, "left")
    lm = solver._ik.seed_ik_solver
    if full_interval:
        # Modify only the LM's internal iteration bounds before its first CUDA
        # graph capture. Sampler settings and physical actuator stops stay fixed.
        lm.action_min.copy_(lm.joint_limits.position[0])
        lm.action_max.copy_(lm.joint_limits.position[1])
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
                      production_interval=run_group(sim, full_interval=False),
                      full_physical_interval=run_group(sim, full_interval=True))
    print(json.dumps(result, indent=2, allow_nan=False))

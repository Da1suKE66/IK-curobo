"""Run the motor, collision, IK coverage and 2 ms servo diagnostics."""
import hashlib
from importlib.metadata import version

import numpy as np
import torch

from collision_tests import sample_collisions
from hardware import model_paths, verify_hardware
from ik_tests import run_ik_tests
from physics_tests import run_physics_tests
from simulation import SIM_ROOT


def run_audit(sim, controller, trace_path=None):
    if trace_path is not None and trace_path.suffix != ".npz":
        raise ValueError("--trace requires a .npz filename")
    ik = run_ik_tests(sim, controller)
    hardware = verify_hardware(sim, controller)
    geometry = sample_collisions(sim)
    traces = {} if trace_path is not None else None
    physics = run_physics_tests(sim, controller, traces=traces)
    if trace_path is not None:
        np.savez_compressed(trace_path, **traces, sample_period_s=physics["sample_period_s"],
                            joint_order=np.asarray(physics["joint_order"]))
    from torque_tests import locked_rotor_test
    torque_guard = locked_rotor_test()
    cases = physics["cases"]
    findings = dict(
        hardware_mismatch=not hardware["passed"],
        torque_guard_failed=not torque_guard["passed"],
        failed_seeded_ik_trials=ik["failed_seeded_trials"],
        missed_path_steps=ik["missed_path_steps"],
        ik_path_jumps=ik["path_jumps"],
        amplified_ik_steps=ik["amplified_path_steps"],
        contact_during_execution=[c["name"] for c in cases if c["telemetry"]["collision_steps"]],
        overload_duration_exceeded=[c["name"] for c in cases if c["telemetry"]["duration_exceeded"]],
        above_peak_torque=[c["name"] for c in cases if c["telemetry"]["peak_exceeded"]],
        above_arm_speed=[c["name"] for c in cases if c["telemetry"]["velocity_exceeded"]],
        above_command_slew=[c["name"] for c in cases if c["telemetry"]["command_slew_exceeded"]],
        failed_motion_checks=[c["name"] for c in cases if not c["all_monitored_limits_met"]],
    )
    versions = {name: version(name) for name in ("mujoco", "nvidia-curobo", "torch")}
    models = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model_paths(SIM_ROOT)}
    return dict(completed=physics["completed"], findings=findings, versions=versions,
                gpu=torch.cuda.get_device_name(), model_sha256=models, hardware=hardware,
                ik=ik, static_collisions=geometry, physics=physics, torque_guard=torque_guard)

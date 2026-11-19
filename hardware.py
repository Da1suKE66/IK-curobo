"""Apply the Firefly-Y6 joint and output-shaft motor specifications to both models.

Assembly travel and joint speeds: https://docs.hexfellow.com/hex-arm/firefly_y6_data_center_cn/
J1..J4 torque: https://docs.hexfellow.com/hex-motor/hex-4342-data-center_cn/
J5..J7 torque: https://docs.hexfellow.com/hex-motor/hex-4310-data-center_cn/
"""
from functools import lru_cache
from math import radians
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET

import numpy as np


# HexFellow Firefly Y6 assembly specifications. J7 drives the gripper; the
# arm table lists six axes, so the gripper keeps the packaged travel and slew.
ARM_RANGE_DEG = ((-164, 164), (-90, 120), (0, 180), (-90, 90), (-90, 90), (-170, 170))
ARM_SPEED_DEG_S = (320, 320, 320, 320, 720, 720)
# Output-shaft ratings from HEX 4342P (J1..4) and HEX 4310 (J5..7).
# Do not multiply these torques by the built-in planetary reduction again.
RATED_TORQUE_NM = (9., 9., 9., 9., 3., 3., 3.)
PEAK_TORQUE_NM = (30., 30., 30., 30., 7., 7., 7.)


def _save(tree, path):
    contents = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    if path.is_file() and path.read_bytes() == contents:
        return
    # Other CLI processes can open the same model while this one starts.
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(contents)
            stream.flush()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


@lru_cache(maxsize=None)
def model_paths(sim_root):
    """Build matching MJCF/URDF views of the bundled workcell, returning both paths."""
    sim_root = Path(sim_root).resolve()
    source = sim_root / "model"
    output = sim_root.parent / "hardware_model"
    output.mkdir(exist_ok=True)
    scene_path, robot_path = output / "firefly.xml", output / "firefly_dual.urdf"
    scene = ET.parse(source / scene_path.name)
    robot = ET.parse(source / robot_path.name)

    for side in ("left", "right"):
        for j, peak in enumerate(PEAK_TORQUE_NM, 1):
            name = f"{side}_joint_{j}" if j <= 6 else f"{side}_grip_joint_1"
            joint = scene.find(f".//worldbody//joint[@name='{name}']")
            drive = scene.find(f"actuator/position[@joint='{name}']")
            limit = robot.find(f"joint[@name='{name}']/limit")
            if joint is None or drive is None or limit is None:
                raise ValueError(f"Missing joint, position actuator or URDF limit for {name}")
            if j <= 6:
                lower, upper = (str(radians(v)) for v in ARM_RANGE_DEG[j - 1])
                speed = str(radians(ARM_SPEED_DEG_S[j - 1]))
            else:
                lower, upper = joint.attrib["range"].split()
                speed = drive.attrib["user"]
            joint.set("range", f"{lower} {upper}")
            joint.set("actuatorfrcrange", f"{-peak:g} {peak:g}")
            drive.set("ctrlrange", f"{lower} {upper}")
            drive.set("user", speed)
            limit.attrib.update(lower=lower, upper=upper, effort=f"{peak:g}", velocity=speed)

    # Both original models refer to the calibrated meshes relative to their
    # containing directory, including a converted D435 OBJ next to the MJCF.
    for tree, attr in ((scene, "file"), (robot, "filename")):
        for mesh in tree.iter("mesh"):
            mesh.set(attr, os.path.relpath(source / mesh.attrib[attr], output))
    _save(scene, scene_path)
    _save(robot, robot_path)
    return scene_path, robot_path


def verify_hardware(sim, controller):
    """Cross-check the compiled, exported and GPU limits against the joint specifications."""
    from simulation import SIDES, SIM_ROOT

    _, robot_path = model_paths(SIM_ROOT)
    robot = ET.parse(robot_path)
    plant = sim.plant
    model = plant.model
    mismatches = []
    records = {}
    gpu_joint_count = 0

    def check(name, field, observed, expected, atol=1e-12):
        if not np.allclose(observed, expected, rtol=0., atol=atol):
            mismatches.append(dict(joint=name, field=field, observed=np.asarray(observed).tolist(),
                                   expected=np.asarray(expected).tolist()))

    for side in SIDES:
        gpu_limits = controller.solvers[side]._ik.kinematics.get_joint_limits()
        gpu_joint_count += len(gpu_limits.joint_names)
        gpu = {field: getattr(gpu_limits, field).detach().cpu().numpy()
               for field in ("position", "velocity", "effort")}
        names = [f"{side}_joint_{j}" for j in range(1, 7)]
        if set(gpu_limits.joint_names) != set(names):
            mismatches.append(dict(arm=side, field="curobo_joint_names", expected=names,
                                   observed=gpu_limits.joint_names))
        names.append(f"{side}_grip_joint_1")
        for j, name in enumerate(names):
            jid = model.joint(name).id
            aid = model.actuator(name).id
            bounds = list(map(radians, ARM_RANGE_DEG[j])) if j < 6 else [0., .6]
            speed = radians(ARM_SPEED_DEG_S[j]) if j < 6 else 1.5
            peak = PEAK_TORQUE_NM[j]
            records[name] = dict(position_rad=bounds, max_speed_rad_s=speed,
                                 rated_nm=RATED_TORQUE_NM[j], peak_nm=peak)
            check(name, "mujoco_limit_flags", [model.jnt_limited[jid], model.jnt_actfrclimited[jid],
                                               model.actuator_ctrllimited[aid]], [1, 1, 1])
            check(name, "mujoco_position", model.jnt_range[jid], bounds)
            check(name, "mujoco_control_range", model.actuator_ctrlrange[aid], bounds)
            check(name, "sim_command_slew", plant.max_step[aid] / model.opt.timestep, speed)
            check(name, "mujoco_output_torque", model.jnt_actfrcrange[jid], [-peak, peak])
            check(name, "mujoco_transmission", [model.actuator_trnid[aid, 0],
                                                model.actuator_gear[aid, 0]], [jid, 1.])
            limit = robot.find(f"joint[@name='{name}']/limit").attrib
            check(name, "urdf_position_velocity_effort",
                  [float(limit[key]) for key in ("lower", "upper", "velocity", "effort")],
                  [*bounds, speed, peak])
            if name in gpu_limits.joint_names:
                i = gpu_limits.joint_names.index(name)
                for field, expected in (("position", bounds), ("velocity", [-speed, speed]),
                                        ("effort", [-peak, peak])):
                    check(name, f"curobo_{field}", gpu[field][:, i], expected, atol=2e-6)
    check("both_arms", "mujoco_drive_count", model.nu, len(records))
    return dict(passed=not mismatches, mujoco_drives=model.nu, curobo_arm_joints=gpu_joint_count,
                specifications=records, mismatches=mismatches)

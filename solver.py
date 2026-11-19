"""GPU inverse kinematics for the two calibrated Firefly arm chains (cuRobo 0.9)."""

import xml.etree.ElementTree as ET

import numpy as np
from hardware import model_paths
from pose_refinement import ArmGeometry


def _robot_config(sim_root, side):
    """Use the exported URDF for the arm and the MuJoCo site for its tool frame."""
    scene_path, urdf_path = model_paths(sim_root)
    tip, tcp = f"{side}_link_6", f"{side}_tcp"
    scene = ET.parse(scene_path)
    site = scene.find(f".//body[@name='{tip}']/site[@name='{tcp}']")
    if site is None:
        raise ValueError(f"The Firefly scene must define {tcp} on {tip}")
    pos = [float(x) for x in site.attrib["pos"].split()]
    quat = [float(x) for x in site.get("quat", "1 0 0 0").split()]
    return {
        "kinematics": {
            "urdf_path": str(urdf_path),
            "base_link": "workcell_base",
            "tool_frames": [tcp],
            "extra_links": {
                tcp: {
                    "link_name": tcp,
                    "parent_link_name": tip,
                    "joint_name": f"{tcp}_fixed",
                    "joint_type": "FIXED",
                    "fixed_transform": pos + quat,
                },
            },
        },
    }


def _vector(values, length, name):
    array = np.asarray(values, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {length} finite numbers")
    return array


class Solver:
    """One reusable cuRobo solver per arm. Joint inputs and outputs are in radians."""

    def __init__(self, sim_root, side, *, num_seeds=32):
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        if not isinstance(num_seeds, int) or num_seeds < 1:
            raise ValueError("num_seeds must be a positive integer")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Firefly cuRobo IK requires a CUDA-enabled PyTorch runtime and a visible NVIDIA GPU")
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

        config = InverseKinematicsCfg.create(
            robot=_robot_config(sim_root, side),
            num_seeds=num_seeds,
            seed_solver_num_seeds=num_seeds,
            position_tolerance=0.0002,
            orientation_tolerance=0.002,
            self_collision_check=False,
            load_collision_spheres=False,
        )
        self._ik = InverseKinematics(config)
        # cuRobo 0.9's LM seeds use an inset at each stop. The calibrated
        # plant permits both endpoints; we recheck them in float64 below.
        lm = self._ik.seed_ik_solver
        lm.action_min.copy_(lm.joint_limits.position[0])
        lm.action_max.copy_(lm.joint_limits.position[1])
        self.geometry = ArmGeometry(model_paths(sim_root)[0], side)
        self._num_seeds = num_seeds
        self._tcp = f"{side}_tcp"
        names = [f"{side}_joint_{i}" for i in range(1, 7)]
        solver_names = self._ik.joint_names
        if sorted(solver_names) != sorted(names):
            raise ValueError(f"Unexpected cuRobo joints for {side} arm: {solver_names}")
        self._to_solver = [names.index(name) for name in solver_names]
        self._to_sim = [solver_names.index(name) for name in names]

    def _gpu_seeds(self, xyz, quat, seed):
        from curobo.types import GoalToolPose, JointState, Pose

        to_device = self._ik.device_cfg.to_device
        pose = Pose(position=to_device(xyz[None]), quaternion=to_device(quat[None]))
        goal = GoalToolPose.from_poses({self._tcp: pose}, num_goalset=1)
        current = JointState.from_position(
            to_device(seed[self._to_solver][None]), joint_names=self._ik.joint_names
        )
        result = self._ik.solve_pose(goal, current_state=current, return_seeds=self._num_seeds)
        valid = result.success[0].detach().cpu().numpy().astype(bool)
        solutions = result.solution[0].detach().cpu().numpy()[:, self._to_sim]
        return solutions[valid & np.isfinite(solutions).all(axis=1)].astype(float)

    def _candidates(self, xyz_m, quat_wxyz, seed_rad, *, all_branches):
        xyz = _vector(xyz_m, 3, "xyz_m")
        quat = _vector(quat_wxyz, 4, "quat_wxyz")
        scale = np.max(np.abs(quat))
        if scale == 0:
            raise ValueError("quat_wxyz must have a finite nonzero norm")
        quat = quat / scale
        quat /= np.linalg.norm(quat)
        seed = _vector(seed_rad, 6, "seed_rad")
        gpu = self._gpu_seeds(xyz, quat, seed)
        geometry = self.geometry
        # Feedback carries the wrist phase that a TCP on a pole cannot observe.
        local = geometry.refine(xyz, quat, seed)
        if not all_branches and local is not None and np.linalg.norm(local - seed) < .04:
            return [local]
        candidates = [] if local is None else [local]
        starts = [*geometry.planar_seeds(xyz, quat, seed), *gpu]
        starts.sort(key=lambda q: np.linalg.norm(q - seed))
        tried = []
        for guess in starts:
            if any(np.max(np.abs(guess - previous)) < 1e-6 for previous in tried):
                continue
            tried.append(guess)
            candidate = geometry.refine(xyz, quat, guess)
            if candidate is not None and all(np.max(np.abs(candidate - other)) > 1e-6 for other in candidates):
                candidates.append(candidate)
        return sorted(candidates, key=lambda q: np.linalg.norm(q - seed))

    def solve(self, xyz_m, quat_wxyz, seed_rad):
        """Continue the measured arm branch, preferring small joint displacement."""
        candidates = self._candidates(xyz_m, quat_wxyz, seed_rad, all_branches=False)
        return candidates[0] if candidates else None

    def solve_candidates(self, xyz_m, quat_wxyz, seed_rad):
        """Return distinct, verified branches ordered by distance from the seed."""
        return self._candidates(xyz_m, quat_wxyz, seed_rad, all_branches=True)

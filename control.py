"""Solve Firefly TCP goals with cuRobo and execute them on one MuJoCo instance."""
import argparse
from contextlib import closing, redirect_stdout
import json
from pathlib import Path
import sys

from simulation import (
    SIDES, SIM_ROOT, Simulation, normalized_pose, pose_error, within_tolerance,
)


def arm_state(joints, pose):
    position, orientation = pose
    return dict(joints_rad=joints.tolist(), xyz_m=position.tolist(), quat_wxyz=orientation.tolist())


class Controller:
    """Reuse one solver per requested arm and seed from its measured joints."""

    def __init__(self, sim):
        self.sim = sim
        self.solvers = {}

    def run(self, targets_by_arm, *, move=False):
        if not targets_by_arm or not set(targets_by_arm).issubset(SIDES):
            raise ValueError("Supply a TCP goal for left, right, or both arms")
        goals = {side: normalized_pose(pose) for side, pose in targets_by_arm.items()}
        solutions = {}
        arms = {}
        for side, (xyz, quat) in goals.items():
            if side not in self.solvers:
                from solver import Solver
                self.solvers[side] = Solver(SIM_ROOT, side)
            solver = self.solvers[side]
            seed = self.sim.joints(side)
            if move:
                candidates = solver.solve_candidates(xyz, quat, seed)
                if not candidates:
                    return dict(status="ik_failed", arm=side, sim_time_s=float(self.sim.plant.data.time))
                solutions[side] = candidates
                continue
            joints = solver.solve(xyz, quat, seed)
            if joints is None:
                return dict(status="ik_failed", arm=side, sim_time_s=float(self.sim.plant.data.time))
            actual = self.sim.fk(side, joints)
            errors = pose_error(actual, (xyz, quat))
            arms[side] = dict(**arm_state(joints, actual), position_error_m=errors[0],
                              orientation_error_rad=errors[1])
            if not within_tolerance(errors):
                return dict(status="candidate_fk_failed", arm=side, arms=arms)

        if move:
            from motion import move_to
            return move_to(self.sim, solutions, goals)
        return dict(status="solved", arms=arms)

    def track(self, times_s, targets_by_arm, *, hold_s=0.4):
        """Track a sequence of workcell TCP poses at the supplied timestamps."""
        from trajectory import track
        return track(self, times_s, targets_by_arm, hold_s=hold_s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-kg", nargs=2, type=float, metavar=("LEFT", "RIGHT"),
                        help="known mass at each TCP in kg (move/track/demo)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("state", help="read measured joints and TCPs at home")
    commands.add_parser("demo", help="move both TCPs by +18 mm in X and +8 mm in Z")
    commands.add_parser("verify", help="run the packaged IK cases and bilateral motion checks")
    track = commands.add_parser("track", help="follow timestamped workcell TCP waypoints")
    track.add_argument("file", type=Path, help="JSON with times_s, targets_by_arm and optional hold_s")
    audit = commands.add_parser("audit", help="test IK coverage, collisions, torque and joint continuity")
    audit.add_argument("--trace", type=Path, metavar="FILE.npz", help="save per-step motor telemetry")
    for name, help_text in (("solve", "compute and check IK"), ("move", "solve and drive the simulation")):
        command = commands.add_parser(name, help=help_text)
        for side in SIDES:
            command.add_argument(f"--{side}", type=float, nargs=7,
                                 metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
                                 help=f"{side} TCP goal in workcell_base (metres, quaternion wxyz)")
    args = parser.parse_args()
    if args.payload_kg and args.command not in ("move", "track", "demo"):
        parser.error("--payload-kg applies to move, track or demo")
    payload = dict(zip(SIDES, args.payload_kg)) if args.payload_kg else {}
    targets = {}
    if args.command in ("solve", "move"):
        for side in SIDES:
            values = getattr(args, side)
            if values is not None:
                targets[side] = (values[:3], values[3:])
        if not targets:
            parser.error("supply --left, --right, or both")

    try:
        # Keep the CLI's stdout machine-readable even on the first CUDA kernel build.
        with redirect_stdout(sys.stderr), closing(Simulation(payload_kg=payload)) as sim:
            controller = Controller(sim)
            if args.command == "state":
                result = dict(status="ready", arms={s: arm_state(sim.joints(s), sim.tcp(s)) for s in SIDES})
            elif args.command == "verify":
                from verify import verify
                result = verify(sim, controller)
            elif args.command == "track":
                packet = json.loads(args.file.read_text())
                result = controller.track(packet["times_s"], packet["targets_by_arm"], hold_s=packet.get("hold_s", 0.4))
            elif args.command == "audit":
                from audit import run_audit
                result = run_audit(sim, controller, args.trace)
            else:
                if args.command == "demo":
                    for side in SIDES:
                        xyz, quat = sim.tcp(side)
                        targets[side] = (xyz + [0.018, 0, 0.008], quat)
                result = controller.run(targets, move=args.command != "solve")
    except (ValueError, KeyError, RuntimeError, ImportError, OSError) as exc:
        print(json.dumps(dict(status="error", message=str(exc)), ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if args.command == "verify":
        return 0 if result["passed"] else 1
    if args.command == "audit":
        return 0 if result["completed"] and not any(result["findings"].values()) else 1
    return 0 if result["status"] in ("ready", "solved", "reached") else 1


if __name__ == "__main__":
    sys.exit(main())

"""Sample simultaneous arm configurations using the workcell's MuJoCo collision geoms."""
from collections import Counter

import mujoco
import numpy as np


def contact_kind(model, contact):
    bodies = [model.body(int(model.geom_bodyid[gid])).name for gid in (contact.geom1, contact.geom2)]
    sides = [name.partition("_")[0] for name in bodies]
    arm_count = sum(side in ("left", "right") for side in sides)
    if arm_count == 2:
        kind = "self" if sides[0] == sides[1] else "inter_arm"
    else:
        kind = "environment" if arm_count else "fixed"
    return kind, *bodies


def sample_collisions(sim):
    plant = sim.plant
    drives = np.stack([plant.arm_drives[s] for s in ("left", "right")])
    addresses = plant.qadr[drives]
    home = plant.model.key("home").qpos.copy()
    scratch = mujoco.MjData(plant.model)
    scratch.qpos[:] = home
    arm_home = home[addresses]
    threshold_m = 1e-6

    def contacts_at(joints):
        scratch.qpos[addresses] = joints
        mujoco.mj_fwdPosition(plant.model, scratch)
        contacts = {}
        for contact in scratch.contact:
            if contact.dist >= -threshold_m:
                continue
            kind, a, b = contact_kind(plant.model, contact)
            if kind != "fixed":
                contacts.setdefault(kind, dict(bodies=[a, b], penetration_m=float(-contact.dist)))
        return contacts

    seed, count = 4310, 256
    rng = np.random.default_rng(seed)
    lower, upper = plant.limits[drives].transpose(2, 0, 1)
    counts = Counter()
    witnesses = {}
    clear_endpoints = []
    for index in range(count):
        q = rng.uniform(lower, upper)
        contacts = contacts_at(q)
        if not contacts:
            clear_endpoints.append(q)
        for kind, contact in contacts.items():
            counts[kind] += 1
            if kind not in witnesses:
                witnesses[kind] = dict(sample=index, joints_rad=q.tolist(), **contact)

    # Verify whether clear endpoints suffice for a straight joint-space segment.
    home_contacts = contacts_at(arm_home)
    sweep = None
    paths_checked = 0
    if not home_contacts:
        for endpoint in clear_endpoints:
            paths_checked += 1
            segments = max(1, int(np.ceil(np.max(np.abs(endpoint - arm_home)) / np.deg2rad(1))))
            for step in range(1, segments):
                fraction = step / segments
                contacts = contacts_at(arm_home + fraction * (endpoint - arm_home))
                if contacts:
                    sweep = dict(from_joints_rad=arm_home.tolist(), to_joints_rad=endpoint.tolist(),
                                 fraction=fraction, contacts=contacts)
                    break
            if sweep:
                break
    return dict(seed=seed, samples=count, joint_order=["left", "right"],
                threshold_m=threshold_m, any_penetration_samples=count - len(clear_endpoints),
                counts_by_kind=dict(counts), example_by_kind=witnesses, home_contacts=home_contacts,
                clear_endpoint_paths_checked=paths_checked, max_joint_increment_deg=1,
                clear_endpoints_with_intermediate_penetration=sweep)

"""Double-precision TCP checks and bounded branch refinement for the Firefly arm."""

import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np


# Residuals applied after clipping to the actual float64 actuator interval.
POSITION_TOL_M = 5e-9
ANGLE_TOL_RAD = 3e-8


def _rotate_y(pitch):
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotate_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class ArmGeometry:
    """Scratch FK/Jacobian of the calibrated MJCF chain, independent of the live plant."""

    def __init__(self, scene, side):
        root = ET.parse(scene).getroot()
        tcp = side + "_tcp"
        site = root.find(f".//site[@name='{tcp}']")
        parents = {child: parent for parent in root.iter() for child in parent}
        nodes = []
        node = parents[site]
        while node.tag == "body":
            nodes.append(node)
            node = parents[node]
        # A single 6-DoF chain needs neither meshes nor the second arm for FK.
        chain = ET.Element("mujoco")
        chain.append(root.find("compiler"))
        parent = ET.SubElement(chain, "worldbody")
        for node in reversed(nodes):
            parent = ET.SubElement(parent, "body", node.attrib)
            for child in node:
                if child.tag in ("joint", "inertial"):
                    parent.append(child)
        parent.append(site)
        self.model = mj.MjModel.from_xml_string(ET.tostring(chain, encoding="unicode"))
        self.data = mj.MjData(self.model)
        self.lower, self.upper = self.model.jnt_range.T.copy()
        self.site = self.data.site(tcp)
        mj.mj_kinematics(self.model, self.data)
        base = self.data.body(side + "_base_link")
        self.base_position = base.xpos.copy()
        self.base_rotation = base.xmat.reshape(3, 3).copy()
        bodies = [self.model.body(f"{side}_link_{j}") for j in range(1, 7)]
        self.offsets = np.stack([body.pos for body in bodies])
        self.tool = self.model.site(tcp).pos.copy()
        self.tool_rotation = np.empty(9)
        mj.mju_quat2Mat(self.tool_rotation, self.model.site(tcp).quat)
        self.tool_rotation = self.tool_rotation.reshape(3, 3)
        # Only this axis arrangement admits the planar decomposition below.
        axes = [[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        self.planar = (np.allclose(self.model.jnt_axis, axes, atol=1e-13, rtol=0)
                       and np.max(np.abs(self.model.jnt_pos)) < 1e-13
                       and np.max(np.abs(self.offsets[1:, 1])) < 1e-13
                       and all(np.linalg.norm(body.quat[1:]) < 1e-13 for body in bodies))

    def _wrist_pitches(self, center, local, bend, seed_pitch):
        """Sample each pitch interval between reach, joint-limit and roll boundaries."""
        _, _, p3, p4, p5, p6 = self.offsets
        tool = p5 + [p6[0], 0., 0.]
        l3, l4 = np.linalg.norm(p3), np.linalg.norm(p4)
        c, s = np.cos(bend), np.sin(bend)
        roll_at_zero = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ local
        roll = np.arctan2(roll_at_zero[1, 0], roll_at_zero[0, 0])
        angles = [seed_pitch, *(np.sign(bend) * (roll - self.model.jnt_range[5]))]

        def add_tangencies(origin, vector, radius):
            # Roots of |origin - Ry(pitch) vector| = radius in the XZ plane.
            dot = origin @ vector
            cross = origin[0] * vector[2] - origin[2] * vector[0]
            amplitude = np.hypot(dot, cross)
            if amplitude < 1e-15:
                return
            cosine = (origin @ origin + vector @ vector - radius**2) / (2 * amplitude)
            if abs(cosine) <= 1 + 1e-7:
                phase, half_width = np.arctan2(cross, dot), np.arccos(np.clip(cosine, -1., 1.))
                angles.extend([phase - half_width, phase + half_width])

        for radius in (l3 + l4, abs(l3 - l4)):
            add_tangencies(center, tool, radius)
        for j2 in self.model.jnt_range[1]:
            add_tangencies(center - _rotate_y(j2) @ p3, tool, l4)
        for j3 in self.model.jnt_range[2]:
            add_tangencies(center, tool, np.linalg.norm(p3 + _rotate_y(j3) @ p4))
        for j4 in self.model.jnt_range[3]:
            add_tangencies(center, tool + _rotate_y(-j4) @ p4, l3)
        edges = np.unique(np.mod(angles, 2 * np.pi))
        midpoints = (edges + np.r_[edges[1:], edges[0] + 2*np.pi]) / 2
        return np.r_[edges, midpoints]

    def planar_seeds(self, xyz, quat, seed):
        """Use the flange Z axis and a two-link triangle to seed both elbow branches."""
        if not self.planar:
            return
        rotation = np.empty(9)
        mj.mju_quat2Mat(rotation, quat)
        flange = self.base_rotation.T @ rotation.reshape(3, 3) @ self.tool_rotation.T
        p1, p2, p3, p4, p5, p6 = self.offsets
        wrist = self.base_rotation.T @ (xyz - self.base_position) - flange @ self.tool - p1
        wrist -= flange[:, 2] * p6[2]
        yaw = np.arctan2(wrist[1], wrist[0])
        yaws = [yaw, yaw + np.pi]
        if np.hypot(*wrist[:2]) < 1e-8:
            yaws.extend([seed[0], *np.linspace(self.lower[0], self.upper[0], 7)])
        length3, length4 = np.linalg.norm(p3), np.linalg.norm(p4)
        tilt3, tilt4 = np.arctan2(p3[0], p3[2]), np.arctan2(p4[0], p4[2])
        for yaw in yaws:
            local = _rotate_z(yaw).T @ flange
            direction = local[:, 2]
            radial = np.hypot(direction[0], direction[2])
            pitch = np.arctan2(direction[0], direction[2])
            bend = np.arctan2(direction[1], radial)
            wrists = [(pitch, bend), (pitch + np.pi, np.pi - bend)]
            center = _rotate_z(yaw).T @ wrist - p2
            center[1] = 0.  # Project numerical noise onto the shoulder-elbow plane.
            if radial < 1e-7:
                # At the wrist pole, pitch can trade off with J6. Include the
                # measured pitch to preserve continuity across adjacent targets.
                wrists.extend((p, bend) for p in self._wrist_pitches(center, local, bend, sum(seed[1:4])))
            for pitch, bend in wrists:
                ry = _rotate_y(pitch)
                point = center - ry @ (p5 + [p6[0], 0., 0.])
                cosine = (point @ point - length3**2 - length4**2) / (2 * length3 * length4)
                # Tiny errors in wrist phase magnify near the stretched elbow.
                # Project onto the tangent, then test the original full pose.
                if abs(cosine) > 1 + 1e-6:
                    continue
                elbow = np.arccos(np.clip(cosine, -1., 1.))
                c, s = np.cos(bend), np.sin(bend)
                rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
                residual = (ry @ rx).T @ local
                roll = np.arctan2(residual[1, 0], residual[0, 0])
                for turn in (elbow, -elbow):
                    shoulder = (np.arctan2(point[0], point[2]) - tilt3
                                - np.arctan2(length4*np.sin(turn), length3 + length4*np.cos(turn)))
                    j3 = turn + tilt3 - tilt4
                    q = np.array([yaw, shoulder, j3, pitch - shoulder - j3, bend, roll])
                    q += 2*np.pi * np.rint(((self.lower + self.upper) * .5 - q) / (2*np.pi))
                    if np.all(q >= self.lower - 1e-7) and np.all(q <= self.upper + 1e-7):
                        yield np.clip(q, self.lower, self.upper)

    def residual(self, xyz, quat, q):
        self.data.qpos[:] = q
        mj.mj_kinematics(self.model, self.data)
        rotation = np.empty(4)
        mj.mju_mat2Quat(rotation, self.site.xmat)
        rotation[1:] *= -1
        difference = np.empty(4)
        mj.mju_mulQuat(difference, quat, rotation)
        omega = np.empty(3)
        mj.mju_quat2Vel(omega, difference, 1.)
        return np.r_[xyz - self.site.xpos, omega]

    def refine(self, xyz, quat, initial):
        """Newton correction with a bounded step and shortest-angle world-frame residual."""
        q = np.clip(initial, self.lower, self.upper)
        error = self.residual(xyz, quat, q)
        jac = np.empty((6, 6))
        for _ in range(32):
            if np.linalg.norm(error[:3]) <= POSITION_TOL_M and np.linalg.norm(error[3:]) <= ANGLE_TOL_RAD:
                return q
            mj.mj_comPos(self.model, self.data)
            mj.mj_jacSite(self.model, self.data, jac[:3], jac[3:], self.site.id)
            delta = np.linalg.lstsq(jac, error, rcond=1e-8)[0]
            delta /= max(1., np.max(np.abs(delta)) / .35)
            loss = error @ error
            for fraction in (1., .5, .25, .125, .0625, .03125):
                next_q = np.clip(q + fraction*delta, self.lower, self.upper)
                next_error = self.residual(xyz, quat, next_q)
                if next_error @ next_error < loss:
                    q, error = next_q, next_error
                    break
            else:
                return None
        return None

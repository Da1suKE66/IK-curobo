"""Conservative convex parts for the concave elbow links, cut from the source STL."""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull


def _clip(triangles, axis, plane, direction):
    """Clip triangles to a half-space, triangulating each clipped quadrilateral."""
    inside = (triangles[:, :, axis] - plane) * direction >= 0
    counts = inside.sum(axis=1)
    kept = [triangles[counts == 3]]
    for count in (1, 2):
        partial = counts == count
        faces = triangles[partial]
        unique_corner = inside[partial] if count == 1 else ~inside[partial]
        row, column = np.nonzero(unique_corner)
        a, b, c = (faces[row, (column + i) % 3] for i in range(3))
        # a lies on the opposite side of the plane from both b and c.
        ab, ac = (a + (v - a) * ((plane - a[:, axis]) / (v[:, axis] - a[:, axis]))[:, None]
                  for v in (b, c))
        if count == 1:
            kept.append(np.stack([a, ab, ac], axis=1))
        else:
            kept.extend([np.stack([b, c, ab], axis=1), np.stack([c, ac, ab], axis=1)])
    return np.concatenate(kept)


def _convex_parts(path, divisions):
    dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    with path.open('rb') as stream:
        stream.seek(80)
        count = int.from_bytes(stream.read(4), 'little')
        triangles = np.fromfile(stream, dtype, count=count)['vertices'].astype(float)
    groups = [triangles]
    for axis, bins in divisions:
        bounds = np.linspace(triangles[:, :, axis].min(), triangles[:, :, axis].max(), bins + 1)
        sections = []
        for group in groups:
            for lo, hi in zip(bounds, bounds[1:]):
                section = _clip(_clip(group, axis, lo, 1), axis, hi, -1)
                if len(section):
                    sections.append(section)
        groups = sections
    for group in groups:
        # Partitions span the unsliced axis. The hull covers the clipped triangle
        # surfaces and the solid between opposing surfaces in the same column.
        points = np.unique(group.reshape(-1, 3), axis=0)
        yield points[ConvexHull(points).vertices]


def refine_collision_meshes(scene, model_dir):
    assets = scene.find("asset")
    # The link-3 fork has two separated cheeks along local Y. Splitting only
    # along its length bridges that opening when the wrist folds against it.
    for name, divisions in (("link_3", ((2, 6), (1, 3))), ("link_5", ((0, 6),))):
        mesh = assets.find(f"mesh[@name='{name}']")
        parts = []
        for i, vertices in enumerate(_convex_parts(model_dir / mesh.attrib["file"], divisions)):
            part = f"{name}_collision_{i}"
            ET.SubElement(assets, "mesh", name=part, vertex=" ".join(f"{v:.10g}" for v in vertices.flat))
            parts.append(part)
        for body in scene.iter("body"):
            for geom in body.findall(f"geom[@mesh='{name}']"):
                if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                    continue
                body.remove(geom)
                for part in parts:
                    ET.SubElement(body, "geom", {**geom.attrib, "mesh": part})

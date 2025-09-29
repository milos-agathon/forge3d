# examples/f11_subdivision_demo.py
# Demonstrate subdivision (midpoint splits) on a simple mesh and export to OBJ.

import os
import tempfile

import numpy as np

import forge3d.io as fio
from forge3d.geometry import MeshBuffers, subdivide_mesh


def main() -> None:
    # Unit square (two triangles)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    uvs = positions[:, :2].copy()
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    mesh = MeshBuffers(positions=positions, normals=np.empty((0, 3), np.float32), uvs=uvs, indices=indices)

    for levels in (1, 2):
        out = subdivide_mesh(mesh, levels=levels)
        print(f"Levels={levels} -> vertices={out.vertex_count}, triangles={out.triangle_count}")

    # Save highest level as OBJ
    out = subdivide_mesh(mesh, levels=2)
    with tempfile.TemporaryDirectory() as td:
        obj_path = os.path.join(td, "subdiv2.obj")
        fio.save_obj(out, obj_path)
        print("Wrote:", obj_path)


if __name__ == "__main__":
    main()

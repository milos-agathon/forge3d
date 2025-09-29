# examples/f6_stl_export_demo.py
# Export a tetrahedron to binary STL and validate watertightness.

import os
import tempfile

import numpy as np

import forge3d.io as fio
from forge3d.geometry import MeshBuffers


def main() -> None:
    # Tetrahedron (4 faces)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.uint32,
    )
    mesh = MeshBuffers(
        positions=positions,
        normals=np.empty((0, 3), np.float32),
        uvs=np.empty((0, 2), np.float32),
        indices=indices,
    )

    with tempfile.TemporaryDirectory() as td:
        stl_path = os.path.join(td, "tetra.stl")
        watertight = fio.save_stl(mesh, stl_path, validate=True)
        print("Wrote:", stl_path)
        print("Watertight:", watertight)


if __name__ == "__main__":
    main()

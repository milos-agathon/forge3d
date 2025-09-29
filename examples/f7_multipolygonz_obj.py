# examples/f7_multipolygonz_obj.py
# Convert MultiPolygonZ rings to a mesh, unwrap UVs, and export OBJ.

import os
import tempfile

import numpy as np

import forge3d.io as fio
import forge3d.converters as conv
from forge3d.geometry import MeshBuffers, unwrap_planar


def main() -> None:
    # Two disjoint squares as simple MultiPolygonZ
    r1 = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float32)
    r2 = np.array([[2,0,0],[3,0,0],[3,1,0],[2,1,0]], dtype=np.float32)

    mesh = conv.multipolygonz_to_mesh([r1, r2])

    # Planar unwrap along Z
    mesh_u = unwrap_planar(mesh, axis=2)

    with tempfile.TemporaryDirectory() as td:
        obj_path = os.path.join(td, "multipolygon.obj")
        fio.save_obj(mesh_u, obj_path)
        print("Wrote:", obj_path)


if __name__ == "__main__":
    main()

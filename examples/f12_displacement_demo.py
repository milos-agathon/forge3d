# examples/f12_displacement_demo.py
# Demonstrate heightmap and procedural displacement on a mesh.

import os
import tempfile

import numpy as np

import forge3d.io as fio
from forge3d.geometry import primitive_mesh, displace_heightmap, displace_procedural


def main() -> None:
    # Base plane mesh
    mesh = primitive_mesh("plane", resolution=(32, 32))

    # Heightmap: radial bump
    H, W = 128, 128
    y = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    Y, X = np.meshgrid(y, x, indexing="ij")
    R = np.sqrt(X * X + Y * Y)
    hm = np.exp(-4.0 * R * R).astype(np.float32)  # gaussian bump

    mesh_hm = displace_heightmap(mesh, hm, scale=0.25)
    mesh_proc = displace_procedural(mesh, amplitude=0.1, frequency=6.0)

    print("Base:", mesh.vertex_count, mesh.triangle_count)
    print("Heightmap:", mesh_hm.vertex_count, mesh_hm.triangle_count)
    print("Procedural:", mesh_proc.vertex_count, mesh_proc.triangle_count)

    # Save both variants as OBJ
    with tempfile.TemporaryDirectory() as td:
        path_hm = os.path.join(td, "plane_heightmap.obj")
        path_proc = os.path.join(td, "plane_procedural.obj")
        fio.save_obj(mesh_hm, path_hm)
        fio.save_obj(mesh_proc, path_proc)
        print("Wrote:", path_hm)
        print("Wrote:", path_proc)


if __name__ == "__main__":
    main()

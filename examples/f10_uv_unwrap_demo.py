# examples/f10_uv_unwrap_demo.py
# Demonstrate planar and spherical unwrap on a unit cube.

import numpy as np

from forge3d.geometry import MeshBuffers, unwrap_planar, unwrap_spherical


def main() -> None:
    positions = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ], dtype=np.float32)
    indices = np.array([
        [0,1,2],[0,2,3],
        [4,5,6],[4,6,7],
        [0,1,5],[0,5,4],
        [2,3,7],[2,7,6],
        [1,2,6],[1,6,5],
        [3,0,4],[3,4,7],
    ], dtype=np.uint32)
    mesh = MeshBuffers(
        positions=positions,
        normals=np.empty((0,3), np.float32),
        uvs=np.empty((0,2), np.float32),
        indices=indices,
    )

    planar = unwrap_planar(mesh, axis=2)
    spherical = unwrap_spherical(mesh)

    # Print summary of UV ranges
    print("Planar UV min/max:", planar.uvs.min(), planar.uvs.max())
    print("Spherical UV min/max:", spherical.uvs.min(), spherical.uvs.max())


if __name__ == "__main__":
    main()

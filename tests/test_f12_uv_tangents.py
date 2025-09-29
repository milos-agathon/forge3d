# tests/test_f12_uv_tangents.py
# Validate UV-space displacement equivalence on planar unwrap and tangent generation

import numpy as np
import pytest

from forge3d.geometry import (
    primitive_mesh,
    unwrap_planar,
    displace_heightmap,
    generate_tangents,
)


@pytest.mark.geometry
def test_uv_space_matches_xy_for_planar_unwrap() -> None:
    mesh = primitive_mesh("plane", resolution=(16, 16))
    mesh = unwrap_planar(mesh, axis=2)

    H, W = 64, 64
    x = np.linspace(0, 2 * np.pi, W, dtype=np.float32)
    y = np.linspace(0, 2 * np.pi, H, dtype=np.float32)
    Y, X = np.meshgrid(y, x, indexing="ij")
    hm = (np.sin(X) * np.cos(Y)).astype(np.float32)

    a = displace_heightmap(mesh, hm, scale=0.05, uv_space=False)
    b = displace_heightmap(mesh, hm, scale=0.05, uv_space=True)

    # With planar unwrap on XY plane, UV-based sampling should closely match XY-based
    assert np.allclose(a.positions, b.positions, atol=1e-5), "UV-space displacement should match XY sampling on planar unwrap"


@pytest.mark.geometry
def test_generate_tangents_output_valid() -> None:
    mesh = primitive_mesh("plane", resolution=(8, 8))
    mesh = unwrap_planar(mesh, axis=2)
    tans = generate_tangents(mesh)

    assert tans.shape == (mesh.vertex_count, 4)
    # Tangent length ~1 for xyz
    lengths = np.linalg.norm(tans[:, :3], axis=1)
    assert np.all(lengths > 0.5)
    assert np.all(lengths < 1.5)

    # Tangents orthogonal to normals
    dots = np.sum(tans[:, :3] * mesh.normals, axis=1)
    assert np.all(np.abs(dots) < 1e-3)

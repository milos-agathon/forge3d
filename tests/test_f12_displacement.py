# tests/test_f12_displacement.py
# Displacement modifiers: heightmap and procedural

import numpy as np
import pytest

from forge3d.geometry import MeshBuffers, primitive_mesh, displace_heightmap, displace_procedural


@pytest.mark.geometry
def test_displace_heightmap_on_plane() -> None:
    # Generate a unit plane with some resolution
    mesh = primitive_mesh("plane", resolution=(8, 8))
    # Heightmap: simple sinusoid
    H, W = 32, 32
    y = np.linspace(0, np.pi * 2, H, dtype=np.float32)
    x = np.linspace(0, np.pi * 2, W, dtype=np.float32)
    Y, X = np.meshgrid(y, x, indexing="ij")
    hm = (np.sin(X) * np.cos(Y)).astype(np.float32)

    out = displace_heightmap(mesh, hm, scale=0.1)

    # Geometry unchanged in index topology
    assert out.indices.shape == mesh.indices.shape
    # Z should vary now
    assert np.std(out.positions[:, 2]) > 0.0
    # Normals preserved in count and normalized
    assert out.normals.shape == (out.vertex_count, 3)
    assert np.all(np.linalg.norm(out.normals, axis=1) > 0)


@pytest.mark.geometry
def test_displace_procedural_changes_positions() -> None:
    mesh = primitive_mesh("plane", resolution=(4, 4))
    out = displace_procedural(mesh, amplitude=0.05, frequency=2.0)
    # Positions changed along Z or other axes based on normals
    assert not np.allclose(out.positions, mesh.positions)

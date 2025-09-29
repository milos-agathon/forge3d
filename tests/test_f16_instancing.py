import numpy as np
import pytest

from forge3d.geometry import primitive_mesh, instance_mesh


@pytest.mark.geometry
def test_f16_instancing_basic_counts_and_transform():
    # Base: unit box
    base = primitive_mesh("box")
    assert base.vertex_count > 0
    tri0 = base.triangle_count

    # Two instances: identity and +X translation
    I = np.eye(4, dtype=np.float32).reshape(-1)
    T = I.copy()
    # Row-major 4x4: translation occupies the last column indices (3,7,11)
    T[3] = 1.0  # translate x by +1
    T[7] = 0.0
    T[11] = 0.0
    transforms = np.stack([I, T], axis=0).astype(np.float32)

    inst = instance_mesh(base, transforms)

    # Counts
    assert inst.vertex_count == base.vertex_count * 2
    assert inst.triangle_count == tri0 * 2

    # First half equals base
    n = base.vertex_count
    assert np.allclose(inst.positions[:n], base.positions, atol=1e-6)
    # Second half equals base translated by +X
    assert np.allclose(inst.positions[n:, 0], base.positions[:, 0] + 1.0, atol=1e-6)
    assert np.allclose(inst.positions[n:, 1], base.positions[:, 1], atol=1e-6)
    assert np.allclose(inst.positions[n:, 2], base.positions[:, 2], atol=1e-6)

    # Normals preserved under pure translation
    if base.normals.size:
        assert np.allclose(inst.normals[:n], base.normals, atol=1e-6)
        assert np.allclose(inst.normals[n:], base.normals, atol=1e-6)

# tests/test_f5_obj_export.py
# Minimal OBJ export test (F5) and round-trip via import.

import os
import tempfile

import numpy as np
import pytest

import forge3d.io as fio
from forge3d.geometry import MeshBuffers

_requires_native = pytest.mark.skipif(
    getattr(fio, "_forge3d", None) is None,
    reason="forge3d native extension is not available",
)


@_requires_native
@pytest.mark.io
def test_export_then_import_roundtrip() -> None:
    # Build a tiny triangle mesh in Python
    positions = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]], dtype=np.float32)
    uvs = np.array([[0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    mesh = MeshBuffers(positions=positions, normals=normals, uvs=uvs, indices=indices)

    with tempfile.TemporaryDirectory() as td:
        obj_path = os.path.join(td, "tri_export.obj")
        fio.save_obj(mesh, obj_path)
        assert os.path.exists(obj_path)

        mesh2, materials, material_groups, g_groups, o_groups = fio.load_obj(obj_path, return_metadata=True)
        # Basic shape checks
        assert mesh2.vertex_count == 3
        assert mesh2.triangle_count == 1
        assert mesh2.positions.shape == (3, 3)
        assert mesh2.indices.shape == (1, 3)
        # Fidelity checks
        assert np.allclose(mesh2.positions, positions)
        assert np.allclose(mesh2.uvs, uvs)
        assert np.allclose(mesh2.normals, normals)
        # No materials were exported, so metadata groups should be empty
        assert isinstance(materials, list)
        assert len(materials) == 0
        assert isinstance(material_groups, dict) and len(material_groups) == 0
        assert isinstance(g_groups, dict) and len(g_groups) == 0
        assert isinstance(o_groups, dict) and len(o_groups) == 0

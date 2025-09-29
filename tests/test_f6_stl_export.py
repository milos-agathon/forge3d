# tests/test_f6_stl_export.py
# STL export tests (binary), with watertightness validation.

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
def test_stl_export_tetrahedron_watertight() -> None:
    # Tetrahedron (4 faces)
    positions = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [0.0, 0.0, 1.0],  # 3
    ], dtype=np.float32)
    indices = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ], dtype=np.uint32)
    mesh = MeshBuffers(positions=positions, normals=np.empty((0,3), np.float32), uvs=np.empty((0,2), np.float32), indices=indices)

    with tempfile.TemporaryDirectory() as td:
        stl_path = os.path.join(td, "tetra.stl")
        watertight = fio.save_stl(mesh, stl_path, validate=True)
        assert os.path.exists(stl_path)
        assert watertight is True

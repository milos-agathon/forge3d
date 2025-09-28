# tests/test_f4_obj_import_neg_indices.py
# OBJ import tests covering negative indices per OBJ spec.

import os
import tempfile

import numpy as np
import pytest

import forge3d.io as fio

_requires_native = pytest.mark.skipif(
    getattr(fio, "_forge3d", None) is None,
    reason="forge3d native extension is not available",
)


@_requires_native
@pytest.mark.io
def test_negative_indices_positions_only() -> None:
    with tempfile.TemporaryDirectory() as td:
        obj_path = os.path.join(td, "neg_positions.obj")
        with open(obj_path, "w", encoding="utf-8") as f:
            f.write(
                """
# 3 vertices, one face with negative indices
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f -3 -2 -1
""".strip()
            )
        mesh = fio.load_obj(obj_path)
        assert mesh.vertex_count == 3
        assert mesh.triangle_count == 1
        assert mesh.positions.shape == (3, 3)
        assert mesh.indices.shape == (1, 3)


@_requires_native
@pytest.mark.io
def test_negative_indices_full_vt_vn() -> None:
    with tempfile.TemporaryDirectory() as td:
        obj_path = os.path.join(td, "neg_full.obj")
        with open(obj_path, "w", encoding="utf-8") as f:
            f.write(
                """
# vertices
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
# texcoords
vt 0.0 0.0
vt 1.0 0.0
vt 0.0 1.0
# single normal
vn 0.0 0.0 1.0
# face uses negative indices for v/vt and references last normal via -1
f -3/-3/-1 -2/-2/-1 -1/-1/-1
""".strip()
            )
        mesh = fio.load_obj(obj_path)
        assert mesh.vertex_count == 3
        assert mesh.triangle_count == 1
        assert mesh.uvs.shape == (3, 2)
        assert mesh.normals.shape == (3, 3)

# tests/test_f4_obj_import_polygons.py
# OBJ import tests covering quads/ngons triangulation and missing UV/normal cases.

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
def test_quad_triangulation_positions_only() -> None:
    with tempfile.TemporaryDirectory() as td:
        obj_path = os.path.join(td, "quad.obj")
        with open(obj_path, "w", encoding="utf-8") as f:
            f.write(
                """
# quad in CCW order
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3 4
""".strip()
            )
        mesh = fio.load_obj(obj_path)
        assert mesh.triangle_count == 2
        assert mesh.indices.shape == (2, 3)


@_requires_native
@pytest.mark.io
def test_ngon_triangulation_with_vt_vn() -> None:
    with tempfile.TemporaryDirectory() as td:
        obj_path = os.path.join(td, "ngon.obj")
        with open(obj_path, "w", encoding="utf-8") as f:
            f.write(
                """
# pentagon with vt/vn
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.2 0.7 0.0
v 0.5 1.2 0.0
v -0.2 0.7 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.5 1.2
vt 0.0 1.0
vn 0.0 0.0 1.0
f 1/1/1 2/2/1 3/3/1 4/4/1 5/5/1
""".strip()
            )
        mesh = fio.load_obj(obj_path)
        assert mesh.triangle_count == 3
        assert mesh.uvs.shape[0] == mesh.positions.shape[0]
        assert mesh.normals.shape[0] == mesh.positions.shape[0]


@_requires_native
@pytest.mark.io
def test_missing_uv_and_normal() -> None:
    with tempfile.TemporaryDirectory() as td:
        obj_path = os.path.join(td, "missing_uv_normal.obj")
        with open(obj_path, "w", encoding="utf-8") as f:
            f.write(
                """
# triangle without vt/vn
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
""".strip()
            )
        mesh = fio.load_obj(obj_path)
        assert mesh.vertex_count == 3
        assert mesh.uvs.shape == (0, 2)
        assert mesh.normals.shape == (0, 3)

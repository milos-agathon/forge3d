# tests/test_f4_obj_import.py
# Minimal OBJ import test (F4) with tiny inline fixtures.

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
def test_load_obj_minimal_triangle() -> None:
    with tempfile.TemporaryDirectory() as td:
        obj_path = os.path.join(td, "tri.obj")
        mtl_path = os.path.join(td, "tri.mtl")

        with open(mtl_path, "w", encoding="utf-8") as f:
            f.write(
                """
# minimal material
newmtl red
Kd 1.0 0.0 0.0
Ka 0.0 0.0 0.0
Ks 0.0 0.0 0.0
map_Kd diffuse.png
""".strip()
            )

        with open(obj_path, "w", encoding="utf-8") as f:
            f.write(
                """
mtllib tri.mtl
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 0.0 1.0
vn 0.0 0.0 1.0
usemtl red
f 1/1/1 2/2/1 3/3/1
""".strip()
            )

        mesh, materials, material_groups, g_groups, o_groups = fio.load_obj(obj_path, return_metadata=True)
        assert mesh.vertex_count == 3
        assert mesh.triangle_count == 1
        assert mesh.positions.shape == (3, 3)
        assert mesh.indices.shape == (1, 3)
        # UVs and normals present
        assert mesh.uvs.shape == (3, 2)
        assert mesh.normals.shape == (3, 3)

        # Materials parsed from MTL
        assert len(materials) == 1
        mat = materials[0]
        assert mat.name == "red"
        assert np.allclose(mat.diffuse_color, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert np.allclose(mat.ambient_color, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert np.allclose(mat.specular_color, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert mat.diffuse_texture == "diffuse.png"

        # Groups contain triangle ordinals for the material
        assert "red" in material_groups
        assert material_groups["red"].dtype == np.uint32
        assert material_groups["red"].tolist() == [0]

        # No g/o groups in this fixture
        assert isinstance(g_groups, dict) and len(g_groups) == 0
        assert isinstance(o_groups, dict) and len(o_groups) == 0

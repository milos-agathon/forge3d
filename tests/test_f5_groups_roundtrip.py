# tests/test_f5_groups_roundtrip.py
# OBJ export/import round-trip with materials and g/o groups.

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
def test_groups_and_materials_roundtrip() -> None:
    # Build a simple quad split into two triangles, assign different groups and materials
    positions = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
    ], dtype=np.float32)
    uvs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1))
    # Two triangles: (0,1,2) and (0,2,3)
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    mesh = MeshBuffers(positions=positions, normals=normals, uvs=uvs, indices=indices)

    # Materials and groups
    red = fio.ObjMaterial(name="red", diffuse_color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                          ambient_color=np.zeros(3, dtype=np.float32),
                          specular_color=np.zeros(3, dtype=np.float32),
                          diffuse_texture=None)
    green = fio.ObjMaterial(name="green", diffuse_color=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                            ambient_color=np.zeros(3, dtype=np.float32),
                            specular_color=np.zeros(3, dtype=np.float32),
                            diffuse_texture="albedo.png")

    material_groups = {"red": np.array([0], dtype=np.uint32), "green": np.array([1], dtype=np.uint32)}
    g_groups = {"left": np.array([0], dtype=np.uint32), "right": np.array([1], dtype=np.uint32)}
    o_groups = {"quad": np.array([0, 1], dtype=np.uint32)}

    with tempfile.TemporaryDirectory() as td:
        obj_path = os.path.join(td, "quad.obj")
        fio.save_obj(
            mesh,
            obj_path,
            materials=[red, green],
            material_groups=material_groups,
            g_groups=g_groups,
            o_groups=o_groups,
        )
        assert os.path.exists(obj_path)
        # The MTL file should be written
        assert os.path.exists(os.path.join(td, "quad.mtl"))

        mesh2, materials2, mat_groups2, g_groups2, o_groups2 = fio.load_obj(obj_path, return_metadata=True)
        # Mesh fidelity
        assert np.allclose(mesh2.positions, positions)
        assert np.allclose(mesh2.uvs, uvs)
        assert np.allclose(mesh2.normals, normals)
        assert np.allclose(mesh2.indices, indices)

        # Materials round-trip (by name presence; order is not guaranteed)
        names = {m.name for m in materials2}
        assert names == {"red", "green"}

        # Material groups preserved
        assert set(mat_groups2.keys()) == {"red", "green"}
        assert mat_groups2["red"].tolist() == [0]
        assert mat_groups2["green"].tolist() == [1]

        # g/o groups preserved
        assert set(g_groups2.keys()) == {"left", "right"}
        assert g_groups2["left"].tolist() == [0]
        assert g_groups2["right"].tolist() == [1]
        assert set(o_groups2.keys()) == {"quad"}
        assert o_groups2["quad"].tolist() == [0, 1]

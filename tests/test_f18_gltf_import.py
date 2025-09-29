import base64
import json
import os
import tempfile

import numpy as np
import pytest

from forge3d.io import import_gltf


def _make_minimal_gltf_triangle(tmpdir: str) -> str:
    # Positions (3 vertices, float32)
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint16)

    # Pack buffers: positions then indices
    pos_bytes = positions.tobytes(order="C")
    idx_bytes = indices.tobytes(order="C")
    # glTF requires buffer length to cover both
    buffer_data = pos_bytes + idx_bytes
    uri = "data:application/octet-stream;base64," + base64.b64encode(buffer_data).decode("ascii")

    # Buffer and views
    buffer = {"uri": uri, "byteLength": len(buffer_data)}
    # Positions start at 0
    bv_positions = {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bytes), "target": 34962}  # ARRAY_BUFFER
    # Indices start after positions
    bv_indices = {"buffer": 0, "byteOffset": len(pos_bytes), "byteLength": len(idx_bytes), "target": 34963}  # ELEMENT_ARRAY_BUFFER

    # Accessors
    acc_positions = {
        "bufferView": 0,
        "componentType": 5126,  # FLOAT
        "count": 3,
        "type": "VEC3",
        "min": [0.0, 0.0, 0.0],
        "max": [1.0, 1.0, 0.0],
    }
    acc_indices = {
        "bufferView": 1,
        "componentType": 5123,  # UNSIGNED_SHORT
        "count": 3,
        "type": "SCALAR",
    }

    # Primitive
    prim = {
        "attributes": {"POSITION": 0},
        "indices": 1,
        "mode": 4  # TRIANGLES
    }

    # Assemble document
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [buffer],
        "bufferViews": [bv_positions, bv_indices],
        "accessors": [acc_positions, acc_indices],
        "meshes": [{"primitives": [prim]}],
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "scene": 0,
    }

    path = os.path.join(tmpdir, "triangle.gltf")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return path


@pytest.mark.io
def test_f18_gltf_import_minimal_triangle():
    with tempfile.TemporaryDirectory() as td:
        path = _make_minimal_gltf_triangle(td)
        mesh = import_gltf(path)
        assert mesh.positions.shape[0] == 3
        assert mesh.indices.shape[0] == 1
        # Primitive triangle shape
        tri = mesh.indices[0]
        assert set(tri.tolist()) == {0, 1, 2}

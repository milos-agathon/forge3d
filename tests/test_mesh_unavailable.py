import numpy as np
import pytest
from forge3d import mesh


def test_no_fabricated_bvh_or_upload(monkeypatch):
    monkeypatch.setattr(mesh, "get_native_symbol", lambda _name: None)
    vertices, indices = mesh.create_triangle_mesh()
    data = mesh.make_mesh(vertices, indices)
    bvh = mesh.build_bvh_cpu(data)
    assert bvh["triangle_count"] == 1
    assert bvh["node_count"] == bvh["leaf_count"] == 1
    assert bvh["nodes"].size == 40
    np.testing.assert_array_equal(bvh["tri_indices"], [0])
    handle = mesh.MeshHandle(data, bvh)
    assert handle.uploaded is False
    assert handle.gpu_memory_bytes == 0
    with pytest.raises(RuntimeError, match="GPU mesh upload requires the compiled forge3d extension"):
        mesh.upload_mesh(data, bvh)

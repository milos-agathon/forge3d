"""Contract and behavior tests for `forge3d.mesh` BVH construction and upload.

`build_bvh_cpu` must report a measured median-split tree (not the retired
`(triangle_count + 3) // 4 * 2 - 1` estimate) and `upload_mesh` must allocate
real tracked GPU buffers or fail loudly. The pure-Python fallback must agree
with the native builder byte for byte.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

import forge3d as f3d
from forge3d import mesh as M
from forge3d._native import get_native_symbol

_BVH_NODE_BYTES = 40


def _triangle_mesh():
    vertices, indices = M.create_triangle_mesh()
    return M.make_mesh(vertices, indices)


def _quad_mesh():
    vertices, indices = M.create_quad_mesh()
    return M.make_mesh(vertices, indices)


def _cube_mesh():
    vertices, indices = M.create_cube_mesh()
    return M.make_mesh(vertices, indices)


def _clustered_mesh(clusters: int = 12):
    vertices = []
    indices = []
    for cluster in range(clusters):
        base = len(vertices)
        x = cluster * 1000.0
        vertices.extend([[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 1.0, 0.0]])
        indices.append([base, base + 1, base + 2])
    return M.make_mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )


def _packed_nodes(bvh):
    nodes = np.asarray(bvh["nodes"], dtype=np.uint8).reshape(-1)
    assert nodes.size % _BVH_NODE_BYTES == 0
    return nodes


def _node_records(bvh):
    """Parse the packed 40-byte nodes into their `BvhNode` fields."""
    nodes = _packed_nodes(bvh)
    records = []
    for offset in range(0, nodes.size, _BVH_NODE_BYTES):
        (
            min_x, min_y, min_z, left,
            max_x, max_y, max_z, right,
            flags, pad,
        ) = struct.unpack_from("<3f I 3f I I I", nodes, offset)
        records.append(
            {
                "aabb_min": (min_x, min_y, min_z),
                "left": left,
                "aabb_max": (max_x, max_y, max_z),
                "right": right,
                "flags": flags,
                "pad": pad,
            }
        )
    return records


# ---------------------------------------------------------------------------
# build_bvh_cpu: a measured tree over the real mesh
# ---------------------------------------------------------------------------


def test_build_bvh_cpu_reports_a_real_tree():
    bvh = M.build_bvh_cpu(_cube_mesh())

    assert bvh["method"] == "median"
    assert bvh["triangle_count"] == 12
    assert bvh["node_count"] > 0
    # A full binary tree: internal nodes are leaves minus one.
    assert bvh["node_count"] == 2 * bvh["leaf_count"] - 1
    assert bvh["internal_count"] == bvh["leaf_count"] - 1
    assert bvh["max_depth"] >= 1
    assert bvh["avg_leaf_size"] == pytest.approx(12.0 / bvh["leaf_count"])

    # Memory is the packed node array plus the triangle order, not an estimate.
    assert bvh["memory_usage_bytes"] == (
        bvh["node_count"] * _BVH_NODE_BYTES + bvh["triangle_count"] * 4
    )
    assert _packed_nodes(bvh).size == bvh["node_count"] * _BVH_NODE_BYTES

    # The triangle order is a permutation of every triangle.
    order = np.sort(np.asarray(bvh["tri_indices"], dtype=np.uint32))
    assert np.array_equal(order, np.arange(12, dtype=np.uint32))

    # The world AABB is the real mesh extent, not the vertex array extent.
    assert bvh["world_aabb_min"] == [0.0, 0.0, 0.0]
    assert bvh["world_aabb_max"] == [1.0, 1.0, 1.0]


def test_build_bvh_cpu_counts_leaves_that_cover_every_triangle_once():
    bvh = M.build_bvh_cpu(_cube_mesh())
    coverage = [0] * bvh["triangle_count"]
    leaf_count = 0
    for node in _node_records(bvh):
        if node["flags"] & 1:
            leaf_count += 1
            # Leaf bounds must enclose the triangles the leaf claims.
            assert all(node["aabb_min"][axis] <= node["aabb_max"][axis] for axis in range(3))
            for offset in range(node["right"]):
                coverage[node["left"] + offset] += 1
    assert leaf_count == bvh["leaf_count"]
    assert all(seen == 1 for seen in coverage), coverage


def test_build_bvh_cpu_is_not_the_retired_formula():
    bvh = M.build_bvh_cpu(_clustered_mesh(12))
    triangle_count = bvh["triangle_count"]
    retired_leaves = (triangle_count + 3) // 4
    retired_nodes = retired_leaves * 2 - 1

    assert bvh["node_count"] != retired_nodes
    assert bvh["node_count"] == 2 * bvh["leaf_count"] - 1
    assert bvh["max_depth"] > 1


def test_build_bvh_cpu_rejects_invalid_requests():
    cube = _cube_mesh()
    with pytest.raises(ValueError):
        M.build_bvh_cpu(cube, method="sah")
    with pytest.raises(ValueError):
        M.build_bvh_cpu(cube, max_leaf_size=0)
    with pytest.raises(ValueError):
        M.build_bvh_cpu({"vertices": cube["vertices"], "indices": cube["indices"], "triangle_count": 0})


# ---------------------------------------------------------------------------
# The pure-Python fallback must agree with the native builder
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    get_native_symbol("mesh_build_bvh_cpu") is None,
    reason="native BVH builder not available",
)
@pytest.mark.parametrize(
    "mesh_factory", [_triangle_mesh, _quad_mesh, _cube_mesh, _clustered_mesh]
)
def test_python_fallback_matches_native_build(mesh_factory):
    mesh = mesh_factory()
    native = M.build_bvh_cpu(mesh)
    fallback = M._python_build_bvh(mesh["vertices"], mesh["indices"], 4)

    for key in ("triangle_count", "node_count", "leaf_count", "internal_count", "max_depth"):
        assert native[key] == fallback[key], key
    assert np.array_equal(_packed_nodes(native), _packed_nodes(fallback))
    assert np.array_equal(
        np.asarray(native["tri_indices"], dtype=np.uint32),
        np.asarray(fallback["tri_indices"], dtype=np.uint32),
    )


# ---------------------------------------------------------------------------
# upload_mesh: real tracked buffers, or a loud failure
# ---------------------------------------------------------------------------


def test_mesh_handle_reports_upload_state_without_uploading():
    handle = M.MeshHandle(_cube_mesh(), M.build_bvh_cpu(_cube_mesh()))
    assert handle.uploaded is False
    assert handle.gpu_memory_bytes == 0


def test_upload_mesh_requires_native_extension(monkeypatch):
    monkeypatch.setattr(M, "get_native_symbol", lambda name: None)
    with pytest.raises(RuntimeError):
        M.upload_mesh(_cube_mesh(), M.build_bvh_cpu(_cube_mesh()))


def test_upload_mesh_rejects_bvh_missing_packed_data():
    mesh = _cube_mesh()
    bvh = M.build_bvh_cpu(mesh)
    stripped = {key: value for key, value in bvh.items() if key not in ("nodes", "tri_indices")}
    with pytest.raises(ValueError):
        M.upload_mesh(mesh, stripped)


def test_upload_mesh_rejects_mismatched_triangle_counts():
    mesh = _cube_mesh()
    bvh = M.build_bvh_cpu(mesh)
    bvh = dict(bvh)
    bvh["triangle_count"] = bvh["triangle_count"] + 1
    with pytest.raises(ValueError):
        M.upload_mesh(mesh, bvh)


@pytest.mark.skipif(
    get_native_symbol("mesh_build_bvh_cpu") is None,
    reason="native BVH builder not available",
)
def test_native_builder_rejects_wrongly_shaped_arrays():
    native = get_native_symbol("mesh_build_bvh_cpu")
    # A 4-wide vertex array must not be read as the first three columns.
    with pytest.raises(ValueError):
        native(np.zeros((3, 4), dtype=np.float32), np.zeros((1, 3), dtype=np.uint32), "median", 4)
    # A 2-wide index array must not panic on the third column.
    with pytest.raises(ValueError):
        native(np.zeros((3, 3), dtype=np.float32), np.zeros((1, 2), dtype=np.uint32), "median", 4)


@pytest.mark.skipif(
    get_native_symbol("mesh_upload_gpu") is None,
    reason="native mesh upload not available",
)
def test_upload_mesh_rejects_a_duplicate_triangle_order():
    mesh = _cube_mesh()
    bvh = M.build_bvh_cpu(mesh)
    duplicated = np.array(bvh["tri_indices"], dtype=np.uint32)
    duplicated[0] = duplicated[1]  # duplicate one triangle, drop another
    corrupt = dict(bvh, tri_indices=duplicated)
    with pytest.raises(ValueError):
        M.upload_mesh(mesh, corrupt)


@pytest.mark.skipif(
    get_native_symbol("mesh_upload_gpu") is None,
    reason="native mesh upload not available",
)
def test_upload_mesh_rejects_a_corrupt_bvh_before_allocating():
    mesh = _cube_mesh()
    bvh = M.build_bvh_cpu(mesh)
    corrupt = dict(bvh)
    corrupt["nodes"] = np.zeros(7, dtype=np.uint8)  # not a multiple of 40 bytes
    with pytest.raises(ValueError):
        M.upload_mesh(mesh, corrupt)


@pytest.mark.skipif(not f3d.has_gpu(), reason="no GPU adapter available")
def test_upload_mesh_allocates_real_tracked_buffers():
    mesh = _cube_mesh()
    bvh = M.build_bvh_cpu(mesh)

    handle = M.upload_mesh(mesh, bvh)

    assert handle.uploaded is True
    assert handle.gpu_memory_bytes > 0
    assert handle.triangle_count == 12
    assert handle.vertex_count == 8
    assert handle.node_count == bvh["node_count"]

    info = M.mesh_info(handle)
    assert info["uploaded"] is True
    assert info["gpu_memory_bytes"] == handle.gpu_memory_bytes
    assert info["bvh_node_count"] == bvh["node_count"]


def test_mesh_info_reports_unuploaded_handles():
    handle = M.MeshHandle(_cube_mesh(), M.build_bvh_cpu(_cube_mesh()))
    info = M.mesh_info(handle)
    assert info["uploaded"] is False
    assert info["gpu_memory_bytes"] == 0
    assert info["world_bounds"]["min"] == [0.0, 0.0, 0.0]

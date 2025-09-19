#!/usr/bin/env python3
"""GPU test for LBVH node invariants after link (P7).

Gated by FORGE3D_RUN_WAVEFRONT=1, cargo present, and a GPU adapter.
Runs the Rust example `accel_lbvh_refit` to produce out/lbvh_nodes.bin and
out/lbvh_indices.bin, then validates basic invariants:
- node_count == 2*prim_count - 1
- internal nodes (0..prim_count-2) have kind=0 and valid children
- leaves (prim_count-1..node_count-1) have kind=1, right=1, left < prim_count
- there is exactly one root internal node with parent=0xffffffff
- for each internal node i, children have parent==i
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import forge3d

REPO_ROOT = Path(__file__).resolve().parents[1]


def _has_gpu_adapter() -> bool:
    try:
        return bool(forge3d.enumerate_adapters())
    except Exception:
        return False


@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_WAVEFRONT", "0") != "1"
    or shutil.which("cargo") is None
    or not _has_gpu_adapter(),
    reason="GPU run disabled unless FORGE3D_RUN_WAVEFRONT=1, cargo present, and a GPU adapter is available",
)
def test_lbvh_nodes_invariants():
    cwd = str(REPO_ROOT)
    cmd = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "accel_lbvh_refit",
    ]
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    assert res.returncode == 0, f"accel_lbvh_refit failed: {res.stderr}"

    nodes_path = REPO_ROOT / "out" / "lbvh_nodes.bin"
    indices_path = REPO_ROOT / "out" / "lbvh_indices.bin"
    assert nodes_path.exists(), f"Missing nodes dump: {nodes_path}"
    assert indices_path.exists(), f"Missing indices dump: {indices_path}"

    # Load indices to determine primitive count
    indices = np.fromfile(indices_path, dtype=np.uint32)
    prim_count = int(indices.size)
    assert prim_count >= 2, "Expect at least 2 primitives for test"

    # Define BvhNode dtype (matches Rust repr(C))
    # struct Aabb { min[3]f32; pad0 f32; max[3]f32; pad1 f32; } => 32B
    # struct BvhNode { Aabb (32); kind u32; left u32; right u32; parent u32; } => 48B
    node_dtype = np.dtype([
        ("min", "<f4", (3,)), ("_pad0", "<f4"),
        ("max", "<f4", (3,)), ("_pad1", "<f4"),
        ("kind", "<u4"), ("left", "<u4"), ("right", "<u4"), ("parent", "<u4"),
    ])
    raw = np.fromfile(nodes_path, dtype=np.uint8)
    # Ensure size is multiple of node size
    assert raw.size % node_dtype.itemsize == 0, "Node dump size not aligned to BvhNode size"
    node_count = raw.size // node_dtype.itemsize
    nodes = raw.view(node_dtype)

    # Invariant: node_count == 2*prim_count - 1
    assert node_count == 2 * prim_count - 1, (
        f"node_count={node_count}, prim_count={prim_count} => expected {2*prim_count-1}"
    )

    internal_count = prim_count - 1
    leaf_start = internal_count
    MAX_U32 = np.uint32(0xFFFFFFFF)

    # Internal nodes range [0 .. prim_count-2]
    internals = nodes[:internal_count]
    assert np.all(internals["kind"] == 0), "All internal nodes must have kind=0"

    # Leaves range [prim_count-1 .. node_count-1]
    leaves = nodes[leaf_start:]
    assert np.all(leaves["kind"] == 1), "All leaves must have kind=1"
    assert np.all(leaves["right"] == 1), "All leaves must have right_idx=1 (primitive count)"
    assert np.all(leaves["left"] < prim_count), "Leaf left_idx must be primitive index < prim_count"

    # Exactly one root internal node (parent == 0xffffffff)
    internal_parents = internals["parent"]
    roots = np.where(internal_parents == MAX_U32)[0]
    assert roots.size == 1, f"Expected exactly one root internal node, found {roots.size}"

    # For each internal node, children must point back to this node as parent
    for i in range(internal_count):
        left = int(internals["left"][i])
        right = int(internals["right"][i])
        assert 0 <= left < node_count, f"Invalid left child index at internal {i}: {left}"
        assert 0 <= right < node_count, f"Invalid right child index at internal {i}: {right}"
        assert int(nodes["parent"][left]) == i, f"Child left parent mismatch at internal {i}"
        assert int(nodes["parent"][right]) == i, f"Child right parent mismatch at internal {i}"

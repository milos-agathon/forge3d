#!/usr/bin/env python3
"""GPU test for minimal LBVH GPU build + refit (P7).

Gated by FORGE3D_RUN_WAVEFRONT=1, cargo present, and a GPU adapter.
Runs the Rust example `accel_lbvh_refit` and asserts that the printed world AABB
changes after refit when we perturb triangle vertices.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import forge3d

REPO_ROOT = Path(__file__).resolve().parents[1]


def _has_gpu_adapter() -> bool:
    try:
        return bool(forge3d.enumerate_adapters())
    except Exception:
        return False


def _parse_aabb(line: str) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    # Expected: AABB before: min=[x,y,z], max=[x,y,z]
    m = re.search(r"min=\[([\d\.+\-eE]+),([\d\.+\-eE]+),([\d\.+\-eE]+)\], max=\[([\d\.+\-eE]+),([\d\.+\-eE]+),([\d\.+\-eE]+)\]", line)
    if not m:
        return None
    vals = [float(g) for g in m.groups()]
    return (vals[0], vals[1], vals[2]), (vals[3], vals[4], vals[5])


@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_WAVEFRONT", "0") != "1"
    or shutil.which("cargo") is None
    or not _has_gpu_adapter(),
    reason="GPU run disabled unless FORGE3D_RUN_WAVEFRONT=1, cargo present, and a GPU adapter is available",
)
def test_lbvh_refit_updates_world_aabb():
    cwd = str(REPO_ROOT)
    cmd = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "accel_lbvh_refit",
    ]
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    assert res.returncode == 0, f"accel_lbvh_refit failed: {res.stderr}"

    before = None
    after = None
    for line in (res.stdout or "").splitlines():
        if "AABB before:" in line:
            parsed = _parse_aabb(line)
            assert parsed is not None, f"Failed to parse before AABB: {line}"
            before = parsed
        elif "AABB after:" in line:
            parsed = _parse_aabb(line)
            assert parsed is not None, f"Failed to parse after AABB: {line}"
            after = parsed

    assert before is not None and after is not None, "Missing before/after AABB logs"

    # At minimum, world max should expand after the vertex perturbation
    (bmin, bmax) = before
    (amin, amax) = after
    assert amax[0] >= bmax[0] and amax[1] >= bmax[1], (
        f"Expected world AABB to expand after refit: before={before}, after={after}"
    )
    # And at least one component should strictly increase
    assert (amax[0] > bmax[0]) or (amax[1] > bmax[1]) or (amax[2] > bmax[2]), (
        f"Expected some expansion: before={before}, after={after}"
    )

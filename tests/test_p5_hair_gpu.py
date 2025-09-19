#!/usr/bin/env python3
"""GPU tests for P5 Hair: verify hair demo affects the rendered image.

Gated by FORGE3D_RUN_WAVEFRONT=1 and FORGE3D_CI_GPU=1.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import forge3d

from .test_p2_validation import REPO_ROOT


@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_WAVEFRONT", "0") != "1"
    or os.environ.get("FORGE3D_CI_GPU", "0") != "1"
    or shutil.which("cargo") is None
    or not forge3d.enumerate_adapters(),
    reason="GPU run disabled unless FORGE3D_RUN_WAVEFRONT=1 and FORGE3D_CI_GPU=1, cargo present, and GPU adapter available",
)
def test_hair_demo_changes_image(tmp_path: Path):
    cwd = str(REPO_ROOT)
    img_path = REPO_ROOT / "out" / "wavefront_instances.png"

    # Baseline
    cmd_base = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances",
    ]
    r0 = subprocess.run(cmd_base, cwd=cwd, capture_output=True, text=True)
    assert r0.returncode == 0, f"Baseline failed: {r0.stderr}"
    base = img_path.read_bytes()

    # With hair demo enabled
    cmd_hair = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances", "--", "--hair-demo",
    ]
    r1 = subprocess.run(cmd_hair, cwd=cwd, capture_output=True, text=True)
    assert r1.returncode == 0, f"Hair run failed: {r1.stderr}"
    hair = img_path.read_bytes()

    assert base != hair, "Hair demo should change the rendered image"

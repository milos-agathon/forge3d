#!/usr/bin/env python3
"""GPU test for P6 QMC/Owen + adaptive SPP in wavefront raygen.

Gated by FORGE3D_RUN_WAVEFRONT=1, cargo present, and at least one GPU adapter.
Runs the Python example wrapper twice (Halton/VDC vs Sobol) and asserts the
rendered PNG differs when forcing a single sample per pixel via --spp-limit.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

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
    reason="Wavefront run disabled unless FORGE3D_RUN_WAVEFRONT=1, cargo present, and a GPU adapter is available",
)
def test_qmc_mode_and_spp_limit_change_image(tmp_path: Path):
    cwd = str(REPO_ROOT)
    img_path = REPO_ROOT / "out" / "wavefront_instances.png"

    # Baseline: Halton/VDC with spp-limit=1
    cmd_a = [
        sys.executable,
        "examples/wavefront_instances.py",
        "--qmc-mode=0",
        "--spp-limit=1",
    ]
    r0 = subprocess.run(cmd_a, cwd=cwd, capture_output=True, text=True)
    assert r0.returncode == 0, f"Baseline run failed: {r0.stderr}"
    assert img_path.exists(), f"Expected output not found: {img_path}"
    a_bytes = img_path.read_bytes()
    (tmp_path / "qmc0.png").write_bytes(a_bytes)

    # Sobol: qmc-mode=1 with same spp-limit=1
    cmd_b = [
        sys.executable,
        "examples/wavefront_instances.py",
        "--qmc-mode=1",
        "--spp-limit=1",
    ]
    r1 = subprocess.run(cmd_b, cwd=cwd, capture_output=True, text=True)
    assert r1.returncode == 0, f"Sobol run failed: {r1.stderr}"
    assert img_path.exists(), f"Expected output not found after Sobol: {img_path}"
    b_bytes = img_path.read_bytes()
    (tmp_path / "qmc1.png").write_bytes(b_bytes)

    # Images should differ between Halton/VDC vs Sobol jitter at 1 spp
    assert a_bytes != b_bytes, "Expected QMC mode change to modify the rendered image"

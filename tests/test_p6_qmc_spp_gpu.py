#!/usr/bin/env python3
"""GPU test for P6: Verify spp-limit clamp changes image under same QMC mode.

Gated by FORGE3D_RUN_WAVEFRONT=1, cargo present, and at least one GPU adapter.
Runs the Python example wrapper twice with Sobol (qmc-mode=1) but different
spp-limit values (1 vs 4) and asserts the rendered PNG differs.
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
def test_spp_limit_changes_image_under_same_qmc(tmp_path: Path):
    cwd = str(REPO_ROOT)
    img_path = REPO_ROOT / "out" / "wavefront_instances.png"

    def run_with_spp_limit(limit: int) -> bytes:
        cmd = [
            sys.executable,
            "examples/wavefront_instances.py",
            "--qmc-mode=1",  # Sobol
            f"--spp-limit={int(limit)}",
        ]
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        assert r.returncode == 0, f"Run failed (spp={limit}): {r.stderr}"
        assert img_path.exists(), f"Expected output not found: {img_path}"
        return img_path.read_bytes()

    a = run_with_spp_limit(1)
    (tmp_path / "sobol_spp1.png").write_bytes(a)

    b = run_with_spp_limit(4)
    (tmp_path / "sobol_spp4.png").write_bytes(b)

    assert a != b, "Expected spp-limit change (1 vs 4) to modify the rendered image under same QMC mode"

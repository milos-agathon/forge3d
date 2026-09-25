"""Run SIDERA's Rust GPU render and adapter-qualified night certificate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parent.parent
CERTIFICATE = ROOT / "tests/golden/sidera/night_certificate.json"


def test_sidera_night_gpu_golden_is_repeatable() -> None:
    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    assert len(certificate["rgba_sha256"]) == 64
    env = dict(os.environ)
    env["WGPU_BACKEND"] = certificate["backend"].lower()
    result = subprocess.run(
        ["cargo", "test", "--lib", "fixed_night_sky_gpu_golden_is_repeatable", "--", "--nocapture"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    if "SIDERA night golden ABSENT" in output:
        pytest.skip("SIDERA certified GPU frame unavailable on this runner")
    assert f"RGBA SHA-256 {certificate['rgba_sha256']}" in output, output

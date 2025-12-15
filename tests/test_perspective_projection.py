"""CLI-level perspective projection checks using terrain_demo."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest


def _render_with_probe(
    camera_mode: str,
    fov: float,
    theta: float,
    output_name: str,
    *,
    phi: float = 135.0,
    debug_mode: int = 41,
) -> str:
    """Render terrain with projection probe and return MD5 hash."""
    output_dir = Path(__file__).parent.parent / "examples" / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "examples" / "terrain_demo.py"),
        "--size",
        "128",
        "128",
        "--hdr",
        str(Path(__file__).parent.parent / "assets" / "hdri" / "snow_field_4k.hdr"),
        "--camera-mode",
        camera_mode,
        "--cam-fov",
        str(fov),
        "--cam-theta",
        str(theta),
        "--cam-phi",
        str(phi),
        "--debug-mode",
        str(debug_mode),
        "--shadows",
        "none",
        "--msaa",
        "1",
        "--output",
        str(output_path),
        "--overwrite",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if result.returncode != 0:
        pytest.fail(f"Render failed: {result.stderr[:500]}")

    return hashlib.md5(output_path.read_bytes()).hexdigest()


class TestPerspectiveProjectionCli:
    """Ensure CLI plumbing preserves perspective controls."""

    def test_mesh_mode_fov_changes_probe(self):
        """FOV variation should change NDC-depth probe in mesh mode."""
        h1 = _render_with_probe("mesh", fov=30, theta=45, output_name="cli_probe_fov30.png")
        h2 = _render_with_probe("mesh", fov=90, theta=45, output_name="cli_probe_fov90.png")
        assert h1 != h2

    def test_mesh_mode_theta_changes_probe(self):
        """Theta variation should change probe."""
        h1 = _render_with_probe("mesh", fov=55, theta=25, output_name="cli_probe_theta25.png")
        h2 = _render_with_probe("mesh", fov=55, theta=75, output_name="cli_probe_theta75.png")
        assert h1 != h2

    def test_mesh_mode_phi_rotates_probe(self):
        """Phi variation should rotate probe output."""
        h1 = _render_with_probe("mesh", fov=55, theta=45, phi=0.0, output_name="cli_probe_phi0.png")
        h2 = _render_with_probe("mesh", fov=55, theta=45, phi=90.0, output_name="cli_probe_phi90.png")
        assert h1 != h2

    def test_screen_vs_mesh_differ(self):
        """Legacy screen mode should differ from mesh mode for same angles."""
        h_screen = _render_with_probe("screen", fov=55, theta=45, output_name="cli_probe_screen.png", debug_mode=41)
        h_mesh = _render_with_probe("mesh", fov=55, theta=45, output_name="cli_probe_mesh.png", debug_mode=41)
        assert h_screen != h_mesh

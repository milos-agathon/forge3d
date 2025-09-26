#!/usr/bin/env python3
# tests/test_text3d_basic.py
# Small test for 3D text mesh pipeline (native). Verifies non-zero vertices/indices.

import os
from pathlib import Path
import pytest

try:
    from forge3d._forge3d import Scene  # native
except Exception:
    Scene = None

pytestmark = pytest.mark.skipif(Scene is None, reason="native module not available")


def _find_font_bytes() -> bytes | None:
    repo = Path(__file__).parent.parent
    font_path = repo / "assets" / "fonts" / "Roboto-Regular.ttf"
    if font_path.is_file():
        return font_path.read_bytes()
    # try common system locations
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        str(Path.home() / "Library/Fonts/Arial.ttf"),
    ):
        if os.path.isfile(p):
            return Path(p).read_bytes()
    return None


def test_text3d_build_and_stats():
    font_bytes = _find_font_bytes()
    if font_bytes is None:
        pytest.skip("No test font found; place one at assets/fonts/Roboto-Regular.ttf")

    s = Scene(320, 240, grid=32, colormap="terrain")
    s.enable_text_meshes()
    s.add_text_mesh(
        "Test",
        font_bytes,
        size_px=48.0,
        depth=0.15,
        position=(0.0, 0.0, 0.0),
        color=(1.0, 1.0, 1.0, 1.0),
    )
    instances, verts, inds = s.get_text_mesh_stats()
    assert instances >= 1
    assert verts > 0
    assert inds > 0

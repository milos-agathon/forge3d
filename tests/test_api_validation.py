# T01-BEGIN:tests
import os
import json
import pathlib
import pytest

import vulkan_forge as vf

def test_public_exports_exist():
    assert hasattr(vf, "Renderer")
    assert hasattr(vf, "render_triangle_rgba")
    assert hasattr(vf, "render_triangle_png")
    assert hasattr(vf, "__version__")

def test_size_validation_errors(tmp_path):
    with pytest.raises(ValueError):
        vf.render_triangle_png(tmp_path/"x.png", 0, 10)
    with pytest.raises(ValueError):
        vf.render_triangle_png(tmp_path/"x.png", 10, -1)
    with pytest.raises(ValueError):
        vf.render_triangle_png(tmp_path/"x.jpg", 10, 10)

def test_rgba_and_png(tmp_path):
    arr = vf.render_triangle_rgba(32, 24)
    assert arr.shape == (24, 32, 4)
    assert arr.dtype == getattr(__import__("numpy"), "uint8")
    out = tmp_path/"tri.png"
    vf.render_triangle_png(str(out), 32, 24)
    assert out.exists() and out.stat().st_size > 0

@pytest.mark.skipif(not hasattr(vf, "TerrainSpike"), reason="terrain feature not built")
def test_terrain_validation(tmp_path):
    with pytest.raises(ValueError):
        vf.make_terrain(64, 64, 1)  # grid must be >= 2
    t = vf.make_terrain(64, 48, 16)
    out = tmp_path/"t.png"
    t.render_png(str(out))
    assert out.exists() and out.stat().st_size > 0
# T01-END:tests
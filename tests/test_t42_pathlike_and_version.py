import pathlib
import numpy as np
import vulkan_forge._vulkan_forge as vf

def test_scene_render_png_accepts_pathlike(tmp_path):
    p = tmp_path / "scene.png"
    s = vf.Scene(8, 6)
    s.render_png(p)   # pass Path object directly
    a = vf.png_to_numpy(p)
    assert a.shape == (6, 8, 4)

def test_renderer_render_triangle_png_accepts_pathlike(tmp_path):
    p = tmp_path / "tri.png"
    r = vf.Renderer(8, 6)
    r.render_triangle_png(p)
    a = vf.png_to_numpy(p)
    assert a.shape == (6, 8, 4)

def test_version_export():
    assert hasattr(vf, "__version__")
    assert vf.__version__ == "0.1.0"
import pathlib
import numpy as np
import forge3d as f3d

def test_scene_render_png_accepts_pathlike(tmp_path):
    p = tmp_path / "scene.png"
    s = f3d.Scene(8, 6)
    s.render_png(p)   # pass Path object directly
    a = f3d.png_to_numpy(p)
    assert a.shape == (6, 8, 4)

def test_renderer_render_triangle_png_accepts_pathlike(tmp_path):
    p = tmp_path / "tri.png"
    r = f3d.Renderer(8, 6)
    r.render_triangle_png(p)
    a = f3d.png_to_numpy(p)
    assert a.shape == (6, 8, 4)

def test_version_export():
    assert hasattr(f3d, "__version__")
    assert f3d.__version__ == "0.79.0"

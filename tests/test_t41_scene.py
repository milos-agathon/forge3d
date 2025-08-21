import os, numpy as np
import forge3d as f3d

def test_t41_scene_renders_png(tmp_path):
    out = tmp_path / "scene_smoke.png"
    scn = f3d.Scene(320, 240, grid=64, colormap="viridis")
    scn.render_png(str(out))
    assert out.exists()
    # should be non-trivial in size (> 4 KB like T3 smoke)
    assert out.stat().st_size > 4096

def test_t41_height_upload_changes_output(tmp_path):
    out1 = tmp_path / "scene1.png"
    out2 = tmp_path / "scene2.png"
    scn = f3d.Scene(320, 240, grid=64, colormap="viridis")
    scn.render_png(str(out1))
    h = (np.sin(np.linspace(0, 4*np.pi, 128))[:,None] * np.cos(np.linspace(0, 4*np.pi, 128))[None,:]).astype("float32") * 0.25
    scn.set_height_from_r32f(h)
    scn.render_png(str(out2))
    assert out1.exists() and out2.exists()
    # Different pixels → different file sizes likely
    assert out1.stat().st_size != out2.stat().st_size
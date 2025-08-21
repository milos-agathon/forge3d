import os, numpy as np
import forge3d as f3d

def test_t42_numpy_to_png_and_back_rgba(tmp_path):
    h, w = 33, 77
    # deterministic RGBA ramp
    y, x = np.mgrid[0:h, 0:w]
    r = ((x + 13) % 256).astype(np.uint8)
    g = ((y + 29) % 256).astype(np.uint8) 
    b = (((x * 7) ^ (y * 11)) % 256).astype(np.uint8)
    a = np.full((h,w), 200, dtype=np.uint8)
    rgba = np.stack([r,g,b,a], axis=-1)

    p = tmp_path / "roundtrip.png"
    f3d.numpy_to_png(str(p), rgba)
    out = f3d.png_to_numpy(str(p))
    assert out.dtype == np.uint8 and out.shape == (h,w,4)
    assert np.array_equal(out, rgba)

def test_t42_scene_render_rgba_matches_png(tmp_path):
    out = tmp_path / "scene.png"
    scn = f3d.Scene(160, 120, grid=32, colormap="viridis")
    # save PNG
    scn.render_png(str(out))
    assert out.exists()
    # fetch pixels
    arr = scn.render_rgba()
    assert arr.dtype == np.uint8 and arr.shape == (120,160,4)
    # read back file and compare pixels
    file_pixels = f3d.png_to_numpy(str(out))
    assert np.array_equal(arr, file_pixels)
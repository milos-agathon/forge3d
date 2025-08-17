import numpy as np
import pathlib
import pytest
import vulkan_forge._vulkan_forge as vf

def test_numpy_to_png_rejects_wrong_dtype(tmp_path):
    p = tmp_path / "bad_dtype.png"
    arr = (np.zeros((4, 5, 3), dtype=np.float32))  # wrong dtype
    with pytest.raises(RuntimeError) as e:
        vf.numpy_to_png(str(p), arr)  # str path still works
    assert "unsupported array; expected uint8 (H,W), (H,W,3) or (H,W,4)" in str(e.value)

def test_numpy_to_png_rejects_wrong_channels(tmp_path):
    p = tmp_path / "bad_channels.png"
    arr = np.zeros((4, 5, 2), dtype=np.uint8)  # 2 channels not allowed
    with pytest.raises(RuntimeError) as e:
        vf.numpy_to_png(str(p), arr)
    assert "expected last dimension to be 3 (RGB) or 4 (RGBA)" in str(e.value)

def test_numpy_to_png_rejects_noncontiguous_gray(tmp_path):
    p = tmp_path / "bad_gray.png"
    gray_f = np.zeros((6, 7), dtype=np.uint8, order="F")
    with pytest.raises(RuntimeError) as e:
        vf.numpy_to_png(str(p), gray_f)
    assert "C-contiguous" in str(e.value)

def test_helpers_accept_pathlike(tmp_path):
    p = tmp_path / "rgb_pathlike.png"
    rgb = np.stack([np.full((3,4), 10, np.uint8),
                    np.full((3,4), 20, np.uint8),
                    np.full((3,4), 30, np.uint8)], axis=-1)
    vf.numpy_to_png(p, rgb)             # pass Path object directly
    out = vf.png_to_numpy(pathlib.Path(p))
    assert out.shape == (3,4,4)
    assert (out[..., :3] == rgb).all()
    assert (out[..., 3] == 255).all()

def test_scene_render_rgba_is_c_contiguous():
    s = vf.Scene(8, 6)
    a = s.render_rgba()
    assert a.flags['C_CONTIGUOUS']
    assert a.shape == (6, 8, 4)
    assert a.dtype == np.uint8
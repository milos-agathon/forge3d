import numpy as np
import pytest
import forge3d as f3d


def test_numpy_to_png_rgb_roundtrip(tmp_path):
    path = tmp_path / "rgb.png"
    # 2x3 RGB with diverse values
    rgb = np.array([
        [[255,   0,   0], [  0, 255,   0], [  0,   0, 255]],
        [[ 12,  34,  56], [ 78,  90, 123], [200, 150, 100]],
    ], dtype=np.uint8)

    f3d.numpy_to_png(str(path), rgb)
    out = f3d.png_to_numpy(str(path))
    assert out.shape == (2, 3, 4)
    assert out.dtype == np.uint8
    # RGB channels are identical; A is opaque
    assert np.array_equal(out[..., :3], rgb)
    assert np.all(out[..., 3] == 255)


def test_numpy_to_png_gray_roundtrip(tmp_path):
    path = tmp_path / "gray.png"
    gray = np.array([
        [  0,  64, 128],
        [192, 255,  33],
    ], dtype=np.uint8)

    f3d.numpy_to_png(str(path), gray)
    out = f3d.png_to_numpy(str(path))
    assert out.shape == (2, 3, 4)
    # Gray expands to RGBA with equal RGB and alpha 255
    assert np.array_equal(out[..., 0], gray)
    assert np.array_equal(out[..., 1], gray)
    assert np.array_equal(out[..., 2], gray)
    assert np.all(out[..., 3] == 255)


def test_numpy_to_png_rejects_non_contiguous(tmp_path):
    path = tmp_path / "bad.png"
    # Fortran-order makes it non C-contiguous
    rgb_f = np.zeros((4, 5, 3), dtype=np.uint8, order="F")
    with pytest.raises(RuntimeError) as e:
        f3d.numpy_to_png(str(path), rgb_f)
    assert "C-contiguous" in str(e.value)
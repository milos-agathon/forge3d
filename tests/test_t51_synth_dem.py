import math
import numpy as np
import pytest

import forge3d as f3d


def _gpu_or_skip():
    info = f3d.device_probe()
    if not isinstance(info, dict) or info.get("status") != "ok":
        msg = info.get("message", "No suitable GPU adapter") if isinstance(info, dict) else "Unknown device status"
        pytest.skip(f"Skipping GPU-dependent tests: {msg}")


def plane(h, w, value=1.0, dtype=np.float32):
    return np.full((h, w), value, dtype=dtype)


def ramp_x(h, w, dtype=np.float32):
    # 0..1 along X, constant along Y
    return np.tile(np.linspace(0.0, 1.0, w, dtype=dtype)[None, :], (h, 1))


def ramp_y(h, w, dtype=np.float32):
    # 0..1 along Y, constant along X
    return np.tile(np.linspace(0.0, 1.0, h, dtype=dtype)[:, None], (1, w)).astype(dtype)


def gaussian2d(h, w, sigma_ratio=0.25, dtype=np.float32):
    # Centered Gaussian peak; sigma scaled by min(h,w)
    y = np.arange(h, dtype=dtype) - (h - 1) / 2.0
    x = np.arange(w, dtype=dtype) - (w - 1) / 2.0
    X, Y = np.meshgrid(x, y, indexing="xy")
    sigma = dtype(min(h, w) * sigma_ratio)
    g = np.exp(-0.5 * (X**2 + Y**2) / (sigma * sigma))
    return g.astype(dtype)


@pytest.mark.parametrize("h,w,val", [(8, 11, 42.0), (3, 2, 0.0)])
def test_plane_stats_and_minmax_flat_epsilon(h, w, val):
    _gpu_or_skip()
    r = f3d.Renderer(16, 16)
    dem = plane(h, w, val)
    r.add_terrain(dem, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")

    mn, mx, mean, std = r.terrain_stats()
    assert abs(mn - val) < 1e-6
    assert abs(mx - val) < 1e-6
    assert abs(mean - val) < 1e-6
    assert abs(std - 0.0) < 1e-7

    # MinMax on a flat field should produce all zeros due to eps handling
    r.normalize_terrain("minmax", range=(0.0, 1.0))
    mn2, mx2, mean2, std2 = r.terrain_stats()
    assert abs(mn2 - 0.0) < 1e-7
    assert abs(mx2 - 0.0) < 1e-7
    assert abs(mean2 - 0.0) < 1e-7
    assert abs(std2 - 0.0) < 1e-7

    # Upload & full readback should match exactly (R32Float)
    r.upload_height_r32f()
    tex = r.read_full_height_texture()
    assert tex.shape == (h, w)
    # After minmax of flat: exact zeros
    assert np.all(tex == 0.0)


@pytest.mark.parametrize("h,w", [(9, 10), (16, 5), (32, 33)])
def test_ramp_x_stats_and_zscore(h, w):
    _gpu_or_skip()
    r = f3d.Renderer(32, 32)
    dem = ramp_x(h, w)
    r.add_terrain(dem, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")

    # Expected stats from numpy for discrete ramp grid
    exp_mean = float(dem.mean())
    exp_std = float(dem.std())
    mn, mx, mean, std = r.terrain_stats()
    assert abs(mn - 0.0) < 1e-7
    assert abs(mx - 1.0) < 1e-7
    assert abs(mean - exp_mean) < 1e-6
    assert abs(std - exp_std) < 1e-6

    r.normalize_terrain("zscore", eps=1e-8)
    mn2, mx2, mean2, std2 = r.terrain_stats()
    assert abs(mean2 - 0.0) < 2e-6
    assert abs(std2 - 1.0) < 2e-6

    r.upload_height_r32f()
    tex = r.read_full_height_texture()
    assert tex.shape == (h, w)
    # z-score => mean ≈ 0, std ≈ 1
    assert abs(float(tex.mean()) - 0.0) < 2e-6
    assert abs(float(tex.std()) - 1.0) < 2e-6


@pytest.mark.parametrize("h,w", [(19, 21), (32, 32)])
def test_gaussian_minmax_peak_near_one(h, w):
    _gpu_or_skip()
    r = f3d.Renderer(64, 64)
    dem = gaussian2d(h, w)
    r.add_terrain(dem, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")

    r.normalize_terrain("minmax", range=(0.0, 1.0))
    r.upload_height_r32f()
    tex = r.read_full_height_texture()
    center = (h - 1) // 2, (w - 1) // 2
    assert tex.shape == (h, w)
    # Peak should be near 1; edges near 0
    assert tex[center] > 0.98
    edge_samples = [tex[0, 0], tex[0, -1], tex[-1, 0], tex[-1, -1]]
    assert all(es < 0.05 for es in edge_samples)


@pytest.mark.parametrize("h,w", [(2, 63), (7, 63), (7, 257), (1, 257), (2, 3)])
def test_upload_readback_padding_exact_parity(h, w):
    """
    Exercise COPY_BYTES_PER_ROW_ALIGNMENT via widths that require padding.
    Expect exact parity after upload/readback (R32Float).
    """
    _gpu_or_skip()
    r = f3d.Renderer(8, 8)
    rng = np.random.default_rng(0)
    dem = rng.uniform(low=-123.0, high=456.0, size=(h, w)).astype(np.float32)
    r.add_terrain(dem, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")
    r.upload_height_r32f()
    tex = r.read_full_height_texture()
    assert tex.shape == (h, w)
    # Bitwise exact for 32-bit floats
    assert tex.dtype == np.float32
    assert np.array_equal(tex.view(np.uint32), dem.view(np.uint32))


def test_add_terrain_validation_errors():
    _gpu_or_skip()
    r = f3d.Renderer(16, 16)

    # Fortran order should be rejected (not C-contiguous)
    f_arr = np.asfortranarray(np.ones((4, 5), dtype=np.float32))
    with pytest.raises(RuntimeError) as ei1:
        r.add_terrain(f_arr, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")
    assert "heightmap must be a 2-D NumPy array" in str(ei1.value)

    # Empty arrays should be rejected
    empty = np.empty((0, 5), dtype=np.float32)
    with pytest.raises(RuntimeError) as ei2:
        r.add_terrain(empty, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")
    assert "heightmap cannot be empty" in str(ei2.value)

    # Unknown colormap should be rejected
    arr = np.ones((3, 3), dtype=np.float32)
    with pytest.raises(RuntimeError) as ei3:
        r.add_terrain(arr, spacing=(1.0, 1.0), exaggeration=1.0, colormap="definitely_not_a_cmap")
    assert "Unknown colormap" in str(ei3.value)

    # Invalid height range (min >= max)
    r.add_terrain(arr, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")
    with pytest.raises(ValueError):
        r.set_height_range(1.0, 1.0)


def test_scene_accepts_height_and_renders_rgba_shape():
    _gpu_or_skip()
    # Small scene to keep things light; height map 5x7
    sc = f3d.Scene(64, 64, grid=8, colormap="viridis")
    dem = ramp_y(5, 7)  # float32, C-contiguous
    sc.set_height_from_r32f(dem)
    rgba = sc.render_rgba()
    assert rgba.dtype == np.uint8
    assert rgba.shape == (64, 64, 4)
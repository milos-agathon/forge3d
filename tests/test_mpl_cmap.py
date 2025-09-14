# tests/test_mpl_cmap.py
# Tests for matplotlib colormap and linear normalization adapters.
# This exists to validate Workstream R1 acceptance criteria with SSIM/PSNR checks.
# RELEVANT FILES:python/forge3d/adapters/mpl_cmap.py,python/forge3d/adapters/__init__.py,docs/integration/matplotlib.md

import math
import numpy as np
import pytest

from tests._ssim import ssim


def _psnr(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def _map_with_lut(normed: np.ndarray, lut: np.ndarray) -> np.ndarray:
    # normed in [0,1]; lut shape (N,4) uint8, return HxWx4 uint8
    n = lut.shape[0]
    idx = np.clip((normed * (n - 1)).round().astype(np.int32), 0, n - 1)
    return lut[idx]


@pytest.mark.parametrize("cmap_name", ["viridis", "plasma", "magma"]) 
def test_accepts_name_and_object(cmap_name):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from forge3d.adapters.mpl_cmap import matplotlib_to_forge3d_colormap

    lut_from_name = matplotlib_to_forge3d_colormap(cmap_name, n_colors=256)
    lut_from_obj = matplotlib_to_forge3d_colormap(plt.get_cmap(cmap_name), n_colors=256)

    assert lut_from_name.shape == (256, 4)
    assert lut_from_name.dtype == np.uint8
    assert np.array_equal(lut_from_name, lut_from_obj)


def test_linear_normalize_parity():
    pytest.importorskip("matplotlib")
    import matplotlib.colors as mcolors
    from forge3d.adapters.mpl_cmap import matplotlib_normalize

    rng = np.random.default_rng(123)
    for _ in range(10):
        a = rng.normal(size=(64, 64)).astype(np.float32)
        vmin, vmax = float(np.min(a)), float(np.max(a))
        ours = matplotlib_normalize(a, 'linear', vmin=vmin, vmax=vmax, clip=False)
        ref = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)(a)
        max_abs_diff = float(np.max(np.abs(ours - ref)))
        assert max_abs_diff <= 1e-7


def test_ramp_rgba_parity_ssim_or_psnr():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from forge3d.adapters.mpl_cmap import (
        matplotlib_to_forge3d_colormap, matplotlib_normalize
    )

    # 1024x32 ramp across width
    w, h = 1024, 32
    x = np.linspace(0.0, 1.0, w, dtype=np.float64)
    ramp = np.tile(x, (h, 1))

    cmap = plt.get_cmap('viridis')
    lut = matplotlib_to_forge3d_colormap(cmap, 256)
    normed = matplotlib_normalize(ramp, 'linear', vmin=0.0, vmax=1.0, clip=True)

    ours_rgba = _map_with_lut(normed, lut)

    ref_rgba = (np.clip(cmap(normed), 0.0, 1.0) * 255.0).round().astype(np.uint8)

    # Prefer SSIM; fallback to PSNR if SSIM fails for some reason
    try:
        value = ssim(ours_rgba, ref_rgba, data_range=255.0)
        assert value >= 0.999
    except Exception:
        value = _psnr(ours_rgba, ref_rgba, data_range=255.0)
        assert value >= 45.0


def test_optional_dep_behavior_message_or_skip():
    try:
        from forge3d.adapters import is_matplotlib_available, matplotlib_to_forge3d_colormap
    except Exception:
        pytest.skip("forge3d not importable in test env")

    if is_matplotlib_available():
        pytest.skip("Matplotlib present; optional-missing behavior not testable here")
    with pytest.raises(ImportError) as ei:
        matplotlib_to_forge3d_colormap('viridis', 256)
    assert "pip install matplotlib" in str(ei.value)


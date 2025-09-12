# tests/test_mpl_display.py
# Tests for Matplotlib display helpers (imshow_rgba) behavior and zero-copy.
# This exists to validate Workstream R4 acceptance criteria.
# RELEVANT FILES:python/forge3d/helpers/mpl_display.py,python/forge3d/helpers/__init__.py,examples/mpl_imshow_demo.py

import numpy as np
import pytest


@pytest.mark.skipif(
    __import__('importlib').util.find_spec('matplotlib') is None,
    reason="Matplotlib not available"
)
def test_imshow_rgba_zero_copy_uint8_and_extent_dpi(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from forge3d.helpers.mpl_display import imshow_rgba

    h, w = 32, 64
    # C-contiguous uint8 RGBA data
    rgba = (np.random.default_rng(0).integers(0, 256, size=(h, w, 4))).astype(np.uint8)
    assert rgba.flags['C_CONTIGUOUS']

    fig, ax = plt.subplots(figsize=(4, 2))
    extent = (1.0, 5.0, 2.0, 6.0)
    dpi = 123
    captured = {}
    original_imshow = ax.imshow
    def _wrapped_imshow(data, *args, **kwargs):
        captured['data'] = data
        return original_imshow(data, *args, **kwargs)
    ax.imshow = _wrapped_imshow  # type: ignore
    img = imshow_rgba(ax, rgba, extent=extent, dpi=dpi)
    # Validate our helper passed the same object (no pre-copy)
    assert 'data' in captured
    assert captured['data'] is rgba

    # Extent and DPI honored
    assert tuple(img.get_extent()) == extent
    assert fig.dpi == dpi

    out = tmp_path / "imshow.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


@pytest.mark.skipif(
    __import__('importlib').util.find_spec('matplotlib') is None,
    reason="Matplotlib not available"
)
def test_imshow_default_aspect_and_origin():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from forge3d.helpers.mpl_display import imshow_rgba

    # Small RGB image to avoid implicit alpha concatenation (no copy)
    rgb = (np.array([[[255, 0, 0], [0, 255, 0]],
                     [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8))

    fig, ax = plt.subplots(figsize=(2, 2))
    img = imshow_rgba(ax, rgb)

    # Origin default 'upper' if available on object
    if hasattr(img, 'get_origin'):
        assert img.get_origin() == 'upper'
    # Aspect default is 'equal' (may be represented as 1.0)
    aspect = ax.get_aspect()
    assert aspect == 'equal' or aspect == 1.0
    plt.close(fig)

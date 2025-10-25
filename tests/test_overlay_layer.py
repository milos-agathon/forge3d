# tests/test_overlay_layer.py
# Tests for the native OverlayLayer wrapper
# Exists to validate overlay creation and parameter enforcement
# RELEVANT FILES: src/overlay_layer.rs, src/colormap1d.rs, python/forge3d/__init__.py, tests/test_colormap1d.py
import pytest

import forge3d as f3d


if not f3d.has_gpu() or not hasattr(f3d, "OverlayLayer"):
    pytest.skip("OverlayLayer requires GPU-backed native module", allow_module_level=True)


def _make_colormap():
    return f3d.Colormap1D.from_stops(
        stops=[(0.0, "#000000"), (0.5, "#ffaa00"), (1.0, "#ffffff")],
        domain=(0.0, 1.0),
    )


def test_overlay_layer_creation():
    cmap = _make_colormap()
    overlay = f3d.OverlayLayer.from_colormap1d(
        cmap,
        strength=0.8,
        offset=0.1,
        blend_mode="Alpha",
        domain=(0.0, 1.0),
    )
    assert overlay.kind == "Colormap1D"
    assert overlay.blend_mode == "Alpha"
    assert overlay.domain == (0.0, 1.0)
    assert overlay.colormap is not None


def test_overlay_layer_invalid_blend_mode():
    cmap = _make_colormap()
    with pytest.raises(ValueError):
        f3d.OverlayLayer.from_colormap1d(cmap, blend_mode="invalid")


def test_overlay_layer_invalid_domain():
    cmap = _make_colormap()
    with pytest.raises(ValueError):
        f3d.OverlayLayer.from_colormap1d(cmap, domain=(1.0, 0.0))


def test_overlay_layer_negative_strength():
    cmap = _make_colormap()
    with pytest.raises(ValueError):
        f3d.OverlayLayer.from_colormap1d(cmap, strength=-0.1)

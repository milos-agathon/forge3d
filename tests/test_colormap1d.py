# tests/test_colormap1d.py
# Unit tests for Colormap1D PyO3 wrapper
# Ensures 1D colormap creation and validation behave as expected
# RELEVANT FILES: src/colormap1d.rs, src/terrain/mod.rs, python/forge3d/__init__.py, tests/test_session.py
import pytest

import forge3d as f3d


if not f3d.has_gpu():
    pytest.skip("GPU adapter not available", allow_module_level=True)
if not hasattr(f3d, "Colormap1D"):
    pytest.skip("Colormap1D not available in this build", allow_module_level=True)


def test_colormap1d_creation() -> None:
    cmap = f3d.Colormap1D.from_stops(
        stops=[(0.0, "#000000"), (1.0, "#ffffff")],
        domain=(0.0, 1.0),
    )
    assert cmap.domain == (0.0, 1.0)


def test_colormap1d_multi_stops() -> None:
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (200.0, "#313695"),
            (800.0, "#4575b4"),
            (1500.0, "#74add1"),
            (2200.0, "#fdae61"),
        ],
        domain=(200.0, 2200.0),
    )
    assert cmap.domain == (200.0, 2200.0)


def test_colormap1d_invalid_color() -> None:
    with pytest.raises(ValueError):
        f3d.Colormap1D.from_stops(
            stops=[(0.0, "invalid")],
            domain=(0.0, 1.0),
        )


def test_colormap1d_invalid_domain() -> None:
    with pytest.raises(ValueError):
        f3d.Colormap1D.from_stops(
            stops=[(0.0, "#000000"), (1.0, "#ffffff")],
            domain=(1.0, 0.0),
        )

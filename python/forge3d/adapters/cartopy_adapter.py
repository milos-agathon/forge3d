# python/forge3d/adapters/cartopy_adapter.py
# M5: Cartopy Integration
# - Rasterize Cartopy GeoAxes to RGBA
# - Query axes CRS and compute extents in target CRS

from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np

try:
    import cartopy.crs as ccrs  # type: ignore
    from cartopy.mpl.geoaxes import GeoAxes  # type: ignore
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False
    GeoAxes = object  # type: ignore


def is_cartopy_available() -> bool:
    return _HAS_CARTOPY


def _require_cartopy():
    if not _HAS_CARTOPY:
        raise ImportError("Cartopy is required. Install with: pip install cartopy")


def rasterize_geoaxes(ax: 'GeoAxes', *, dpi: Optional[int] = None, facecolor: Optional[str] = None) -> np.ndarray:
    """Rasterize a Cartopy GeoAxes to an RGBA numpy array (H,W,4 uint8)."""
    _require_cartopy()
    assert isinstance(ax, GeoAxes), "ax must be a Cartopy GeoAxes"
    fig = ax.figure
    if dpi is not None:
        fig.set_dpi(int(dpi))
    if facecolor is not None:
        try:
            fig.set_facecolor(facecolor)
        except Exception:
            pass
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf.copy()


def get_axes_crs(ax: 'GeoAxes') -> Any:
    """Return the CRS object for a GeoAxes."""
    _require_cartopy()
    assert isinstance(ax, GeoAxes), "ax must be a Cartopy GeoAxes"
    return ax.projection


def get_extent_in_crs(ax: 'GeoAxes', target_crs: Any) -> Tuple[float, float, float, float]:
    """Get the axes extent transformed into the target CRS.

    Returns extent as (west, east, south, north) in target CRS coordinates.
    """
    _require_cartopy()
    assert isinstance(ax, GeoAxes), "ax must be a Cartopy GeoAxes"
    # Cartopy uses (x0, x1, y0, y1) in the CRS coordinates
    ex = ax.get_extent(crs=target_crs)
    # Return as (left, right, bottom, top)
    return (float(ex[0]), float(ex[1]), float(ex[2]), float(ex[3]))

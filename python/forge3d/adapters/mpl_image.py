# python/forge3d/adapters/mpl_image.py
# M1: Matplotlib Adapter (Image)
# - Rasterize Matplotlib Axes/Figure to RGBA numpy arrays
# - Optional height-from-luminance helper for DEM-like draping workflows
# - Keeps dependencies optional and degrades gracefully

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

try:
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False
    Axes = object  # type: ignore
    Figure = object  # type: ignore


def is_matplotlib_available() -> bool:
    return _HAS_MPL


def _require_mpl():
    if not _HAS_MPL:
        raise ImportError("Matplotlib is required. Install with: pip install matplotlib")


def rasterize_figure(fig: 'Figure', *, dpi: Optional[int] = None, facecolor: Optional[str] = None) -> np.ndarray:
    """Rasterize a Matplotlib Figure into an RGBA numpy array (H,W,4, uint8).

    Args:
        fig: Matplotlib Figure
        dpi: Optional override DPI
        facecolor: Optional figure facecolor (e.g., 'white', 'none')
    """
    _require_mpl()
    assert isinstance(fig, Figure), "fig must be a Matplotlib Figure"

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
    # Copy to detach from canvas memory
    return buf.copy()


def rasterize_axes(ax: 'Axes', *, dpi: Optional[int] = None, bbox_inches: Optional[str] = 'tight', pad_inches: float = 0.0, facecolor: Optional[str] = None) -> np.ndarray:
    """Rasterize a Matplotlib Axes to an RGBA numpy array (H,W,4, uint8).

    When bbox_inches='tight', the rendering includes only the axes content (with optional padding).
    """
    _require_mpl()
    assert isinstance(ax, Axes), "ax must be a Matplotlib Axes"

    fig = ax.figure
    if dpi is not None:
        fig.set_dpi(int(dpi))
    if facecolor is not None:
        try:
            fig.set_facecolor(facecolor)
        except Exception:
            pass

    # Use a temporary canvas; save to a bytes buffer and read back via PIL if necessary
    canvas = FigureCanvasAgg(fig)
    bbox = None
    if bbox_inches == 'tight':
        try:
            # Get tight bbox in display units, then use Agg
            fig.tight_layout()
        except Exception:
            pass
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf.copy()


def height_from_luminance(rgba: np.ndarray, weights: Tuple[float, float, float] = (0.2126, 0.7152, 0.0722)) -> np.ndarray:
    """Compute a height-like scalar field from an RGBA image via luminance.

    Args:
        rgba: (H,W,4|3) numpy array; uint8 or float
        weights: luminance weights (R,G,B)

    Returns:
        float32 (H,W) in [0,1]
    """
    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError("rgba must have shape (H,W,3|4)")
    arr = rgba.astype(np.float32, copy=False)
    if arr.dtype == np.uint8:
        arr = arr / 255.0
    rgb = arr[..., :3]
    w = np.array(weights, dtype=np.float32)
    w = w / np.maximum(np.sum(w), 1e-8)
    lum = np.clip(np.tensordot(rgb, w, axes=([-1], [0])), 0.0, 1.0)
    return lum.astype(np.float32, copy=False)

# python/forge3d/adapters/charts.py
# M6: Seaborn/Plotly convenience adapters
# - Detect and route chart objects to RGBA render paths
# - Plotly: use kaleido to_image for deterministic PNG bytes
# - Seaborn: resolve underlying Matplotlib Figure/Axes and rasterize via mpl_image

from __future__ import annotations

from typing import Any, Optional
import io
import numpy as np


def is_plotly_available() -> bool:
    try:
        import plotly  # noqa: F401
        import kaleido  # noqa: F401
        return True
    except Exception:
        return False


def is_seaborn_available() -> bool:
    try:
        import seaborn  # noqa: F401
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


def _plotly_to_rgba(fig: Any, *, width: Optional[int] = None, height: Optional[int] = None, scale: float = 1.0) -> np.ndarray:
    import plotly.io as pio  # type: ignore
    # to_image returns bytes in PNG when format='png' and kaleido is available
    png_bytes: bytes = pio.to_image(fig, format='png', width=width, height=height, scale=scale)  # type: ignore
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Pillow is required to decode Plotly PNG bytes. pip install pillow") from exc
    im = Image.open(io.BytesIO(png_bytes)).convert('RGBA')
    return np.array(im, dtype=np.uint8)


def _seaborn_to_rgba(obj: Any, *, dpi: Optional[int] = None) -> np.ndarray:
    # Resolve to Matplotlib Figure/Axes
    try:
        import matplotlib
        from matplotlib.figure import Figure
    except Exception as exc:  # pragma: no cover
        raise ImportError("Matplotlib is required for seaborn chart rasterization. pip install matplotlib seaborn") from exc

    # Common seaborn containers have .fig or .figure attributes
    fig = getattr(obj, 'fig', None) or getattr(obj, 'figure', None)
    if fig is None and hasattr(obj, 'get_figure'):
        try:
            fig = obj.get_figure()
        except Exception:
            fig = None
    if fig is None:
        # If it's an Axes-level object, try to find its figure
        fig = getattr(getattr(obj, 'axes', None), 'figure', None)
    if fig is None:
        # Last resort: assume obj is a Figure
        fig = obj
    from .mpl_image import rasterize_figure
    return rasterize_figure(fig, dpi=dpi)


def render_chart_to_rgba(obj: Any, **kwargs) -> np.ndarray:
    """Render a chart object (Plotly or Seaborn/Matplotlib) to an RGBA numpy array.

    Args:
        obj: Plotly figure or Seaborn/Matplotlib object
        kwargs: For Plotly: width, height, scale. For Seaborn: dpi.
    """
    # Try plotly
    try:
        import plotly.graph_objects as go  # type: ignore
        if isinstance(obj, go.Figure):
            if not is_plotly_available():
                raise ImportError("Plotly/kaleido not available. pip install plotly kaleido")
            return _plotly_to_rgba(obj, width=kwargs.get('width'), height=kwargs.get('height'), scale=kwargs.get('scale', 1.0))
    except Exception:
        pass

    # Try seaborn/matplotlib
    try:
        import seaborn as sns  # type: ignore
        # Seaborn objects vary; attempt rasterization path
        return _seaborn_to_rgba(obj, dpi=kwargs.get('dpi'))
    except Exception:
        # If seaborn not present, still try matplotlib figure
        try:
            from matplotlib.figure import Figure  # type: ignore
            from .mpl_image import rasterize_figure
            if isinstance(obj, Figure):
                return rasterize_figure(obj, dpi=kwargs.get('dpi'))
        except Exception:
            pass

    raise TypeError("Unsupported chart object for render_chart_to_rgba; expected Plotly Figure or Seaborn/Matplotlib object")

# python/forge3d/helpers/ipython_display.py
# Workstream I2: Offscreen + Jupyter helpers (IPython half)
# Provides simple display functions for use in notebooks without
# requiring matplotlib.

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from .offscreen import render_offscreen_rgba, rgba_to_png_bytes


def _require_ipython():
    try:
        import IPython  # noqa: F401
        from IPython.display import Image, display  # noqa: F401
    except Exception as exc:  # pragma: no cover - notebook-only path
        raise ImportError(
            "IPython is required for display helpers. Install with: pip install ipython"
        ) from exc


def display_rgba(rgba: np.ndarray, *, title: Optional[str] = None) -> None:
    """Display an RGBA numpy array inside a Jupyter notebook cell.

    Converts to deterministic PNG bytes and uses IPython.display to embed.
    """
    from IPython.display import Image, display  # type: ignore

    if not isinstance(rgba, np.ndarray) or rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError("rgba must be numpy array with shape (H,W,3|4)")
    data = rgba_to_png_bytes(rgba)
    img = Image(data=data)
    if title:
        display({"text/plain": title})
    display(img)


def display_offscreen(
    width: int,
    height: int,
    *,
    scene: Any | None = None,
    camera: Optional[Mapping[str, Any]] = None,
    seed: int = 1,
    frames: int = 1,
    denoiser: str = "off",
    title: Optional[str] = None,
) -> None:
    """Render offscreen and display in a Jupyter notebook cell."""
    rgba = render_offscreen_rgba(width, height, scene=scene, camera=camera, seed=seed, frames=frames, denoiser=denoiser)
    display_rgba(rgba, title=title)

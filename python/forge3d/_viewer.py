from __future__ import annotations

import numpy as np


def open_viewer_image(
    rgba: np.ndarray,
    *,
    width: int | None = None,
    height: int | None = None,
    title: str = "forge3d Image Preview",
    vsync: bool = True,
    fov_deg: float = 45.0,
    znear: float = 0.1,
    zfar: float = 1000.0,
) -> None:
    """Open the interactive viewer initialized with an RGBA image via the native module.

    This mirrors `forge3d.open_viewer_image()` but is placed in a separate module to avoid
    circular imports when called from `render.py`.
    """
    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("rgba must have shape (H, W, 4)")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=True)
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)

    try:
        from . import _forge3d as _native  # type: ignore[attr-defined]
        if hasattr(_native, "open_viewer_image"):
            _native.open_viewer_image(
                arr,
                width=None if width is None else int(width),
                height=None if height is None else int(height),
                title=str(title),
                vsync=bool(vsync),
                fov_deg=float(fov_deg),
                znear=float(znear),
                zfar=float(zfar),
            )
        else:
            raise RuntimeError(
                "Interactive image viewer not available. "
                "This feature requires the native extension built with viewer support."
            )
    except ImportError as e:
        raise RuntimeError(
            f"Native module not available: {e}. "
            "Install forge3d with: pip install forge3d"
        ) from e

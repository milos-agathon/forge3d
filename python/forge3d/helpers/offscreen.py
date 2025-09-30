# python/forge3d/helpers/offscreen.py
# Workstream I2: Offscreen + Jupyter helpers (offscreen half)
# - Headless render wrappers returning numpy RGBA
# - Deterministic PNG writer for stable hashing
# RELEVANT FILES: python/forge3d/path_tracing.py, python/forge3d/__init__.py

from __future__ import annotations

from typing import Any, Mapping, Optional
import io
import numpy as np

from ..path_tracing import render_rgba as _fallback_render_rgba

try:
    from .. import _forge3d as _native  # type: ignore[attr-defined]
except Exception:
    _native = None  # type: ignore


def render_offscreen_rgba(
    width: int,
    height: int,
    *,
    scene: Any | None = None,
    camera: Optional[Mapping[str, Any]] = None,
    seed: int = 1,
    frames: int = 1,
    denoiser: str = "off",
) -> np.ndarray:
    """Render an RGBA image offscreen and return a numpy array.

    Uses the native module when available; otherwise falls back to the
    deterministic CPU path tracer for tests and notebooks.
    """
    w = int(width); h = int(height)
    if w <= 0 or h <= 0:
        raise ValueError("width and height must be positive")

    # Prefer native path if present (headless offscreen), else Python fallback
    if _native is not None and hasattr(_native, "render_rgba"):
        try:
            return _native.render_rgba(w, h, scene=scene, camera=camera, seed=int(seed), frames=int(frames), denoiser=str(denoiser))
        except Exception:
            # Fall back silently to Python path
            pass

    return _fallback_render_rgba(w, h, scene=scene, camera=camera, seed=int(seed), frames=int(frames), denoiser=str(denoiser))


def save_png_deterministic(path: str | bytes | "os.PathLike[str]", rgba: np.ndarray) -> None:
    """Save RGBA as PNG with deterministic bytes for hashing.

    Ensures stable PNG output by using fixed parameters and avoiding metadata.
    """
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Pillow is required for save_png_deterministic()") from exc

    if not isinstance(rgba, np.ndarray) or rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError("rgba must be numpy array with shape (H,W,3|4)")

    data = rgba
    if data.dtype == np.uint8:
        arr = data
    elif data.dtype in (np.float32, np.float64):
        arr = (np.clip(data, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    else:
        arr = data.astype(np.uint8)

    mode = "RGBA" if arr.shape[2] == 4 else "RGB"
    img = Image.fromarray(arr, mode=mode)

    # Write without optimization or ancillary chunks; PIL by default writes deterministic PNG
    # given identical bytes and parameters. Explicitly pass a fixed compress_level for stability.
    img.save(path, format="PNG", optimize=False, compress_level=6)


def rgba_to_png_bytes(rgba: np.ndarray) -> bytes:
    """Convert RGBA array to PNG bytes deterministically."""
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Pillow is required for rgba_to_png_bytes()") from exc

    data = rgba
    if data.dtype == np.uint8:
        arr = data
    elif data.dtype in (np.float32, np.float64):
        arr = (np.clip(data, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    else:
        arr = data.astype(np.uint8)

    mode = "RGBA" if arr.shape[2] == 4 else "RGB"
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()

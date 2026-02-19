# python/forge3d/helpers/offscreen.py
# Workstream I2: Offscreen + Jupyter helpers (offscreen half)
# - Headless render wrappers returning numpy RGBA
# - Deterministic PNG writer for stable hashing
# RELEVANT FILES: python/forge3d/path_tracing.py, python/forge3d/__init__.py

from __future__ import annotations

from typing import Any, Mapping, Optional
import io
import sys
import warnings
import numpy as np

from ..path_tracing import render_rgba as _fallback_render_rgba

# Resolve the native Scene class for isinstance checks.
# Scene.render_rgba() is an instance method, not a module-level function.
try:
    from .._native import get_native_module as _get_native_module
    _nm = _get_native_module()
    _NativeScene = getattr(_nm, "Scene", None) if _nm is not None else None
    del _nm
except Exception:
    _NativeScene = None


def _is_native_scene(obj: Any) -> bool:
    """Return True if *obj* is a native (Rust/PyO3) Scene instance."""
    return _NativeScene is not None and isinstance(obj, _NativeScene)


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

    When *scene* is a native ``Scene`` instance, delegates to
    ``scene.render_rgba()`` (GPU path).  The native Scene owns its
    framebuffer dimensions, so *width*/*height* are only used by the
    CPU fallback path.  A warning is emitted if the caller supplies
    explicit dimensions that will be ignored on the native path.

    Otherwise, falls back to the deterministic CPU path tracer
    (``forge3d.path_tracing.render_rgba``).
    """
    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise ValueError("width and height must be positive")

    # --- Native path: Scene.render_rgba() is an instance method. ---
    if _is_native_scene(scene):
        try:
            return scene.render_rgba()
        except (RuntimeError, OSError) as exc:
            print(
                f"[forge3d] native render_rgba failed, using CPU fallback: {exc}",
                file=sys.stderr,
            )

    # --- Fallback: deterministic CPU path tracer. ---
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


def save_png_with_exif(path: str | bytes | "os.PathLike[str]", rgba: np.ndarray, metadata: Optional[Mapping[str, Any]] = None) -> None:
    """Save RGBA as PNG with EXIF metadata.

    Embeds camera and exposure metadata in PNG for reproducibility and documentation.

    Args:
        path: Output path for PNG file
        rgba: RGBA array (H, W, 3|4) uint8 or float32
        metadata: Optional dict with keys:
            - camera: dict with eye, target, up, fov_deg
            - exposure: dict with mode, stops, gamma

    Example:
        metadata = {
            "camera": {
                "eye": [10.0, 20.0, 30.0],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0],
                "fov_deg": 45.0
            },
            "exposure": {
                "mode": "ACES",
                "stops": 0.0,
                "gamma": 2.2
            }
        }
        save_png_with_exif("render.png", rgba, metadata)
    """
    try:
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Pillow is required for save_png_with_exif()") from exc

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

    # Embed metadata as PNG text chunks (tEXt/zTXt)
    pnginfo = PngInfo()

    if metadata:
        # Camera metadata
        if "camera" in metadata:
            cam = metadata["camera"]
            if "eye" in cam:
                pnginfo.add_text("forge3d:camera:eye", str(cam["eye"]))
            if "target" in cam:
                pnginfo.add_text("forge3d:camera:target", str(cam["target"]))
            if "up" in cam:
                pnginfo.add_text("forge3d:camera:up", str(cam["up"]))
            if "fov_deg" in cam:
                pnginfo.add_text("forge3d:camera:fov_deg", str(cam["fov_deg"]))

        # Exposure metadata
        if "exposure" in metadata:
            exp = metadata["exposure"]
            if "mode" in exp:
                pnginfo.add_text("forge3d:exposure:mode", str(exp["mode"]))
            if "stops" in exp:
                pnginfo.add_text("forge3d:exposure:stops", str(exp["stops"]))
            if "gamma" in exp:
                pnginfo.add_text("forge3d:exposure:gamma", str(exp["gamma"]))

        # General metadata
        if "description" in metadata:
            pnginfo.add_text("Description", str(metadata["description"]))
        if "software" in metadata:
            pnginfo.add_text("Software", str(metadata["software"]))

    # Write PNG with metadata
    img.save(path, format="PNG", pnginfo=pnginfo, optimize=False, compress_level=6)

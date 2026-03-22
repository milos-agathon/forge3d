"""TV12: Optional OIDN (Open Image Denoise) integration.

Requires: pip install pyoidn   (or: pip install oidn)
Falls back cleanly when the package is not installed.
"""
from __future__ import annotations
from typing import Optional
import numpy as np


def oidn_available() -> bool:
    """Return True if an OIDN Python binding is importable."""
    try:
        import oidn
        return True
    except ImportError:
        pass
    try:
        import pyoidn
        return True
    except ImportError:
        pass
    return False


def oidn_denoise(
    beauty: np.ndarray,
    albedo: Optional[np.ndarray] = None,
    normal: Optional[np.ndarray] = None,
    *,
    hdr: bool = True,
    quality: str = "high",
) -> np.ndarray:
    """Denoise using Intel Open Image Denoise.

    Args:
        beauty: (H, W, 3) float32 linear HDR color.
        albedo: optional (H, W, 3) float32 albedo guide.
        normal: optional (H, W, 3) float32 normal guide.
        hdr: True if beauty is HDR (values can exceed 1.0).
        quality: "default" or "high".

    Returns:
        Denoised (H, W, 3) float32 array.

    Raises:
        ValueError: if input shapes are wrong.
        ImportError: if no OIDN package is available.
    """
    if beauty.ndim != 3 or beauty.shape[2] != 3:
        raise ValueError("beauty must be (H, W, 3) float32")
    h, w = beauty.shape[:2]
    if albedo is not None and albedo.shape != (h, w, 3):
        raise ValueError(f"albedo shape {albedo.shape} must match beauty ({h}, {w}, 3)")
    if normal is not None and normal.shape != (h, w, 3):
        raise ValueError(f"normal shape {normal.shape} must match beauty ({h}, {w}, 3)")

    oidn_mod = None
    try:
        import oidn as oidn_mod
    except ImportError:
        try:
            import pyoidn as oidn_mod
        except ImportError:
            raise ImportError(
                "OIDN denoising requires the 'oidn' or 'pyoidn' package. "
                "Install with: pip install pyoidn"
            )

    device = oidn_mod.NewDevice()
    device.Commit()

    beauty_f32 = np.ascontiguousarray(beauty, dtype=np.float32)
    output = np.empty_like(beauty_f32)

    filt = device.NewFilter("RT")
    filt.SetImage("color", beauty_f32)
    filt.SetImage("output", output)
    if albedo is not None:
        filt.SetImage("albedo", np.ascontiguousarray(albedo, dtype=np.float32))
    if normal is not None:
        filt.SetImage("normal", np.ascontiguousarray(normal, dtype=np.float32))
    filt.SetBool("hdr", hdr)
    if quality == "high":
        filt.SetInt("quality", 1)
    filt.Commit()
    filt.Execute()

    return output

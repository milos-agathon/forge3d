"""Deterministic, error-bounded F3DZ DEM compression."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._native import get_native_module

__all__ = ["compress_dem", "decompress_dem", "verify_dem"]


def _require_native():
    native = get_native_module()
    if native is None:
        raise RuntimeError("forge3d native extension is required for forge3d.codec")
    return native


def _dem_f32(dem: Any) -> np.ndarray:
    array = np.asarray(dem, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"DEM must be a two-dimensional array, got shape {array.shape}")
    return np.ascontiguousarray(array)


def compress_dem(
    dem: Any,
    eps: float,
    progressive: bool = True,
) -> bytes:
    """Compress a two-dimensional height grid into deterministic F3DZ v1."""
    return bytes(_require_native().compress_dem(_dem_f32(dem), float(eps), bool(progressive)))


def decompress_dem(data: bytes | bytearray | memoryview) -> tuple[np.ndarray, dict[str, Any]]:
    """Fail-closed F3DZ decode returning ``(height_array, stream_info)``."""
    heights, info = _require_native().decompress_dem(bytes(data))
    return np.asarray(heights, dtype=np.float32), dict(info)


def verify_dem(
    data: bytes | bytearray | memoryview,
    source: Any | None = None,
) -> dict[str, Any]:
    """Recheck structure/CRCs and optionally every source error and page bound."""
    native_source = None if source is None else _dem_f32(source)
    return dict(_require_native().verify_dem(bytes(data), native_source))

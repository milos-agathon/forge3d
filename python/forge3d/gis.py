"""Thin Python wrappers for Rust-backed GIS raster operations."""

from __future__ import annotations

import os
from typing import Any

from ._native import get_native_module


_native = get_native_module()

if _native is not None and hasattr(_native, "RasterInfo"):
    RasterInfo = _native.RasterInfo
else:  # pragma: no cover - exercised only without the compiled extension
    RasterInfo = None


def _require_native():
    native = get_native_module()
    if native is None:
        raise RuntimeError("forge3d native extension is required for forge3d.gis")
    return native


def read_raster_info(path: os.PathLike[str] | str):
    """Read authoritative local GeoTIFF raster metadata."""
    return _require_native().read_raster_info(os.fspath(path))


def write_raster(
    path: os.PathLike[str] | str,
    array: Any,
    *,
    crs: str | dict[str, Any] | None = None,
    transform: tuple[float, float, float, float, float, float] | None = None,
    nodata: float | list[float] | tuple[float, ...] | None = None,
    driver: str = "GTiff",
    overwrite: bool = False,
    creation_options: dict[str, Any] | None = None,
    like_path: os.PathLike[str] | str | None = None,
    like_info: Any | None = None,
):
    """Write a local GeoTIFF and return reopened metadata.

    Accepted array shapes are (height, width) and band-first
    (bands, height, width). HWC arrays are not a separate G-002a1 mode.
    """
    like_path_value = None if like_path is None else os.fspath(like_path)
    return _require_native().write_raster(
        os.fspath(path),
        array,
        crs=crs,
        transform=transform,
        nodata=nodata,
        driver=driver,
        overwrite=overwrite,
        creation_options=creation_options,
        like_path=like_path_value,
        like_info=like_info,
    )


__all__ = ["RasterInfo", "read_raster_info", "write_raster"]

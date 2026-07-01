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

if _native is not None and hasattr(_native, "VectorInfo"):
    VectorInfo = _native.VectorInfo
else:  # pragma: no cover - exercised only without the compiled extension
    VectorInfo = None

if _native is not None and hasattr(_native, "AffineTransform"):
    AffineTransform = _native.AffineTransform
else:  # pragma: no cover - exercised only without the compiled extension
    AffineTransform = None

if _native is not None and hasattr(_native, "CrsTransform"):
    CrsTransform = _native.CrsTransform
else:  # pragma: no cover - exercised only without the compiled extension
    CrsTransform = None

RasterReadResult = dict


def _require_native():
    native = get_native_module()
    if native is None:
        raise RuntimeError("forge3d native extension is required for forge3d.gis")
    return native


def read_raster_info(path: os.PathLike[str] | str):
    """Read authoritative local GeoTIFF raster metadata."""
    return _require_native().read_raster_info(os.fspath(path))


def read_raster(
    path: os.PathLike[str] | str,
    bands: int | list[int] | tuple[int, ...] | None = None,
    window: tuple[int, int, int, int] | dict[str, int] | None = None,
    masked: bool = False,
):
    """Read a local TIFF/GeoTIFF as band-first (bands, height, width) data.

    Band numbers are 1-based. Pixel windows use
    (col_off, row_off, width, height); boundless reads remain
    read_raster_window-only.
    """
    return _require_native().read_raster(
        os.fspath(path),
        bands=bands,
        window=window,
        masked=masked,
    )


def read_vector(
    path: os.PathLike[str] | str,
    *,
    layer: str | None = None,
    columns: list[str] | tuple[str, ...] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    limit: int | None = None,
):
    """Read a local GeoJSON vector source and return a FeatureCollection-like dict."""
    return _require_native().read_vector(
        os.fspath(path),
        layer=layer,
        columns=columns,
        bbox=bbox,
        limit=limit,
    )


def reproject_vector(
    input: os.PathLike[str] | str | dict[str, Any],
    dst_crs: str | int | dict[str, Any],
    src_crs: str | int | dict[str, Any] | None = None,
):
    """Reproject a vector FeatureCollection through the native built-in CRS path."""
    return _require_native().reproject_vector(_path_or_self(input), dst_crs, src_crs)


def geometry_type(source: os.PathLike[str] | str | Any, *, layer: str | None = None) -> str:
    """Return a stable geometry type string for a vector source or GeoJSON-like object."""
    return _require_native().geometry_type(_path_or_self(source), layer=layer)


def vector_schema(source: os.PathLike[str] | str | Any, *, layer: str | None = None):
    """Return vector field schema metadata."""
    return _require_native().vector_schema(_path_or_self(source), layer=layer)


def feature_count(source: os.PathLike[str] | str | Any, *, layer: str | None = None) -> int:
    """Return the feature count for a vector source or FeatureCollection-like dict."""
    return _require_native().feature_count(_path_or_self(source), layer=layer)


def vector_crs(source: os.PathLike[str] | str | Any, *, layer: str | None = None):
    """Return CRS metadata for a vector source without guessing missing CRS."""
    return _require_native().vector_crs(_path_or_self(source), layer=layer)


def vector_bounds(
    source: os.PathLike[str] | str | Any,
    *,
    layer: str | None = None,
) -> tuple[float, float, float, float]:
    """Return vector bounds as (left, bottom, right, top)."""
    return _require_native().vector_bounds(_path_or_self(source), layer=layer)


def _path_or_self(value: Any):
    return os.fspath(value) if isinstance(value, (str, os.PathLike)) else value


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


def parse_crs(value: str | int | dict[str, Any]):
    """Parse a CRS literal without inspecting dataset metadata."""
    return _require_native().parse_crs(value)


def inspect_crs(source: os.PathLike[str] | str | Any):
    """Inspect CRS metadata or a CRS literal without guessing missing CRS."""
    return _require_native().inspect_crs(_path_or_self(source))


def raster_crs(source: os.PathLike[str] | str | Any):
    """Return CRS metadata for a raster path or RasterInfo."""
    return _require_native().raster_crs(_path_or_self(source))


def create_crs_transformer(
    src_crs: str | int | dict[str, Any],
    dst_crs: str | int | dict[str, Any],
    *,
    always_xy: bool = True,
):
    """Create a CRS transformer with explicit axis-order policy."""
    return _require_native().create_crs_transformer(src_crs, dst_crs, always_xy=always_xy)


def transform_bounds(
    src_crs: str | int | dict[str, Any],
    dst_crs: str | int | dict[str, Any],
    bounds: tuple[float, float, float, float],
    *,
    densify: int | None = None,
) -> tuple[float, float, float, float]:
    """Transform bounds; densify must be None or 0 for the built-in backend."""
    return _require_native().transform_bounds(src_crs, dst_crs, bounds, densify=densify)


def bounds(source: os.PathLike[str] | str | Any) -> tuple[float, float, float, float]:
    """Return raster bounds as (left, bottom, right, top)."""
    return raster_bounds(source)


def raster_transform(source: os.PathLike[str] | str | Any):
    """Return a raster affine transform as (a, b, c, d, e, f)."""
    return _require_native().raster_transform(_path_or_self(source))


def transform_from_origin(
    west: float,
    north: float,
    xsize: float,
    ysize: float,
) -> tuple[float, float, float, float, float, float]:
    """Build a north-up affine transform from origin and pixel size."""
    return _require_native().transform_from_origin(west, north, xsize, ysize)


def transform_from_bounds(
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float, float, float]:
    """Build a north-up affine transform from bounds and shape."""
    return _require_native().transform_from_bounds(bounds, width, height)


def array_bounds(
    height: int,
    width: int,
    transform: Any,
) -> tuple[float, float, float, float]:
    """Return array bounds as (left, bottom, right, top)."""
    return _require_native().array_bounds(height, width, transform)


def raster_bounds(source: os.PathLike[str] | str | Any) -> tuple[float, float, float, float]:
    """Return raster bounds as (left, bottom, right, top)."""
    return _require_native().raster_bounds(_path_or_self(source))


def raster_resolution(source: os.PathLike[str] | str | Any) -> tuple[float, float]:
    """Return positive raster pixel resolution."""
    return _require_native().raster_resolution(_path_or_self(source))


def validate_transform(transform: Any, *, require_north_up: bool = False):
    """Validate an affine transform."""
    return _require_native().validate_transform(transform, require_north_up=require_north_up)


def pixel_convention(transform: Any):
    """Return the affine/pixel offset convention."""
    return _require_native().pixel_convention(transform)


def rowcol(transform: Any, x: float, y: float) -> tuple[int, int]:
    """Return (row, col) for world coordinates."""
    return _require_native().rowcol(transform, x, y)


def xy(transform: Any, row: int, col: int) -> tuple[float, float]:
    """Return pixel-center world coordinates."""
    return _require_native().xy(transform, row, col)


def index(transform: Any, x: float, y: float) -> tuple[int, int]:
    """Alias for rowcol using common GIS naming."""
    return _require_native().index(transform, x, y)


def assign_crs(
    source_or_path: os.PathLike[str] | str | Any,
    crs: str | dict[str, Any],
    *,
    overwrite: bool = False,
):
    """Assign CRS metadata without reprojection."""
    return _require_native().assign_crs(
        _path_or_self(source_or_path),
        crs,
        overwrite=overwrite,
    )


def window_from_bounds(
    info_or_path: os.PathLike[str] | str | Any,
    bounds: tuple[float, float, float, float] | dict[str, Any],
    *,
    boundless: bool = False,
) -> dict[str, Any]:
    """Convert geospatial bounds to a pixel window."""
    return _require_native().window_from_bounds(
        _path_or_self(info_or_path),
        bounds,
        boundless=boundless,
    )


def apply_nodata(array: Any, nodata: float | list[float | None] | None, *, mask: Any | None = None):
    """Apply nodata and an optional true-valid mask."""
    return _require_native().apply_nodata(array, nodata, mask=mask)


def read_raster_mask(path: os.PathLike[str] | str, band: int | None = None):
    """Read a true-valid raster mask."""
    return _require_native().read_raster_mask(os.fspath(path), band)


def resample_raster(
    source: os.PathLike[str] | str | Any,
    shape_or_resolution: tuple[int, int] | float | tuple[float, float] | dict[str, Any],
    *,
    method: str | None = None,
) -> dict[str, Any]:
    """Resample; result["info"] is a serialized RasterInfo dict."""
    return _require_native().resample_raster(
        _path_or_self(source),
        shape_or_resolution,
        method=method,
    )


def assert_grid_compatible(left: Any, right: Any, *, compare_nodata: bool = True):
    """Return grid compatibility diagnostics for two rasters."""
    return _require_native().assert_grid_compatible(
        _path_or_self(left),
        _path_or_self(right),
        compare_nodata=compare_nodata,
    )


def align_raster_grid(
    source: os.PathLike[str] | str | Any,
    target_info: os.PathLike[str] | str | Any,
    *,
    resampling: str | None = None,
) -> dict[str, Any]:
    """Sample a raster onto an explicit target grid; result["info"] is a serialized dict."""
    return _require_native().align_raster_grid(
        _path_or_self(source),
        _path_or_self(target_info),
        resampling=resampling,
    )


def align_raster_to(
    source: os.PathLike[str] | str | Any,
    target_info: os.PathLike[str] | str | Any,
    *,
    resampling: str | None = None,
) -> dict[str, Any]:
    """Compatibility alias for align_raster_grid."""
    return align_raster_grid(source, target_info, resampling=resampling)


def reproject_raster(
    source: os.PathLike[str] | str | Any,
    dst_crs: str | dict[str, Any],
    *,
    resampling: str | None = None,
) -> dict[str, Any]:
    """Reproject a raster; result["info"] is a serialized RasterInfo dict."""
    return _require_native().reproject_raster(
        _path_or_self(source),
        dst_crs,
        resampling=resampling,
    )


def calculate_default_transform(
    src_info: os.PathLike[str] | str | Any,
    dst_crs: str | int | dict[str, Any],
    *,
    resolution: float | tuple[float, float] | tuple[int, int] | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate destination transform, shape, bounds, and CRS metadata."""
    return _require_native().calculate_default_transform(
        _path_or_self(src_info),
        dst_crs,
        resolution=resolution,
    )


def read_raster_window(
    path: os.PathLike[str] | str,
    bounds_or_window: tuple[float, float, float, float] | tuple[int, int, int, int] | dict[str, Any],
    *,
    boundless: bool = False,
    masked: bool = False,
) -> dict[str, Any]:
    """Read a raster window; result["info"] is a serialized RasterInfo dict."""
    return _require_native().read_raster_window(
        os.fspath(path),
        bounds_or_window,
        boundless=boundless,
        masked=masked,
    )


def window_transform(
    info_or_path: os.PathLike[str] | str | Any,
    window: tuple[int, int, int, int],
) -> tuple[float, float, float, float, float, float]:
    """Return the affine transform for a pixel window."""
    return _require_native().window_transform(_path_or_self(info_or_path), window)


def web_mercator_bounds(
    bounds: tuple[float, float, float, float],
    src_crs: str | dict[str, Any],
) -> tuple[float, float, float, float]:
    """Transform bounds to EPSG:3857."""
    return _require_native().web_mercator_bounds(bounds, src_crs)


__all__ = [
    "RasterInfo",
    "VectorInfo",
    "AffineTransform",
    "CrsTransform",
    "RasterReadResult",
    "read_raster_info",
    "read_raster",
    "read_vector",
    "reproject_vector",
    "geometry_type",
    "vector_schema",
    "feature_count",
    "vector_crs",
    "vector_bounds",
    "write_raster",
    "parse_crs",
    "inspect_crs",
    "raster_crs",
    "assign_crs",
    "create_crs_transformer",
    "transform_bounds",
    "web_mercator_bounds",
    "raster_transform",
    "transform_from_origin",
    "transform_from_bounds",
    "array_bounds",
    "raster_bounds",
    "raster_resolution",
    "validate_transform",
    "pixel_convention",
    "rowcol",
    "xy",
    "index",
    "apply_nodata",
    "read_raster_mask",
    "resample_raster",
    "assert_grid_compatible",
    "align_raster_grid",
    "align_raster_to",
    "reproject_raster",
    "calculate_default_transform",
    "window_from_bounds",
    "read_raster_window",
    "window_transform",
    "bounds",
]

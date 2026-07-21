"""Thin Python wrappers for Rust-backed GIS raster operations."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

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
_MISSING = object()
_MAX_GEOMETRY_COORDINATES = 250_000


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


def derive_water_mask(
    heightmap: Any,
    *,
    level: float | None = None,
    quantile: float = 0.15,
    slope_threshold: float = 0.02,
) -> np.ndarray:
    """Derive a simple float32 water mask from low, flat DEM regions."""

    dem = np.asarray(heightmap, dtype=np.float32)
    if dem.ndim != 2:
        raise ValueError("heightmap must be a 2D array")
    finite = np.isfinite(dem)
    if not finite.any():
        return np.zeros(dem.shape, dtype=np.float32)
    threshold = float(level) if level is not None else float(np.nanquantile(dem[finite], float(quantile)))
    gy, gx = np.gradient(np.where(finite, dem, threshold))
    slope = np.hypot(gx, gy)
    mask = finite & (dem <= threshold) & (slope <= float(slope_threshold))
    return np.ascontiguousarray(mask.astype(np.float32))


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


def _coordinate_count(value: Any, limit: int) -> int:
    """Count nested coordinate records without expanding oversized sequences."""
    if isinstance(value, Mapping):
        if "coordinates" in value:
            return _coordinate_count(value["coordinates"], limit)
        total = 0
        for item in value.get("geometries", ()) or ():
            total += _coordinate_count(item, max(0, limit - total))
            if total > limit:
                break
        return total
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return 0
        if value.shape[0] > limit:
            return limit + 1
        return int(np.prod(value.shape[:-1])) if value.ndim > 1 else 1
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > limit:
            return limit + 1
        if len(value) and not isinstance(value[0], (Mapping, Sequence, np.ndarray)):
            return 1
        total = 0
        for item in value:
            total += _coordinate_count(item, max(0, limit - total))
            if total > limit:
                break
        return total
    return 0
def validate_geometry(geometry: dict[str, Any]):
    """Validate a GeoJSON-like geometry object."""
    coordinate_count = _coordinate_count(geometry, _MAX_GEOMETRY_COORDINATES)
    if coordinate_count > _MAX_GEOMETRY_COORDINATES:
        raise ValueError(
            f"geometry coordinate count exceeds the safe limit of {_MAX_GEOMETRY_COORDINATES}"
        )
    return _require_native().validate_geometry(geometry)


def is_valid(geometry: dict[str, Any]) -> dict[str, Any]:
    """Return the exact polygonal validity report for a GeoJSON geometry."""
    return _require_native().is_valid(geometry)


def repair_geometry(geometry: dict[str, Any], *, method: str = "make_valid"):
    """Repair a GeoJSON-like geometry object through the native topology backend."""
    return _require_native().repair_geometry(geometry, method=method)


def geometry_measure(
    geometry: dict[str, Any],
    *,
    crs: str | int | dict[str, Any],
    metrics: tuple[str, ...] | list[str] = ("area", "length"),
) -> dict[str, Any]:
    """Measure area and length for a GeoJSON-like geometry object.

    The CRS is mandatory (MENSURA): EPSG:4326 coordinates are measured with
    Karney geodesics on the WGS84 ellipsoid (metres / m^2); projected CRSs are
    measured planar in the CRS units. Geographic CRSs other than EPSG:4326
    raise rather than silently reporting square degrees.
    """
    return _require_native().geometry_measure(geometry, crs=crs, metrics=metrics)


def measure_geometries(
    geometry: dict[str, Any],
    *,
    crs: str | int | dict[str, Any],
    metrics: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Measure geometry; compatibility alias for :func:`geometry_measure`."""
    return geometry_measure(geometry, crs=crs, metrics=metrics)


def geometry_centroid(
    geometry: dict[str, Any],
    *,
    crs: str | int | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the centroid for a GeoJSON-like geometry object.

    ``crs`` selects geographic (EPSG:4326, with antimeridian handling) versus
    planar handling. It is required unless the input carries embedded CRS
    metadata; coordinate ranges are never used to infer it.
    """
    return _require_native().geometry_centroid(geometry, crs=crs)


def representative_point(
    geometry: dict[str, Any],
    *,
    crs: str | int | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a representative point for a GeoJSON-like geometry object.

    See :func:`geometry_centroid` for the ``crs`` contract.
    """
    return _require_native().representative_point(geometry, crs=crs)


def interpolate_line(
    geometry: dict[str, Any],
    distance: float,
    *,
    normalized: bool = False,
    crs: str | int | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Interpolate a point along a LineString or MultiLineString.

    See :func:`geometry_centroid` for the ``crs`` contract. Under EPSG:4326
    ``distance`` is geodesic metres along the line (never degrees); under a
    projected CRS it is planar CRS units. ``normalized=True`` distances are a
    unitless fraction of the total length in either mode.
    """
    return _require_native().interpolate_line(
        geometry,
        distance,
        normalized=normalized,
        crs=crs,
    )


def union_geometries(
    geometries: list[dict[str, Any]] | tuple[dict[str, Any], ...] | dict[str, Any],
    *,
    crs: str | int | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Union GeoJSON-like geometries through the native topology backend.

    Accepts a sequence of geometries/Features or a single FeatureCollection.
    See :func:`geometry_centroid` for the ``crs`` contract; items declaring
    conflicting embedded CRSs raise ``crs_mismatch``.
    """
    return _require_native().union_geometries(geometries, crs=crs)


def union(
    geometries: list[dict[str, Any]] | tuple[dict[str, Any], ...] | dict[str, Any],
    *,
    crs: str | int | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact snap-rounded union of polygonal geometries."""
    return _require_native().union(geometries, crs=crs)


def intersection(
    geometries: list[dict[str, Any]] | tuple[dict[str, Any], ...] | dict[str, Any],
    *,
    crs: str | int | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact snap-rounded intersection of polygonal geometries."""
    return _require_native().intersection(geometries, crs=crs)


def difference(
    geometries: list[dict[str, Any]] | tuple[dict[str, Any], ...] | dict[str, Any],
    *,
    crs: str | int | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Subtract each later polygonal geometry from the first."""
    return _require_native().difference(geometries, crs=crs)


def symmetric_difference(
    geometries: list[dict[str, Any]] | tuple[dict[str, Any], ...] | dict[str, Any],
    *,
    crs: str | int | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact snap-rounded symmetric difference."""
    return _require_native().symmetric_difference(geometries, crs=crs)


def dissolve_vector(source: os.PathLike[str] | str | dict[str, Any], *, by=None):
    """Dissolve a vector source through the native topology backend."""
    return _require_native().dissolve_vector(_path_or_self(source), by=by)


def buffer_geometry(
    geometry: dict[str, Any],
    distance: float,
    *,
    quad_segs: int = 8,
    crs: str | int | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Buffer a GeoJSON-like geometry through the native topology backend.

    See :func:`geometry_centroid` for the ``crs`` contract.
    """
    return _require_native().buffer_geometry(
        geometry, distance, quad_segs=quad_segs, crs=crs
    )


def clip_vector(
    source: os.PathLike[str] | str | dict[str, Any],
    clip_geometry: dict[str, Any],
    *,
    clip_crs: str | int | dict[str, Any] | None = None,
):
    """Clip a vector source through the native topology backend."""
    return _require_native().clip_vector(
        _path_or_self(source),
        clip_geometry,
        clip_crs=clip_crs,
    )


def intersect_vectors(
    left: os.PathLike[str] | str | dict[str, Any],
    right: os.PathLike[str] | str | dict[str, Any],
    *,
    suffixes: tuple[str, str] = ("_left", "_right"),
):
    """Intersect vector sources through the native topology backend."""
    return _require_native().intersect_vectors(
        _path_or_self(left),
        _path_or_self(right),
        suffixes=suffixes,
    )


def simplify_geometry(
    geometry: dict[str, Any],
    tolerance: float,
    *,
    preserve_topology: bool = True,
    crs: str | int | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simplify a GeoJSON-like geometry through the native topology backend.

    See :func:`geometry_centroid` for the ``crs`` contract.
    """
    return _require_native().simplify_geometry(
        geometry,
        tolerance,
        preserve_topology=preserve_topology,
        crs=crs,
    )


def load_boundary(
    path: os.PathLike[str] | str,
    *,
    layer: str | None = None,
    where: str | None = None,
    union: bool = True,
    dst_crs: str | int | dict[str, Any] | None = None,
):
    """Load a vector boundary through the native topology backend."""
    return _require_native().load_boundary(
        os.fspath(path),
        layer=layer,
        where=where,
        union=union,
        dst_crs=dst_crs,
    )


def rasterize_vectors(
    vectors: os.PathLike[str] | str | dict[str, Any],
    target_info: Any,
    *,
    value: float = 1,
    attribute: str | None = None,
    burn_values: float | list[float] | tuple[float, ...] | None = None,
    dtype: str = "uint8",
    fill: float = 0,
    all_touched: bool = False,
    merge_alg: str = "replace",
):
    """Rasterize polygonal vector features onto an explicit target grid."""
    return _require_native().rasterize_vectors(
        _path_or_self(vectors),
        target_info,
        value=value,
        attribute=attribute,
        burn_values=burn_values,
        dtype=dtype,
        fill=fill,
        all_touched=all_touched,
        merge_alg=merge_alg,
    )


def geometry_mask(
    geometries: dict[str, Any],
    target_info: Any,
    *,
    invert: bool = False,
    all_touched: bool = False,
    mask_polarity: str = "true_inside",
):
    """Create a boolean geometry mask on an explicit target grid."""
    return _require_native().geometry_mask(
        _path_or_self(geometries),
        target_info,
        invert=invert,
        all_touched=all_touched,
        mask_polarity=mask_polarity,
    )


def mask_raster(
    source: os.PathLike[str] | str | Any,
    mask: Any,
    *,
    mask_polarity=_MISSING,
    crop: bool = False,
    fill: float | None = None,
    nodata: float | list[float | None] | tuple[float | None, ...] | None = None,
):
    """Apply an explicit true-retain boolean mask to raster data."""
    polarity = None if mask_polarity is _MISSING else mask_polarity
    return _require_native().mask_raster(
        _path_or_self(source),
        mask,
        mask_polarity=polarity,
        crop=crop,
        fill=fill,
        nodata=nodata,
    )


def normalize_raster(
    source: os.PathLike[str] | str | Any,
    *,
    method: str = "minmax",
    valid_mask: Any | None = None,
    nodata: float | list[float | None] | tuple[float | None, ...] | None = None,
    clip: tuple[float, float] | None = None,
):
    """Normalize raster values through the native thematic backend."""
    return _require_native().normalize_raster(
        _path_or_self(source),
        method=method,
        valid_mask=valid_mask,
        nodata=nodata,
        clip=clip,
    )


def classify_raster(
    source: os.PathLike[str] | str | Any,
    *,
    bins: Any = None,
    labels: list[str] | tuple[str, ...] | None = None,
    right: bool = False,
    valid_mask: Any | None = None,
    nodata: float | list[float | None] | tuple[float | None, ...] | None = None,
    dtype: str = "uint16",
):
    """Classify raster values through the native thematic backend."""
    return _require_native().classify_raster(
        _path_or_self(source),
        bins=bins,
        labels=labels,
        right=right,
        valid_mask=valid_mask,
        nodata=nodata,
        dtype=dtype,
    )


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
    height_system: str | None = None,
):
    """Write a local GeoTIFF and return reopened metadata.

    Accepted array shapes are (height, width) and band-first
    (bands, height, width). HWC arrays are not a separate G-002a1 mode.

    ``height_system`` (MENSURA) persists the vertical datum
    ("ellipsoidal" / "orthometric_egm96" / "chart_datum") as a private GeoTIFF
    tag so it survives a write->read round trip; the default inherits from
    ``like_info`` or stays "unspecified".
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
        height_system=height_system,
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
    """Transform bounds, densifying each edge into ``densify`` segments.

    ``densify=None`` samples only the corners (legacy behaviour); pass a
    positive segment count for projections with curved parallels/meridians.
    """
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
    on_transform_error: str = "raise",
) -> dict[str, Any]:
    """Reproject a raster; result["info"] is a serialized RasterInfo dict.

    ``on_transform_error`` controls what happens when a pixel's coordinate
    transform fails: ``"raise"`` (default) raises TransformFailed with the
    failure count and first failing pixel; ``"nodata"`` fills the failing
    pixels with nodata and records a diagnostic.
    """
    return _require_native().reproject_raster(
        _path_or_self(source),
        dst_crs,
        resampling=resampling,
        on_transform_error=on_transform_error,
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


def warped_vrt_info(
    source: os.PathLike[str] | str | Any,
    dst_crs: str | int | dict[str, Any],
    *,
    resampling: str | None = None,
    resolution: float | tuple[float, float] | tuple[int, int] | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return metadata for a virtual destination grid without materializing pixels."""
    return _require_native().warped_vrt_info(
        _path_or_self(source),
        dst_crs,
        resampling=resampling,
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


def fetch_remote_geodata(
    url: str,
    cache: os.PathLike[str] | str | dict[str, Any] | None = None,
    timeout: float | None = None,
    checksum: str | None = None,
) -> dict[str, Any]:
    cache_value = _cache_or_none(cache)
    return _require_native().fetch_remote_geodata(
        url,
        cache=cache_value,
        timeout=timeout,
        checksum=checksum,
    )


def cache_geodata(
    key_or_url: str,
    cache_dir: os.PathLike[str] | str,
    refresh: bool = False,
) -> dict[str, Any]:
    return _require_native().cache_geodata(key_or_url, os.fspath(cache_dir), refresh)


def fetch_vector(
    url: str,
    cache: os.PathLike[str] | str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _require_native().fetch_vector(url, cache=_cache_or_none(cache))


def read_cog(
    path_or_url: os.PathLike[str] | str,
    window: tuple[int, int, int, int] | None = None,
    overview: int | None = None,
) -> dict[str, Any]:
    return _require_native().read_cog(os.fspath(path_or_url), window=window, overview=overview)


def slippy_tile_index(
    bounds: tuple[float, float, float, float],
    zoom: int,
    crs: str = "EPSG:4326",
) -> dict[str, Any]:
    return _require_native().slippy_tile_index(bounds, zoom, crs)


def query_osm_features(
    aoi: tuple[float, float, float, float],
    tags: dict[str, Any],
    cache: dict[str, Any] | None = None,
    *,
    endpoint: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    return _require_native().query_osm_features(
        aoi, tags, cache=cache, endpoint=endpoint, timeout=timeout
    )


def parse_osm_features(osm_json: dict[str, Any] | str, tags: dict[str, Any] | None = None) -> dict[str, Any]:
    return _require_native().parse_osm_features(osm_json, tags=tags)


def load_context_vectors(
    path_or_features: os.PathLike[str] | str | dict[str, Any],
    layers: str | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return _require_native().load_context_vectors(_path_or_self(path_or_features), layers=layers)


def prepare_osm_scene(
    aoi: tuple[float, float, float, float],
    tags: dict[str, Any] | None = None,
    cache: dict[str, Any] | None = None,
    *,
    endpoint: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    return _require_native().prepare_osm_scene(
        aoi, tags=tags, cache=cache, endpoint=endpoint, timeout=timeout
    )


def prepare_dem(
    source: os.PathLike[str] | str | Any,
    target_info: os.PathLike[str] | str | Any | None = None,
    nodata: float | None = None,
) -> dict[str, Any]:
    return _require_native().prepare_dem(
        _path_or_self(source),
        target_info=_path_or_self(target_info) if target_info is not None else None,
        nodata=nodata,
    )


def prepare_terrain_derivatives(
    dem: os.PathLike[str] | str | Any,
    derivatives: list[str] | tuple[str, ...] = ("slope", "hillshade"),
) -> dict[str, Any]:
    return _require_native().prepare_terrain_derivatives(_path_or_self(dem), derivatives=derivatives)


def read_gridded_dataset(path: os.PathLike[str] | str, variable: str | None = None) -> dict[str, Any]:
    return _require_native().read_gridded_dataset(os.fspath(path), variable=variable)


def subset_grid(
    source: os.PathLike[str] | str | Any,
    bounds_or_coords: tuple[float, float, float, float],
    variable: str | None = None,
) -> dict[str, Any]:
    return _require_native().subset_grid(_path_or_self(source), bounds_or_coords, variable=variable)


def decode_terrarium_dem(rgb_array_or_path: os.PathLike[str] | str | Any) -> dict[str, Any]:
    return _require_native().decode_terrarium_dem(_path_or_self(rgb_array_or_path))


def build_terrarium_dem(
    bounds: tuple[float, float, float, float],
    zoom: int,
    cache: os.PathLike[str] | str | dict[str, Any] | None = None,
    *,
    url_template: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    return _require_native().build_terrarium_dem(
        bounds,
        zoom,
        cache=_cache_or_none(cache),
        url_template=url_template,
        timeout=timeout,
    )


def prepare_landcover_raster(
    source: os.PathLike[str] | str | Any,
    target_info: os.PathLike[str] | str | Any,
    classes: dict[int, str] | None = None,
) -> dict[str, Any]:
    return _require_native().prepare_landcover_raster(
        _path_or_self(source),
        _path_or_self(target_info),
        classes=classes,
    )


def prepare_population_raster(
    source: os.PathLike[str] | str | Any,
    target_info: os.PathLike[str] | str | Any | None = None,
    normalization: str | None = None,
) -> dict[str, Any]:
    return _require_native().prepare_population_raster(
        _path_or_self(source),
        target_info=_path_or_self(target_info) if target_info is not None else None,
        normalization=normalization,
    )


def load_building_footprints(
    path_or_features: os.PathLike[str] | str | dict[str, Any],
    dst_crs: str | int | dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _require_native().load_building_footprints(_path_or_self(path_or_features), dst_crs=dst_crs)


def extract_building_heights(
    features: dict[str, Any],
    defaults: dict[str, float] | None = None,
) -> dict[str, Any]:
    return _require_native().extract_building_heights(features, defaults=defaults)


def estimate_local_utm(
    bounds_or_geometry: tuple[float, float, float, float] | dict[str, Any],
) -> dict[str, Any]:
    return _require_native().estimate_local_utm(bounds_or_geometry)


def _cache_or_none(value: os.PathLike[str] | str | dict[str, Any] | None):
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    if "cache_dir" in value:
        return {**value, "cache_dir": os.fspath(value["cache_dir"])}
    return value


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
    "validate_geometry",
    "is_valid",
    "repair_geometry",
    "geometry_measure",
    "measure_geometries",
    "geometry_centroid",
    "representative_point",
    "interpolate_line",
    "union_geometries",
    "union",
    "intersection",
    "difference",
    "symmetric_difference",
    "dissolve_vector",
    "buffer_geometry",
    "clip_vector",
    "intersect_vectors",
    "simplify_geometry",
    "load_boundary",
    "rasterize_vectors",
    "geometry_mask",
    "mask_raster",
    "normalize_raster",
    "classify_raster",
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
    "warped_vrt_info",
    "window_from_bounds",
    "read_raster_window",
    "window_transform",
    "bounds",
    "fetch_remote_geodata",
    "cache_geodata",
    "fetch_vector",
    "read_cog",
    "slippy_tile_index",
    "query_osm_features",
    "parse_osm_features",
    "load_context_vectors",
    "prepare_osm_scene",
    "prepare_dem",
    "prepare_terrain_derivatives",
    "read_gridded_dataset",
    "subset_grid",
    "decode_terrarium_dem",
    "build_terrarium_dem",
    "prepare_landcover_raster",
    "prepare_population_raster",
    "load_building_footprints",
    "extract_building_heights",
    "estimate_local_utm",
]

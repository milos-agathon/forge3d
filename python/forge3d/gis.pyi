from __future__ import annotations

import os
from typing import Any, TypedDict

import numpy as np


class RasterInfo:
    @property
    def path(self) -> str: ...
    @property
    def driver(self) -> str: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def band_count(self) -> int: ...
    @property
    def dtype_per_band(self) -> list[str]: ...
    @property
    def crs_wkt(self) -> str | None: ...
    @property
    def crs_authority(self) -> dict[str, str] | None: ...
    @property
    def transform(self) -> tuple[float, float, float, float, float, float] | None: ...
    @property
    def bounds(self) -> tuple[float, float, float, float] | None: ...
    @property
    def resolution(self) -> tuple[float, float] | None: ...
    @property
    def nodata_per_band(self) -> list[float | None]: ...
    @property
    def block_size(self) -> list[tuple[int, int]] | None: ...
    @property
    def tiling(self) -> str | None: ...
    @property
    def compression(self) -> str | None: ...
    @property
    def is_georeferenced(self) -> bool: ...
    @property
    def warnings(self) -> list[dict[str, str | None]]: ...
    def as_dict(self) -> dict[str, Any]: ...


class VectorInfo:
    @property
    def path(self) -> str: ...
    @property
    def driver(self) -> str: ...
    @property
    def layer_name(self) -> str | None: ...
    @property
    def layer_count(self) -> int: ...
    @property
    def geometry_type(self) -> str: ...
    @property
    def feature_count(self) -> int: ...
    @property
    def schema(self) -> list[dict[str, Any]]: ...
    @property
    def crs_wkt(self) -> str | None: ...
    @property
    def crs_authority(self) -> dict[str, str] | None: ...
    @property
    def bounds(self) -> tuple[float, float, float, float] | None: ...
    @property
    def is_georeferenced(self) -> bool: ...
    @property
    def warnings(self) -> list[dict[str, str | None]]: ...
    def as_dict(self) -> dict[str, Any]: ...


class AffineTransform:
    def __init__(self, a: float, b: float, c: float, d: float, e: float, f: float) -> None: ...
    @property
    def coefficients(self) -> tuple[float, float, float, float, float, float]: ...
    @property
    def resolution(self) -> tuple[float, float]: ...
    @property
    def rotated_or_sheared(self) -> bool: ...


class CrsTransform:
    @staticmethod
    def from_crs(
        src_crs: str | int | dict[str, Any],
        dst_crs: str | int | dict[str, Any],
        *,
        always_xy: bool = ...,
    ) -> CrsTransform: ...
    @property
    def src_crs(self) -> str: ...
    @property
    def dst_crs(self) -> str: ...
    @property
    def src_authority(self) -> dict[str, str] | None: ...
    @property
    def dst_authority(self) -> dict[str, str] | None: ...
    @property
    def axis_order_policy(self) -> str: ...
    def transform_point(self, x: float, y: float) -> tuple[float, float]: ...
    def transform_bounds(
        self,
        bounds: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]: ...


class RasterReadResult(TypedDict):
    array: np.ndarray
    info: dict[str, Any]
    bands: tuple[int, ...]
    window: tuple[int, int, int, int] | None
    window_transform: tuple[float, float, float, float, float, float] | None
    mask: np.ndarray | None
    mask_polarity: str | None
    nodata_per_band: list[float | None]
    warnings: list[dict[str, str | None]]


def read_raster_info(path: os.PathLike[str] | str) -> RasterInfo: ...


def read_raster(
    path: os.PathLike[str] | str,
    bands: int | list[int] | tuple[int, ...] | None = ...,
    window: tuple[int, int, int, int] | dict[str, int] | None = ...,
    masked: bool = ...,
) -> RasterReadResult: ...


def read_vector(
    path: os.PathLike[str] | str,
    *,
    layer: str | None = ...,
    columns: list[str] | tuple[str, ...] | None = ...,
    bbox: tuple[float, float, float, float] | None = ...,
    limit: int | None = ...,
) -> dict[str, Any]: ...


def reproject_vector(
    input: os.PathLike[str] | str | dict[str, Any],
    dst_crs: str | int | dict[str, Any],
    src_crs: str | int | dict[str, Any] | None = ...,
) -> dict[str, Any]: ...


def geometry_type(
    source: os.PathLike[str] | str | VectorInfo | dict[str, Any],
    *,
    layer: str | None = ...,
) -> str: ...


def vector_schema(
    source: os.PathLike[str] | str | VectorInfo | dict[str, Any],
    *,
    layer: str | None = ...,
) -> list[dict[str, Any]]: ...


def feature_count(
    source: os.PathLike[str] | str | VectorInfo | dict[str, Any],
    *,
    layer: str | None = ...,
) -> int: ...


def vector_crs(
    source: os.PathLike[str] | str | VectorInfo | dict[str, Any],
    *,
    layer: str | None = ...,
) -> dict[str, Any]: ...


def vector_bounds(
    source: os.PathLike[str] | str | VectorInfo | dict[str, Any],
    *,
    layer: str | None = ...,
) -> tuple[float, float, float, float]: ...


def validate_geometry(geometry: dict[str, Any]) -> dict[str, Any]: ...


def repair_geometry(
    geometry: dict[str, Any],
    *,
    method: str = ...,
) -> dict[str, Any]: ...


def geometry_measure(
    geometry: dict[str, Any],
    *,
    metrics: tuple[str, ...] | list[str] = ...,
) -> dict[str, Any]: ...


def geometry_centroid(geometry: dict[str, Any]) -> dict[str, Any]: ...


def representative_point(geometry: dict[str, Any]) -> dict[str, Any]: ...


def interpolate_line(
    geometry: dict[str, Any],
    distance: float,
    *,
    normalized: bool = ...,
) -> dict[str, Any]: ...


def union_geometries(geometries: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> dict[str, Any]: ...


def dissolve_vector(
    source: os.PathLike[str] | str | dict[str, Any],
    *,
    by: str | list[str] | tuple[str, ...] | None = ...,
) -> dict[str, Any]: ...


def buffer_geometry(
    geometry: dict[str, Any],
    distance: float,
    *,
    quad_segs: int = ...,
) -> dict[str, Any]: ...


def clip_vector(
    source: os.PathLike[str] | str | dict[str, Any],
    clip_geometry: dict[str, Any],
    *,
    clip_crs: str | int | dict[str, Any] | None = ...,
) -> dict[str, Any]: ...


def intersect_vectors(
    left: os.PathLike[str] | str | dict[str, Any],
    right: os.PathLike[str] | str | dict[str, Any],
    *,
    suffixes: tuple[str, str] = ...,
) -> dict[str, Any]: ...


def simplify_geometry(
    geometry: dict[str, Any],
    tolerance: float,
    *,
    preserve_topology: bool = ...,
) -> dict[str, Any]: ...


def load_boundary(
    path: os.PathLike[str] | str,
    *,
    layer: str | None = ...,
    where: str | None = ...,
) -> dict[str, Any]: ...


def rasterize_vectors(
    vectors: os.PathLike[str] | str | dict[str, Any],
    target_info: RasterInfo | dict[str, Any],
    *,
    value: float = ...,
    attribute: str | None = ...,
    dtype: str = ...,
    fill: float = ...,
    all_touched: bool = ...,
) -> dict[str, Any]: ...


def geometry_mask(
    geometries: os.PathLike[str] | str | dict[str, Any],
    target_info: RasterInfo | dict[str, Any],
    *,
    invert: bool = ...,
    all_touched: bool = ...,
    mask_polarity: str = ...,
) -> dict[str, Any]: ...


def mask_raster(
    source: os.PathLike[str] | str | np.ndarray | dict[str, Any],
    mask: np.ndarray,
    *,
    mask_polarity: str,
    crop: bool = ...,
    fill: float | None = ...,
    nodata: float | list[float | None] | tuple[float | None, ...] | None = ...,
) -> dict[str, Any]: ...


def write_raster(
    path: os.PathLike[str] | str,
    array: np.ndarray,
    *,
    crs: str | dict[str, Any] | None = ...,
    transform: tuple[float, float, float, float, float, float] | None = ...,
    nodata: float | list[float] | tuple[float, ...] | None = ...,
    driver: str = ...,
    overwrite: bool = ...,
    creation_options: dict[str, Any] | None = ...,
    like_path: os.PathLike[str] | str | None = ...,
    like_info: RasterInfo | None = ...,
) -> RasterInfo: ...


def parse_crs(value: str | int | dict[str, Any]) -> dict[str, Any]: ...


def inspect_crs(source: os.PathLike[str] | str | RasterInfo | dict[str, Any]) -> dict[str, Any]: ...


def raster_crs(source: os.PathLike[str] | str | RasterInfo) -> dict[str, Any]: ...


def assign_crs(
    source_or_path: os.PathLike[str] | str | RasterInfo,
    crs: str | dict[str, Any],
    *,
    overwrite: bool = ...,
) -> RasterInfo: ...


def create_crs_transformer(
    src_crs: str | int | dict[str, Any],
    dst_crs: str | int | dict[str, Any],
    *,
    always_xy: bool = ...,
) -> CrsTransform: ...


def transform_bounds(
    src_crs: str | int | dict[str, Any],
    dst_crs: str | int | dict[str, Any],
    bounds: tuple[float, float, float, float],
    *,
    densify: int | None = ...,
) -> tuple[float, float, float, float]: ...


def web_mercator_bounds(
    bounds: tuple[float, float, float, float],
    src_crs: str | dict[str, Any],
) -> tuple[float, float, float, float]: ...


def raster_transform(
    source: os.PathLike[str] | str | RasterInfo,
) -> tuple[float, float, float, float, float, float]: ...


def transform_from_origin(
    west: float,
    north: float,
    xsize: float,
    ysize: float,
) -> tuple[float, float, float, float, float, float]: ...


def transform_from_bounds(
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float, float, float]: ...


def array_bounds(
    height: int,
    width: int,
    transform: AffineTransform | tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float]: ...


def raster_bounds(source: os.PathLike[str] | str | RasterInfo) -> tuple[float, float, float, float]: ...


def raster_resolution(source: os.PathLike[str] | str | RasterInfo) -> tuple[float, float]: ...


def validate_transform(
    transform: AffineTransform | tuple[float, float, float, float, float, float],
    *,
    require_north_up: bool = ...,
) -> dict[str, Any]: ...


def pixel_convention(
    transform: AffineTransform | tuple[float, float, float, float, float, float],
) -> dict[str, Any]: ...


def rowcol(
    transform: AffineTransform | tuple[float, float, float, float, float, float],
    x: float,
    y: float,
) -> tuple[int, int]: ...


def xy(
    transform: AffineTransform | tuple[float, float, float, float, float, float],
    row: int,
    col: int,
) -> tuple[float, float]: ...


def index(
    transform: AffineTransform | tuple[float, float, float, float, float, float],
    x: float,
    y: float,
) -> tuple[int, int]: ...


def apply_nodata(
    array: np.ndarray,
    nodata: float | list[float | None] | None,
    *,
    mask: np.ndarray | None = ...,
) -> dict[str, Any]: ...


def read_raster_mask(path: os.PathLike[str] | str, band: int | None = ...) -> dict[str, Any]: ...


def resample_raster(
    source: os.PathLike[str] | str | RasterInfo | np.ndarray,
    shape_or_resolution: tuple[int, int] | float | tuple[float, float] | dict[str, Any],
    *,
    method: str | None = ...,
) -> dict[str, Any]: ...


def assert_grid_compatible(
    left: os.PathLike[str] | str | RasterInfo,
    right: os.PathLike[str] | str | RasterInfo,
    *,
    compare_nodata: bool = ...,
) -> dict[str, Any]: ...


def align_raster_grid(
    source: os.PathLike[str] | str | RasterInfo | dict[str, Any],
    target_info: os.PathLike[str] | str | RasterInfo,
    *,
    resampling: str | None = ...,
) -> dict[str, Any]: ...


def align_raster_to(
    source: os.PathLike[str] | str | RasterInfo | dict[str, Any],
    target_info: os.PathLike[str] | str | RasterInfo,
    *,
    resampling: str | None = ...,
) -> dict[str, Any]: ...


def reproject_raster(
    source: os.PathLike[str] | str | RasterInfo,
    dst_crs: str | dict[str, Any],
    *,
    resampling: str | None = ...,
) -> dict[str, Any]: ...


def calculate_default_transform(
    src_info: os.PathLike[str] | str | RasterInfo,
    dst_crs: str | int | dict[str, Any],
    *,
    resolution: float | tuple[float, float] | tuple[int, int] | dict[str, Any] | None = ...,
) -> dict[str, Any]: ...


def window_from_bounds(
    info_or_path: os.PathLike[str] | str | RasterInfo,
    bounds: tuple[float, float, float, float] | dict[str, Any],
    *,
    boundless: bool = ...,
) -> dict[str, Any]: ...


def read_raster_window(
    path: os.PathLike[str] | str,
    bounds_or_window: tuple[float, float, float, float] | tuple[int, int, int, int] | dict[str, Any],
    *,
    boundless: bool = ...,
    masked: bool = ...,
) -> dict[str, Any]: ...


def window_transform(
    info_or_path: os.PathLike[str] | str | RasterInfo,
    window: tuple[int, int, int, int],
) -> tuple[float, float, float, float, float, float]: ...


def bounds(source: os.PathLike[str] | str | RasterInfo) -> tuple[float, float, float, float]: ...

"""Typed thematic rasters. Classification/normalization use the native GIS API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from . import gis
from .legend import categorical_legend, bivariate_legend

__all__ = ["RasterHeightSurfaceStyle", "CategoricalRasterStyle", "BivariateRasterStyle", "RasterStyleResult"]


def _finite(value: Any, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_argument: {name} must be a finite number") from exc
    if not np.isfinite(value):
        raise ValueError(f"invalid_argument: {name} must be finite")
    return 0.0 if value == 0 else value


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid_argument: {name} must be nonempty text")
    return value


def _rgba(color: Sequence[int]) -> tuple[int, int, int, int]:
    try:
        valid = len(color) == 4 and all(not isinstance(v, (bool, np.bool_)) and isinstance(v, (int, np.integer)) and 0 <= v <= 255 for v in color)
    except TypeError:
        valid = False
    if not valid:
        raise ValueError("invalid_argument: palette colors must be four integer RGBA bytes in [0, 255]")
    return tuple(int(v) for v in color)


def _nodata(value: float | None) -> float | None:
    return None if value is None else _finite(value, "nodata")


def _single(source: Any) -> np.ndarray:
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        source = gis.read_raster(source, masked=True)
    if isinstance(source, Mapping):
        arr = np.asarray(source["array"])
        mask = source.get("mask")
        if mask is not None:
            arr = arr.astype(np.float64)
            arr[~np.asarray(mask, dtype=bool)] = np.nan
        source = arr
    arr = np.asarray(source)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or not arr.size:
        raise ValueError("shape_mismatch: thematic sources must be nonempty single-band 2D rasters")
    return arr


def _normalized(source: Any, nodata: float | None, valid_mask: Any) -> tuple[np.ndarray, np.ndarray, dict]:
    arr = _single(source)
    result = gis.normalize_raster(arr, nodata=nodata, valid_mask=valid_mask)
    return arr, np.isfinite(result["array"][0]), result


@dataclass(frozen=True)
class RasterStyleResult:
    rgba: np.ndarray
    valid_mask: np.ndarray
    legend: dict[str, Any]
    heights: np.ndarray | None = None
    classes: np.ndarray | None = None
    x_classes: np.ndarray | None = None
    y_classes: np.ndarray | None = None


@dataclass(frozen=True)
class RasterHeightSurfaceStyle:
    """Height = population × height_scale (height_units per source_units).

    Shade interpolates low/high RGBA using native minmax normalization.
    Constant sources receive the low color; invalid cells are holes.
    """
    height_scale: float
    source_units: str
    height_units: str = "m"
    nodata: float | None = None
    low_color: tuple[int, int, int, int] = (32, 32, 32, 255)
    high_color: tuple[int, int, int, int] = (240, 240, 240, 255)

    def __post_init__(self) -> None:
        scale = _finite(self.height_scale, "height_scale")
        if scale <= 0:
            raise ValueError("invalid_argument: height_scale must be positive")
        object.__setattr__(self, "height_scale", scale)
        _text(self.source_units, "source_units")
        _text(self.height_units, "height_units")
        object.__setattr__(self, "nodata", _nodata(self.nodata))
        for key in ("low_color", "high_color"):
            object.__setattr__(self, key, _rgba(getattr(self, key)))

    def apply(self, source: Any, *, valid_mask: Any = None) -> RasterStyleResult:
        arr, valid, result = _normalized(source, self.nodata, valid_mask)
        if np.any(arr[valid] < 0):
            raise ValueError("invalid_argument: population values must be nonnegative outside nodata")
        with np.errstate(over="ignore", invalid="ignore"):
            heights64 = arr.astype(np.float64) * self.height_scale
        if np.any(~np.isfinite(heights64[valid])) or np.any(heights64[valid] > np.finfo(np.float32).max):
            raise ValueError("invalid_argument: population heights exceed finite float32 range")
        heights = np.full(arr.shape, np.nan, dtype=np.float32)
        heights[valid] = heights64[valid]
        t = np.nan_to_num(result["array"][0])[..., None]
        low, high = np.asarray(self.low_color), np.asarray(self.high_color)
        rgba = np.rint(low + t * (high - low)).astype(np.uint8)
        rgba[~valid] = 0
        legend = {"kind": "population_height_shade", "source_units": self.source_units,
                  "height_units": self.height_units, "height_scale": self.height_scale,
                  "source_range": [result["min"], result["max"]],
                  "height_range": [float(heights[valid].min()), float(heights[valid].max())],
                  "low_rgba": list(self.low_color), "high_rgba": list(self.high_color)}
        return RasterStyleResult(rgba, valid, legend, heights=heights)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "population_height_shade", "height_scale": self.height_scale,
                "source_units": self.source_units, "height_units": self.height_units,
                "nodata": self.nodata, "low_color": list(self.low_color), "high_color": list(self.high_color)}


@dataclass(frozen=True)
class CategoricalRasterStyle:
    palette: Mapping[int, Sequence[int]]
    labels: Mapping[int, str] | None = None
    nodata: float | None = None
    units: str = ""

    def __post_init__(self) -> None:
        from types import MappingProxyType
        if not self.palette or any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) for k in self.palette):
            raise ValueError("invalid_argument: categorical palette needs integer class IDs")
        palette = {int(k): _rgba(v) for k, v in sorted(self.palette.items())}
        labels = dict(self.labels) if self.labels is not None else {k: str(k) for k in palette}
        if labels.keys() != palette.keys():
            raise ValueError("invalid_argument: categorical labels must match palette class IDs")
        for label in labels.values():
            _text(label, "class label")
        if not isinstance(self.units, str):
            raise ValueError("invalid_argument: units must be text")
        object.__setattr__(self, "palette", MappingProxyType(palette))
        object.__setattr__(self, "labels", MappingProxyType(labels))
        object.__setattr__(self, "nodata", _nodata(self.nodata))

    def legend(self) -> dict[str, Any]:
        return categorical_legend(self.palette, self.labels, units=self.units)

    def apply(self, source: Any, *, valid_mask: Any = None) -> RasterStyleResult:
        arr, valid, _ = _normalized(source, self.nodata, valid_mask)
        values = arr[valid]
        if np.any(values != np.floor(values)):
            raise ValueError("invalid_argument: categorical raster must contain integer class IDs")
        keys = list(self.palette)
        unknown = np.unique(values[~np.isin(values, keys)])
        if unknown.size:
            raise ValueError(f"invalid_argument: categorical palette has no class IDs {unknown.tolist()}")
        # The classifier numbers valid bins from 1 and reserves 0 for nodata.
        # right=True maps each exact source ID to its sorted palette entry.
        if len(keys) == 1:
            classes = np.where(valid, 1, 0).astype(np.uint16)
        else:
            classes = gis.classify_raster(arr, bins=keys[:-1], right=True, valid_mask=valid, nodata=self.nodata)["array"][0]
        colors = np.asarray([(0, 0, 0, 0), *self.palette.values()], dtype=np.uint8)
        return RasterStyleResult(colors[classes], valid, self.legend(), classes=classes)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "categorical", "classes": self.legend()["entries"], "nodata": self.nodata, "units": self.units}


@dataclass(frozen=True)
class BivariateRasterStyle:
    """Fixed 3×3: palette[y_bin][x_bin], low to high on both axes."""
    x_bins: Sequence[float]
    y_bins: Sequence[float]
    palette: Sequence[Sequence[Sequence[int]]]
    x_label: str
    y_label: str
    x_labels: Sequence[str] = ("low", "middle", "high")
    y_labels: Sequence[str] = ("low", "middle", "high")
    x_units: str = ""
    y_units: str = ""
    right: bool = False
    x_nodata: float | None = None
    y_nodata: float | None = None

    def __post_init__(self) -> None:
        for key in ("x_bins", "y_bins"):
            try:
                bins = tuple(_finite(v, key) for v in getattr(self, key))
            except TypeError as exc:
                raise ValueError(f"invalid_argument: {key} must contain two increasing finite boundaries for 3 bins") from exc
            if len(bins) != 2 or bins[0] >= bins[1]:
                raise ValueError(f"invalid_argument: {key} must contain two increasing finite boundaries for 3 bins")
            object.__setattr__(self, key, bins)
        try:
            matrix_valid = len(self.palette) == 3 and all(len(row) == 3 for row in self.palette)
        except TypeError:
            matrix_valid = False
        if not matrix_valid:
            raise ValueError("shape_mismatch: bivariate palette must be 3x3, indexed [y][x]")
        object.__setattr__(self, "palette", tuple(tuple(_rgba(c) for c in row) for row in self.palette))
        for key in ("x_label", "y_label"):
            _text(getattr(self, key), key)
        for key in ("x_labels", "y_labels"):
            try:
                labels = tuple(getattr(self, key))
            except TypeError as exc:
                raise ValueError(f"invalid_argument: {key} needs three bin labels") from exc
            if len(labels) != 3:
                raise ValueError(f"invalid_argument: {key} needs three bin labels")
            for label in labels:
                _text(label, key)
            object.__setattr__(self, key, labels)
        for key in ("x_units", "y_units"):
            if not isinstance(getattr(self, key), str):
                raise ValueError(f"invalid_argument: {key} must be text")
        if not isinstance(self.right, bool):
            raise ValueError("invalid_argument: right must be boolean")
        for key in ("x_nodata", "y_nodata"):
            object.__setattr__(self, key, _nodata(getattr(self, key)))

    def legend(self) -> dict[str, Any]:
        return bivariate_legend(self.to_dict())

    def apply(self, x: Any, y: Any, *, valid_mask: Any = None) -> RasterStyleResult:
        x, y = _single(x), _single(y)
        if x.shape != y.shape:
            raise ValueError("shape_mismatch: bivariate raster axes must have identical grids")
        x_result = gis.classify_raster(x, bins=self.x_bins, labels=self.x_labels, right=self.right, nodata=self.x_nodata, valid_mask=valid_mask)
        y_result = gis.classify_raster(y, bins=self.y_bins, labels=self.y_labels, right=self.right, nodata=self.y_nodata, valid_mask=valid_mask)
        xc, yc = x_result["array"][0], y_result["array"][0]
        valid = (xc != 0) & (yc != 0)
        if not valid.any():
            raise ValueError("empty_raster: bivariate axes have no jointly valid cells")
        xc, yc = np.where(valid, xc, 0), np.where(valid, yc, 0)
        rgba = np.zeros((*x.shape, 4), dtype=np.uint8)
        rgba[valid] = np.asarray(self.palette, dtype=np.uint8)[yc[valid] - 1, xc[valid] - 1]
        return RasterStyleResult(rgba, valid, self.legend(), classes=np.where(valid, (yc - 1) * 3 + xc, 0), x_classes=xc, y_classes=yc)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "bivariate", "x_bins": list(self.x_bins), "y_bins": list(self.y_bins),
                "palette": [[list(c) for c in row] for row in self.palette],
                "x_label": self.x_label, "y_label": self.y_label,
                "x_labels": list(self.x_labels), "y_labels": list(self.y_labels),
                "x_units": self.x_units, "y_units": self.y_units, "right": self.right,
                "x_nodata": self.x_nodata, "y_nodata": self.y_nodata}


def raster_style_from_dict(data: Mapping[str, Any] | None):
    if data is None:
        return None
    values = dict(data)
    kind = values.pop("kind")
    if kind == "population_height_shade":
        return RasterHeightSurfaceStyle(**values)
    if kind == "categorical":
        rows = values.pop("classes")
        return CategoricalRasterStyle(palette={r["class_id"]: r["rgba"] for r in rows}, labels={r["class_id"]: r["label"] for r in rows}, **values)
    if kind == "bivariate":
        return BivariateRasterStyle(**values)
    raise ValueError(f"invalid_argument: unknown raster style {kind!r}")


def raster_values_to_dict(data: Any) -> dict | None:
    """Encode nonfinite source cells as null holes, never JSON NaN."""
    if data is None:
        return None
    arr = _single(data)
    if arr.dtype.kind not in "iuf":
        raise ValueError("unsupported_dtype: thematic raster values must be numeric")
    values = arr.astype(object)
    values[~np.isfinite(arr)] = None
    values[arr == 0] = 0  # canonical negative zero
    return {"dtype": str(arr.dtype), "values": values.tolist()}


def raster_values_from_dict(data: Mapping | None):
    if data is None:
        return None
    values = np.asarray(data["values"], dtype=data["dtype"])
    # Serialized parameters cannot smuggle JSON NaN or Infinity into a recipe.
    raw = np.asarray(data["values"], dtype=object)
    finite_input = np.fromiter((v is not None for v in raw.flat), dtype=bool).reshape(raw.shape)
    if np.any(~np.isfinite(values[finite_input])):
        raise ValueError("invalid_argument: nonfinite raster values must be encoded as null nodata cells")
    return values

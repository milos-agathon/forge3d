"""Local, semantic data preparation for MapScene, independent of rendering.

GeoTIFF uses the existing rasterio alignment backend. Prepared NumPy grids and
GeoJSON require declared coordinates. Counts retain their native cells; only a
display of those cells may subsequently be sampled onto the terrain grid.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np

from .alignment import _affine_from_metadata, _require_rasterio, _resampling_method
from .diagnostics import Diagnostic

__all__ = [
    "PreparationError", "PreparationGrid", "PreparationSource", "PreparationRecipe",
    "PreparedLayer", "PreparedScene", "prepare_scene",
]

_KINDS = {"elevation", "continuous", "categorical", "count", "imagery", "vector", "observation"}
_METRES = {"m": 1.0, "ft": 0.3048, "us_survey_ft": 1200.0 / 3937.0}


class PreparationError(ValueError):
    """A machine-readable failure with a correction a mapmaker can perform."""

    def __init__(self, code: str, message: str, remediation: str, source_id: str | None = None):
        self.diagnostic = Diagnostic(code, "error", message, remediation, layer_id=source_id)
        super().__init__(f"{code}: {message} {remediation}")


def _fail(code: str, message: str, fix: str, source: str | None = None) -> NoReturn:
    raise PreparationError(code, message, fix, source)


def _json(value: Any) -> str:
    def canonical(item: Any) -> Any:
        if isinstance(item, float) and item == 0:
            return 0.0
        if isinstance(item, dict):
            return {key: canonical(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [canonical(child) for child in item]
        return item
    return json.dumps(canonical(value), sort_keys=True, indent=2, allow_nan=False, ensure_ascii=True)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local(path: str | Path, base: Path) -> Path:
    text = str(path)
    if "://" in text or text.startswith(("/vsi", "\\\\")):
        _fail("unsupported_source_route", text, "Use a recorded local GeoTIFF, .npy grid or GeoJSON file.")
    candidate = Path(text)
    if candidate.suffix.lower() not in {".tif", ".tiff", ".npy", ".json", ".geojson"}:
        _fail("unsupported_source_route", candidate.suffix, "Use a local GeoTIFF, .npy grid/mask or GeoJSON file.")
    candidate = candidate if candidate.is_absolute() else base / candidate
    if not candidate.is_file():
        _fail("missing_local_asset", str(candidate), "Correct the recipe path or restore the recorded asset.")
    return candidate.resolve()


@dataclass(frozen=True)
class PreparationGrid:
    """Pixel-area grid. ``geotransform`` uses GDAL order, at pixel corners."""

    crs: str
    width: int
    height: int
    geotransform: Sequence[float]

    def __post_init__(self) -> None:
        rasterio = _require_rasterio()
        try:
            parsed_crs = rasterio.crs.CRS.from_user_input(self.crs)
        except (ValueError, rasterio.errors.CRSError) as exc:
            raise PreparationError("invalid_crs", str(self.crs), "Supply a verified CRS definition such as EPSG:28992.") from exc
        wkt = parsed_crs.to_wkt()
        if any(wkt.startswith(prefix) for prefix in ("COMPD_CS[", "COMPOUNDCRS[", "VERT_CS[", "VERTCRS[")):
            _fail("compound_crs_requires_preparation", "Vertical/compound CRS warps are not supported in this route.",
                  "Use a verified horizontal 2D CRS and declare height units/datum separately, avoiding implicit double vertical conversions.")
        # WKT1 may hide a geographic CRS's third axis. Read the explicit WKT2
        # coordinate-system dimension before admitting a two-dimensional grid.
        dimension = re.search(r"\bCS\s*\[\s*[^,\]]+,\s*(\d+)", parsed_crs.to_wkt(version="WKT2_2019"))
        if dimension is None or int(dimension[1]) != 2:
            _fail("unsupported_grid_crs", "Raster grids require a horizontal two-dimensional CRS.",
                  "Prepare a verified 2D horizontal grid and declare height units/datum separately; geographic/geocentric 3D axes are not implicitly discarded.")
        if (isinstance(self.width, bool) or isinstance(self.height, bool)
                or int(self.width) != self.width or int(self.height) != self.height
                or self.width <= 0 or self.height <= 0):
            _fail("invalid_grid", "Grid dimensions must be positive integers.", "Supply the actual raster width and height.")
        gt = tuple(float(v) for v in self.geotransform)
        if len(gt) != 6 or not all(math.isfinite(v) for v in gt):
            _fail("invalid_grid", "A finite six-value GDAL transform is required.", "Copy the source corner transform or specify the target grid.")
        if gt[1] * gt[5] - gt[2] * gt[4] == 0:
            _fail("invalid_grid", "Grid transform is singular.", "Supply nonzero independent pixel axes.")
        object.__setattr__(self, "geotransform", gt)
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))

    @property
    def transform(self) -> Any:
        return _affine_from_metadata(self.geotransform)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["geotransform"] = list(self.geotransform)
        return result

    def same_as(self, other: PreparationGrid) -> bool:
        crs = _require_rasterio().crs.CRS
        return (self.width == other.width and self.height == other.height
                and self.geotransform == other.geotransform
                and crs.from_user_input(self.crs) == crs.from_user_input(other.crs))

    def xy(self, column: float, row: float) -> tuple[float, float]:
        """World position of a sample centre, including rotated/sheared grids."""
        return self.transform * (column + 0.5, row + 0.5)


@dataclass(frozen=True)
class PreparationSource:
    source_id: str
    path: str | Path
    kind: str
    crs: str | None = None
    grid: PreparationGrid | None = None
    units: str | None = None
    vertical_reference: str | None = None
    nodata: float | int | None = None
    mask_path: str | Path | None = None
    resampling: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id or self.kind not in _KINDS:
            _fail("invalid_source", "A stable source_id and supported kind are required.", f"Choose one of {sorted(_KINDS)}.")
        if self.kind in {"elevation", "continuous", "count", "observation"} and not self.units:
            _fail("missing_units", "Measurement units are missing.", "Declare units; counts use an explicit unit such as people/cell.", self.source_id)
        if self.kind == "elevation" and (not self.vertical_reference or self.vertical_reference == "unspecified"):
            _fail("missing_vertical_reference", "Elevation vertical reference is missing.", "Declare the measured datum (for example NAP, LN02, ellipsoidal or orthometric_egm96).", self.source_id)
        if self.kind == "elevation" and self.units not in _METRES:
            _fail("unsupported_height_units", str(self.units), "Use m, ft or us_survey_ft, or explicitly prepare the source externally.", self.source_id)
        if self.nodata is not None and not math.isfinite(self.nodata):
            _fail("invalid_nodata", "Explicit nodata must be finite.", "Omit nodata for IEEE NaN/infinity; nonfinite samples are always masked.", self.source_id)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        result["mask_path"] = str(self.mask_path) if self.mask_path is not None else None
        if self.grid:
            result["grid"] = self.grid.to_dict()
        return result


@dataclass(frozen=True)
class PreparationRecipe:
    target_grid: PreparationGrid
    sources: Sequence[PreparationSource]
    vertical_reference: str
    # Each checkpoint supplies source_id, source_xy, expected_target_xy and an
    # owner/source-accuracy-derived tolerance in target CRS units.
    checkpoints: Sequence[Mapping[str, Any]] = ()
    display_styles: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [s.source_id for s in self.sources]
        if len(ids) != len(set(ids)) or sum(s.kind == "elevation" for s in self.sources) != 1:
            _fail("invalid_recipe", "Exactly one elevation source and unique source IDs are required.", "Choose one terrain and give every input a stable identity.")
        if not self.vertical_reference or self.vertical_reference == "unspecified":
            _fail("missing_vertical_reference", "Target height reference is missing.", "Declare the target datum; retain the source datum when no qualified conversion exists.")
        _validate_display_styles(self.display_styles, self.sources)
        for check in self.checkpoints:
            required = {"source_id", "expected_target_xy", "tolerance"}
            allowed = required | {"source_xy", "source_pixel"}
            if (not isinstance(check, Mapping) or not required <= check.keys()
                    or check.keys() - allowed or ("source_xy" in check) == ("source_pixel" in check)):
                _fail("invalid_recipe", "Malformed checkpoint fields.",
                      "Supply source_id, expected_target_xy, tolerance and exactly one of source_xy or source_pixel.")
            try:
                points = np.asarray([check.get("source_xy", check.get("source_pixel")), check["expected_target_xy"]], dtype=float)
                tolerance = float(check["tolerance"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise PreparationError("invalid_recipe", "Malformed checkpoint coordinates or tolerance.",
                                       "Use two finite coordinates and a finite nonnegative tolerance.") from exc
            if points.shape != (2, 2) or not np.all(np.isfinite(points)) or not math.isfinite(tolerance) or tolerance < 0:
                _fail("invalid_recipe", "Malformed checkpoint coordinates or tolerance.",
                      "Use two finite coordinates and a finite nonnegative tolerance.")
            source = next((s for s in self.sources if s.source_id == check["source_id"]), None)
            if source is None or ("source_pixel" in check and source.kind in {"vector", "observation"}):
                _fail("invalid_recipe", "Pixel checkpoints require a raster source; checkpoint source_id must exist.",
                      "For vectors and observations use source_xy, or choose a raster source for source_pixel.")
        try:
            _json(self.to_dict())
        except (TypeError, ValueError, OverflowError) as exc:
            raise PreparationError("invalid_recipe", f"Recipe is not finite JSON: {exc}",
                                   "Use finite JSON values in metadata and settings; replace NaN/infinity with an explicit missing-value representation.") from exc

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "target_grid": self.target_grid.to_dict(),
                "sources": [s.to_dict() for s in self.sources],
                "vertical_reference": self.vertical_reference, "checkpoints": list(self.checkpoints),
                "display_styles": dict(self.display_styles)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PreparationRecipe:
        try:
            return cls._from_dict(value)
        except PreparationError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
            raise PreparationError("invalid_recipe", f"Malformed recipe: {exc}",
                                   "Check required fields and spelling against PreparationRecipe schema 1; use the recorded recipe as a template.") from exc

    @classmethod
    def _from_dict(cls, value: Mapping[str, Any]) -> PreparationRecipe:
        if not isinstance(value, Mapping):
            _fail("invalid_recipe", "The recipe must be a JSON object.", "Use a schema 1 PreparationRecipe object.")
        allowed = {"schema_version", "target_grid", "sources", "vertical_reference", "checkpoints", "display_styles"}
        if value.keys() - allowed:
            _fail("invalid_recipe", f"Unknown recipe fields: {sorted(value.keys() - allowed)}.", "Correct the field names; unknown fields are not ignored.")
        if value.get("schema_version") != 1:
            _fail("unsupported_recipe_version", str(value.get("schema_version")), "Use preparation recipe schema 1.")
        sources = []
        for item in value["sources"]:
            item = dict(item)
            if item.get("grid"):
                item["grid"] = PreparationGrid(**item["grid"])
            sources.append(PreparationSource(**item))
        return cls(PreparationGrid(**value["target_grid"]), tuple(sources),
                   value["vertical_reference"], tuple(value.get("checkpoints", ())), value.get("display_styles", {}))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(_json(self.to_dict()) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PreparationRecipe:
        """Resolve relative source/mask paths relative to the recipe file."""
        path = Path(path).resolve()
        try:
            value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda v:
                               _fail("invalid_recipe", f"Non-finite JSON constant {v}.", "Replace it with finite metadata or an explicit missing-value representation."))
        except (OSError, ValueError) as exc:
            raise PreparationError("invalid_recipe", f"Cannot read recipe {path}: {exc}",
                                   "Supply a readable UTF-8 schema 1 JSON recipe.") from exc
        recipe = cls.from_dict(value)
        value = recipe.to_dict()
        for source in value["sources"]:
            for key in ("path", "mask_path"):
                if source.get(key) and not Path(source[key]).is_absolute():
                    source[key] = str(path.parent / source[key])
        return cls.from_dict(value)


@dataclass
class PreparedLayer:
    source_id: str
    kind: str
    grid: PreparationGrid | None
    data: np.ndarray | None
    valid_mask: np.ndarray | None
    features: list[dict[str, Any]] | None
    metadata: dict[str, Any]


def _read_raster(source: PreparationSource, path: Path, base: Path) -> tuple[np.ndarray, np.ndarray, PreparationGrid, dict[str, Any]]:
    try:
        return _decode_raster(source, path, base)
    except PreparationError:
        raise
    except (OSError, ValueError, _require_rasterio().errors.RasterioError) as exc:
        raise PreparationError("raster_read_failed", str(exc),
                               "Check the local format, numeric bands, CRS and declared mask/grid; no alternate source is selected.", source.source_id) from exc


def _decode_raster(source: PreparationSource, path: Path, base: Path) -> tuple[np.ndarray, np.ndarray, PreparationGrid, dict[str, Any]]:
    extra: dict[str, Any] = {}
    if path.suffix.lower() in {".tif", ".tiff"}:
        with _require_rasterio().open(path) as ds:
            if ds.crs is None:
                _fail("missing_crs", "GeoTIFF has no CRS.", "Assign the verified source CRS in an external preparation step.", source.source_id)
            if source.crs and _require_rasterio().crs.CRS.from_user_input(source.crs) != ds.crs:
                _fail("conflicting_crs", "Declared CRS differs from GeoTIFF metadata.", "Correct the input metadata or source declaration; do not relabel coordinates.", source.source_id)
            grid = PreparationGrid(str(ds.crs), ds.width, ds.height, ds.transform.to_gdal())
            if source.grid and not source.grid.same_as(grid):
                _fail("conflicting_grid", "Declared grid differs from GeoTIFF metadata.", "Correct the grid declaration or the source geotransform.", source.source_id)
            data = ds.read()
            valid = ds.read_masks() != 0
            if source.nodata is not None:
                valid &= data != source.nodata
            extra = {"tags": ds.tags(), "scales": list(ds.scales), "offsets": list(ds.offsets),
                     "file_units": list(ds.units), "file_nodata": [str(n) if n is not None else None for n in ds.nodatavals]}
            unit_aliases = {"metre": "m", "metres": "m", "meter": "m", "meters": "m", "feet": "ft", "foot": "ft"}
            for file_unit in ds.units:
                if file_unit and source.units:
                    declared = unit_aliases.get(source.units.lower(), source.units.lower())
                    encoded = unit_aliases.get(file_unit.lower(), file_unit.lower())
                    if declared != encoded:
                        _fail("conflicting_units", f"File units {file_unit!r} differ from declared units {source.units!r}.",
                              "Correct the declaration or explicitly convert the source first.", source.source_id)
            file_vertical = ds.tags().get("vertical_reference") or ds.tags().get("height_system")
            if file_vertical and source.kind == "elevation" and file_vertical.casefold() != str(source.vertical_reference).casefold():
                _fail("conflicting_vertical_reference", f"File height reference {file_vertical!r} differs from {source.vertical_reference!r}.",
                      "Correct the declaration or supply a documented external datum conversion.", source.source_id)
            extra["source_pixel_registration"] = ds.tags().get("AREA_OR_POINT", "not declared")
            extra["source_geotransform_convention"] = "GDAL corner-normalized; samples at half-pixel centres"
            extra["ancillary_identities"] = [{"path": str(Path(p).resolve()), "sha256": _sha(Path(p))}
                                             for p in ds.files if Path(p).resolve() != path]
            if any(s != 1 for s in ds.scales) or any(o != 0 for o in ds.offsets):
                if source.kind in {"categorical", "count"}:
                    _fail("encoded_discrete_values", "Discrete data has scale/offset encoding.", "Decode and verify class/count meaning externally before preparation.", source.source_id)
                data = data.astype(np.float64)
                for band, (scale, offset) in enumerate(zip(ds.scales, ds.offsets)):
                    data[band] = data[band] * scale + offset
    elif path.suffix.lower() == ".npy":
        if source.grid is None:
            _fail("missing_grid", "NumPy arrays have no geographic metadata.", "Declare a source grid with CRS and corner transform.", source.source_id)
        grid = source.grid
        if source.crs and _require_rasterio().crs.CRS.from_user_input(source.crs) != _require_rasterio().crs.CRS.from_user_input(grid.crs):
            _fail("conflicting_crs", "Source CRS differs from its grid.", "Correct the source CRS declaration.", source.source_id)
        data = np.load(path, allow_pickle=False)
        if data.ndim == 2:
            data = data[None, ...]
        valid = np.ones(data.shape, dtype=bool)
        if source.nodata is not None:
            valid &= data != source.nodata
    else:
        _fail("unsupported_source_route", path.suffix, "Use GeoTIFF or an explicitly georeferenced .npy grid.", source.source_id)
    if data.dtype.kind not in "iuf" or data.ndim != 3 or data.shape[1:] != (grid.height, grid.width):
        _fail("invalid_raster", "Raster must be numeric, band-first, and match its grid.", "Correct the array shape/type or grid dimensions.", source.source_id)
    if source.kind != "imagery" and data.shape[0] != 1:
        _fail("invalid_band_count", "A semantic band must be single-band.", "Declare each measurement/class band as a separate source.", source.source_id)
    if source.kind == "imagery" and data.shape[0] not in {1, 3, 4}:
        _fail("invalid_band_count", "Imagery requires 1, 3 or 4 bands.", "Prepare an explicit grayscale/RGB/RGBA composite.", source.source_id)
    valid &= np.isfinite(data)
    if source.mask_path:
        mask = np.load(_local(source.mask_path, base), allow_pickle=False)
        if mask.dtype != np.bool_ or mask.shape != data.shape[1:]:
            _fail("invalid_mask", "Validity mask must be boolean and match the source grid.", "Supply a .npy mask with True for usable cells, including disclosure rules.", source.source_id)
        valid &= mask[None, ...]
    if not np.any(valid):
        _fail("empty_source", "No valid samples remain.", "Check nodata and disclosure masks or select a covered region.", source.source_id)
    return data, valid, grid, extra


def _warp(data: np.ndarray, valid: np.ndarray, source: PreparationGrid,
          target: PreparationGrid, resampling: str) -> tuple[np.ndarray, np.ndarray]:
    """Use the same GDAL warp/transform convention as alignment.py, with masks.

    Float64 scratch values avoid narrowing scientific inputs. Discrete warps
    require exactly representable values and cast back only after nearest copy.
    """
    from rasterio.warp import reproject
    destination = np.full((data.shape[0], target.height, target.width), np.nan, dtype=np.float64)
    for band in range(data.shape[0]):
        values = data[band].astype(np.float64)
        values[~valid[band]] = np.nan
        reproject(values, destination[band], src_transform=source.transform, src_crs=source.crs,
                  src_nodata=np.nan, dst_transform=target.transform, dst_crs=target.crs,
                  dst_nodata=np.nan, resampling=_resampling_method(resampling))
    return destination, np.isfinite(destination)


def _height_conversion(data: np.ndarray, valid: np.ndarray, grid: PreparationGrid,
                       source: PreparationSource, target_reference: str) -> np.ndarray:
    result = data.astype(np.float64) * _METRES[str(source.units)]
    if source.vertical_reference == target_reference:
        return result
    pair = (source.vertical_reference, target_reference)
    if pair not in {("orthometric_egm96", "ellipsoidal"), ("ellipsoidal", "orthometric_egm96")}:
        _fail("unsupported_vertical_transform", f"{pair[0]} -> {pair[1]} is not implemented.",
              "Retain the measured datum or supply an externally transformed, documented source (NAP/LN02 conversions are not inferred).", source.source_id)
    from .crs import transform_coords, geoid_undulation
    rows, cols = np.nonzero(valid[0])
    xy = np.array([grid.xy(int(c), int(r)) for r, c in zip(rows, cols)])
    lonlat = transform_coords(xy, grid.crs, "EPSG:4326")
    sign = 1 if pair[0] == "orthometric_egm96" else -1
    for row, col, (lon, lat) in zip(rows, cols, lonlat):
        result[0, row, col] += sign * geoid_undulation(float(lat), float(lon))
    return result


def _transform_geometry_checked(geometry: Mapping[str, Any], source_crs: str, target_crs: str) -> dict[str, Any]:
    from .alignment import transform_geometry
    kind = geometry.get("type")
    if kind in {"Point", "LineString", "Polygon"}:
        # Existing transform_geometry preserves Z; here Z is refused because
        # an undeclared vector height datum cannot be transformed honestly.
        def check(coords: Any) -> None:
            if not isinstance(coords, (list, tuple)) or not coords:
                _fail("invalid_geometry", "Empty or malformed coordinates.", "Supply a valid 2D GeoJSON geometry.")
            if isinstance(coords[0], (int, float)):
                if len(coords) != 2 or not all(math.isfinite(float(v)) for v in coords):
                    _fail("invalid_geometry", "Only finite 2D coordinates are supported.", "Prepare 3D features with a documented height operation externally.")
            else:
                for child in coords:
                    check(child)
        check(geometry.get("coordinates"))
        coords = geometry["coordinates"]
        if kind == "LineString" and len(coords) < 2:
            _fail("invalid_geometry", "A line needs at least two points.", "Repair the source GeoJSON geometry.")
        if kind == "Polygon" and any(len(ring) < 4 or ring[0] != ring[-1] for ring in coords):
            _fail("invalid_geometry", "Polygon rings must be closed and contain at least four positions.", "Repair the source rings; no synthetic closure is inserted.")
        result = transform_geometry(geometry, source_crs, target_crs)
        check(result.get("coordinates"))
        return result
    singles = {"MultiPoint": "Point", "MultiLineString": "LineString", "MultiPolygon": "Polygon"}
    if kind in singles:
        return {"type": kind, "coordinates": [
            _transform_geometry_checked({"type": singles[kind], "coordinates": c}, source_crs, target_crs)["coordinates"]
            for c in geometry["coordinates"]]}
    if kind == "GeometryCollection":
        return {"type": kind, "geometries": [_transform_geometry_checked(g, source_crs, target_crs) for g in geometry["geometries"]]}
    _fail("unsupported_geometry", str(kind), "Use GeoJSON point, line, polygon, multi-geometry or geometry collection.")


def prepare_scene(recipe: PreparationRecipe, *, base_dir: str | Path = ".") -> PreparedScene:
    """Execute a recipe. No network access, rendering, or provider guessing."""
    base = Path(base_dir).resolve()
    layers: dict[str, PreparedLayer] = {}
    reports = []
    crs_type = _require_rasterio().crs.CRS
    coverage: dict[str, Any]
    for source in recipe.sources:
        path = _local(source.path, base)
        identity = {"path": str(path), "sha256": _sha(path)}
        metadata = dict(source.metadata)
        metadata.update({"source_id": source.source_id, "identity": identity, "units": source.units,
                         "source_vertical_reference": source.vertical_reference})
        operations: list[dict[str, Any]] = []
        if source.mask_path:
            mask_path = _local(source.mask_path, base)
            metadata["mask_identity"] = {"path": str(mask_path), "sha256": _sha(mask_path)}
        if source.kind in {"vector", "observation"}:
            if path.suffix.lower() not in {".json", ".geojson"} or not source.crs:
                _fail("missing_vector_crs", "Local GeoJSON and an explicit CRS are required.", "Declare crs even for longitude/latitude input.", source.source_id)
            crs_type.from_user_input(source.crs)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise PreparationError("invalid_geojson", str(exc), "Correct the local UTF-8 GeoJSON file.", source.source_id) from exc
            if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
                _fail("invalid_geojson", "A FeatureCollection is required.", "Export a GeoJSON FeatureCollection.", source.source_id)
            # A legacy GeoJSON CRS, when present, must agree with the declaration.
            legacy = payload.get("crs", {}).get("properties", {}).get("name")
            if legacy and crs_type.from_user_input(legacy) != crs_type.from_user_input(source.crs):
                _fail("conflicting_crs", "GeoJSON CRS differs from the declaration.", "Correct source CRS metadata.", source.source_id)
            features = []
            for feature in payload["features"]:
                if feature.get("type") != "Feature" or not isinstance(feature.get("geometry"), Mapping):
                    _fail("invalid_geometry", "Feature has no usable geometry.", "Remove or repair empty geometries explicitly.", source.source_id)
                if source.kind == "observation" and feature["geometry"].get("type") != "Point":
                    _fail("invalid_observation", "Prepared observations must be point features.", "Supply point geometry and retain values/dates/disclosure flags in properties.", source.source_id)
                item = dict(feature)
                item["geometry"] = _transform_geometry_checked(feature["geometry"], source.crs, recipe.target_grid.crs)
                # Bounding boxes refer to old coordinates and must not survive.
                item.pop("bbox", None)
                features.append(item)
            metadata.update({"source_crs": source.crs, "crs": recipe.target_grid.crs})
            operations.append({"operation": "vector_transform", "from_crs": source.crs, "to_crs": recipe.target_grid.crs, "axis_order": "xy"})
            layer = PreparedLayer(source.source_id, source.kind, None, None, None, features, metadata)
            coverage = {"feature_count": len(features)}
        else:
            data, valid, original_grid, extra = _read_raster(source, path, base)
            metadata.update(extra)
            metadata["source_grid"] = original_grid.to_dict()
            if source.kind == "count":
                if source.resampling not in {None, "native"}:
                    _fail("count_resampling_refused", "Resampling counts would change cell meaning.", "Keep native cells or externally perform documented conservative aggregation.", source.source_id)
                if np.any(data[valid] < 0) or np.any(data[valid] != np.floor(data[valid])):
                    _fail("invalid_count", "Valid count cells must be nonnegative integers.", "Mark withheld/missing cells in the validity mask; declare densities as continuous.", source.source_id)
                grid = original_grid
                operations.append({"operation": "retain_native_count_cells", "analysis_resampling": "none"})
            else:
                grid = recipe.target_grid
                method = source.resampling or ("nearest" if source.kind == "categorical" else "bilinear")
                _resampling_method(method)
                if source.kind == "categorical":
                    if data.dtype.kind not in "iu" or method != "nearest":
                        _fail("category_resampling_refused", "Categories require integer IDs and nearest sampling.", "Use nearest; do not interpolate class identifiers.", source.source_id)
                    if (not original_grid.same_as(grid)
                            and (int(data[valid].min()) < -(2**53) or int(data[valid].max()) > 2**53)):
                        _fail("category_precision", "Warp scratch cannot exactly represent these IDs.", "Remap IDs to exactly representable integers with a recorded lookup table.", source.source_id)
                if source.kind == "elevation":
                    data = _height_conversion(data, valid, original_grid, source, recipe.vertical_reference)
                    metadata.update({"units": "m", "vertical_reference": recipe.vertical_reference})
                    operations.append({"operation": "height_reference", "from": source.vertical_reference,
                                       "to": recipe.vertical_reference, "input_units": source.units, "output_units": "m"})
                if not original_grid.same_as(grid):
                    dtype = data.dtype
                    data, valid = _warp(data, valid, original_grid, grid, method)
                    if source.kind == "categorical":
                        data = np.where(valid, data, 0).astype(dtype)
                    elif source.kind == "imagery" and dtype == np.uint8:
                        data = np.rint(np.where(valid, data, 0)).clip(0, 255).astype(np.uint8)
                    operations.append({"operation": "geographic_warp", "resampling": method,
                                       "from_grid": original_grid.to_dict(), "to_grid": grid.to_dict()})
                else:
                    operations.append({"operation": "retain_grid", "resampling": "none"})
            if not np.any(valid):
                _fail("no_coverage", "Source has no valid target coverage.", "Correct CRS/grid metadata or choose an overlapping extent.", source.source_id)
            if data.shape[0] == 1:
                data, valid = data[0], valid[0]
            metadata.update({"crs": grid.crs, "source_crs": original_grid.crs, "grid": grid.to_dict(),
                             "nodata_policy": "separate_validity_mask", "pixel_registration": "area"})
            layer = PreparedLayer(source.source_id, source.kind, grid, data, valid, None, metadata)
            coverage = {"valid_samples": int(np.count_nonzero(valid)), "total_samples": int(valid.size),
                        "valid_fraction": float(np.count_nonzero(valid) / valid.size),
                        "native_grid_retained": source.kind == "count"}
            target_valid = valid if valid.ndim == 2 else np.all(valid, axis=0)
            if not grid.same_as(recipe.target_grid):
                _, target_mask = _warp(np.ones((1, grid.height, grid.width)), target_valid[None, ...],
                                       grid, recipe.target_grid, "nearest")
                target_valid = target_mask[0]
                operations.append({"operation": "coverage_mask_to_target", "resampling": "nearest", "measurement_values_resampled": False})
            coverage["target_valid_cells"] = int(np.count_nonzero(target_valid))
            coverage["target_cell_count"] = recipe.target_grid.height * recipe.target_grid.width
            if coverage["target_valid_cells"] == 0:
                _fail("no_coverage", "Source has no valid cells in the target extent.", "Choose overlapping inputs or correct the source/target transform.", source.source_id)
        metadata["operations"] = operations
        layers[source.source_id] = layer
        reports.append({"source_id": source.source_id, "kind": source.kind, "coverage": coverage,
                        "metadata": metadata})
    checks = []
    from .crs import transform_coords
    for check in recipe.checkpoints:
        checkpoint_source = next((s for s in recipe.sources if s.source_id == check["source_id"]), None)
        if checkpoint_source is None:
            _fail("invalid_checkpoint", "Unknown checkpoint source.", "Use a source_id present in the recipe.")
        source_crs = checkpoint_source.crs or layers[checkpoint_source.source_id].metadata.get("source_crs")
        if not source_crs:
            _fail("invalid_checkpoint", "Checkpoint source CRS is missing.", "Declare the source CRS before checking positions.")
        if "source_pixel" in check:
            source_grid = PreparationGrid(**layers[checkpoint_source.source_id].metadata["source_grid"])
            source_xy = source_grid.xy(*check["source_pixel"])
        else:
            source_xy = check["source_xy"]
        actual = transform_coords(np.asarray([source_xy], dtype=np.float64), source_crs, recipe.target_grid.crs)[0]
        expected = np.asarray(check["expected_target_xy"], dtype=np.float64)
        tolerance = float(check["tolerance"])
        if expected.shape != (2,) or not np.all(np.isfinite(expected)) or not math.isfinite(tolerance) or tolerance < 0:
            _fail("invalid_checkpoint", "Finite expected XY and nonnegative tolerance are required.", "Derive tolerance from source accuracy or transform precision.")
        residual = float(np.linalg.norm(actual - expected))
        if not math.isfinite(residual) or residual > tolerance:
            _fail("checkpoint_failed", f"{checkpoint_source.source_id}: residual {residual} exceeds {tolerance} target units.", "Check axis order, CRS and shifted geotransforms.", checkpoint_source.source_id)
        checks.append({**dict(check), "actual_target_xy": actual.tolist(), "residual": residual})
    target_crs = crs_type.from_user_input(recipe.target_grid.crs)
    horizontal_units = ({"name": target_crs.linear_units_factor[0], "to_metres": target_crs.linear_units_factor[1]}
                        if target_crs.is_projected else {"name": "degrees", "to_metres": None})
    report = {"schema_version": 1, "target_grid": recipe.target_grid.to_dict(), "horizontal_units": horizontal_units,
              "vertical_reference": recipe.vertical_reference, "layers": reports, "checkpoints": checks,
              "diagnostics": [], "render_verified": False}
    return PreparedScene(recipe, layers, report)


@dataclass
class PreparedScene:
    recipe: PreparationRecipe
    layers: dict[str, PreparedLayer]
    report: dict[str, Any]
    root: Path | None = None

    def save(self, directory: str | Path) -> Path:
        """Save a self-contained prepared snapshot. Never replace an existing folder."""
        root = Path(directory).resolve()
        if root.exists():
            _fail("output_exists", str(root), "Choose a new output directory; existing files are never overwritten.")
        # Validate serializable metadata before creating any output. Raster
        # measurements retain their own masks and may contain IEEE nodata.
        try:
            _json({"recipe": self.recipe.to_dict(), "report": self.report,
                   "metadata": [layer.metadata for layer in self.layers.values()]})
        except (TypeError, ValueError, OverflowError) as exc:
            raise PreparationError("invalid_recipe", f"Snapshot metadata is not finite JSON: {exc}",
                                   "Correct the metadata before saving; no output was published.") from exc
        try:
            root.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f".{root.name}.", dir=root.parent) as temporary:
                staging = Path(temporary).resolve()
                if not staging.is_relative_to(root.parent):
                    _fail("snapshot_write_failed", "Staging directory is outside the output parent.", "Choose a writable output parent.")
                self._write_snapshot(staging)
                staging.rename(root)
        except FileExistsError as exc:
            raise PreparationError("output_exists", str(root), "Choose a new output directory; existing files are never overwritten.") from exc
        except (OSError, TypeError, ValueError) as exc:
            if isinstance(exc, PreparationError):
                raise
            raise PreparationError("snapshot_write_failed", str(exc), "Correct the output path or metadata and retry; the incomplete staging directory was removed.") from exc
        self.root = root
        return root

    def _write_snapshot(self, root: Path) -> None:
        records = []
        for layer in self.layers.values():
            stem = hashlib.sha256(layer.source_id.encode("utf-8")).hexdigest()
            files = {}
            for key, array in (("data", layer.data), ("valid_mask", layer.valid_mask)):
                if array is not None:
                    path = root / f"{stem}.{key}.npy"
                    np.save(path, array, allow_pickle=False)
                    files[key] = {"path": path.name, "sha256": _sha(path)}
            if layer.features is not None:
                path = root / f"{stem}.geojson"
                path.write_text(_json({"type": "FeatureCollection", "features": layer.features}), encoding="utf-8")
                files["features"] = {"path": path.name, "sha256": _sha(path)}
            records.append({"source_id": layer.source_id, "kind": layer.kind,
                            "grid": layer.grid.to_dict() if layer.grid else None,
                            "metadata": layer.metadata, "files": files})
        manifest: dict[str, Any] = {"schema_version": 1, "recipe": self.recipe.to_dict(), "report": self.report, "layers": records}
        # The executable recipe in a snapshot must resolve to its recorded inputs,
        # regardless of where the original recipe's relative paths were rooted.
        for source in manifest["recipe"]["sources"]:
            metadata = self.layers[source["source_id"]].metadata
            source["path"] = metadata["identity"]["path"]
            if "mask_identity" in metadata:
                source["mask_path"] = metadata["mask_identity"]["path"]
        (root / "preparation.json").write_text(_json(manifest) + "\n", encoding="utf-8")
        (root / "recipe.json").write_text(_json(manifest["recipe"]) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> PreparedScene:
        """Reuse prepared assets without reopening or requiring original sources."""
        root = Path(directory).resolve()
        manifest = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            _fail("unsupported_preparation_version", str(manifest.get("schema_version")), "Use preparation snapshot schema 1.")
        layers = {}
        for record in manifest["layers"]:
            values: dict[str, Any] = {"data": None, "valid_mask": None, "features": None}
            for key, entry in record["files"].items():
                path = (root / entry["path"]).resolve()
                if not path.is_relative_to(root) or not path.is_file() or _sha(path) != entry["sha256"]:
                    _fail("prepared_asset_integrity", entry["path"], "Restore the original snapshot or run preparation into a new directory.", record["source_id"])
                values[key] = (json.loads(path.read_text(encoding="utf-8"))["features"] if key == "features"
                               else np.load(path, allow_pickle=False))
            grid = PreparationGrid(**record["grid"]) if record["grid"] else None
            if grid and (values["data"] is None or values["valid_mask"] is None
                         or values["valid_mask"].dtype != np.bool_
                         or values["data"].shape != values["valid_mask"].shape
                         or values["data"].shape[-2:] != (grid.height, grid.width)):
                _fail("invalid_prepared_asset", "Prepared arrays do not match their grid/mask.", "Regenerate the snapshot from recorded sources.", record["source_id"])
            layers[record["source_id"]] = PreparedLayer(record["source_id"], record["kind"], grid,
                                                        metadata=record["metadata"], **values)
        return cls(PreparationRecipe.from_dict(manifest["recipe"]), layers, manifest["report"], root)

    def coverage_preview(self, path: str | Path) -> Path:
        """Write a geographic HTML coverage report; no render or alignment claim."""
        import html
        polygons = []
        for layer in self.layers.values():
            if layer.grid:
                grid = layer.grid
                points = np.asarray([grid.transform * p for p in ((0, 0), (grid.width, 0), (grid.width, grid.height), (0, grid.height))])
                from .crs import transform_coords
                points = transform_coords(np.asarray(points), grid.crs, self.recipe.target_grid.crs)
                polygons.append((layer.source_id, points))
        all_points = np.concatenate([p for _, p in polygons])
        lo, hi = all_points.min(axis=0), all_points.max(axis=0)
        span = hi - lo
        paths = []
        for name, points in polygons:
            points = (points - lo) / span
            text = " ".join(f"{float(x)*800},{(1-float(y))*500}" for x, y in points)
            paths.append(f'<polygon points="{text}"><title>{html.escape(name)}</title></polygon>')
        content = ('<!doctype html><meta charset="utf-8"><title>Preparation coverage</title>'
                   '<style>body{font-family:system-ui}polygon{fill:steelblue;fill-opacity:.12;stroke:#235}pre{white-space:pre-wrap}</style>'
                   '<h1>Preparation coverage</h1><p>Footprints in target CRS; inspect validity fractions and independent checkpoints below.</p>'
                   '<svg viewBox="0 0 800 500">' + ''.join(paths) + '</svg><pre>'
                   + html.escape(_json(self.report)) + '</pre>')
        path = Path(path)
        path.write_text(content, encoding="utf-8")
        return path

    def to_scene(self, *, lighting: Any, output: Any, camera: Any = None,
                 styles: Mapping[str, Mapping[str, Any]] | None = None,
                  terrain_void_fill: float | None = None,
                  assets_dir: str | Path | None = None) -> Any:
        """Create the existing MapScene seam; styles declare fixed display meaning.

        Classes use ``{"classes": {id: [r,g,b,a]}}``. Measurements use
        ``{"range": [low, high], "colors": [[r,g,b,a], [r,g,b,a]]}``.
        Optional ``scale="log1p"`` maps new measurement values without a class
        lookup; endpoint colours saturate outside the fixed range. Analysis
        values are never clipped or transformed by a display style.
        Imagery without a style must already be uint8 RGB/RGBA/grayscale.

        Display rasters are GeoTIFFs on their prepared grid, so MapScene's
        terrain-UV raster drape registers them to the terrain in any camera
        view. Vectors keep target-CRS coordinates for its terrain projection.
        """
        from .map_scene import MapScene, TerrainSource, RasterOverlay, VectorOverlay
        _require_render_grid(self.recipe.target_grid)
        if self.root is None:
            _fail("unsaved_preparation", "Scene assets need a persistent directory.", "Call prepared.save(directory) before to_scene().")
        styles = self.recipe.display_styles if styles is None else styles
        _validate_display_styles(styles, self.recipe.sources)
        assets_root = (Path(assets_dir).resolve() if assets_dir is not None
                       else self.root.with_name(self.root.name + ".scene-assets"))
        if assets_root.is_relative_to(self.root):
            _fail("invalid_scene_assets_directory", str(assets_root), "Choose a directory outside the saved snapshot.")
        terrain = None
        overlays: list[RasterOverlay | VectorOverlay] = []
        for layer in self.layers.values():
            if layer.kind == "elevation":
                layer_grid, data, valid_mask = _layer_raster(layer)
                heights = np.array(data, copy=True)
                if not np.all(valid_mask):
                    if terrain_void_fill is None or not math.isfinite(terrain_void_fill):
                        _fail("terrain_voids", "The terrain contains invalid cells.", "Declare a finite terrain_void_fill for rendering; original values/masks remain in the snapshot.", layer.source_id)
                    heights[~valid_mask] = terrain_void_fill
                metadata = {**layer.metadata, **layer_grid.to_dict(),
                            "resolution": [layer_grid.geotransform[1], -layer_grid.geotransform[5]], "preparation": self.report,
                            "render_void_fill": terrain_void_fill}
                terrain = TerrainSource(data=heights, crs=layer_grid.crs, metadata=metadata,
                                        dtype=str(heights.dtype), elevation_sampling_available=True)
            elif layer.features is not None:
                overlays.append(VectorOverlay(layer.source_id, features=layer.features,
                                              crs=self.recipe.target_grid.crs,
                                              metadata={**layer.metadata, "preparation_grid": self.recipe.target_grid.to_dict()},
                                              style=styles.get(layer.source_id)))
            else:
                layer_grid, _, _ = _layer_raster(layer)
                rgba = _display_rgba(layer, styles.get(layer.source_id))
                display_id = hashlib.sha256(rgba.tobytes() + _json(layer_grid.to_dict()).encode()).hexdigest()
                assets_root.mkdir(parents=True, exist_ok=True)
                path = assets_root / f"{display_id}.display.tif"
                rasterio = _require_rasterio()
                with rasterio.open(path, "w", driver="GTiff", width=layer_grid.width, height=layer_grid.height,
                                   count=4, dtype="uint8", crs=layer_grid.crs, transform=layer_grid.transform) as ds:
                    ds.write(np.moveaxis(rgba, -1, 0))
                overlays.append(RasterOverlay(layer.source_id, path=path, crs=layer_grid.crs,
                                             metadata={**layer.metadata, "display_style": styles.get(layer.source_id),
                                                       "alignment_resampling": "nearest"}))
        return MapScene(terrain=terrain, lighting=lighting, output=output, camera=camera,
                        layers=overlays, target_crs=self.recipe.target_grid.crs)

    def viewer_assets(self, directory: str | Path, *, styles: Mapping[str, Mapping[str, Any]] | None = None,
                      terrain_void_fill: float | None = None) -> dict[str, Any]:
        """Write registered terrain/overlay assets for the existing 3D viewer.

        ViewerHandle.load_overlay drapes these target-grid textures on the
        terrain during camera movement when the viewer's PBR terrain path is
        enabled (set_terrain_pbr enabled=True).
        Vector display uses a declared fixed RGBA ``color``; original features
        remain in the prepared snapshot. Display rasterization never replaces
        analysis values, count cells or observation properties.
        """
        from ._png import encode_png
        from rasterio.features import rasterize
        grid = self.recipe.target_grid
        _require_render_grid(grid)
        styles = self.recipe.display_styles if styles is None else styles
        _validate_display_styles(styles, self.recipe.sources)
        terrain = next(layer for layer in self.layers.values() if layer.kind == "elevation")
        _, terrain_data, terrain_valid = _layer_raster(terrain)
        heights = np.array(terrain_data, copy=True)
        if not np.all(terrain_valid):
            if terrain_void_fill is None or not math.isfinite(terrain_void_fill):
                _fail("terrain_voids", "The terrain contains invalid cells.", "Declare a finite render fill; original measurements and masks remain unchanged.")
            heights[~terrain_valid] = terrain_void_fill
        images = []
        for layer in self.layers.values():
            if layer.kind == "elevation":
                continue
            if layer.features is not None:
                style = styles.get(layer.source_id, {})
                if "color" not in style:
                    _fail("missing_vector_display_style", "Viewer vectors require a fixed RGBA color.", "Declare display_styles[source_id] = {'color': [r,g,b,a]}.", layer.source_id)
                color = _color(style["color"])
                mask = rasterize([(feature["geometry"], 1) for feature in layer.features],
                                 out_shape=(grid.height, grid.width), transform=grid.transform,
                                 fill=0, dtype="uint8") if layer.features else np.zeros((grid.height, grid.width), dtype=np.uint8)
                rgba = np.zeros((grid.height, grid.width, 4), dtype=np.uint8)
                rgba[mask != 0] = color
            else:
                layer_grid, _, _ = _layer_raster(layer)
                rgba = _display_rgba(layer, styles.get(layer.source_id))
                if not layer_grid.same_as(grid):
                    bands = np.moveaxis(rgba, -1, 0)
                    warped, valid = _warp(bands, np.ones(bands.shape, dtype=bool), layer_grid, grid, "nearest")
                    rgba = np.moveaxis(np.where(valid, warped, 0).astype(np.uint8), 0, -1)
            images.append((layer, rgba))
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=False)
        terrain_path = root / "terrain.tif"
        with _require_rasterio().open(terrain_path, "w", driver="GTiff", width=grid.width, height=grid.height,
                                     count=1, dtype=str(heights.dtype), crs=grid.crs, transform=grid.transform) as ds:
            ds.write(heights, 1)
            ds.update_tags(AREA_OR_POINT="Area", vertical_reference=self.recipe.vertical_reference, units="m")
        overlays = []
        for index, (layer, rgba) in enumerate(images):
            path = root / f"overlay_{index}.png"
            path.write_bytes(encode_png(rgba))
            overlays.append({"source_id": layer.source_id, "path": str(path), "extent": [0, 0, 1, 1],
                             "sha256": _sha(path), "operation": "display_only_on_target_grid"})
        result = {"schema_version": 1, "terrain_path": str(terrain_path), "terrain_sha256": _sha(terrain_path),
                  "overlays": overlays, "preparation": self.report, "display_styles": styles,
                  "terrain_void_fill": terrain_void_fill, "viewer_render_verified": False}
        (root / "viewer-assets.json").write_text(_json(result) + "\n", encoding="utf-8")
        return result


def _require_render_grid(grid: PreparationGrid) -> None:
    crs = _require_rasterio().crs.CRS.from_user_input(grid.crs)
    gt = grid.geotransform
    if (not crs.is_projected or crs.linear_units_factor[1] != 1
            or gt[2] != 0 or gt[4] != 0 or gt[1] <= 0 or gt[5] >= 0):
        _fail("unsupported_render_grid", "Prepared render assets require a north-up projected grid in metres.",
              "Declare a suitable metric target CRS/grid; source feet/degrees/rotations remain supported by preparation warping.")


def _validate_display_styles(styles: Mapping[str, Mapping[str, Any]], sources: Sequence[PreparationSource]) -> None:
    """Validate declared display syntax before any preparation or file writes."""
    try:
        if not isinstance(styles, Mapping):
            raise ValueError("display_styles must be an object")
        by_id = {source.source_id: source for source in sources}
        for source_id, style in styles.items():
            if source_id not in by_id or not isinstance(style, Mapping):
                raise ValueError(f"Unknown display source or malformed style: {source_id}")
            if by_id[source_id].kind in {"vector", "observation"}:
                if set(style) != {"color"}:
                    raise ValueError(f"{source_id}: vector/observation display requires a fixed color")
                _color(style["color"])
            elif "classes" in style:
                if set(style) != {"classes"} or not isinstance(style["classes"], Mapping) or not style["classes"]:
                    raise ValueError(f"{source_id}: classes must be a nonempty ID-to-RGBA object")
                for key, value in style["classes"].items():
                    number = int(key)
                    if isinstance(key, float) and number != key:
                        raise ValueError("Class IDs must be integers")
                    _color(value)
            else:
                if not {"range", "colors"} <= style.keys() or style.keys() - {"range", "colors", "scale"}:
                    raise ValueError(f"{source_id}: use range/colors/scale or classes; unknown keys are not ignored")
                low, high = (float(v) for v in style["range"])
                if not math.isfinite(low) or not math.isfinite(high) or high <= low:
                    raise ValueError("Display range must be finite and increasing")
                if len(style["colors"]) != 2:
                    raise ValueError("Supply exactly two endpoint colors")
                for value in style["colors"]:
                    _color(value)
                scale = style.get("scale", "linear")
                if scale not in {"linear", "log1p"} or (scale == "log1p" and low <= -1):
                    raise ValueError("Use linear or log1p with a range greater than -1")
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise PreparationError("invalid_recipe", f"Invalid display_styles: {exc}",
                               "Use classes with RGBA byte colors, range/colors with optional linear or log1p scale, or a fixed color for vectors/observations.") from exc


def _layer_raster(layer: PreparedLayer) -> tuple[PreparationGrid, np.ndarray, np.ndarray]:
    if layer.grid is None or layer.data is None or layer.valid_mask is None:
        _fail("invalid_prepared_asset", "Raster grid, measurements or validity mask are missing.",
              "Regenerate the prepared raster from its recorded recipe.", layer.source_id)
    return layer.grid, layer.data, layer.valid_mask


def _display_rgba(layer: PreparedLayer, style: Mapping[str, Any] | None) -> np.ndarray:
    _, data, valid = _layer_raster(layer)
    mask = valid if valid.ndim == 2 else np.all(valid, axis=0)
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    if style and "classes" in style:
        if data.ndim != 2:
            _fail("invalid_display_style", "Class styles require one band.", "Choose a single categorical band.", layer.source_id)
        assigned = np.zeros(mask.shape, dtype=bool)
        for key, color in style["classes"].items():
            selected = (data == int(key)) & mask
            rgba[selected] = _color(color)
            assigned |= selected
        if np.any(mask & ~assigned):
            _fail("unmapped_class", "Valid class IDs lack display colors.", "Declare all class colors explicitly.", layer.source_id)
    elif style and "range" in style:
        lo, hi = (float(v) for v in style["range"])
        if data.ndim != 2 or not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            _fail("invalid_display_style", "A finite increasing range for one band is required.", "Declare a fixed measurement display range.", layer.source_id)
        colors = np.array([_color(c) for c in style["colors"]], dtype=np.float64)
        if colors.shape != (2, 4):
            _fail("invalid_display_style", "Exactly two endpoint colors are required.", "Supply two RGBA colors.", layer.source_id)
        scale = style.get("scale", "linear")
        values = np.where(mask, data, lo)
        if scale == "log1p":
            if lo <= -1 or np.any(values <= -1):
                _fail("invalid_display_style", "log1p requires values and range endpoints greater than -1.", "Use a suitable fixed range or linear scale.", layer.source_id)
            values, lo, hi = np.log1p(values), np.log1p(lo), np.log1p(hi)
        elif scale != "linear":
            _fail("invalid_display_style", f"Unknown display scale: {scale}.", "Choose linear or log1p.", layer.source_id)
        t = np.clip((values - lo) / (hi - lo), 0, 1)[..., None]
        rgba = np.rint(colors[0] + t * (colors[1] - colors[0])).astype(np.uint8)
    elif layer.kind == "imagery" and data.dtype == np.uint8:
        bands = data[None, ...] if data.ndim == 2 else data
        rgba[..., :3] = np.moveaxis(bands[:3], 0, -1) if bands.shape[0] >= 3 else bands[0, ..., None]
        rgba[..., 3] = bands[3] if bands.shape[0] == 4 else 255
    else:
        _fail("missing_display_style", "Measurements need an explicit display mapping.", "Supply fixed classes or a range/colors style; no data-dependent normalization is inferred.", layer.source_id)
    rgba[~mask] = 0
    return rgba


def _color(value: Sequence[int]) -> np.ndarray:
    color = np.asarray(value)
    if color.shape != (4,) or color.dtype.kind not in "iu" or np.any(color < 0) or np.any(color > 255):
        _fail("invalid_display_color", "Colors must be four integer channels in [0,255].", "Supply RGBA byte colors.")
    return color.astype(np.uint8)

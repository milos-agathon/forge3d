# G-002c GIS Vector, Rasterization, Masks, and Thematic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the G-002c Rust-first GIS vector, rasterization, mask, and thematic raster APIs on top of the G-002a1/G-002b raster foundation.

**Architecture:** Backend GIS behavior belongs in Rust under `src/gis/`; Python wrappers in `python/forge3d/gis.py` stay thin `os.fspath`/argument marshaling shims over PyO3. Heavy GIS capabilities are feature-gated and must fail with stable `BackendUnavailable` errors when unavailable rather than silently falling back to Python libraries.

**Tech Stack:** Rust, PyO3, NumPy at the Python boundary, existing `tiff`/`ndarray` raster code, existing CRS/affine helpers, optional GDAL/OGR for vector IO/reprojection/rasterization, optional GEOS-equivalent topology for robust geometry operations, and optional reference checks using Python GIS libraries in tests only.

---

## Source Note

The prompt names these canonical sources:

- `docs/carto-engine/rust-gis-implementation-plan.md`
- `docs/carto-engine/gis-operation-api-crosswalk.md`
- `docs/carto-engine/g-002b-support-matrix.md`

Only `docs/carto-engine/g-002b-support-matrix.md` exists in this worktree and in local `g-002b` history. This plan uses that file, the current `src/gis/` and `python/forge3d/gis.py` surfaces, and the prompt's contract list. The two absent source docs should be added or restored before implementation if they contain decisions not reflected here.

## Pre-Edit Planning Matrix

| G-002c API | priority | Rust module | proposed Python wrapper | required inputs | required outputs / metadata | diagnostics / errors | dependencies | test bundle | implementation order | explicit non-goals | open questions |
|---|---:|---|---|---|---|---|---|---|---:|---|---|
| `read_vector` | P0 | `src/gis/vector.rs` | `forge3d.gis.read_vector(path, layer=None, columns=None, bbox=None, limit=None)` | path, optional layer/filter | GeoJSON-like feature collection plus `VectorInfo` dict | `NotFound`, `unsupported_driver`, `missing_layer`, `empty_feature_set`, `missing_crs`, `invalid_geometry` | `gdal-vector`, `geojson`, `serde_json` | T-vector-io | C1.1 | no mutation, no reprojection, no topology | Should C1 support GeoJSON only or GeoJSON plus GPKG? |
| `VectorInfo` | P0 | `src/gis/vector.rs` or `src/gis/types.rs` | native `VectorInfo` class | path/layer inspection | path, driver, layer, geometry type, schema, count, CRS, bounds, warnings | `missing_crs`, `missing_layer`, `empty_feature_set` | `gdal-vector` | T-vector-io | C1.1 | no feature payload | None; register the class in C1 |
| `geometry_type` | P0 | `src/gis/vector.rs` | `geometry_type(source, layer=None)` | path/`VectorInfo`/read result/geometry | stable geometry type string | `empty_geometry`, `unsupported_geometry_type` | GDAL for datasets, `geojson` for geometry | T-vector-io | C1.2 | no validation or repair | Exact mixed-geometry spelling |
| `vector_schema` | P0 | `src/gis/vector.rs` | `vector_schema(source, layer=None)` | path/`VectorInfo`/read result | field list with OGR-compatible metadata | `missing_layer`, `unsupported_driver` | `gdal-vector` | T-vector-io | C1.2 | no pandas/geopandas | OGR type to public type mapping |
| `feature_count` | P0 | `src/gis/vector.rs` | `feature_count(source, layer=None)` | path/`VectorInfo`/read result | integer count | `empty_feature_set`, `missing_layer` | `gdal-vector` | T-vector-io | C1.2 | no scan-heavy analytics | Should approximate OGR counts be exposed? |
| `vector_crs` | P0 | `src/gis/vector.rs`, `src/gis/crs.rs` | `vector_crs(source, layer=None)` | path/`VectorInfo`/read result | `CrsInfo`-compatible dict | `missing_crs`, `invalid_crs` | `gdal-vector`, existing CRS helpers | T-vector-io | C1.3 | no guessing | Authority normalization parity with rasters |
| `vector_bounds` | P0 | `src/gis/vector.rs` | `vector_bounds(source, layer=None)` | path/`VectorInfo`/read result | `(left, bottom, right, top)` | `empty_feature_set`, `missing_crs` warning | `gdal-vector` | T-vector-io | C1.3 | no reprojection | Bounds behavior for invalid geometries |
| `reproject_vector` | P1 | `src/gis/vector.rs`, `src/gis/crs.rs` | `reproject_vector(source, dst_crs, src_crs=None, always_xy=True)` | vector source, destination CRS, optional source CRS | reprojected features plus `VectorInfo` dict | `missing_crs`, `invalid_crs`, `crs_mismatch`, `TransformFailed` | `gdal-vector` or `proj` path | T-vector-crs | C2 | separate from raster reprojection | Whether to require GDAL transforms or reuse existing PROJ subset |
| `validate_geometry` | P1 | `src/gis/vector.rs` | `validate_geometry(geometry)` | GeoJSON-like geometry | validity bool, reason, type | `empty_geometry`, `unsupported_geometry_type` | `geos-topology` for robust validity | T-vector-geom | C3.1 | no automatic repair | Feature name for topology backend |
| `repair_geometry` | P1 | `src/gis/vector.rs` | `repair_geometry(geometry, method="make_valid")` | geometry, repair method | repaired geometry and `GeometryOperationInfo` | `invalid_geometry`, `geometry_repaired`, `geometry_repair_changed_type` | `geos-topology` | T-vector-geom | C3.2 | no silent type changes | Fallback behavior when GEOS is missing |
| `geometry_measure` | P1 | `src/gis/vector.rs` | `geometry_measure(geometry, metrics=("area", "length"))` | geometry, metric list | numeric measurements and units note | `empty_geometry`, `invalid_geometry`, `unsupported_geometry_type` | `geo` for planar metrics | T-vector-geom | C3.3 | no geodesic measurement in C3 | Whether planar-only is acceptable for first pass |
| `geometry_centroid` | P1 | `src/gis/vector.rs` | `geometry_centroid(geometry)` | geometry | point geometry | `empty_geometry`, `invalid_geometry` | `geo` | T-vector-geom | C3.4 | no label placement | Empty polygon result policy |
| `representative_point` | P1 | `src/gis/vector.rs` | `representative_point(geometry)` | geometry | point geometry | `empty_geometry`, `invalid_geometry` | GEOS preferred | T-vector-geom | C3.4 | no cartographic labeling | Allow fallback only when inside/on guarantee holds? |
| `interpolate_line` | P1 | `src/gis/vector.rs` | `interpolate_line(geometry, distance, normalized=False)` | line geometry, distance | point plus measure metadata | `empty_geometry`, `unsupported_geometry_type`, `invalid_argument` | `geo` | T-vector-geom | C3.5 | no routing/network analysis | MultiLineString distance semantics |
| `union_geometries` | P2 | `src/gis/vector.rs` | `union_geometries(geometries)` | iterable geometries | union geometry and `GeometryOperationInfo` | `empty_geometry`, `invalid_geometry`, `unsupported_geometry_type` | `geos-topology` | T-vector-geom | C4.1 | no attribute dissolve | Robust topology dependency |
| `dissolve_vector` | P2 | `src/gis/vector.rs` | `dissolve_vector(source, by=None)` | vector source, optional group fields | grouped features and info | `missing_layer`, `invalid_geometry`, `empty_feature_set` | GDAL/OGR plus GEOS | T-vector-geom | C4.2 | no full ETL engine | Attribute aggregation rules |
| `buffer_geometry` | P2 | `src/gis/vector.rs` | `buffer_geometry(geometry, distance, quad_segs=8)` | geometry, distance | polygon geometry and operation info | `empty_geometry`, `invalid_geometry` | `geos-topology` | T-vector-geom | C4.3 | no geodesic buffers | Default segment count |
| `clip_vector` | P2 | `src/gis/vector.rs` | `clip_vector(source, clip_geometry, clip_crs=None)` | vector source, clip geometry | clipped features and info | `crs_mismatch`, `empty_feature_set`, `invalid_geometry` | GDAL/OGR plus GEOS | T-vector-geom | C4.4 | no raster clip | Require same CRS unless explicit `clip_crs`? |
| `intersect_vectors` | P2 | `src/gis/vector.rs` | `intersect_vectors(left, right, suffixes=("_left", "_right"))` | two vector sources | intersection features and joined attrs | `crs_mismatch`, `missing_layer`, `invalid_geometry` | GDAL/OGR plus GEOS | T-vector-geom | C4.5 | no spatial index tuning first | Attribute conflict naming |
| `simplify_geometry` | P2 | `src/gis/vector.rs` | `simplify_geometry(geometry, tolerance, preserve_topology=True)` | geometry, tolerance | simplified geometry and operation info | `invalid_argument`, `invalid_geometry` | GEOS if topology-preserving, `geo` otherwise | T-vector-geom | C4.6 | no scale-dependent styling | Preserve-topology default |
| `load_boundary` | P2 | `src/gis/vector.rs` | `load_boundary(path, layer=None, where=None)` | vector source and optional filter | boundary geometry/features and info | `missing_layer`, `empty_feature_set`, `missing_crs` | GDAL/OGR plus GEOS optional | T-vector-geom | C4.7 | no named domain downloads | Should this remain a small read/filter helper? |
| `rasterize_vectors` | P1 | `src/gis/rasterize.rs` | `rasterize_vectors(vectors, target_info, value=1, attribute=None, dtype="uint8", fill=0, all_touched=False)` | vectors/geometries, explicit `RasterInfo` grid | array and `RasterizationResult` | `missing_crs`, `crs_mismatch`, `mask_polarity_explicit`, `unsupported_dtype` | GDAL rasterize first | T-rasterize-mask | C5.1 | no implicit grid creation | GDAL rasterize vs pure Rust scanline |
| `geometry_mask` | P1 | `src/gis/rasterize.rs` | `geometry_mask(geometries, target_info, invert=False, all_touched=False, mask_polarity="true_inside")` | geometries and explicit grid | bool mask and `MaskResult` | `mask_polarity_explicit`, `empty_geometry`, `crs_mismatch` | rasterizer backend | T-rasterize-mask | C5.2 | no hidden polarity defaults | `true_inside` vs `true_valid` for public spelling |
| `mask_raster` | P1 | `src/gis/rasterize.rs` | `mask_raster(source, mask, mask_polarity, crop=False, fill=None, nodata=None)` | raster path/array/info, explicit mask/polarity | masked array, info, crop/nodata metadata | `mask_polarity_explicit`, `empty_raster`, `unsupported_dtype`, `InvalidNodata` | existing raster helpers | T-rasterize-mask | C5.3 | no vector rasterization hidden inside | Crop semantics |
| `normalize_raster` | P2 | `src/gis/thematic.rs` | `normalize_raster(source, method="minmax", valid_mask=None, nodata=None, clip=None)` | raster path/array/info, method | normalized array and `ThematicResult` | `empty_raster`, `unsupported_dtype`, `invalid_argument` | existing raster plus `ndarray` | T-thematic | C6.1 | no color mapping/rendering | Supported methods first |
| `classify_raster` | P2 | `src/gis/thematic.rs` | `classify_raster(source, bins=None, labels=None, right=False, valid_mask=None, nodata=None, dtype="uint16")` | raster and explicit class rules | class array, class table, counts, result info | `empty_raster`, `unsupported_dtype`, `invalid_argument` | existing raster plus `ndarray` | T-thematic | C6.2 | no landcover/domain recipes | Explicit bins only for first pass? |

Common non-goals for every row: remote fetching, cache management, live network services, OSM querying, Terrarium, slippy tiles, DEM/landcover/population/building domain helpers, recipe manifests, gallery goldens, MapScene recipe-family work, and network travel-time analysis.

## Phase Split

### C1: Vector Metadata/Read Foundation

- Implement `VectorInfo`, `read_vector`, `vector_crs`, `vector_bounds`, `geometry_type`, `vector_schema`, and `feature_count`.
- Keep all backend behavior in Rust under `src/gis/vector.rs`.
- Python wrappers only normalize `os.PathLike` and pass through keyword arguments.
- Required diagnostics: `missing_layer`, `missing_crs`, `empty_feature_set`, `unsupported_driver`, `invalid_geometry`.
- This slice does not mutate vectors, transform coordinates, repair geometry, rasterize, classify, render, or write vector files.

### C2: Vector CRS

- Implement `reproject_vector`.
- Source CRS is required unless `src_crs` is explicitly supplied.
- Axis order is always explicit and defaults to `always_xy=True`.
- Output metadata must validate destination CRS, transformed bounds, and feature count.
- Vector reprojection remains a separate API from `reproject_raster`.

### C3: Vector Geometry Validity and Measurement

Focused execution plan: `docs/carto-engine/g-002c-c3-implementation-plan.md`.

- Implement `validate_geometry`, `repair_geometry`, `geometry_measure`, `geometry_centroid`, `representative_point`, and `interpolate_line`.
- Handle empty, invalid, and mixed geometries with explicit diagnostics.
- Keep measurements planar unless a later plan adds geodesic measurement support.
- Never silently change geometry type during repair; emit `geometry_repair_changed_type` when it happens.

### C4: Vector Overlay Operations

- Implement `union_geometries`, `dissolve_vector`, `buffer_geometry`, `clip_vector`, `intersect_vectors`, `simplify_geometry`, and `load_boundary`.
- Require CRS compatibility for operations combining sources.
- Use robust topology only when the planned topology backend is enabled.
- Do not add generalized ETL, remote boundary downloads, or rendering behavior.

### C5: Rasterization and Masks

- Implement `rasterize_vectors`, `geometry_mask`, and `mask_raster`.
- Target grid is always an explicit `RasterInfo` or serialized `RasterInfo` dict.
- Mask polarity is always explicit. Accepted public spellings are `true_inside`, `true_outside`, and `true_valid` where applicable.
- Crop, fill, and nodata behavior must be present in output metadata.

### C6: Thematic Raster

- Implement `normalize_raster` and `classify_raster`.
- Valid-pixel handling is nodata- and mask-aware.
- `classify_raster` returns a class table and per-class counts.
- This phase does not add colormap rendering, domain recipes, gallery goldens, or MapScene recipe-family work.

## Type Design

### Python Boundary Geometry Format

First implementation should accept a deliberately small GeoJSON-like subset:

- Geometry dicts with `{"type": "...", "coordinates": ...}`.
- Feature dicts with `{"type": "Feature", "geometry": ..., "properties": ...}`.
- FeatureCollection dicts returned by `read_vector`.

Do not accept WKB bytes, WKT strings, or internal Rust-backed geometry objects in C1-C3. They can be added later once parsers and ownership rules are explicit. GeoJSON-like dicts are boring, inspectable, match `read_vector` output, and avoid a Python-side GIS dependency.

### `VectorInfo`

`VectorInfo` is a PyO3 class mirroring the style of `RasterInfo`.

Fields:

- `path: str`
- `driver: str`
- `layer_name: str | None`
- `layer_count: int`
- `geometry_type: str`
- `feature_count: int`
- `schema: list[dict[str, Any]]`
- `crs_wkt: str | None`
- `crs_authority: dict[str, str] | None`
- `bounds: tuple[float, float, float, float] | None`
- `is_georeferenced: bool`
- `warnings: list[dict[str, str | None]]`

Methods:

- `as_dict() -> dict[str, Any]`
- `__repr__() -> str`

Schema field entries:

- `name: str`
- `type: str`
- `nullable: bool | None`
- `width: int | None`
- `precision: int | None`

### `CrsInfo` Reuse

Do not add a separate vector CRS type. `vector_crs` and `reproject_vector` use the existing CRS dict shape from `parse_crs`, `inspect_crs`, and `raster_crs`:

- `input`
- `source_kind`
- `crs_wkt`
- `authority`
- `canonical`
- `warnings`

Missing CRS is reported with `missing_crs`; it is never guessed from file names, bounds, or driver defaults.

### `GeometryOperationInfo`

Returned under the `operation` key for geometry operations.

Fields:

- `name: str`
- `input_geometry_type: str | list[str]`
- `output_geometry_type: str | None`
- `input_count: int`
- `output_count: int`
- `changed: bool`
- `crs: dict[str, Any] | None`
- `warnings: list[dict[str, str | None]]`

### `RasterizationResult`

Returned by `rasterize_vectors`.

Fields:

- `array: np.ndarray`
- `info: dict[str, Any]`
- `target_shape: tuple[int, int]`
- `target_transform: tuple[float, float, float, float, float, float]`
- `target_bounds: tuple[float, float, float, float]`
- `dtype: str`
- `fill: int | float`
- `burned_pixels: int`
- `all_touched: bool`
- `warnings: list[dict[str, str | None]]`

### `MaskResult`

Returned by `geometry_mask` and `mask_raster`.

Fields:

- `mask: np.ndarray | None`
- `array: np.ndarray | None`
- `info: dict[str, Any]`
- `mask_polarity: str`
- `true_count: int`
- `false_count: int`
- `crop_window: tuple[int, int, int, int] | None`
- `fill: int | float | None`
- `nodata: float | list[float | None] | None`
- `warnings: list[dict[str, str | None]]`

### `ThematicResult`

Returned by `normalize_raster` and `classify_raster`.

Fields:

- `array: np.ndarray`
- `info: dict[str, Any]`
- `method: str`
- `valid_count: int`
- `nodata_count: int`
- `min: float | None`
- `max: float | None`
- `mean: float | None`
- `std: float | None`
- `class_table: list[dict[str, Any]] | None`
- `warnings: list[dict[str, str | None]]`

### Class Table Schema

`classify_raster` returns `class_table` entries with:

- `class_id: int`
- `label: str | None`
- `left: float | None`
- `right: float | None`
- `right_inclusive: bool`
- `count: int`
- `nodata: bool`

## Backend Dependency Decision

### Required Default Path

- C6 thematic raster uses existing Rust raster/`ndarray` code and should not need heavy system GIS dependencies.
- C5 `mask_raster` should reuse existing raster read/window/nodata helpers when the caller passes an explicit mask.

### Optional Heavy Backends

- Add a Cargo feature named `gdal-vector` for GDAL/OGR-backed vector file IO, vector CRS inspection, vector reprojection, and GDAL rasterization.
- Add a Cargo feature named `geos-topology` for robust validity, repair, buffer, overlay, representative-point, and topology-preserving simplify.
- Keep broad arbitrary CRS transformation behind the existing `proj` feature or the GDAL path. Built-in EPSG:4326/EPSG:3857 transforms remain a limited fallback only when the operation can be proven correct.

### Fallback Behavior

- Public functions should remain importable even when optional backends are disabled.
- Backend-required functions must raise `BackendUnavailable` with messages containing the relevant stable diagnostic, such as `backend_unavailable: gdal-vector feature required`.
- Do not implement backend GIS behavior in Python using `rasterio`, `geopandas`, `shapely`, `rioxarray`, `xarray`, or `terra`.
- Those libraries may appear only in optional reference tests or docs examples, skipped cleanly when unavailable.

### What Not To Overclaim

- Do not claim GDAL/OGR support unless the feature is wired, registered through PyO3, included in the intended wheel feature set, and tested.
- Do not claim GEOS-grade topology for union, dissolve, buffer, clip, intersect, repair, or representative point unless `geos-topology` is enabled and tests cover the path.
- Do not claim arbitrary CRS support from the built-in CRS code. Existing built-ins cover same-CRS and EPSG:4326/EPSG:3857 only.

## Error and Diagnostic Vocabulary

Use existing `GisError` variants where possible and include stable lowercase diagnostic tokens in messages or warning codes.

Required stable diagnostics:

- `missing_crs`: vector or raster source has no CRS metadata where CRS is required.
- `invalid_crs`: CRS literal cannot be parsed or backend rejects it.
- `crs_mismatch`: two sources are combined without matching CRS.
- `empty_geometry`: input geometry is empty.
- `invalid_geometry`: input geometry is invalid for the requested operation.
- `geometry_repaired`: repair succeeded and changed coordinate topology.
- `geometry_repair_changed_type`: repair changed the geometry type.
- `unsupported_geometry_type`: geometry type is not accepted for the operation.
- `missing_layer`: requested layer is absent.
- `empty_feature_set`: vector source or filtered result has zero features.
- `mask_polarity_explicit`: caller omitted required mask polarity or used an unknown polarity.
- `empty_raster`: raster has no valid pixels for the requested operation.
- `unsupported_dtype`: dtype cannot represent the requested output.
- `unsupported_driver`: vector/raster driver is not supported by the active backend.
- `unsupported_option`: option is recognized but not implemented for G-002c.

No new diagnostic spelling is needed for the first implementation. Use `BackendUnavailable` as the error code for disabled optional backends and include `backend_unavailable` in the message.

## API Contracts

### Vector Metadata / IO

| API | function signature | accepted input forms | output schema and metadata fields | stable diagnostics and errors | unsupported cases and reference comparison | minimum tests |
|---|---|---|---|---|---|---|
| `read_vector` | `read_vector(path, *, layer=None, columns=None, bbox=None, limit=None) -> dict[str, Any]` | `str` or `os.PathLike`; `bbox` uses `(left,bottom,right,top)` in source CRS | `{"features": [Feature], "info": VectorInfo.as_dict(), "warnings": [...]}`; features are GeoJSON-like | `NotFound`, `UnsupportedDriver`, `InvalidBounds`, `missing_layer`, `missing_crs`, `empty_feature_set`, `invalid_geometry` | No writes, SQL, reprojection, remote paths, or Python GIS fallback. Compare count/schema/bounds to Fiona/GeoPandas only when installed. | tiny GeoJSON happy path, missing file, missing layer, column filter, bbox filter, empty result, missing CRS warning |
| `VectorInfo` | native class, returned through `read_vector()["info"]` and registered as `_forge3d.VectorInfo` | vector dataset path and optional layer | fields listed in Type Design; `as_dict()` preserves same keys | `UnsupportedDriver`, `missing_layer`, `missing_crs`, `empty_feature_set` | No geometry payload and no extra public helper in C1. Compare metadata with GDAL/Fiona when available. | construct via read, property getters, `as_dict`, repr, missing CRS warning |
| `geometry_type` | `geometry_type(source, *, layer=None) -> str` | vector path, `VectorInfo`, read result, Feature, Geometry | stable string: `Point`, `LineString`, `Polygon`, `MultiPoint`, `MultiLineString`, `MultiPolygon`, `GeometryCollection`, `Mixed`, `Unknown`, `Empty` | `empty_geometry`, `unsupported_geometry_type` where source cannot be inspected | No repair or validation. Reference against Shapely `geom_type` for geometry dicts when installed. | each basic geometry type, mixed collection, empty geometry |
| `vector_schema` | `vector_schema(source, *, layer=None) -> list[dict[str, Any]]` | vector path, `VectorInfo`, read result | schema entries listed in Type Design | `UnsupportedDriver`, `missing_layer` | No pandas dtype inference. Compare names/types to OGR/Fiona when available. | string/int/float fields, empty schema, missing layer |
| `feature_count` | `feature_count(source, *, layer=None) -> int` | vector path, `VectorInfo`, read result | non-negative integer | `missing_layer`, `empty_feature_set` warning for zero | Do not expose approximate counts in C1; if backend only supplies approximate, scan tiny fixtures. Compare to Fiona length when available. | zero, one, many, filtered read result |
| `vector_crs` | `vector_crs(source, *, layer=None) -> dict[str, Any]` | vector path, `VectorInfo`, read result | `CrsInfo`-compatible dict | `missing_crs`, `invalid_crs`, `UnsupportedDriver` | No CRS guessing. Compare EPSG authority to pyproj/Fiona when available. | EPSG CRS, missing CRS, invalid CRS metadata fixture |
| `vector_bounds` | `vector_bounds(source, *, layer=None) -> tuple[float, float, float, float]` | vector path, `VectorInfo`, read result | bounds order is always `(left,bottom,right,top)` | `empty_feature_set`, `missing_layer`, `invalid_geometry` warning when backend reports it | No reprojection. Compare to Shapely/Fiona bounds when installed. | point/polygon bounds, empty source, bbox-filtered bounds |

### Vector CRS

| API | function signature | accepted input forms | output schema and metadata fields | stable diagnostics and errors | unsupported cases and reference comparison | minimum tests |
|---|---|---|---|---|---|---|
| `reproject_vector` | `reproject_vector(source, dst_crs, *, src_crs=None, always_xy=True) -> dict[str, Any]` | vector path, read result, FeatureCollection dict; CRS strings/ints/dicts | `{"features": [...], "info": VectorInfo dict, "src_crs": CrsInfo, "dst_crs": CrsInfo, "warnings": [...]}` | `missing_crs`, `invalid_crs`, `TransformFailed`, `BackendUnavailable` | No raster reprojection, no implicit source CRS guessing, no writes. Compare coordinates/bounds to pyproj/Fiona/GeoPandas when available. | EPSG:4326 to EPSG:3857, same CRS passthrough, missing source CRS, explicit `src_crs`, invalid CRS |

### Vector Geometry

| API | function signature | accepted input forms | output schema and metadata fields | stable diagnostics and errors | unsupported cases and reference comparison | minimum tests |
|---|---|---|---|---|---|---|
| `validate_geometry` | `validate_geometry(geometry) -> dict[str, Any]` | GeoJSON Geometry or Feature | `{"valid": bool, "reason": str | None, "geometry_type": str, "warnings": [...]}` | `empty_geometry`, `unsupported_geometry_type`, `BackendUnavailable` if robust backend required | No repair. Compare to Shapely `is_valid` when installed. | valid polygon, self-intersecting polygon, empty geometry |
| `repair_geometry` | `repair_geometry(geometry, *, method="make_valid") -> dict[str, Any]` | GeoJSON Geometry or Feature | `{"geometry": Geometry, "operation": GeometryOperationInfo, "valid": bool}` | `invalid_geometry`, `geometry_repaired`, `geometry_repair_changed_type`, `BackendUnavailable` | No silent type changes. Compare to Shapely `make_valid` when installed. | bowtie polygon repair, already valid unchanged, changed type diagnostic |
| `geometry_measure` | `geometry_measure(geometry, *, metrics=("area", "length")) -> dict[str, Any]` | GeoJSON Geometry or Feature | `{"area": float | None, "length": float | None, "units": "source_crs_planar", "operation": ...}` | `empty_geometry`, `invalid_geometry`, `unsupported_geometry_type` | No geodesic measurement. Compare planar area/length to Shapely when installed. | unit square area, line length, point unsupported metric |
| `geometry_centroid` | `geometry_centroid(geometry) -> dict[str, Any]` | GeoJSON Geometry or Feature | Point geometry plus `GeometryOperationInfo` | `empty_geometry`, `invalid_geometry` | No representative-point guarantee. Compare to Shapely centroid when installed. | polygon centroid, line centroid, empty geometry |
| `representative_point` | `representative_point(geometry) -> dict[str, Any]` | GeoJSON Geometry or Feature | Point geometry plus `GeometryOperationInfo` | `empty_geometry`, `invalid_geometry`, `BackendUnavailable` if guarantee needs GEOS | Must be on/in geometry; no label collision logic. Compare to Shapely `representative_point` when installed. | concave polygon point inside, line point on line, empty geometry |
| `interpolate_line` | `interpolate_line(geometry, distance, *, normalized=False) -> dict[str, Any]` | LineString or MultiLineString Geometry/Feature | Point geometry, `distance`, `normalized`, `operation` | `unsupported_geometry_type`, `invalid_argument`, `empty_geometry` | No network routing. Compare to Shapely interpolate when installed. | absolute distance, normalized distance, negative/out-of-range rejection |

### Vector Overlay Operations

| API | function signature | accepted input forms | output schema and metadata fields | stable diagnostics and errors | unsupported cases and reference comparison | minimum tests |
|---|---|---|---|---|---|---|
| `union_geometries` | `union_geometries(geometries) -> dict[str, Any]` | iterable GeoJSON geometries/features | union geometry and `GeometryOperationInfo` | `empty_geometry`, `invalid_geometry`, `unsupported_geometry_type`, `BackendUnavailable` | No attribute dissolve. Compare to Shapely unary union when installed. | overlapping polygons, disjoint polygons, empty list |
| `dissolve_vector` | `dissolve_vector(source, *, by=None) -> dict[str, Any]` | vector source/read result, optional property names | FeatureCollection plus `VectorInfo` dict and operation info | `missing_layer`, `empty_feature_set`, `invalid_geometry`, `BackendUnavailable` | Aggregation is only grouping geometry; no stats aggregation in first pass. Compare to GeoPandas dissolve when installed. | dissolve all, dissolve by one field, missing field |
| `buffer_geometry` | `buffer_geometry(geometry, distance, *, quad_segs=8) -> dict[str, Any]` | GeoJSON Geometry or Feature | polygon geometry and operation info | `empty_geometry`, `invalid_geometry`, `invalid_argument`, `BackendUnavailable` | No geodesic buffers. Compare area/bounds to Shapely buffer when installed. | point buffer, negative line buffer if backend supports, invalid distance |
| `clip_vector` | `clip_vector(source, clip_geometry, *, clip_crs=None) -> dict[str, Any]` | vector source/read result plus clip geometry | FeatureCollection plus `VectorInfo` dict and operation info | `crs_mismatch`, `empty_feature_set`, `invalid_geometry`, `BackendUnavailable` | No raster clipping. Compare to GeoPandas clip when installed. | polygon clipped by box, CRS mismatch, no intersections |
| `intersect_vectors` | `intersect_vectors(left, right, *, suffixes=("_left", "_right")) -> dict[str, Any]` | two vector sources/read results | FeatureCollection with joined properties and `VectorInfo` dict | `crs_mismatch`, `missing_layer`, `invalid_geometry`, `BackendUnavailable` | No optimized spatial index requirement in first pass. Compare to GeoPandas overlay intersection when installed. | one overlap, no overlap, property name collision |
| `simplify_geometry` | `simplify_geometry(geometry, tolerance, *, preserve_topology=True) -> dict[str, Any]` | GeoJSON Geometry or Feature | simplified geometry and operation info | `invalid_argument`, `invalid_geometry`, `BackendUnavailable` if preserve topology requires GEOS | No scale-driven cartographic simplification. Compare to Shapely simplify when installed. | line simplification, polygon preserve topology, negative tolerance |
| `load_boundary` | `load_boundary(path, *, layer=None, where=None) -> dict[str, Any]` | vector path and optional attribute filter string | FeatureCollection/boundary geometry plus `VectorInfo` dict | `missing_layer`, `empty_feature_set`, `missing_crs`, `unsupported_option` for unsupported filters | No built-in downloads, no named domain datasets. Compare to `read_vector` count/bounds. | read whole boundary, filter unsupported, missing layer |

### Rasterization / Masks

| API | function signature | accepted input forms | output schema and metadata fields | stable diagnostics and errors | unsupported cases and reference comparison | minimum tests |
|---|---|---|---|---|---|---|
| `rasterize_vectors` | `rasterize_vectors(vectors, target_info, *, value=1, attribute=None, dtype="uint8", fill=0, all_touched=False) -> dict[str, Any]` | vector path/read result/geometries and `RasterInfo`/dict target grid | `RasterizationResult` fields | `missing_crs`, `crs_mismatch`, `unsupported_dtype`, `unsupported_geometry_type`, `BackendUnavailable` | No implicit target grid. Compare to `rasterio.features.rasterize` when installed. | polygon burn, attribute burn, all_touched toggle, CRS mismatch |
| `geometry_mask` | `geometry_mask(geometries, target_info, *, invert=False, all_touched=False, mask_polarity="true_inside") -> dict[str, Any]` | geometries and explicit target grid | `MaskResult` with bool mask and polarity | `mask_polarity_explicit`, `empty_geometry`, `crs_mismatch`, `BackendUnavailable` | No hidden polarity defaults. Compare to `rasterio.features.geometry_mask` when installed. | true_inside mask, inverted mask, missing polarity |
| `mask_raster` | `mask_raster(source, mask, *, mask_polarity, crop=False, fill=None, nodata=None) -> dict[str, Any]` | raster path/array/info and bool mask | `MaskResult` with output array, info, crop window, nodata/fill | `mask_polarity_explicit`, `empty_raster`, `unsupported_dtype`, `InvalidNodata`, `ShapeMismatch` | Does not call vector rasterization internally; caller passes mask. Compare to rasterio mask semantics when installed. | true_valid mask, crop, fill, nodata per band, shape mismatch |

### Thematic Raster

| API | function signature | accepted input forms | output schema and metadata fields | stable diagnostics and errors | unsupported cases and reference comparison | minimum tests |
|---|---|---|---|---|---|---|
| `normalize_raster` | `normalize_raster(source, *, method="minmax", valid_mask=None, nodata=None, clip=None) -> dict[str, Any]` | raster path, `RasterInfo`, serialized raster result, or NumPy array | `ThematicResult` with float output and stats | `empty_raster`, `unsupported_dtype`, `invalid_argument`, `ShapeMismatch` | No color mapping/rendering. Compare to NumPy reference calculations. | minmax, zscore if enabled, nodata excluded, all invalid |
| `classify_raster` | `classify_raster(source, *, bins=None, labels=None, right=False, valid_mask=None, nodata=None, dtype="uint16") -> dict[str, Any]` | raster path, `RasterInfo`, serialized raster result, or NumPy array | `ThematicResult` with integer class array and class table | `empty_raster`, `unsupported_dtype`, `invalid_argument`, `ShapeMismatch` | First pass should use explicit bins only; quantile/natural breaks are later. Compare to NumPy `digitize`. | explicit bins, labels, nodata class, bad bins, counts |

## Test Bundles and Fixture Strategy

### T-vector-io

- Fixtures: generated tiny GeoJSON files with point, line, polygon, mixed, empty, and missing-CRS cases; optional GPKG fixture generated only when GDAL is available.
- Happy paths: `read_vector`, `VectorInfo`, schema, count, CRS, bounds.
- Missing/invalid inputs: missing path, missing layer, unsupported extension, invalid GeoJSON.
- CRS mismatch behavior: not applicable except confirming no reprojection occurs.
- Empty input behavior: empty feature collection emits `empty_feature_set`.
- Invalid geometry behavior: invalid features are reported through warnings in C1.
- Metadata validation: all `VectorInfo.as_dict()` fields have stable keys and bounds order.
- Reference checks: Fiona/GeoPandas optional and skipped if unavailable.

### T-vector-crs

- Fixtures: tiny EPSG:4326 vector, missing-CRS vector, and explicit `src_crs` FeatureCollection.
- Happy paths: same-CRS passthrough and EPSG:4326 to EPSG:3857.
- Missing/invalid inputs: missing source CRS without `src_crs`, invalid destination CRS.
- CRS mismatch behavior: explicit mismatch diagnostics when combining vector sources in later phases.
- Empty input behavior: empty feature collection preserves CRS and count.
- Invalid geometry behavior: invalid geometries fail before coordinate transform unless backend can safely transform coordinates.
- Metadata validation: output CRS, transformed bounds, and feature count.
- Reference checks: pyproj/GeoPandas optional and skipped if unavailable.

### T-vector-geom

- Fixtures: inline GeoJSON dicts for unit square, bowtie polygon, concave polygon, disjoint polygons, lines, empty geometry, and properties for dissolve/intersect.
- Happy paths: validate, repair, measure, centroid, representative point, interpolate, union, dissolve, buffer, clip, intersect, simplify, load boundary.
- Missing/invalid inputs: unknown geometry type, missing geometry key, bad coordinates, missing dissolve field.
- CRS mismatch behavior: clip/intersect fail on mismatched CRS metadata unless explicit transform occurs in C2.
- Empty input behavior: empty geometry and empty collection diagnostics.
- Invalid geometry behavior: invalid geometry rejected or repaired only by `repair_geometry`.
- Output metadata validation: `GeometryOperationInfo` fields and changed/type-change diagnostics.
- Reference checks: Shapely/GeoPandas optional and skipped if unavailable.

### T-rasterize-mask

- Fixtures: existing tiny raster fixtures from G-002a1/G-002b plus inline polygons aligned to a small target grid.
- Happy paths: burn constant values, burn attribute values, create geometry mask, apply mask to raster.
- Missing/invalid inputs: missing target transform, shape mismatch, unsupported dtype, missing mask polarity.
- CRS mismatch behavior: vector CRS and target grid CRS mismatch fails.
- Empty input behavior: empty geometry creates all-fill raster or explicit empty diagnostic, as documented per API.
- Invalid geometry behavior: invalid geometry rejected before rasterization.
- Output metadata validation: array shape, transform, bounds, dtype, burned pixel counts, crop window.
- Reference checks: rasterio optional and skipped if unavailable.

### T-thematic

- Fixtures: tiny NumPy arrays and tiny rasters with nodata and masks.
- Happy paths: minmax normalize, optional zscore normalize, explicit-bin classify, class labels, class counts.
- Missing/invalid inputs: unsupported dtype, bad bins, label count mismatch, shape mismatch between mask and raster.
- CRS mismatch behavior: not applicable; thematic operations do not combine CRS-bearing sources.
- Empty input behavior: all nodata/all masked raises or reports `empty_raster`.
- Invalid geometry behavior: not applicable.
- Output metadata validation: stats exclude invalid pixels, class table counts sum to valid pixels.
- Reference checks: NumPy calculations only; no external GIS dependency needed.

## Strict Phase Hygiene

G-002c explicitly excludes:

- Remote fetching.
- Cache management.
- Live network services.
- OSM querying.
- Terrarium.
- Slippy tiles.
- DEM, landcover, population, and building domain helpers.
- Recipe manifests.
- Gallery goldens.
- MapScene recipe-family work.
- Network travel-time analysis.
- Runtime code in this planning change-set.
- PyO3 exports, Python wrappers, tests, examples, shaders, rendering behavior, or visual goldens in this planning change-set.

## Backward Compatibility With G-002a1/G-002b

G-002c must not regress:

- `read_raster_info`
- `read_raster`
- `write_raster`
- `RasterInfo`
- CRS helpers: `parse_crs`, `inspect_crs`, `raster_crs`, `assign_crs`, `create_crs_transformer`, `transform_bounds`, `web_mercator_bounds`
- Affine helpers: `AffineTransform`, `raster_transform`, `transform_from_origin`, `transform_from_bounds`, `array_bounds`, `raster_bounds`, `raster_resolution`, `validate_transform`, `pixel_convention`, `rowcol`, `xy`, `index`
- Nodata and mask helpers: `apply_nodata`, `read_raster_mask`
- Resample, alignment, and reprojection helpers: `resample_raster`, `assert_grid_compatible`, `align_raster_grid`, `align_raster_to`, `reproject_raster`, `calculate_default_transform`
- Windowing helpers: `window_from_bounds`, `read_raster_window`, `window_transform`
- Compatibility aliases: `bounds`, `align_raster_to`

Contract tests must keep existing API presence checks and add vector APIs only when their phase intentionally registers them.

## Implementation Order

1. C1.1: Add `VectorInfo` type and feature-gated vector metadata reader.
2. C1.2: Add C1 Python wrappers and `.pyi` stubs for read/schema/count/type helpers.
3. C1.3: Add vector CRS and bounds helpers using existing CRS/bounds conventions.
4. C1.4: Add C1 tests and contract locks.
5. C2.1: Add `reproject_vector` backend and always-xy CRS tests.
6. C3.1: Add geometry parsing and `validate_geometry`.
7. C3.2: Add `repair_geometry`.
8. C3.3: Add planar measure, centroid, representative point, and line interpolation.
9. C4.1: Add robust topology-backed union and dissolve.
10. C4.2: Add buffer, clip, intersect, simplify, and load boundary.
11. C5.1: Add rasterization to explicit `RasterInfo` grids.
12. C5.2: Add geometry masks with explicit polarity.
13. C5.3: Add raster masking/crop/fill/nodata handling.
14. C6.1: Add nodata-aware raster normalization.
15. C6.2: Add explicit-bin raster classification and class tables.

## Proposed First Implementation Prompt: C1 Only

Use this as the next Codex implementation prompt after Milos completes independent review:

```text
You are working in milos-agathon/forge3d.

Task: Implement G-002c C1 only: vector metadata/read foundation.

Use docs/carto-engine/g-002c-implementation-plan.md as the contract. Do not implement C2-C6.

Scope:
- Add Rust-first vector metadata/read behavior under src/gis/vector.rs or the smallest existing src/gis/ location that fits.
- Add VectorInfo as a PyO3 class mirroring RasterInfo style.
- Add thin Python wrappers and .pyi stubs for:
  - read_vector(path, *, layer=None, columns=None, bbox=None, limit=None)
  - geometry_type(source, *, layer=None)
  - vector_schema(source, *, layer=None)
  - feature_count(source, *, layer=None)
  - vector_crs(source, *, layer=None)
  - vector_bounds(source, *, layer=None)
- Register PyO3 exports and update tests/test_api_contracts.py EXPECTED_FUNCTIONS/EXPECTED_CLASSES for the C1 surface only.
- Keep Python wrappers thin; do not use rasterio, geopandas, shapely, rioxarray, xarray, or terra for backend behavior.
- If GDAL/OGR is feature-gated or unavailable, keep public functions importable and return stable BackendUnavailable errors for backend-required paths.

Required C1 behavior:
- read_vector returns GeoJSON-like features plus VectorInfo metadata.
- Bounds order is always (left, bottom, right, top).
- Missing CRS is reported, never guessed.
- Missing layer reports missing_layer.
- Empty datasets or filters report empty_feature_set.
- Unsupported drivers report unsupported_driver.
- Geometry serialization at the Python boundary is the GeoJSON-like subset from the plan.

Do not implement:
- reproject_vector
- validate_geometry, repair_geometry, measurement, centroid, representative point, interpolate
- union, dissolve, buffer, clip, intersect, simplify, load_boundary
- rasterize_vectors, geometry_mask, mask_raster
- normalize_raster, classify_raster
- vector writes, examples, shaders, rendering, visual goldens
- OSM, Terrarium, remote/cache/fetch, slippy tiles, domain helpers, recipe manifests, gallery goldens, MapScene recipe-family work

Tests:
- Add T-vector-io tests for tiny generated GeoJSON fixtures covering happy path, missing path, missing layer, unsupported driver, empty feature set, missing CRS warning, schema, feature count, geometry type, CRS, and bounds.
- Add optional reference checks only if reference libraries are installed, and skip cleanly otherwise.
- Do not run the full Rust/Python suite unless code/test files require it; for this implementation run the C1 tests plus the existing GIS contract tests.

Review bundle rule:
- After the completed C1 change-set, generate a fresh temporary review bundle containing git status, diff name-status, diff stat, full diff, untracked files, validation logs, command metadata/exit codes, branch, merge base, current commit, timestamp, and cwd.
- The review bundle is temporary evidence only and must never be staged, committed, or included in the PR.
```

## Validation Plan for This Planning Change-Set

Required commands:

- `git status --short`
- `git diff --name-status`
- `git diff --stat`
- `git diff`
- `git ls-files --others --exclude-standard`

Docs validation:

- This repository has Sphinx documentation workflows in `.github/workflows/docs.yml`; run `python -m sphinx -b html docs docs/_build/html` if Sphinx is installed.
- No full Rust/Python test suite is needed for this planning change-set because it changes documentation only.

Review bundle contents:

- `git status --short`
- `git diff --name-status`
- `git diff --stat`
- full `git diff`
- `git ls-files --others --exclude-standard`
- exact validation/test logs
- command metadata and exit codes
- branch name
- merge base
- current commit/worktree base
- timestamp
- cwd

The review bundle is temporary evidence only and must never be staged, committed, or included in the PR.

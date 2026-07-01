# G-002c C3 Vector Geometry Validity and Measurement Implementation Plan

Date: 2026-07-01

## Current State Verified

Current `origin/main` is `37eb1df65ddebfb7e12a5767cce13826f67cf6d2`, the merge commit for G-002c C2 `reproject_vector`. The local tracked tree has no diff against that ref.

File-level evidence from updated main:

- `src/gis/vector.rs` contains C1 `VectorInfo`, `VectorReadOptions`, `VectorReadResult`, `read_vector`, `read_vector_info`, GeoJSON feature normalization, schema inference, geometry type detection, bounds calculation, and C2 `VectorReprojectInput`, `VectorReprojectResult`, and `reproject_vector`.
- `src/gis/vector.rs` exposes PyO3 functions for `read_vector`, `geometry_type`, `vector_schema`, `feature_count`, `vector_crs`, `vector_bounds`, and `reproject_vector`.
- `src/gis/error.rs` already has `InvalidGeometry`, `InvalidArgument`, `MissingCrs`, `CrsMismatch`, `BackendUnavailable`, and the Python error mapping needed by C3.
- `src/gis/mod.rs` re-exports C1/C2 vector types/functions and PyO3 vector functions.
- `src/py_module/functions/gis.rs` registers `crate::gis::reproject_vector_py` with the native module.
- `src/py_module/classes.rs` registers `VectorInfo`; C3 does not require a new public class.
- `python/forge3d/gis.py` exposes a thin `reproject_vector` wrapper and includes it in `__all__`.
- `python/forge3d/gis.pyi` has the `reproject_vector` stub.
- `tests/test_api_contracts.py`, `tests/test_gis_raster.py`, `tests/test_gis_crs_affine.py`, and `tests/test_gis_vector_crs.py` cover the C2 public surface.
- `Cargo.toml` has no `geo`, `geo-types`, or GEOS dependency today. It has an optional `proj` feature, but current CRS/vector transforms use the built-in same-CRS and EPSG:4326/EPSG:3857 path.
- `pyproject.toml` maturin features currently enable `extension-module`, rendering features, and `copc_laz`; they do not enable any GIS topology backend.

Existing planning docs already cross-link C3 at a roadmap level:

- `docs/carto-engine/g-002c-implementation-plan.md`
- `docs/carto-engine/g-002c-c2-reproject-vector-implementation-plan.md`
- `docs/carto-engine/gis-operation-api-crosswalk.md`
- `docs/carto-engine/gis-contract-evidence.md`
- `docs/carto-engine/g-002b-support-matrix.md`

This file is the focused C3-only execution plan and should be linked from `docs/carto-engine/g-002c-implementation-plan.md` after implementation starts.

## C3 Scope

Plan implementation of exactly these public APIs:

- `validate_geometry(geometry) -> dict[str, Any]`
- `repair_geometry(geometry, *, method="make_valid") -> dict[str, Any]`
- `geometry_measure(geometry, *, metrics=("area", "length")) -> dict[str, Any]`
- `geometry_centroid(geometry) -> dict[str, Any]`
- `representative_point(geometry) -> dict[str, Any]`
- `interpolate_line(geometry, distance, *, normalized=False) -> dict[str, Any]`

C3 accepts only the existing GeoJSON-like boundary:

- Geometry dicts with `{"type": "...", "coordinates": ...}`.
- Feature dicts with `{"type": "Feature", "geometry": ..., "properties": ...}`. Feature properties are ignored by C3 operations unless copied through an unchanged existing return shape.
- FeatureCollection/read-vector-result dicts with `{"type": "FeatureCollection", "features": [...]}` where explicitly listed below.

C3 does not add WKB bytes, WKT strings, Python Shapely objects, or Rust-owned geometry object inputs.

C3 must handle Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, and GeometryCollection deliberately. Unsupported or inapplicable geometry types must fail with stable diagnostics rather than falling through as unknown behavior.

## API Contracts

### `validate_geometry`

Return:

```python
{
    "valid": bool,
    "reason": str | None,
    "geometry_type": str,
    "warnings": list[dict[str, str | None]],
}
```

Behavior:

- Accept Geometry, Feature, GeometryCollection, and FeatureCollection/read-vector-result inputs.
- Normalize Feature input to its `geometry` only.
- Validate each FeatureCollection feature and return aggregate validity.
- Detect `empty_geometry`.
- Reject malformed objects with `invalid_geometry`.
- Reject unsupported geometry types with `unsupported_geometry_type`.
- Validate finite numeric positions, required coordinate nesting, minimum point counts, closed polygon rings, minimum ring size, and obvious segment self-intersection for polygon rings.
- Do not claim GEOS-grade topology in the default backend. If the implementation cannot answer robust validity for a geometry type or case without `geos-topology`, raise `BackendUnavailable` with `backend_unavailable: geos-topology feature required for robust validity`; never silently overclaim.

### `repair_geometry`

Return:

```python
{
    "geometry": dict[str, Any],
    "operation": GeometryOperationInfo,
    "valid": bool,
}
```

Behavior:

- Default method is `"make_valid"`.
- Accept Geometry and Feature inputs. Feature input repairs the Feature geometry only and returns a Geometry.
- Reject FeatureCollection in C3 with `unsupported_geometry_type` to avoid inventing a vector-write or feature-preserving repair contract.
- Unsupported methods fail as `InvalidArgument` with `unsupported_option`.
- Without `geos-topology`, `method="make_valid"` raises `BackendUnavailable` with `backend_unavailable: geos-topology feature required for make_valid`.
- With `geos-topology`, use GEOS `make_valid` or the crate-supported equivalent.
- Never silently change geometry type. If output type differs from input type, include `geometry_repair_changed_type` in `operation["warnings"]` and set `operation["changed"] = true`.
- If repair changes coordinate topology but not type, include `geometry_repaired`.
- Do not implement `buffer(0)` as a fake default repair unless a future explicit method is added and tested.

### `geometry_measure`

Return:

```python
{
    "area": float | None,
    "length": float | None,
    "units": "source_crs_planar",
    "operation": GeometryOperationInfo,
}
```

Behavior:

- Accept Geometry, Feature, GeometryCollection, and FeatureCollection/read-vector-result inputs.
- Measurements are planar only. No geodesic measurement in C3.
- Supported metric names are exactly `"area"` and `"length"`.
- Unsupported metric names fail as `InvalidArgument` with `unsupported_option`.
- Polygon and MultiPolygon area use planar ring area: outer rings minus holes.
- Polygon and MultiPolygon length use boundary/perimeter length.
- LineString and MultiLineString length use planar Euclidean segment length.
- Point and MultiPoint return `None` for both area and length because the metrics are not applicable.
- GeometryCollection and FeatureCollection sum applicable component measurements and return `None` only when no component contributes that metric.
- Empty geometry fails with `InvalidGeometry` containing `empty_geometry`.

### `geometry_centroid`

Return:

```python
{
    "geometry": {"type": "Point", "coordinates": [x, y]},
    "operation": GeometryOperationInfo,
}
```

Behavior:

- Accept Geometry, Feature, GeometryCollection, and FeatureCollection/read-vector-result inputs.
- Point returns itself; MultiPoint returns the arithmetic mean of points.
- LineString/MultiLineString use length-weighted segment centroids.
- Polygon/MultiPolygon use planar area-weighted centroids, with hole area subtracted.
- GeometryCollection/FeatureCollection use the same component weighting policy: polygon area first where present, then line length where no polygonal area exists, then point average.
- Empty or zero-measure inputs fail with `empty_geometry` or `invalid_geometry` rather than returning NaN.
- No representative/interior guarantee is made.

### `representative_point`

Return:

```python
{
    "geometry": {"type": "Point", "coordinates": [x, y]},
    "operation": GeometryOperationInfo,
}
```

Behavior:

- Accept Geometry and Feature inputs.
- Point returns itself.
- MultiPoint returns the first non-empty point in input order.
- LineString returns the midpoint by length; MultiLineString uses cumulative length in input order.
- Polygon/MultiPolygon require an inside-or-on guarantee. Without `geos-topology`, polygonal representative point raises `BackendUnavailable` with `backend_unavailable: geos-topology feature required for polygon representative_point`.
- With `geos-topology`, use a GEOS point-on-surface/representative-point operation and test the inside-or-on guarantee.
- GeometryCollection is handled deliberately: use the first non-empty polygonal geometry when topology backend is available; otherwise use the first non-empty line or point if no polygonal member exists. If no supported member exists, fail with `unsupported_geometry_type` or `empty_geometry`.
- FeatureCollection is out of scope for this API in C3 and fails with `unsupported_geometry_type`.
- No cartographic label placement or collision behavior.

### `interpolate_line`

Return:

```python
{
    "geometry": {"type": "Point", "coordinates": [x, y]},
    "distance": float,
    "normalized": bool,
    "operation": GeometryOperationInfo,
}
```

Behavior:

- Accept LineString and MultiLineString Geometry inputs, or Feature inputs whose geometry is LineString/MultiLineString.
- Reject Point, Polygon, MultiPoint, MultiPolygon, GeometryCollection, and FeatureCollection with `unsupported_geometry_type`.
- `normalized=False`: `distance` is planar distance in source units.
- `normalized=True`: `distance` is a fraction in `[0.0, 1.0]`; implementation converts it to cumulative line length internally.
- Reject negative distances with `invalid_argument`.
- Reject normalized values outside `[0.0, 1.0]` with `invalid_argument`.
- Reject non-normalized distances greater than total length with `invalid_argument`.
- MultiLineString semantics are cumulative length in input geometry order.
- Return the first endpoint at distance `0.0` and the last endpoint at total length or normalized `1.0`.
- No routing, snapping, or network analysis.

## `GeometryOperationInfo`

Every operation metadata dict uses this stable shape:

```python
{
    "name": str,
    "input_geometry_type": str,
    "output_geometry_type": str | None,
    "input_count": int,
    "output_count": int,
    "changed": bool,
    "crs": dict[str, Any] | None,
    "warnings": list[dict[str, str | None]],
}
```

Rules:

- `input_count` is the count of normalized input geometries: `1` for Geometry/Feature, member count for GeometryCollection, and feature count for FeatureCollection where supported.
- `output_count` is `1` for point/geometry outputs, `0` only for error paths that never return, and the normalized output member count for optional repair paths that produce collection-like outputs.
- `changed` is true only when coordinates, topology, or type changed.
- `crs` is `None` unless the input is a read-vector result with `info` or `vector_info` carrying CRS metadata. Missing CRS is reported, never guessed.
- `warnings` uses the existing `RasterWarning` dict style: `{"code", "message", "field"}`.

## Backend Dependency Decisions

Default C3 should use a small Rust-first geometry layer under `src/gis/` for parsing, finite coordinate validation, planar area, planar length, centroid, line interpolation, and obvious polygon ring checks. These algorithms are tractable, avoid system dependencies, and fit the current local GeoJSON-only C1/C2 vector backend.

Do not add GDAL for C3.

Do not use `rasterio`, `geopandas`, `shapely`, `rioxarray`, `xarray`, or `terra` as forge3d runtime backend behavior. They may appear only in optional reference tests or documentation examples and must skip cleanly when unavailable.

Optional topology backend:

- Add a Cargo feature named `geos-topology` only when implementing GEOS-backed validity/repair/representative-point behavior.
- If a GEOS crate is added, make it an optional dependency and keep public functions importable without it.
- Do not add `geos-topology` to `pyproject.toml` maturin features unless CI and wheel packaging intentionally support the system dependency.
- Backend-required behavior without the feature must raise `BackendUnavailable` with messages containing both `backend_unavailable` and `geos-topology`.
- Do not claim GEOS-grade validity, make-valid, or polygon representative-point guarantees unless the feature is wired, exported, included in tests, and exercised.

`geo`/`geo-types` decision:

- Do not add them by default for C3 unless implementation proves the manual Rust geometry layer is becoming complex or incorrect.
- If added, justify the dependency in the implementation PR and use it for planar area, length, centroid, and line interpolation only.
- Adding `geo`/`geo-types` does not replace GEOS for robust make-valid or polygon representative-point guarantees.

## File-By-File Deltas

### `src/gis/geometry.rs` or `src/gis/vector_geom.rs`

- Add a new GIS geometry module rather than extending `src/gis/vector.rs`, which is already large after C2.
- Define an internal parsed geometry representation that preserves GeoJSON type names and coordinates.
- Add parsing/normalization for Geometry, Feature, GeometryCollection, and supported FeatureCollection cases.
- Add `GeometryOperationInfo`.
- Add helpers for warnings, geometry type counts, finite coordinate validation, line segment length, ring area, ring closure, polygon self-intersection checks, centroid math, and line interpolation.
- Add core Rust functions:
  - `validate_geometry`
  - `repair_geometry`
  - `geometry_measure`
  - `geometry_centroid`
  - `representative_point`
  - `interpolate_line`
- Add PyO3 wrappers:
  - `validate_geometry_py`
  - `repair_geometry_py`
  - `geometry_measure_py`
  - `geometry_centroid_py`
  - `representative_point_py`
  - `interpolate_line_py`
- Reuse or move the JSON conversion helpers now private in `src/gis/vector.rs` so C2 and C3 share one implementation instead of duplicating PyDict/PyList conversion.

### `src/gis/vector.rs`

- Keep C1/C2 behavior unchanged.
- Move only shared GeoJSON/PyO3 helpers if needed:
  - `py_to_json_strict`
  - `json_to_py`
  - `warnings_to_py`
  - read-vector-result/vector-info extraction helpers where C3 needs CRS metadata.
- Preserve `read_vector` and `reproject_vector` return shapes exactly.

### `src/gis/mod.rs`

- Add the new C3 geometry module.
- Re-export the six Rust C3 functions.
- Re-export the six PyO3 C3 functions under `#[cfg(feature = "extension-module")]`.

### `src/py_module/functions/gis.rs`

- Register the six new native free functions with `wrap_pyfunction!`.
- Keep them as module-level free functions. Do not add methods on `VectorInfo`.

### `src/py_module/classes.rs`

- No new public class is required.
- Do not register `GeometryOperationInfo` as a class; return a dict.

### `src/gis/error.rs`

- Prefer existing `InvalidGeometry`, `InvalidArgument`, and `BackendUnavailable`.
- Add no new error variant unless implementation exposes a materially distinct error that cannot be expressed by existing variants.
- Ensure messages include stable lowercase tokens listed below.

### `src/gis/types.rs`

- Reuse `RasterWarning` for C3 warning dicts.
- Add geometry warning constants only if they are shared across modules. Otherwise keep C3-local constants in the new geometry module.

### `Cargo.toml`

- No default dependency is required for the pure Rust C3 layer.
- Optional only if implementing GEOS-backed paths:
  - Add `geos-topology = ["dep:geos"]` or the crate-specific equivalent.
  - Add an optional GEOS crate dependency.
- Do not add GDAL for C3.

### `pyproject.toml`

- No change for pure Rust C3.
- Do not add `geos-topology` to maturin features unless release packaging is prepared to ship or require GEOS.

### `python/forge3d/gis.py`

- Add thin wrappers only:
  - `validate_geometry(geometry)`
  - `repair_geometry(geometry, *, method="make_valid")`
  - `geometry_measure(geometry, *, metrics=("area", "length"))`
  - `geometry_centroid(geometry)`
  - `representative_point(geometry)`
  - `interpolate_line(geometry, distance, *, normalized=False)`
- Wrappers may call `_path_or_self` only for existing read-vector-result/path-adjacent consistency. They must not implement GIS behavior in Python.
- Add the six names to `__all__`.

### `python/forge3d/gis.pyi`

- Add stubs for the six C3 APIs.
- Keep return types as `dict[str, Any]`.
- Include accepted input type as `dict[str, Any] | VectorInfo` only where the wrapper actually supports it. Do not advertise Shapely/WKB/WKT.

### `tests/test_api_contracts.py`

- Add exactly the six C3 native function names to `EXPECTED_FUNCTIONS`.
- Do not add C4/C5/C6 function names.

### `tests/test_gis_raster.py`

- Add the six C3 APIs to `gis.__all__`.
- Add callable assertions for each.
- Keep `test_public_gis_all_matches_stub_exports` passing.

### `tests/test_gis_crs_affine.py`

- Add the six C3 APIs to the runtime/stub exposure check.

### `tests/test_gis_vector_geom.py`

- Add focused C3 behavior tests described below.

## Diagnostics

Use existing `GisError` variants where possible and include these lowercase diagnostic tokens in messages or warning codes:

- `empty_geometry`
- `invalid_geometry`
- `geometry_repaired`
- `geometry_repair_changed_type`
- `unsupported_geometry_type`
- `unsupported_option`
- `invalid_argument`
- `backend_unavailable`

Mapping policy:

- Structural or malformed geometry: `GisError::InvalidGeometry("invalid_geometry: ...")`.
- Unsupported geometry for the requested operation: `GisError::InvalidGeometry("unsupported_geometry_type: ...")`.
- Unsupported metric/method/option: `GisError::InvalidArgument("unsupported_option: ...")`.
- Bad distance/range/numeric option: `GisError::InvalidArgument("invalid_argument: ...")`.
- Missing optional topology backend: `GisError::BackendUnavailable("backend_unavailable: geos-topology feature required for ...")`.
- Non-fatal repair/topology notes: `RasterWarning` codes `geometry_repaired` and `geometry_repair_changed_type`.

## Test Plan

Add `tests/test_gis_vector_geom.py` with fixtures:

- valid unit square polygon
- self-intersecting bowtie polygon
- concave polygon for representative point
- LineString with known length
- MultiLineString cumulative length case
- point geometry
- empty geometry
- malformed geometry object
- unsupported geometry type
- GeometryCollection
- changed-type repair case
- already-valid repair unchanged case
- negative and out-of-range interpolate distances
- unsupported metric name

Minimum tests:

- `test_validate_geometry_valid_unit_square`
- `test_validate_geometry_detects_bowtie_self_intersection`
- `test_validate_geometry_empty_geometry_reports_empty_geometry`
- `test_validate_geometry_malformed_and_unsupported_geometry_errors`
- `test_geometry_measure_polygon_area_and_boundary_length`
- `test_geometry_measure_line_length`
- `test_geometry_measure_points_return_none_for_area_and_length`
- `test_geometry_measure_geometry_collection_sums_applicable_metrics`
- `test_geometry_measure_rejects_unsupported_metric`
- `test_geometry_centroid_point_line_polygon_and_collection`
- `test_representative_point_point_and_line_without_topology_backend`
- `test_representative_point_polygon_requires_or_uses_geos_topology`
- `test_interpolate_line_linestring_distance`
- `test_interpolate_line_multilinestring_cumulative_order`
- `test_interpolate_line_rejects_negative_out_of_range_and_unsupported_geometry`
- `test_repair_geometry_make_valid_requires_or_uses_geos_topology`
- `test_repair_geometry_changed_type_warning_with_geos_topology`
- `test_repair_geometry_already_valid_unchanged_with_geos_topology`
- `test_public_gis_vector_geom_wrapper_surface`
- `test_no_python_gis_backend_imports_in_forge3d_gis`

Reference-library comparison policy:

- Optional Shapely checks may compare area, length, centroid, representative point, and make-valid outputs.
- Use `pytest.importorskip("shapely.geometry")` or equivalent and skip cleanly when unavailable.
- Reference tests must not become runtime backend behavior.
- Use approximate numeric comparisons for planar floats.
- Do not require GeoPandas for C3 unit tests unless comparing FeatureCollection behavior; if used, skip cleanly.

Rust unit tests:

- Add unit tests for coordinate parsing, ring closure, ring area with holes, segment intersection, centroid weighting, MultiLineString cumulative interpolation, and operation metadata.
- If `geos-topology` is added, guard GEOS-specific tests with `#[cfg(feature = "geos-topology")]`.

## Implementation Checklist

### Planning

- [ ] Confirm `origin/main` contains C1 and C2 before coding.
- [ ] Link this phase-specific plan from `docs/carto-engine/g-002c-implementation-plan.md`.
- [ ] Confirm whether `geos-topology` will be implemented in the same PR or left as importable `BackendUnavailable` behavior.
- [ ] Confirm no C4/C5/C6 API names are added.

### Implementation

- [ ] Add the new Rust C3 geometry module under `src/gis/`.
- [ ] Implement GeoJSON-like input normalization for supported C3 inputs.
- [ ] Implement `GeometryOperationInfo` dict construction.
- [ ] Implement pure Rust validation for empty, malformed, unsupported, finite coordinates, ring defects, and obvious polygon self-intersection.
- [ ] Implement planar area, length, centroid, and line interpolation.
- [ ] Implement `repair_geometry` as GEOS-backed when `geos-topology` is enabled, otherwise stable `BackendUnavailable`.
- [ ] Implement polygonal `representative_point` with GEOS when enabled, otherwise stable `BackendUnavailable`.
- [ ] Add PyO3 wrappers and register them as module-level free functions.
- [ ] Add thin Python wrappers and `.pyi` stubs.
- [ ] Update public surface and native contract tests.

### Tests

- [ ] Add `tests/test_gis_vector_geom.py`.
- [ ] Add wrapper/stub/native contract updates only for the six C3 APIs.
- [ ] Add no-backend-Python-GIS-library tests.
- [ ] Add optional Shapely/GEOS reference tests with clean skips.
- [ ] Keep existing C1/C2 tests passing without return-shape drift.

### Validation

Required validation for the C3 implementation PR:

- [ ] `git status --short`
- [ ] `git diff --name-status`
- [ ] `git diff --stat`
- [ ] `git diff`
- [ ] `git ls-files --others --exclude-standard`
- [ ] `cargo fmt --check`
- [ ] `cargo check`
- [ ] `python3 -m py_compile python/forge3d/gis.py`
- [ ] `python3 -m pytest tests/test_gis_vector_geom.py -v`
- [ ] `python3 -m pytest tests/test_api_contracts.py -v`
- [ ] focused wrapper/stub/no-Python-backend-GIS tests already used by C1/C2
- [ ] C2 vector CRS tests if shared vector parsing, JSON conversion, metadata, or warnings are touched: `python3 -m pytest tests/test_gis_vector_crs.py -v`
- [ ] broader cargo/python tests only if touched files justify them

### Review Bundle

After the planning or implementation change-set, create a fresh temporary review bundle containing:

- git status
- diff name-status
- diff stat
- full diff, including untracked docs
- untracked files
- validation logs
- command metadata and exit codes
- branch
- merge base
- current commit
- timestamp

## Backward Compatibility Requirements

C3 must preserve C1/C2 and all G-002a1/G-002b APIs:

- `read_raster_info`
- `read_raster`
- `write_raster`
- `RasterInfo`
- `VectorInfo`
- `read_vector`
- `geometry_type`
- `vector_schema`
- `feature_count`
- `vector_crs`
- `vector_bounds`
- `reproject_vector`
- `parse_crs`
- `inspect_crs`
- `raster_crs`
- `assign_crs`
- `create_crs_transformer`
- `transform_bounds`
- `web_mercator_bounds`
- affine helpers
- nodata/mask helpers
- resampling/alignment/reprojection/window helpers
- compatibility aliases `bounds` and `align_raster_to`

No existing return shapes may drift unless a test and migration note explicitly justify the change. C3 must not move backend behavior into Python.

## Explicit Non-Goals

C3 must not add or implement:

- `union_geometries`
- `dissolve_vector`
- `buffer_geometry`
- `clip_vector`
- `intersect_vectors`
- `simplify_geometry`
- `load_boundary`
- `rasterize_vectors`
- `geometry_mask`
- `mask_raster`
- `normalize_raster`
- `classify_raster`
- vector writes
- examples
- shaders
- rendering
- visual goldens
- remote/cache/fetch
- OSM
- Terrarium
- slippy tiles
- DEM/landcover/population/building domain helpers
- recipe manifests
- gallery goldens
- MapScene recipe-family work
- network travel-time analysis

## Open Questions

No open question blocks C3 planning. The conservative decision is to ship pure Rust structural/planar geometry behavior by default, expose robust topology-dependent behavior only behind `geos-topology`, and fail with stable `BackendUnavailable` where the optional backend is required.

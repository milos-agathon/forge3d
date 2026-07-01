# G-002c C2 `reproject_vector` Implementation Plan

## 1. Current repository state

- Inspected files: `AGENTS.md`, `Cargo.toml`, `src/lib.rs`, `src/gis/mod.rs`, `src/gis/vector.rs`, `src/gis/crs.rs`, `src/gis/error.rs`, `src/gis/raster_write.rs`, `src/py_module/functions/gis.rs`, `src/py_module/classes.rs`, `python/forge3d/gis.py`, `python/forge3d/gis.pyi`, `tests/test_gis_raster.py`, `tests/test_gis_crs_affine.py`, `docs/carto-engine/g-002b-support-matrix.md`, `docs/carto-engine/g-002c-implementation-plan.md`, `docs/carto-engine/gis-operation-api-crosswalk.md`, and `docs/carto-engine/gis-contract-evidence.md`.
- `src/gis/vector.rs` currently contains `VectorInfo`, `VectorReadOptions`, `VectorReadResult`, `read_vector`, `read_vector_info`, GeoJSON feature normalization, schema inference, geometry type detection, bounds calculation, and PyO3 functions for `read_vector`, `geometry_type`, `vector_schema`, `feature_count`, `vector_crs`, and `vector_bounds`.
- `src/gis/vector.rs` currently supports local `.geojson` and `.json` directly. `.gpkg`, `.shp`, `.fgb`, and `.flatgeobuf` fail with `BackendUnavailable` in the current C1 path because `gdal-vector` is not wired.
- `src/gis/vector.rs` already reads GeoJSON CRS metadata from `crs.properties.name`, normalizes EPSG URN/URL-style names to `EPSG:<code>`, and emits `missing_crs` warnings rather than guessing.
- `src/gis/crs.rs` currently provides `CrsInspection`, CRS equality, `authority_map`, `canonical_label`, `transform_point`, `transform_bounds`, `parse_crs_string`, and `CrsTransform`.
- Built-in CRS transforms currently support same-CRS passthrough and EPSG:4326 <-> EPSG:3857 only. Other parseable CRS pairs fail with `BackendUnavailable` unless a future external backend is wired. `src/gis/raster_write.rs` currently parses only EPSG:4326, EPSG:3857, EPSG:32631, and the limited WGS84 WKT subset.
- `Cargo.toml` defines an optional `proj` feature, but the inspected `src/gis/crs.rs` transform path does not currently use it for arbitrary vector or raster transformations.
- `src/gis/error.rs` has `MissingCrs`, `InvalidCrs`, `CrsMismatch`, `TransformFailed`, and `BackendUnavailable`. It does not have a dedicated `InvalidGeometry` variant; current C1 vector malformed-geometry paths use `InvalidArgument` messages containing `invalid_geometry`.
- `python/forge3d/gis.py`, `python/forge3d/gis.pyi`, `src/gis/mod.rs`, and `src/py_module/functions/gis.rs` do not yet expose or register `reproject_vector`.
- `src/py_module/classes.rs` already registers `VectorInfo` and `CrsTransform`; C2 does not need a new public class.
- Public wrapper-surface tests in `tests/test_gis_raster.py` assert the exact `forge3d.gis.__all__` set, so adding `reproject_vector` requires synchronized wrapper, stub, and test updates.
- Native API contract tests in `tests/test_api_contracts.py` lock registered free functions and classes. Adding a native `reproject_vector` registration requires adding it to `EXPECTED_FUNCTIONS`.
- `docs/carto-engine/gis-operation-api-crosswalk.md` and `docs/carto-engine/gis-contract-evidence.md` identify G-002c `vector_reprojection` / `to_crs` as `reproject_vector`, with 39 evidence hits, 23 scripts, and GeoPandas `to_crs` as reference behavior. These are reference contracts, not backend implementation permission.

## 2. Contract decisions for C2

- Public Python signature: `reproject_vector(input, dst_crs, src_crs=None)`.
- Rust entry point:

```rust
pub fn reproject_vector(
    input: VectorReprojectInput,
    dst_crs: CrsSpec,
    src_crs: Option<CrsSpec>,
) -> GisResult<VectorReprojectResult>
```

- This Rust shape is appropriate because Python-specific extraction stays at the PyO3 boundary, while core reprojection receives normalized features, optional metadata, and parsed CRS values.
- Accepted C2 inputs:
  - local GeoJSON/JSON path, read through existing `read_vector`;
  - `read_vector` result dict or FeatureCollection-like dict containing `features` plus `info` or `vector_info`;
  - raw GeoJSON FeatureCollection, Feature, or Geometry only when `src_crs` is explicitly supplied or supported GeoJSON CRS metadata can be extracted without guessing.
- Reject `VectorInfo` alone for reprojection because it has metadata but no feature payload.
- Source CRS resolution order:
  - If `src_crs` argument is supplied, parse it through the existing CRS parsing path.
  - If input metadata also contains CRS and it conflicts with supplied `src_crs`, fail with `CrsMismatch`; do not reinterpret coordinates.
  - If `src_crs` is not supplied, read CRS from `VectorInfo`, result `info`, or supported GeoJSON CRS metadata.
  - If no CRS is available, fail with `MissingCrs`.
  - Never guess EPSG:4326.
- Destination CRS:
  - `dst_crs` is required.
  - Parse `dst_crs` through the existing CRS parsing path.
  - Invalid destination CRS fails with `InvalidCrs`.
- Coordinate semantics:
  - Transform x/y only.
  - Preserve z and additional ordinates unchanged.
  - Preserve feature properties, ids, and non-geometry members.
  - Recompute geometry-derived metadata: output bounds in `VectorInfo`, result `bounds`, and any existing GeoJSON `bbox` members. Remove an existing `bbox` when the transformed geometry has no bounds.
  - Preserve null and empty geometries as empty output geometries, keep the feature in the output, and add an `empty_geometry` warning.
  - Recompute output bounds from transformed coordinates in `(left, bottom, right, top)` order.
  - Output feature count equals input feature count unless a future explicit invalid-feature skip mode is added. C2 should use fatal `InvalidGeometry` for malformed coordinates instead of silently dropping features.
- Geometry repair:
  - Do not implement geometry repair in C2 because the inspected repo has no stable geometry repair primitive.
  - Treat `geometry_repaired` as optional/future and do not emit it in C2.
  - Invalid coordinate structure fails with the planned `InvalidGeometry` error variant.
- Backend limit:
  - Same CRS returns copied features with metadata updated to the destination CRS.
  - EPSG:4326 -> EPSG:3857 and EPSG:3857 -> EPSG:4326 use the existing built-in `crs::transform_point` path.
  - Other valid CRS pairs fail with the current stable `BackendUnavailable` path unless a separate feature-gated PROJ-backed path is explicitly added with tests. This C2 plan does not add that backend.

## 3. Data structures and helper functions

Planned additions in `src/gis/vector.rs`:

- `pub enum VectorReprojectInput`
  - `Path(PathBuf)` for local GeoJSON/JSON reads.
  - `Features { features: Vec<serde_json::Value>, info: Option<VectorInfo>, source_crs: Option<CrsSpec>, path: String, driver: String, layer_name: Option<String> }` for already-extracted read results and raw GeoJSON inputs.
  - Responsibility: keep Python extraction separate from core reprojection while retaining enough metadata to build output `VectorInfo`.
- `pub struct VectorReprojectResult`
  - Fields: `features: Vec<serde_json::Value>`, `info: VectorInfo`, `src_crs: CrsSpec`, `dst_crs: CrsSpec`, `bounds: Option<(f64, f64, f64, f64)>`, `feature_count: u64`, `warnings: Vec<RasterWarning>`.
  - Responsibility: native result model for Python conversion and Rust tests.
- `pub fn reproject_vector(input, dst_crs, src_crs)`
  - Responsibility: normalize input, resolve source CRS, check CRS conflicts with `crs::crs_equal`, call geometry helpers, build output `VectorInfo`, and return `VectorReprojectResult`.
- `fn resolve_vector_source_crs(explicit: Option<CrsSpec>, metadata: Option<CrsSpec>) -> GisResult<CrsSpec>`
  - Responsibility: implement the exact source CRS resolution order and emit `MissingCrs` or `CrsMismatch`.
- `fn crs_spec_from_vector_info(info: &VectorInfo) -> GisResult<Option<CrsSpec>>`
  - Responsibility: convert `crs_authority` through `CrsSpec::from_parts(Some(name), Some(code), None)` or `crs_wkt` through `CrsSpec::from_parts(None, None, Some(wkt))`.
- `fn geojson_crs_from_value(root: &Value) -> GisResult<Option<CrsSpec>>`
  - Responsibility: expose the existing GeoJSON CRS extraction behavior for raw dict inputs; this can reuse or rename current private `geojson_crs`.
- `fn reproject_feature(feature: &Value, src: &CrsSpec, dst: &CrsSpec, warnings: &mut Vec<RasterWarning>) -> GisResult<Value>`
  - Responsibility: copy the feature object, preserve properties/id/non-geometry members, transform the `geometry` member, and recompute/remove feature `bbox`.
- `fn reproject_geometry(geometry: &Value, src: &CrsSpec, dst: &CrsSpec, warnings: &mut Vec<RasterWarning>) -> GisResult<Value>`
  - Responsibility: handle null, empty, standard GeoJSON geometry objects, and `GeometryCollection` recursion.
- `fn reproject_coordinates_for_geometry_type(kind: &str, coordinates: &Value, src: &CrsSpec, dst: &CrsSpec) -> GisResult<Value>`
  - Responsibility: enforce the coordinate nesting level for Point, LineString, Polygon, MultiPoint, MultiLineString, and MultiPolygon.
- `fn reproject_position(position: &Value, src: &CrsSpec, dst: &CrsSpec) -> GisResult<Value>`
  - Responsibility: require at least numeric finite x/y, call `crs::transform_point`, and preserve all ordinates after index 1 unchanged.
- `fn build_reprojected_vector_info(...) -> VectorInfo`
  - Responsibility: use `bounds_for_features`, `geometry_type_for_features`, `infer_schema`, `crs::authority_map`, and destination `CrsSpec` to build output metadata.
- `fn warning_once(warnings: &mut Vec<RasterWarning>, code: &str, message: &str, field: Option<&str>)`
  - Responsibility: avoid duplicate `empty_geometry` warnings for many null/empty geometries.

Existing helper usage:

- `crs::transform_point` performs all x/y transforms and enforces the current backend limits.
- `crs::crs_equal` decides same-CRS copy behavior and supplied-vs-metadata conflict checks.
- `crs::authority_map` fills output `VectorInfo.crs_authority`.
- `bounds_for_features` recomputes output bounds.
- `geometry_type_for_features` recomputes output geometry type.
- `infer_schema` preserves current schema inference from transformed features.

## 4. PyO3 and Python wrapper plan

Files to change during implementation:

- `src/gis/vector.rs`
  - Add `#[pyfunction(name = "reproject_vector", signature = (input, dst_crs, src_crs = None))]`.
  - Add Python input extraction for path, read-vector result dict, FeatureCollection, Feature, and Geometry.
  - Reject `VectorInfo`-only input with `InvalidArgument` containing `feature payload is required`.
  - Convert `VectorReprojectResult` to a Python dict.
  - Reuse existing PyO3 JSON conversion helpers or move them inside the module so `reproject_vector_py` and existing vector helpers share one conversion path.
- `src/gis/mod.rs`
  - Re-export Rust `reproject_vector`.
  - Re-export `reproject_vector_py` under `#[cfg(feature = "extension-module")]`.
  - Allow `src/gis/vector.rs` to use the existing CRS extraction and CRS inspection conversion helpers, or make small `pub(crate)` helper adjustments if Rust privacy requires it.
- `src/py_module/functions/gis.rs`
  - Register `crate::gis::reproject_vector_py`.
- `python/forge3d/gis.py`
  - Add the thin wrapper:

```python
def reproject_vector(input, dst_crs, src_crs=None):
    return _require_native().reproject_vector(_path_or_self(input), dst_crs, src_crs)
```

  - Add `"reproject_vector"` to `__all__`.
- `python/forge3d/gis.pyi`
  - Add `def reproject_vector(input: os.PathLike[str] | str | dict[str, Any], dst_crs: str | int | dict[str, Any], src_crs: str | int | dict[str, Any] | None = ...) -> dict[str, Any]: ...`.
  - Keep stubs and `__all__` contract tests aligned.
- `tests/test_api_contracts.py`
  - Add `"reproject_vector"` to `EXPECTED_FUNCTIONS`.
- `tests/test_gis_raster.py`
  - Add `"reproject_vector"` to the exact `gis.__all__` assertion and add `assert callable(gis.reproject_vector)`.
- `tests/test_gis_crs_affine.py`
  - Add `"reproject_vector"` to the runtime/stub exposure check if that test remains the central GIS stub surface check.

Planned Python return object:

- `"type": "FeatureCollection"`
- `"features"`
- `"info"` as a serialized `VectorInfo` dict
- `"vector_info"` as a `VectorInfo` object
- `"src_crs"` as CRS inspection/metadata dict
- `"dst_crs"` as CRS inspection/metadata dict
- `"bounds"`
- `"feature_count"`
- `"warnings"`

## 5. Error and diagnostic plan

- `MissingCrs`: source CRS is absent and `src_crs` is not supplied. Existing `GisError::MissingCrs` maps to Python `ValueError`.
- `InvalidCrs`: invalid `dst_crs` or invalid supplied `src_crs`. Existing `GisError::InvalidCrs` maps to Python `ValueError`.
- `CrsMismatch`: supplied `src_crs` conflicts with input CRS metadata. Existing `GisError::CrsMismatch` maps to Python `ValueError`.
- `InvalidGeometry`: unsupported or malformed GeoJSON geometry, non-finite coordinate, non-numeric x/y, too-short position, wrong coordinate nesting, or malformed `GeometryCollection`.
  - Add the smallest enum addition in `src/gis/error.rs`: `InvalidGeometry(String)`.
  - Add `code()` mapping to `"InvalidGeometry"`.
  - Add `message()` passthrough.
  - Map to Python `ValueError`.
  - Use message text containing `invalid_geometry`.
- `BackendUnavailable`: valid but unsupported CRS transform pair under the current built-in backend. Existing `GisError::BackendUnavailable` maps to Python `RuntimeError`.
- `empty_feature_set`: warning for zero features after source normalization/read; C2 should return an empty FeatureCollection if CRS is resolvable.
- `empty_geometry`: warning for null or empty geometries preserved in the output; add a vector warning constant in `src/gis/vector.rs`.
- `invalid_geometry`: warning only if a future non-fatal path skips or preserves an invalid feature in a defined way. C2 should use fatal `InvalidGeometry` for malformed coordinates.
- `geometry_repaired`: out of scope for C2 because no inspected stable geometry repair primitive exists.

Do not overload `InvalidArgument` for new C2 malformed geometry cases after adding `InvalidGeometry`. Existing C1 `invalid_geometry` uses can remain unless touched by the C2 implementation.

## 6. Test plan

Python integration tests, likely in new `tests/test_gis_vector_crs.py`:

- `test_reproject_vector_geojson_path_epsg4326_to_3857`
  - Fixture: small FeatureCollection file with GeoJSON CRS EPSG:4326 and Point, LineString, Polygon features.
  - Assert: `feature_count == 3`, `dst_crs.authority == {"name": "EPSG", "code": "3857"}`, transformed point approximates known Web Mercator values, bounds are `(left, bottom, right, top)`, and properties are preserved.
- `test_reproject_vector_dict_with_explicit_src_crs`
  - Fixture: raw FeatureCollection without CRS metadata.
  - Assert: `src_crs="EPSG:4326"` avoids `MissingCrs` and returns valid transformed features.
- `test_reproject_vector_missing_source_crs_errors`
  - Fixture: raw FeatureCollection without CRS metadata and no `src_crs`.
  - Assert: Python `ValueError` message contains `MissingCrs` or `missing_crs`.
- `test_reproject_vector_invalid_dst_crs_errors`
  - Fixture: valid EPSG:4326 vector with `dst_crs="EPSG:9999"` or malformed CRS.
  - Assert: Python `ValueError` message contains `InvalidCrs`.
- `test_reproject_vector_src_crs_conflict_errors`
  - Fixture: metadata says EPSG:4326, supplied `src_crs="EPSG:3857"`.
  - Assert: Python `ValueError` message contains `CrsMismatch`.
- `test_reproject_vector_same_crs_copies_features_and_updates_metadata`
  - Fixture: EPSG:4326 FeatureCollection.
  - Assert: coordinates unchanged, properties preserved, `src_crs` and `dst_crs` metadata present, and `feature_count` unchanged.
- `test_reproject_vector_preserves_z_coordinate`
  - Fixture: Point `[lon, lat, z]`.
  - Assert: x/y transformed and z unchanged.
- `test_reproject_vector_geometry_collection`
  - Fixture: GeometryCollection with Point and LineString.
  - Assert: recursive transformation and preserved collection shape.
- `test_reproject_vector_rejects_malformed_coordinates`
  - Fixture: string coordinate, non-finite coordinate, and too-short position cases.
  - Assert: Python `ValueError` message contains `InvalidGeometry` or `invalid_geometry`.
- `test_reproject_vector_public_wrapper_surface`
  - Update existing public surface checks to include `reproject_vector`.
- `test_public_gis_all_matches_stub_exports`
  - Must continue passing after stub and wrapper updates.

Rust unit tests in `src/gis/vector.rs`:

- `reproject_position_preserves_extra_ordinates`
  - Assert: x/y transform through EPSG:4326 -> EPSG:3857, z and later ordinates unchanged.
- `reproject_geometry_point_line_polygon_multipolygon`
  - Assert: correct recursive coordinate handling for Point, LineString, Polygon, and MultiPolygon.
- `resolve_vector_source_crs_errors_when_missing`
  - Assert: no explicit CRS and no metadata CRS returns `MissingCrs`.
- `resolve_vector_source_crs_rejects_conflict`
  - Assert: metadata EPSG:4326 plus explicit EPSG:3857 returns `CrsMismatch`.
- `same_crs_reprojection_is_copy_not_transform`
  - Assert: same-CRS output coordinates equal input coordinates and output metadata uses destination CRS.

Reference libraries such as GeoPandas, pyproj, and Shapely may be used only for expected numeric/reference behavior in tests or docs, never as forge3d backend implementation.

## 7. Implementation sequence for the later coding agent

- [ ] `src/gis/error.rs`: add `GisError::InvalidGeometry`, `code()`, `message()`, and PyErr mapping.
- [ ] `src/gis/vector.rs`: add CRS extraction helper for `VectorInfo` and expose/reuse GeoJSON CRS extraction for raw GeoJSON values.
- [ ] `src/gis/vector.rs`: add pure Rust coordinate transformation helpers and Rust unit tests for positions and geometry recursion.
- [ ] `src/gis/vector.rs`: add `VectorReprojectInput`, `VectorReprojectResult`, `resolve_vector_source_crs`, output `VectorInfo` builder, and core `reproject_vector`.
- [ ] `src/gis/vector.rs` and `src/gis/mod.rs`: add PyO3 wrapper extraction/conversion and share CRS inspection conversion without changing the public return shape of existing GIS functions.
- [ ] `src/gis/mod.rs` and `src/py_module/functions/gis.rs`: re-export and register the native function.
- [ ] `python/forge3d/gis.py` and `python/forge3d/gis.pyi`: add the thin wrapper, typed signature, and `__all__` entry.
- [ ] `tests/test_api_contracts.py`, `tests/test_gis_raster.py`, and `tests/test_gis_crs_affine.py`: update wrapper/stub/native contract checks.
- [ ] `tests/test_gis_vector_crs.py`: add Python vector CRS tests for happy path, missing CRS, invalid CRS, CRS mismatch, same CRS, z preservation, geometry collection, and malformed geometry.
- [ ] Run formatting, Rust tests, and Python tests listed below.

## 8. Validation commands

- `cargo fmt --check`
- `cargo test`
- `python -m pytest tests/test_gis_raster.py tests/test_gis_crs_affine.py`
- `python -m pytest tests/test_gis_vector_crs.py`
- `python -m pytest tests/test_api_contracts.py`

Python commands require the compiled `_forge3d` extension; otherwise the relevant GIS/API modules are skipped or unavailable according to existing test guards. If the repo's active CI matrix uses the lean non-viewer selector, also run the established broader selector after the focused GIS commands:

- `python -m pytest tests/ -q --tb=short -m "not viewer and not interactive_viewer and not offscreen"`

## 9. Non-goals

- No raster reprojection changes.
- No vector union, clip, intersection, buffer, or simplify.
- No rasterization or masking.
- No thematic classification.
- No GDAL vector driver expansion unless feature-gated and separately planned.
- No hidden network or remote reads.
- No geometry repair implementation unless an existing primitive is found and a separate plan expands C2.
- No use of GeoPandas, Shapely, rasterio, rioxarray, xarray, or terra as backend implementation.

## 10. Acceptance criteria

- [ ] `forge3d.gis.reproject_vector` exists in native registration, Python wrapper, `.pyi`, and `__all__`.
- [ ] It handles path, read-vector result dict, and raw FeatureCollection with explicit `src_crs`.
- [ ] It never guesses missing CRS.
- [ ] It detects source CRS conflicts.
- [ ] It preserves properties and z ordinates.
- [ ] It recomputes bounds and feature count.
- [ ] It returns source and destination CRS metadata.
- [ ] It has tests for happy path, missing CRS, invalid CRS, CRS mismatch, same CRS, z preservation, recursive geometry collection, malformed geometry, and wrapper/stub export consistency.
- [ ] It documents current backend limits honestly: same CRS and EPSG:4326 <-> EPSG:3857 only under the current built-in path, with other valid CRS pairs failing as `BackendUnavailable`.

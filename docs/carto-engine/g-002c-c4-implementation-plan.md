# G-002c C4 Vector Overlay Operations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add G-002c C4 vector overlay operations on top of the merged C1 vector read/metadata, C2 vector reprojection, and C3 geometry validity/measurement APIs.

**Architecture:** Backend GIS behavior stays in Rust under `src/gis/`. Use the current `src/gis/geometry/*` split for geometry parsing, validation, metadata, and topology helpers; keep vector-source orchestration in `src/gis/vector.rs`. Python wrappers in `python/forge3d/gis.py` remain thin `os.fspath` and argument-marshaling shims over PyO3.

**Tech Stack:** Rust, PyO3, `serde_json` GeoJSON-like dicts at the Python boundary, existing `VectorInfo`/`RasterWarning`/`GisError` contracts, existing C2 CRS helpers, and a new optional robust topology Cargo feature named `geos-topology` or an equivalent robust backend feature with the same public missing-feature token.

---

## Current-State Inventory

### Main Preflight

- Preflight target: fetched `origin/main` at `6eb197294e04a0ba0077f415df845bc7f8709005`.
- Working branch after fast-forward: `codex/g-002c-plan-docs`.
- The exact requested path `docs/carto-engine/g-002c-c4-implementation-plan.md` follows the current `docs/carto-engine/` convention used by:
  - `docs/carto-engine/g-002c-c2-reproject-vector-implementation-plan.md`
  - `docs/carto-engine/g-002c-c3-implementation-plan.md`
  - `docs/carto-engine/g-002c-implementation-plan.md`
- No `docs/carto-engine/index.md` or toctree convention exists, so this change-set should add only this file unless a reviewer asks for a docs index later.

Runtime preflight required a local extension rebuild because Python initially imported an older site-packages `_forge3d.pyd` missing C1-C3 symbols. After `python -m maturin develop --quiet`, this verification passed:

```text
python -m pytest tests/test_api_contracts.py tests/test_gis_vector_io.py tests/test_gis_vector_crs.py tests/test_gis_vector_geom.py -q
345 passed
```

### C1 APIs Present On Main

Python wrappers and stubs are present in `python/forge3d/gis.py` and `python/forge3d/gis.pyi`; PyO3 exports are registered through `src/py_module/functions/gis.rs`; contract tests are in `tests/test_api_contracts.py`.

| API | Current public signature | Backend/export files | Test files |
|---|---|---|---|
| `VectorInfo` | PyO3 class with `path`, `driver`, `layer_name`, `layer_count`, `geometry_type`, `feature_count`, `schema`, `crs_wkt`, `crs_authority`, `bounds`, `is_georeferenced`, `warnings`, `as_dict()` | `src/gis/vector.rs`, `src/py_module/classes.rs` | `tests/test_api_contracts.py`, `tests/test_gis_vector_io.py` |
| `read_vector` | `read_vector(path, *, layer=None, columns=None, bbox=None, limit=None)` | `python/forge3d/gis.py`, `python/forge3d/gis.pyi`, `src/gis/vector.rs`, `src/py_module/functions/gis.rs` | `tests/test_gis_vector_io.py` |
| `geometry_type` | `geometry_type(source, *, layer=None)` | same as above | `tests/test_gis_vector_io.py` |
| `vector_schema` | `vector_schema(source, *, layer=None)` | same as above | `tests/test_gis_vector_io.py` |
| `feature_count` | `feature_count(source, *, layer=None)` | same as above | `tests/test_gis_vector_io.py` |
| `vector_crs` | `vector_crs(source, *, layer=None)` | same as above | `tests/test_gis_vector_io.py` |
| `vector_bounds` | `vector_bounds(source, *, layer=None)` | same as above | `tests/test_gis_vector_io.py` |

C1 currently reads local GeoJSON-like vector sources. GDAL/OGR is not wired.

### C2 APIs Present On Main

| API | Current public signature | Backend/export files | Test files |
|---|---|---|---|
| `reproject_vector` | `reproject_vector(input, dst_crs, src_crs=None)` | `python/forge3d/gis.py`, `python/forge3d/gis.pyi`, `src/gis/vector.rs`, `src/gis/crs.rs`, `src/py_module/functions/gis.rs` | `tests/test_gis_vector_crs.py`, `tests/test_api_contracts.py` |

C2 uses current built-in CRS support. It does not add arbitrary CRS support beyond the existing CRS backend behavior.

### C3 APIs Present On Main

| API | Current public signature | Backend/export files | Test files |
|---|---|---|---|
| `validate_geometry` | `validate_geometry(geometry)` | `python/forge3d/gis.py`, `python/forge3d/gis.pyi`, `src/gis/geometry.rs`, `src/gis/geometry/validate.rs`, `src/gis/geometry/py.rs` | `tests/test_gis_vector_geom.py`, `tests/test_api_contracts.py` |
| `repair_geometry` | `repair_geometry(geometry, *, method="make_valid")` | `src/gis/geometry.rs`, `src/gis/geometry/py.rs` | `tests/test_gis_vector_geom.py`, `tests/test_api_contracts.py` |
| `geometry_measure` | `geometry_measure(geometry, *, metrics=("area", "length"))` | `src/gis/geometry.rs`, `src/gis/geometry/measure.rs`, `src/gis/geometry/math.rs`, `src/gis/geometry/py.rs` | `tests/test_gis_vector_geom.py`, `tests/test_api_contracts.py` |
| `geometry_centroid` | `geometry_centroid(geometry)` | `src/gis/geometry.rs`, `src/gis/geometry/centroid.rs`, `src/gis/geometry/py.rs` | `tests/test_gis_vector_geom.py`, `tests/test_api_contracts.py` |
| `representative_point` | `representative_point(geometry)` | `src/gis/geometry.rs`, `src/gis/geometry/line_ops.rs`, `src/gis/geometry/py.rs` | `tests/test_gis_vector_geom.py`, `tests/test_api_contracts.py` |
| `interpolate_line` | `interpolate_line(geometry, distance, *, normalized=False)` | `src/gis/geometry.rs`, `src/gis/geometry/line_ops.rs`, `src/gis/geometry/py.rs` | `tests/test_gis_vector_geom.py`, `tests/test_api_contracts.py` |

C3 already establishes `backend_unavailable: geos-topology feature required ...` for topology-required paths such as polygon `representative_point` and `repair_geometry(make_valid)`. C4 should reuse that public token and feature name.

### Files Containing Current Shared Behavior

| Concern | Files |
|---|---|
| Python wrappers | `python/forge3d/gis.py` |
| Python type stubs | `python/forge3d/gis.pyi` |
| PyO3 function registration | `src/py_module/functions/gis.rs` |
| PyO3 class registration | `src/py_module/classes.rs` |
| Vector read/metadata/CRS/bounds | `src/gis/vector.rs` |
| CRS parsing/comparison/transforms | `src/gis/crs.rs`, `src/gis/vector.rs` |
| Geometry public entrypoints | `src/gis/geometry.rs` |
| Geometry model and operation metadata | `src/gis/geometry/model.rs` |
| Geometry parsing/normalization | `src/gis/geometry/parse.rs` |
| Geometry validation | `src/gis/geometry/validate.rs` |
| Geometry measurement/centroid/line helpers | `src/gis/geometry/measure.rs`, `src/gis/geometry/centroid.rs`, `src/gis/geometry/line_ops.rs`, `src/gis/geometry/math.rs` |
| JSON/PyO3 conversion | `src/gis/py_json.rs` |
| Error mapping | `src/gis/error.rs` |
| Warning type | `src/gis/types.rs` |
| Contract tests | `tests/test_api_contracts.py` |
| C1-C3 behavior tests | `tests/test_gis_vector_io.py`, `tests/test_gis_vector_crs.py`, `tests/test_gis_vector_geom.py` |

### Reusable C3 Helpers For C4

- `src/gis/geometry/model.rs`
  - `Geometry`, `Coord`, and `NormalizedInput`.
  - Stable diagnostic constants: `empty_geometry`, `invalid_geometry`, `unsupported_geometry_type`, `unsupported_option`, `invalid_argument`, `backend_unavailable`.
  - `operation_value(...)`, which is the current `GeometryOperationInfo` JSON shape.
  - `warnings_value(...)`, `finite_value(...)`, `point_value(...)`, `empty_geometry_error()`, and `polygon_topology_error(...)`.
- `src/gis/geometry/parse.rs`
  - `normalize_input(source, allow_feature_collection)` and `raw_geometry_type(source)`.
  - CRS metadata extraction from read-vector style `info` dicts.
- `src/gis/geometry/validate.rs`
  - `validate_input_or_error(...)` and `validate_geometry_or_error(...)` for rejecting invalid geometries before overlay.
- `src/gis/vector.rs`
  - `VectorInfo`, `VectorReadOptions`, `VectorReadResult`, `read_vector(...)`.
  - Existing bounds order `(left, bottom, right, top)`.
  - Existing GeoJSON Feature/FeatureCollection normalization and schema inference.
  - Existing CRS extraction from `VectorInfo`, read results, and GeoJSON `crs` metadata.
- `src/gis/py_json.rs`
  - `py_to_json_strict(...)` and `json_to_py(...)` for thin PyO3 shims.

### Missing Helpers Needed Before C4 Implementation

C4 should add only the helpers below. Do not redesign C1-C3.

| Helper | File | Purpose |
|---|---|---|
| `geometry_to_value(&Geometry) -> GisResult<Value>` | `src/gis/geometry/model.rs` or `src/gis/geometry/serde.rs` | Serialize arbitrary C3 `Geometry` back to GeoJSON-like dicts; C3 currently only serializes points. |
| `geometry_type_for_geometries(&[Geometry]) -> String` | `src/gis/geometry/model.rs` | Compute stable output type strings including `Empty`, `Mixed`, `Polygon`, `MultiPolygon`, and `GeometryCollection`. |
| `operation_value_with_warnings(...)` reuse wrapper | `src/gis/geometry/model.rs` | Keep C4 operation metadata exactly aligned with C3 `operation_value(...)` instead of copying dict construction. |
| `topology_backend_available() -> bool` | `src/gis/geometry/topology.rs` | Report `cfg(feature = "geos-topology")` for tests and backend gating. |
| `require_topology_backend(operation: &str) -> GisResult<()>` | `src/gis/geometry/topology.rs` | Return `GisError::BackendUnavailable("backend_unavailable: geos-topology feature required for <operation>")` when disabled. |
| `topology::{union, buffer, intersection, simplify}` | `src/gis/geometry/topology.rs` | Isolate robust backend conversions and operations. |
| `resolve_vector_source(source, layer) -> VectorSourceData` | `src/gis/vector.rs` | Accept path, read result FeatureCollection, raw FeatureCollection, or Feature for C4 vector APIs. |
| `vector_info_for_features(...) -> VectorInfo` | `src/gis/vector.rs` | Recompute `VectorInfo` after clip/intersect/dissolve/load_boundary with stable schema, count, CRS, bounds, and warnings. |
| `extract_vector_crs(...) -> Option<CrsSpec>` | `src/gis/vector.rs` | Reuse C2 CRS extraction for C4 CRS compatibility checks. |
| `require_same_vector_crs(left, right, context) -> GisResult<Option<CrsSpec>>` | `src/gis/vector.rs` | Enforce no guessing and return `crs_mismatch` for incompatible CRS. |
| `merge_properties(left, right, suffixes)` | `src/gis/vector.rs` | Implement deterministic property collision behavior for `intersect_vectors`. |

## Backend Decision And Capability Gate

No robust topology backend is wired on current main. `Cargo.toml` has no `geo`, `geo-types`, `geojson`, `geos`, or `gdal` dependency. C4 must not implement polygon overlay with bounding boxes, ring clipping, or approximate hand-rolled topology.

Required C4 backend gate:

- Add an optional Cargo feature named `geos-topology`.
- Wire it to a robust topology backend. GEOS through a Rust crate is the expected implementation unless a reviewer chooses an equivalent robust Rust topology backend before C4. The public missing-feature token remains `geos-topology` either way.
- Public C4 functions must always be importable through Python and `_forge3d`.
- When the backend is disabled, backend-required calls must raise `GisError::BackendUnavailable` mapped to Python `RuntimeError` with both tokens:

```text
BackendUnavailable: backend_unavailable: geos-topology feature required for <operation>
```

- Do not add Python GIS libraries as backend dependencies.
- Do not claim GEOS-grade union, dissolve, buffer, clip, intersection, repair, representative point, or topology-preserving simplify unless the backend is wired, exported, built, and tested.
- Decide during C4.0 whether `pyproject.toml [tool.maturin].features` should include `geos-topology`. If the chosen backend needs system libraries unavailable in CI, do not include it in maturin by default; instead keep semantic tests feature-gated and keep backend-unavailable tests mandatory.

## Shared C4 Semantics

These rules apply to all seven C4 APIs.

- Python boundary accepts GeoJSON-like Geometry dicts, Feature dicts, FeatureCollection dicts, and local vector paths where the API names a vector source.
- Python boundary does not accept WKB, WKT, Shapely objects, GeoPandas objects, or new Rust-backed Python geometry objects.
- Python wrappers only call `_require_native().<name>(...)` after `os.fspath` conversion for path-like arguments.
- Invalid geometries are rejected with `invalid_geometry`; they are not silently repaired.
- Repair remains explicit through C3 `repair_geometry`.
- Missing CRS is never guessed.
- CRS assignment is not reprojection. `clip_crs` assigns CRS metadata to the clip geometry only for compatibility checking.
- Operations combining two CRS-bearing inputs require compatible CRS. If CRS is incompatible and the API has no explicit transform parameter, return `crs_mismatch`.
- Bounds order remains `(left, bottom, right, top)`.
- Empty inputs and empty outputs are distinct:
  - `empty_feature_set`: a vector source has no features before the operation.
  - `empty_geometry`: a geometry input is empty or all geometry members are empty.
  - `empty_input`: an explicit sequence argument, such as `union_geometries([])`, is empty.
  - `empty_output`: valid inputs produced no output geometry/features, such as no intersection.
- Geometry type changes must be visible in `operation.input_geometry_type`, `operation.output_geometry_type`, and an operation warning with code `geometry_type_changed`.

`GeometryOperationInfo` must keep the current C3 shape:

```json
{
  "name": "operation_name",
  "input_geometry_type": "Polygon",
  "output_geometry_type": "MultiPolygon",
  "input_count": 2,
  "output_count": 1,
  "changed": true,
  "crs": null,
  "warnings": []
}
```

For mixed inputs, use `input_geometry_type: "Mixed"`. Do not add new metadata fields in C4 unless a test proves the existing shape cannot represent the operation.

## C4 API Contract Table

| API | Function signature | Accepted input forms | Output schema | Required metadata fields | Required diagnostics/errors | Backend dependency requirements | Reference behavior to compare against | Minimum tests | Explicit unsupported cases |
|---|---|---|---|---|---|---|---|---|---|
| `union_geometries` | `union_geometries(geometries)` | Python sequence of Geometry dicts or Feature dicts; FeatureCollection dict as shorthand for its feature geometries | `{"geometry": Geometry | None, "operation": GeometryOperationInfo}` | `operation.name="union_geometries"`, `input_geometry_type`, `output_geometry_type`, `input_count`, `output_count`, `changed`, `crs`, `warnings` | `empty_input` warning for `[]`; `empty_geometry`; `invalid_geometry`; `unsupported_geometry_type`; `backend_unavailable`; `geometry_type_changed`; non-sequence raises `invalid_argument` | Requires `geos-topology` for non-empty inputs. Public function importable without it. | Shapely `unary_union` for optional area/type checks | overlapping polygons, disjoint polygons, empty list, invalid geometry, backend unavailable | Attribute dissolve, CRS transformation, WKB/WKT/object inputs, bbox-only union |
| `dissolve_vector` | `dissolve_vector(source, *, by=None)` | Vector path, read-vector result dict, FeatureCollection dict; `by=None`, string field name, or non-empty sequence of field names | FeatureCollection dict with `features`, `info`, `operation`, `warnings` | Operation fields above; `info.feature_count` equals output groups; `info.geometry_type` recomputed; output CRS copied from source | `missing_field`; `empty_feature_set`; `invalid_geometry`; `backend_unavailable`; `geometry_type_changed`; invalid `by` raises `invalid_argument` | Requires `geos-topology` for non-empty groups. Uses current GeoJSON vector reader; no GDAL. | GeoPandas `dissolve` optional geometry comparison only; no stats aggregation comparison | dissolve all, dissolve by one field, missing field, invalid geometry, property preservation rules | Stats aggregation, field expressions, vector writes, CRS reprojection, silent invalid-geometry repair |
| `buffer_geometry` | `buffer_geometry(geometry, distance, *, quad_segs=8)` | Geometry dict or Feature dict | `{"geometry": Polygon/MultiPolygon/GeometryCollection | None, "operation": GeometryOperationInfo}` | Operation fields above; `changed=true` unless backend returns identical valid geometry for zero distance; type-change warning when output type differs | `invalid_argument` for non-finite distance or `quad_segs < 1`; `empty_geometry`; `invalid_geometry`; `backend_unavailable`; `empty_output`; `geometry_type_changed` | Requires `geos-topology`. Public function importable without it. | Shapely `buffer(distance, quad_segs=...)` optional area/bounds checks | point buffer, line/polygon buffer if supported, zero/negative/nonfinite distance behavior, invalid geometry, backend unavailable | Geodesic buffers, unit conversion, cap/join style options, implicit repair by zero buffer |
| `clip_vector` | `clip_vector(source, clip_geometry, *, clip_crs=None)` | Source path/read result/FeatureCollection; clip Geometry, Feature, or FeatureCollection; optional CRS literal/dict/int for clip CRS assignment | FeatureCollection dict with clipped features, recomputed `info`, `operation`, `warnings` | Operation fields above; `info.crs_*` copied from source; `info.bounds` recomputed; source properties preserved | `missing_crs` when source or clip CRS is unavailable; `crs_mismatch`; `empty_feature_set`; `empty_output`; `invalid_geometry`; `backend_unavailable`; `geometry_type_changed` | Requires `geos-topology` for non-empty source and clip. Uses C1 read path only. | GeoPandas `clip` optional feature count/area/bounds checks | polygon clipped by polygon/box, no intersections, CRS mismatch, invalid clip geometry, empty feature set | Raster clipping, implicit clip reprojection, bbox-only clipping, `clip_crs` as transform, writes |
| `intersect_vectors` | `intersect_vectors(left, right, *, suffixes=("_left", "_right"))` | Each side path/read result/FeatureCollection | FeatureCollection dict with intersection geometries, joined properties, recomputed `info`, `operation`, `warnings` | Operation fields above; output properties follow deterministic suffix rules; output CRS copied from left after CRS match | `missing_crs`; `crs_mismatch`; `empty_feature_set`; `empty_output`; `invalid_geometry`; `backend_unavailable`; `property_collision`; invalid suffixes raise `invalid_argument`; `geometry_type_changed` | Requires `geos-topology` for non-empty inputs. Spatial index optimization is optional, not a contract. | GeoPandas `overlay(..., how="intersection")` optional geometry/property comparison | one overlap, no overlap, property collision with suffixes, CRS mismatch, invalid geometry | Union/difference/symmetric difference, nearest joins, many-to-many aggregation, implicit reprojection |
| `simplify_geometry` | `simplify_geometry(geometry, tolerance, *, preserve_topology=True)` | Geometry dict or Feature dict | `{"geometry": Geometry | None, "operation": GeometryOperationInfo}` | Operation fields above; `changed` from structural/coordinate change; type-change warning when output type differs | `invalid_argument` for non-finite or negative tolerance; `empty_geometry`; `invalid_geometry`; `backend_unavailable` when backend path required; `geometry_type_changed` | `preserve_topology=True` requires `geos-topology`. Initial C4 may also require `geos-topology` for `False`; if a small non-topology line-only simplifier is added, document its limited supported types in tests. | Shapely `simplify(tolerance, preserve_topology=...)` optional coordinate/type checks | line simplification, polygon preserve-topology behavior, negative tolerance, backend unavailable when topology preservation needs robust backend | Scale-dependent cartographic simplification, geodesic simplification, smoothing, new CRS units |
| `load_boundary` | `load_boundary(path, *, layer=None, where=None)` | Local vector path; optional layer string; `where=None` only in first C4 | `{"geometry": Geometry | None, "features": FeatureCollection, "info": VectorInfo dict, "operation": GeometryOperationInfo, "warnings": [...]}` | Operation fields above; `info` from read source; output geometry is a single unioned boundary when backend is available and input non-empty | `unsupported_option` for `where` until filtering is implemented; `missing_layer`; `empty_feature_set`; `missing_crs` warning copied from source; `invalid_geometry`; `backend_unavailable` if multiple/non-empty boundary union needs topology | Uses C1 local GeoJSON reader. Requires `geos-topology` to merge multiple boundary features; one valid polygon feature may return directly without overlay. | `read_vector` for metadata/count; optional Shapely unary union for multi-feature boundary geometry | read boundary path, layer handling, unsupported where behavior, missing layer, empty result | Remote boundary fetching, OSM queries, named administrative datasets, SQL/filter language, vector writes, reprojection |

## Implementation Slices

The recommended order from the prompt is correct for the current code. The only adjustment is that C4.0 must create a small `src/gis/geometry/topology.rs` split because the current C3 layout already keeps geometry internals out of `vector.rs`.

### C4.0: Backend Gate And Shared Metadata Plumbing

**Files:**
- Modify: `Cargo.toml`
- Modify: `pyproject.toml` only after the backend-default decision is made
- Modify: `src/gis/geometry.rs`
- Create: `src/gis/geometry/topology.rs`
- Modify: `src/gis/geometry/model.rs`
- Modify: `src/gis/vector.rs`
- Modify: `src/gis/geometry/py.rs`
- Modify: `python/forge3d/gis.py`
- Modify: `python/forge3d/gis.pyi`
- Modify: `src/py_module/functions/gis.rs`
- Modify: `tests/test_api_contracts.py`
- Create: `tests/test_gis_vector_overlay.py`

- [ ] Add optional Cargo feature `geos-topology` and the chosen robust topology dependency.
- [ ] Add `topology_backend_available()` and `require_topology_backend(operation)` in `src/gis/geometry/topology.rs`.
- [ ] Add serialization helper `geometry_to_value(&Geometry) -> GisResult<Value>`.
- [ ] Add or reuse a single metadata helper that calls current `operation_value(...)`.
- [ ] Add PyO3 functions and Python wrappers for all seven C4 public APIs, even before backend-enabled behavior is complete. Disabled backend paths must raise `backend_unavailable`.
- [ ] Update `tests/test_api_contracts.py` so the C4 APIs are expected and C5/C6 APIs are explicitly absent.
- [ ] Run contract tests and backend-unavailable tests before implementing semantic overlay operations.

### C4.1: `union_geometries`

**Files:**
- Modify: `src/gis/geometry.rs`
- Modify: `src/gis/geometry/topology.rs`
- Modify: `src/gis/geometry/py.rs`
- Modify: `python/forge3d/gis.py`
- Modify: `python/forge3d/gis.pyi`
- Test: `tests/test_gis_vector_overlay.py`

- [ ] Write tests for overlapping polygons, disjoint polygons, empty list, invalid bowtie polygon, and backend unavailable.
- [ ] Implement input normalization from sequence/FeatureCollection to `Vec<Geometry>`.
- [ ] Validate every non-empty geometry with C3 validation before calling the topology backend.
- [ ] Call topology union only through `geometry/topology.rs`.
- [ ] Return C3-shaped operation metadata and `geometry_type_changed` warning when output type differs.

### C4.2: `buffer_geometry`

**Files:**
- Modify: `src/gis/geometry.rs`
- Modify: `src/gis/geometry/topology.rs`
- Modify: `src/gis/geometry/py.rs`
- Modify: `python/forge3d/gis.py`
- Modify: `python/forge3d/gis.pyi`
- Test: `tests/test_gis_vector_overlay.py`

- [ ] Write tests for point buffer, line or polygon buffer when backend is enabled, zero distance, negative distance, non-finite distance, invalid geometry, and backend unavailable.
- [ ] Validate `distance.is_finite()` and `quad_segs >= 1`.
- [ ] Reject invalid geometry before backend call.
- [ ] Preserve GEOS empty-output semantics with `empty_output` metadata rather than silently dropping the result.

### C4.3: `clip_vector`

**Files:**
- Modify: `src/gis/vector.rs`
- Modify: `src/gis/geometry/topology.rs`
- Modify: `python/forge3d/gis.py`
- Modify: `python/forge3d/gis.pyi`
- Test: `tests/test_gis_vector_overlay.py`

- [ ] Write tests for polygon clipped by polygon/box, no intersections, CRS mismatch, invalid clip geometry, and empty feature set.
- [ ] Add `resolve_vector_source(...)` for path/read-result/FeatureCollection inputs.
- [ ] Extract source CRS and clip CRS. `clip_crs` assigns metadata only; it never reprojects coordinates.
- [ ] Require CRS compatibility before geometry overlay.
- [ ] Preserve source feature properties and recompute vector metadata for output features.

### C4.4: `intersect_vectors`

**Files:**
- Modify: `src/gis/vector.rs`
- Modify: `src/gis/geometry/topology.rs`
- Modify: `python/forge3d/gis.py`
- Modify: `python/forge3d/gis.pyi`
- Test: `tests/test_gis_vector_overlay.py`

- [ ] Write tests for one overlap, no overlap, property collision with suffixes, CRS mismatch, invalid geometry, and backend unavailable.
- [ ] Validate `suffixes` as exactly two strings.
- [ ] Require compatible non-missing CRS for both sources.
- [ ] For each left/right feature pair, compute intersection and keep only non-empty outputs.
- [ ] Merge properties deterministically: unique names unchanged; colliding names become `<name><left_suffix>` and `<name><right_suffix>`; if those generated keys collide, raise `property_collision`.

### C4.5: `simplify_geometry`

**Files:**
- Modify: `src/gis/geometry.rs`
- Modify: `src/gis/geometry/topology.rs`
- Modify: `src/gis/geometry/py.rs`
- Modify: `python/forge3d/gis.py`
- Modify: `python/forge3d/gis.pyi`
- Test: `tests/test_gis_vector_overlay.py`

- [ ] Write tests for line simplification, polygon preserve-topology behavior, negative tolerance, and backend unavailable for topology-preserving simplify.
- [ ] Validate `tolerance.is_finite()` and `tolerance >= 0.0`.
- [ ] Route `preserve_topology=True` through the robust topology backend.
- [ ] If `preserve_topology=False` uses a limited non-topology fallback, restrict it to line geometries and test unsupported polygon behavior. Otherwise route both modes through the backend.

### C4.6: `dissolve_vector`

**Files:**
- Modify: `src/gis/vector.rs`
- Modify: `src/gis/geometry/topology.rs`
- Modify: `python/forge3d/gis.py`
- Modify: `python/forge3d/gis.pyi`
- Test: `tests/test_gis_vector_overlay.py`

- [ ] Write tests for dissolve all, dissolve by one field, missing field, invalid geometry, property preservation, empty source, and backend unavailable.
- [ ] Accept `by=None`, a string, or a non-empty sequence of strings.
- [ ] Group features by the exact property values for `by`; preserve only grouping fields in output properties.
- [ ] For `by=None`, output a single feature with `{}` properties.
- [ ] Union each group's geometries through the topology backend and recompute `VectorInfo`.

### C4.7: `load_boundary`

**Files:**
- Modify: `src/gis/vector.rs`
- Modify: `src/gis/geometry/topology.rs`
- Modify: `python/forge3d/gis.py`
- Modify: `python/forge3d/gis.pyi`
- Test: `tests/test_gis_vector_overlay.py`

- [ ] Write tests for reading a boundary path, layer handling, `where` unsupported behavior, missing layer, empty result, invalid geometry, and backend unavailable for multi-feature union.
- [ ] Reject `where is not None` with `unsupported_option: load_boundary where filtering is not implemented`.
- [ ] Use existing C1 `read_vector` semantics for path/layer.
- [ ] Return one boundary geometry: direct single geometry for one valid feature, topology union for multiple features.
- [ ] Preserve the read FeatureCollection under `features` so callers can inspect original attributes.

## Test Plan

### Mandatory Core Tests Without External Python GIS Dependencies

Add `tests/test_gis_vector_overlay.py`.

Required test groups:

- `union_geometries`
  - overlapping polygons produce one polygonal result when backend is enabled.
  - disjoint polygons produce a multi-part or collection result with type-change metadata.
  - empty list returns `geometry is None`, `input_count == 0`, `output_count == 0`, and `empty_input`.
  - invalid geometry raises `invalid_geometry`.
  - backend disabled raises `RuntimeError` containing `backend_unavailable` and `geos-topology`.
- `buffer_geometry`
  - point buffer returns polygonal geometry when backend is enabled.
  - line or polygon buffer behavior is covered when backend supports it.
  - zero distance is deterministic and documented by tests.
  - negative distance follows backend semantics and records `empty_output` when applicable.
  - non-finite distance and bad `quad_segs` raise `invalid_argument`.
  - invalid geometry raises `invalid_geometry`.
  - backend disabled raises `backend_unavailable`.
- `clip_vector`
  - polygon source clipped by polygon or box.
  - no intersections returns empty FeatureCollection with `empty_output`.
  - CRS mismatch raises `crs_mismatch`.
  - invalid clip geometry raises `invalid_geometry`.
  - empty source returns empty FeatureCollection with `empty_feature_set`.
- `intersect_vectors`
  - one overlap returns joined properties and intersection geometry.
  - no overlap returns empty FeatureCollection with `empty_output`.
  - property collisions use suffixes.
  - generated property-name collision raises `property_collision`.
  - CRS mismatch raises `crs_mismatch`.
  - invalid geometry raises `invalid_geometry`.
- `simplify_geometry`
  - line simplification reduces coordinates or preserves endpoints according to backend behavior.
  - polygon `preserve_topology=True` stays valid when backend is enabled.
  - negative tolerance raises `invalid_argument`.
  - topology-preserving simplify without backend raises `backend_unavailable`.
- `dissolve_vector`
  - dissolve all returns one feature with `{}` properties.
  - dissolve by one field returns one feature per group and preserves only that field.
  - missing field raises `missing_field`.
  - invalid geometry raises `invalid_geometry`.
  - property aggregation beyond grouping is absent by contract.
- `load_boundary`
  - reads a boundary path through current C1 GeoJSON reader.
  - layer handling matches C1 read behavior.
  - `where` raises `unsupported_option` if not implemented.
  - missing layer raises `missing_layer`.
  - empty result carries `empty_feature_set`.
- Contract tests
  - Add C4 API names to `EXPECTED_FUNCTIONS`.
  - Assert C5/C6 names are absent until their phases: `rasterize_vectors`, `geometry_mask`, `mask_raster`, `normalize_raster`, `classify_raster`.
  - Keep C1-C3 API presence tests unchanged.
- Regression tests
  - Existing C1-C3 tests continue passing:
    - `tests/test_gis_vector_io.py`
    - `tests/test_gis_vector_crs.py`
    - `tests/test_gis_vector_geom.py`
    - `tests/test_api_contracts.py`

### Backend-Unavailable Test Strategy

At least one backend-unavailable path must run in the normal CI mode. Choose the smallest reliable method during C4.0:

- If `geos-topology` is not in `pyproject.toml` maturin features, normal Python tests can assert unavailable behavior directly.
- If `geos-topology` is in maturin features, add a Rust unit test compiled without the feature or a Python no-feature build job. The assertion must check both `backend_unavailable` and `geos-topology`.

### Optional Reference Comparisons

Optional tests may import Shapely or GeoPandas only inside tests and skip cleanly when unavailable.

Optional Shapely checks:

- `union_geometries`: compare area and geometry type to `shapely.ops.unary_union`.
- `buffer_geometry`: compare bounds/area within tolerance for point and polygon buffers.
- `simplify_geometry`: compare coordinate count/type for both `preserve_topology` modes.
- `load_boundary`: compare multi-feature boundary union geometry.

Optional GeoPandas checks:

- `clip_vector`: compare output feature count and total area to `geopandas.clip`.
- `intersect_vectors`: compare output feature count/properties/area to `geopandas.overlay(..., how="intersection")`.
- `dissolve_vector`: compare group count and total area to `GeoDataFrame.dissolve`.

Mandatory tests must not require rasterio, geopandas, shapely, rioxarray, xarray, or terra.

## Phase Hygiene

C4 must not plan or implement:

- `rasterize_vectors`
- `geometry_mask`
- `mask_raster`
- `normalize_raster`
- `classify_raster`
- raster clipping hidden inside vector APIs
- vector writes
- OSM querying
- Terrarium
- slippy tiles
- remote fetching
- cache management
- DEM, landcover, population, or building domain helpers
- MapScene recipe-family work
- gallery goldens
- examples, shaders, rendering behavior, or visual goldens

## Backward Compatibility

C4 must preserve all existing public APIs and tests from earlier phases.

Preserve G-002a1 raster read/write:

- `RasterInfo`
- `read_raster_info`
- `read_raster`
- `write_raster`

Preserve G-002b CRS, affine, nodata, mask, alignment, reprojection, and window APIs:

- `parse_crs`
- `inspect_crs`
- `raster_crs`
- `assign_crs`
- `create_crs_transformer`
- `transform_bounds`
- `web_mercator_bounds`
- `AffineTransform`
- `raster_transform`
- `transform_from_origin`
- `transform_from_bounds`
- `array_bounds`
- `raster_bounds`
- `raster_resolution`
- `validate_transform`
- `pixel_convention`
- `rowcol`
- `xy`
- `index`
- `apply_nodata`
- `read_raster_mask`
- `resample_raster`
- `assert_grid_compatible`
- `align_raster_grid`
- `align_raster_to`
- `reproject_raster`
- `calculate_default_transform`
- `window_from_bounds`
- `read_raster_window`
- `window_transform`
- `bounds`

Preserve G-002c C1 vector metadata/read:

- `VectorInfo`
- `read_vector`
- `geometry_type`
- `vector_schema`
- `feature_count`
- `vector_crs`
- `vector_bounds`

Preserve G-002c C2:

- `reproject_vector`

Preserve G-002c C3:

- `validate_geometry`
- `repair_geometry`
- `geometry_measure`
- `geometry_centroid`
- `representative_point`
- `interpolate_line`

## Validation Plan For This Planning Change-Set

Because this task is documentation/planning only, run:

```text
git status --short
git diff --name-status
git diff --stat
git diff
git ls-files --others --exclude-standard
```

Docs build:

- Run `python -c "import sphinx"` first.
- If Sphinx is installed, run `python -m sphinx -b html docs docs/_build/html`.
- If Sphinx is unavailable, record that docs build was skipped because the dependency is missing.

No Rust, PyO3, Python wrapper, stub, or behavior tests are required for this planning-only change after the preflight. The preflight command already verifies C1-C3 on the rebuilt extension:

```text
python -m pytest tests/test_api_contracts.py tests/test_gis_vector_io.py tests/test_gis_vector_crs.py tests/test_gis_vector_geom.py -q
```

## Review Bundle Rule

After the planning change-set, create a fresh temporary review bundle outside the repo. The bundle is evidence only and must never be staged, committed, or included in the PR.

Required bundle contents:

- `git status --short`
- `git diff --name-status`
- `git diff --stat`
- full `git diff`
- `git ls-files --others --exclude-standard`
- validation logs
- command metadata and exit codes
- branch
- merge base
- current commit
- timestamp
- cwd

Suggested path pattern:

```text
%TEMP%/forge3d-g002c-c4-review-bundle-<timestamp>/
```

## Implementation Readiness

C4 is ready for implementation after the C4.0 backend decision is made and documented in the implementation PR.

Current blocker for real overlay semantics:

- No robust topology backend is currently wired in `Cargo.toml` or `pyproject.toml`.

Required backend decision:

- Use `geos-topology` as the public Cargo feature and missing-feature token.
- Select either GEOS through Rust bindings or an equivalent robust Rust topology backend before implementing C4.1-C4.7.
- Decide whether the selected backend is included in maturin's default Python extension features or tested in a separate feature-enabled job.

Next implementation prompt title:

```text
Implement G-002c C4.0 only: topology backend gate and vector overlay metadata plumbing
```

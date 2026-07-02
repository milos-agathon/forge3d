# G-002 Later Domain And Remote Helpers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Plan the Rust-first implementation of every API in `Contract Matrix: Later Domain And Remote Helpers`.

**Architecture:** GIS behavior stays in Rust under `src/gis/`; Python wrappers in `python/forge3d/gis.py` stay thin path normalization and argument marshaling shims over PyO3. Later helpers compose the shipped raster, CRS, affine, vector, rasterization, and thematic primitives instead of duplicating them.

**Tech Stack:** Rust, PyO3, NumPy at the Python boundary, existing `tiff`/`image`/`sha2`/`serde_json`/`ndarray`/COG code, optional existing `reqwest`/`tokio` for explicit remote fetch, optional existing `proj` or `geos-topology` only where already appropriate, and optional Python GIS libraries in reference tests only.

---

## Prerequisite Verification

Result: PASS. The local checkout is on `main` at `8fa189322b48c1557c60fe7e252dcb867a5da5f0`, matching `origin/main` after `git fetch origin main --prune`. GitHub PR verification found #93, #94, #95, #96, and #97 merged to `main`; C4 is present in local `main` history as commits `34513b0`, `47e1602`, `f2ebeac`, `47d6e7c`, `7a7d687`, `7f87efe`, `04b1f0e`, and `8166616`.

File-level evidence:

- G-002a1 raster foundation exists in `src/gis/raster_info.rs`, `src/gis/raster_write.rs`, `src/gis/types.rs`, `src/gis/mod.rs`, `python/forge3d/gis.py`, `python/forge3d/gis.pyi`, `tests/test_gis_raster.py`, and `tests/test_gis_read_raster.py`.
- G-002b CRS, affine, nodata, windowing, alignment, and reprojection exist in `src/gis/crs.rs`, `src/gis/affine.rs`, `src/gis/warp.rs`, `src/gis/mod.rs`, `python/forge3d/gis.py`, `python/forge3d/gis.pyi`, `tests/test_gis_crs_affine.py`, `tests/test_gis_alignment_windowing.py`, and `tests/test_gis_resample_reproject.py`.
- G-002c C1-C4 vector IO, CRS, geometry, and overlay exist in `src/gis/vector.rs`, `src/gis/geometry.rs`, `src/gis/geometry/`, `python/forge3d/gis.py`, `python/forge3d/gis.pyi`, `tests/test_gis_vector_io.py`, `tests/test_gis_vector_crs.py`, `tests/test_gis_vector_geom.py`, and `tests/test_gis_vector_overlay.py`.
- G-002c C5 rasterization and masks exist in `src/gis/rasterize.rs`, wrapper/stub exports, and `tests/test_gis_rasterize_mask.py`.
- G-002c C6 thematic raster exists in `src/gis/thematic.rs`, wrapper/stub exports, and `tests/test_gis_thematic.py`.
- PyO3 registration exists in `src/py_module/functions/gis.rs`; GIS classes are registered in `src/py_module/classes.rs` as `RasterInfo`, `VectorInfo`, `AffineTransform`, and `CrsTransform`.
- Contract locks exist in `tests/test_api_contracts.py`; `LATER_GIS_FUNCTIONS = []`, so none of the Later APIs are currently registered.
- `Cargo.toml` already has `sha2`, `image`, `tiff`, `serde_json`, optional `reqwest`, optional `tokio`, optional `proj`, and optional `geo`; do not add a second HTTP, hash, image, GeoTIFF, CRS, or topology stack.

## Scope

Plan exactly these APIs:

- `fetch_remote_geodata(url, cache=None, timeout=None, checksum=None) -> dict[str, Any]`
- `cache_geodata(key_or_url, cache_dir, refresh=False) -> dict[str, Any]`
- `fetch_vector(url, cache=None) -> dict[str, Any]`
- `read_cog(path_or_url, window=None, overview=None) -> dict[str, Any]`
- `slippy_tile_index(bounds, zoom, crs="EPSG:4326") -> dict[str, Any]`
- `query_osm_features(aoi, tags, cache=None) -> dict[str, Any]`
- `parse_osm_features(osm_json, tags=None) -> dict[str, Any]`
- `load_context_vectors(path_or_features, layers=None) -> dict[str, Any]`
- `prepare_osm_scene(aoi, tags=None, cache=None) -> dict[str, Any]`
- `prepare_dem(source, target_info=None, nodata=None) -> dict[str, Any]`
- `prepare_terrain_derivatives(dem, derivatives=("slope", "hillshade")) -> dict[str, Any]`
- `read_gridded_dataset(path, variable=None) -> dict[str, Any]`
- `subset_grid(source, bounds_or_coords, variable=None) -> dict[str, Any]`
- `decode_terrarium_dem(rgb_array_or_path) -> dict[str, Any]`
- `build_terrarium_dem(bounds, zoom, cache=None) -> dict[str, Any]`
- `prepare_landcover_raster(source, target_info, classes=None) -> dict[str, Any]`
- `prepare_population_raster(source, target_info=None, normalization=None) -> dict[str, Any]`
- `load_building_footprints(path_or_features, dst_crs=None) -> dict[str, Any]`
- `extract_building_heights(features, defaults=None) -> dict[str, Any]`
- `estimate_local_utm(bounds_or_geometry) -> dict[str, Any]`

## Non-Goals

- No routing, travel-time, network analysis, or graph cost modeling.
- No vector writes.
- No rendering, shaders, visual goldens, gallery goldens, examples, recipe manifests, UI, viewer, or MapScene recipe-family work.
- No redesign of existing G-002a1, G-002b, or G-002c C1-C6 APIs beyond reuse.
- No Python GIS backend behavior using `rasterio`, `geopandas`, `shapely`, `rioxarray`, `xarray`, or `terra`.
- No GDAL, NetCDF, HDF5, or other heavy runtime dependency in the first pass.

## Shared Contracts

`RemoteDatasetInfo` is a dict, not a new PyO3 class:

- `url`
- `cache_path`
- `status`
- `content_type`
- `byte_size`
- `checksum`
- `etag`
- `last_modified`
- `from_cache`
- `warnings`

Domain operations return a small metadata dict under `operation`:

- `name`
- `source_kind`
- `input_count`
- `output_count`
- `input_crs`
- `output_crs`
- `target_grid`
- `changed`
- `warnings`

Result shape rules:

- Reuse existing `RasterInfo`, `VectorInfo`, `CrsInfo`-compatible dicts, `RasterWarning`, and geometry/thematic result shapes.
- Keep public functions importable without optional backends. Backend-required behavior raises `GisError::BackendUnavailable` mapped to Python `RuntimeError` with both `backend_unavailable` and the required feature/backend name in the message.
- Prefer warnings as `{"code": "...", "message": "...", "field": ...}` to match existing raster/vector warnings.
- No large one-off result classes.

Stable diagnostic tokens to use when relevant:

`network_timeout`, `cache_miss`, `cache_stale`, `checksum_mismatch`, `malformed_payload`, `metadata_unavailable`, `missing_crs`, `invalid_crs`, `crs_mismatch`, `grid_mismatch`, `missing_transform`, `invalid_transform`, `categorical_resampling_warning`, `unsupported_dtype`, `unsupported_driver`, `unsupported_option`, `unsupported_geometry_type`, `missing_layer`, `empty_feature_set`, `empty_raster`, `invalid_argument`, `invalid_bounds`, `shape_mismatch`, `backend_unavailable`, `unsupported_scheme`, `rate_limited`, `incomplete_way`, `unsupported_relation`, `ambiguous_variable`, `unsupported_layout`, `ambiguous_axes`, `missing_tile`, `partial_mosaic`, `unknown_class`, `invalid_height`, `height_fallback`.

## Dependency Decisions

- Add no new crates for L1-L7 unless a phase proves an existing crate cannot cover a required behavior.
- Add a Cargo feature alias `gis-remote = ["dep:reqwest", "dep:tokio"]` in L1 if remote fetch code needs a clear backend name; this reuses existing optional crates and is not a second HTTP stack.
- Keep `cog_streaming` for COG range-read behavior and reuse `src/terrain/cog/` only behind that feature. Local `read_cog` must work without `cog_streaming`.
- Use existing `sha2` for checksums, `image` for Terrarium PNG decode, `tiff` and existing raster readers for local TIFF/COG, `serde_json` for GeoJSON/OSM/CityJSON payloads, `ndarray`/NumPy conversion patterns already used by GIS wrappers, and existing CRS/vector/raster/thematic helpers.
- Do not add GDAL for GPKG, NetCDF/HDF5 for gridded datasets, or a new OSM parser. Recognize those formats and return `backend_unavailable` or `unsupported_layout` where required.
- Optional Python GIS libraries may be imported only inside skipped reference tests or documentation examples.

## File Deltas

| File | Delta |
|---|---|
| `Cargo.toml` | Add feature alias only if needed; no new dependency stack. Add feature checks to validation notes. |
| `pyproject.toml` | Add `gis-remote` to maturin features only when L1 intentionally ships remote fetch in wheels; do not add Python GIS runtime deps. |
| `src/gis/mod.rs` | Declare/export `remote`, `tiles`, `osm`, `terrarium`, and `domain`; re-export PyO3 functions. |
| `src/gis/error.rs` | Prefer existing variants. Add no variant unless a required stable token cannot be expressed through current variants. |
| `src/gis/remote.rs` | New remote fetch/cache primitives plus `fetch_vector` orchestration. |
| `src/gis/tiles.rs` | New slippy tile math and bounds helpers. |
| `src/gis/osm.rs` | New Overpass query wrapper and OSM JSON parser. |
| `src/gis/terrarium.rs` | New Terrarium decode and mosaic helper. |
| `src/gis/domain.rs` | New domain helpers that compose existing GIS primitives. |
| `src/gis/raster_info.rs` | Add local `read_cog` helper only if it is smaller than a new COG module; otherwise call from `domain`/`remote`. |
| `src/terrain/cog/*` | Reuse for remote COG range reads only if the existing API can be called without widening terrain behavior. |
| `src/py_module/functions/gis.rs` | Register Later PyO3 functions phase by phase. |
| `python/forge3d/gis.py` | Add thin wrappers and `__all__`; only `os.fspath`, path normalization, and keyword marshaling. |
| `python/forge3d/gis.pyi` | Add exact stubs for each phase. |
| `tests/test_api_contracts.py` | Move each phase's APIs into `EXPECTED_FUNCTIONS`; keep not-yet phases in negative contract coverage. |
| `tests/test_gis_remote.py` | New L1/L3 remote/cache tests. |
| `tests/test_gis_cog_tiles.py` | New L2 COG/tile tests. |
| `tests/test_gis_osm.py` | New L3 OSM tests. |
| `tests/test_gis_domain.py` | New L4-L7 domain tests. |

## Phase Workflow

Every phase follows this order:

- [ ] Planning: re-read current touched files and verify no newer main changes alter the contracts.
- [ ] Tests: add the smallest failing contract/wrapper/behavior tests for that phase.
- [ ] Implementation: write the minimal Rust backend and thin Python wrappers needed to pass those tests.
- [ ] Contract: update `tests/test_api_contracts.py`, `python/forge3d/gis.py` `__all__`, and `python/forge3d/gis.pyi` only for APIs intentionally registered in that phase.
- [ ] Validation: run the phase-specific tests plus impacted existing GIS primitive tests.
- [ ] Review bundle: generate a temporary bundle with status, diffs, logs, metadata, branch, merge base, commit, timestamp, and cwd; do not stage it.

## L1: Remote And Cache Primitives

APIs: `fetch_remote_geodata`, `cache_geodata`.

Behavior:

- Support `http` and `https` only for remote fetch. Other URL schemes raise `unsupported_scheme`.
- `cache=None` means no persistent cache for `fetch_remote_geodata`.
- `cache` may be a path or dict with `cache_dir`; cache writes use a same-directory temp file, checksum validation, `fsync` where practical, then atomic rename.
- Cache keys are `sha256(url)` plus safe extension inferred from URL path or content type.
- `cache_geodata(key_or_url, cache_dir, refresh=False)` checks existing cache for plain keys. For URL input, it may fetch on miss or refresh because the API name and URL input make remote caching explicit.
- Stale cache fallback is allowed only when a prior complete cache file exists; return `status="stale"`, `from_cache=true`, and `cache_stale`.
- Checksum supports `sha256:<hex>` and bare 64-hex SHA-256. Mismatch removes the temp file and raises `checksum_mismatch`.
- Timeouts must map to `network_timeout`. HTTP 429 maps to `rate_limited`; other non-2xx statuses map to `malformed_payload` or `network_error` text with stable tokens where applicable.
- Content-type/driver allowlist: GeoTIFF/TIFF, GeoJSON/JSON, PNG for Terrarium, and octet-stream when extension is recognized. Unknown types raise `unsupported_driver`.

Implementation steps:

- [ ] Create `src/gis/remote.rs` with URL validation, cache-key, atomic-write, checksum, and `RemoteDatasetInfo` dict builders.
- [ ] Add a tiny transport seam in Rust for tests; default transport uses existing `reqwest` under `gis-remote`.
- [ ] Add disabled-backend paths returning `BackendUnavailable("backend_unavailable: gis-remote feature required for remote fetch")`.
- [ ] Register PyO3 functions and Python wrappers for only L1 APIs.
- [ ] Update contract tests for L1 only.

Minimum tests:

- Mocked URL fetch, cache hit, cache miss, stale-cache fallback, timeout, unsupported scheme, unsupported content type, checksum mismatch, malformed payload, atomic cache write, refresh behavior, and no-live-network guard.

## L2: COG And Tile Math

APIs: `read_cog`, `slippy_tile_index`.

Behavior:

- `read_cog` local path uses existing TIFF/raster readers and `read_raster_window`; no network feature is needed.
- Local `read_cog` returns `array`, `info`, `window`, `overview`, `is_cog_like`, `tile_info`, and `warnings`.
- `overview=None` reads base resolution; integer overview selects available overview or raises `invalid_argument`/`metadata_unavailable`.
- Remote `read_cog` for `http`/`https` routes through explicit range-read/cache backend. If unavailable, raise `backend_unavailable` with `cog_streaming`.
- `slippy_tile_index` accepts bounds in `crs`, transforms to EPSG:4326/EPSG:3857 with existing CRS helpers, and returns sorted tile dicts with keys `z`, `x`, `y`, `bounds_wgs84`, and `bounds_web_mercator`.
- Zoom must be an integer `0..24`; invalid zoom raises `invalid_argument`.
- Latitude outside `[-90, 90]` raises `invalid_bounds`; latitude outside Web Mercator valid range is clamped to `+/-85.05112878` with an `invalid_bounds` warning.
- Antimeridian-crossing EPSG:4326 bounds (`left > right`) are split into two longitude intervals and reported with `antimeridian_split=true`.
- X wraps modulo `2^z`; Y is clamped to `[0, 2^z - 1]`; output order is `z`, then `x`, then `y`.

Implementation steps:

- [ ] Add `src/gis/tiles.rs` for tile math; reuse `web_mercator_bounds`.
- [ ] Add `read_cog` local helper in `src/gis/raster_info.rs` or call existing helpers from a small GIS COG function.
- [ ] Reuse `src/terrain/cog::RangeReader` for remote path only if feature-gated reuse stays small.
- [ ] Register PyO3 functions and Python wrappers for L2 only.
- [ ] Update contract tests for L2 only.

Minimum tests:

- Local COG read, remote COG backend-unavailable or mocked range-read path, window read, overview selection, invalid window/bounds, slippy tile index at known zooms, latitude validation/clamp, zoom validation, antimeridian handling, and Web Mercator bounds metadata.

## L3: Vector Fetch And OSM Payloads

APIs: `fetch_vector`, `query_osm_features`, `parse_osm_features`.

Behavior:

- `fetch_vector` composes `fetch_remote_geodata` and existing `read_vector` for GeoJSON. Remote GPKG/WFS are recognized but gated: GPKG returns `backend_unavailable: gdal-vector feature required`; WFS or unknown drivers return `unsupported_driver`.
- `query_osm_features` builds a small Overpass JSON request from AOI and tag filters, uses mocked HTTP tests only, and returns raw OSM JSON plus cache/remote metadata.
- `parse_osm_features` supports nodes as Points, open ways as LineStrings, closed ways as Polygons, skipped incomplete ways with `incomplete_way`, unsupported relations with `unsupported_relation`, tag filtering, skipped counts, EPSG:4326 CRS metadata, and bounds.
- Malformed JSON or missing `elements` raises `malformed_payload`.
- Empty parsed output returns an empty FeatureCollection with `empty_feature_set` warning instead of guessing.

Implementation steps:

- [ ] Extend `src/gis/remote.rs` with `fetch_vector`.
- [ ] Create `src/gis/osm.rs` with query builder, payload parser, tag filtering, skipped counters, and metadata.
- [ ] Keep OSM parsing in Rust with `serde_json`; do not add an OSM parser crate.
- [ ] Register PyO3 functions and Python wrappers for L3 only.
- [ ] Update contract tests for L3 only.

Minimum tests:

- Mocked Overpass JSON, empty result, incomplete way, unsupported relation, tag filtering, cache behavior, CRS/bounds metadata, rate-limit diagnostic, malformed payload, and no-live-network guard.

## L4: Local Context And Building Vectors

APIs: `load_context_vectors`, `load_building_footprints`, `extract_building_heights`.

Behavior:

- `load_context_vectors` accepts a local vector path, a `read_vector` result, a FeatureCollection, or a dict of named layers. `layers=None` loads all available layers from the supplied object. Missing requested layers raise `missing_layer`.
- First-pass path support is current GeoJSON. GPKG is recognized but returns `backend_unavailable: gdal-vector feature required for GPKG`. CityJSON is not for context vectors; use `load_building_footprints`.
- `load_building_footprints` supports GeoJSON FeatureCollections and current GeoJSON path reading. GPKG is recognized but gated to `gdal-vector`. CityJSON first pass parses `CityObjects` of type `Building`/`BuildingPart` for simple LoD0/LoD1 footprints; unsupported CityJSON layouts return `unsupported_layout`.
- Invalid or empty rings raise/record `unsupported_geometry_type` or `empty_feature_set` as appropriate.
- `dst_crs` composes `reproject_vector`; missing CRS raises `missing_crs`; CRS mismatch is never silently fixed.
- `extract_building_heights` checks `height`, `building:height`, `render_height`, `roof:height`, `building:levels`, `levels`, and `building:part:levels`.
- Height units support meters by default plus `m`, `meter`, `meters`, `ft`, `feet`; invalid units or negative/zero values produce `invalid_height`.
- Defaults are `height_m=10.0` and `level_height_m=3.0` unless overridden by `defaults`; fallback counts use `height_fallback`.

Implementation steps:

- [ ] Add context/building loaders and height extraction to `src/gis/domain.rs`.
- [ ] Reuse `read_vector`, `reproject_vector`, geometry validation, and current warning structs.
- [ ] Add PyO3 functions and wrappers for L4 only.
- [ ] Update contract tests for L4 only.

Minimum tests:

- GeoJSON context layers, missing layer, empty features, building footprint GeoJSON, GPKG backend-unavailable, CityJSON simple footprint, invalid rings, missing CRS, destination CRS reprojection path, height attributes, units, levels-to-height, invalid height, negative/zero height, missing attribute fallback, and metadata counts.

## L5: DEM, Terrarium, And Terrain Derivatives

APIs: `prepare_dem`, `decode_terrarium_dem`, `build_terrarium_dem`, `prepare_terrain_derivatives`.

Behavior:

- `prepare_dem` accepts raster path, `read_raster`-style dict, or array plus `RasterInfo`-compatible metadata. It outputs float32 height array, `RasterInfo`, true-valid mask, nodata summary, scale metadata, and `operation`.
- If `target_info` is supplied, align through existing `align_raster_grid` with bilinear resampling unless caller metadata marks categorical data, which is invalid for DEM.
- Missing CRS/transform uses existing `missing_crs`/`missing_transform`; no CRS guessing. Empty valid pixels raise `empty_raster`.
- `decode_terrarium_dem` formula: `height_m = red * 256.0 + green + blue / 256.0 - 32768.0`.
- Terrarium input accepts `uint8` arrays shaped `(height, width, 3)` or PNG paths decoded through existing `image`; unsupported dtype raises `unsupported_dtype`; bad rank/channel count raises `shape_mismatch`.
- Terrarium has no intrinsic nodata; valid mask is all true unless a future explicit nodata parameter is added.
- `build_terrarium_dem` composes `slippy_tile_index`, explicit tile cache/source policy, `fetch_remote_geodata`, `decode_terrarium_dem`, and mosaic. There is no hidden default public tile service: `cache` may be a cache path or dict containing `cache_dir` and optional `url_template`; without a source or cached tile, return `cache_miss`/`missing_tile`.
- Partial tile failures return `partial_mosaic` only when at least one tile succeeds; total failure raises.
- `prepare_terrain_derivatives` supports `slope` and `hillshade` first. Slope uses central differences and returns degrees: `atan(sqrt(dzdx^2 + dzdy^2)) * 180/pi`. Hillshade uses azimuth 315 degrees and altitude 45 degrees with standard Lambertian shading scaled `0..255`.
- Missing resolution or transform raises `missing_transform`. Geographic CRS emits a units warning in `warnings` and keeps units metadata explicit.

Implementation steps:

- [ ] Add DEM and derivative helpers to `src/gis/domain.rs`.
- [ ] Add Terrarium decode/mosaic helpers to `src/gis/terrarium.rs`.
- [ ] Reuse raster read, nodata, window, align, and tile helpers.
- [ ] Add PyO3 functions and wrappers for L5 only.
- [ ] Update contract tests for L5 only.

Minimum tests:

- DEM path/dict/array source, nodata mask, target-grid alignment, dtype conversion, empty raster, missing CRS/transform, Terrarium formula, invalid dtype/shape/path, tile manifest, cached tile mosaic, missing tile, partial mosaic, slope/hillshade formulas on a tiny DEM, unsupported derivative, missing resolution, and geographic-units warning.

## L6: Gridded Datasets

APIs: `read_gridded_dataset`, `subset_grid`.

Behavior:

- First pass supports raster-like 2D/3D local GeoTIFF through existing `read_raster` and reports variables as `band_1`, `band_2`, etc.
- NetCDF, HDF5, OPeNDAP, Zarr, multi-time, curvilinear, and ambiguous multidimensional layouts are recognized but not faked. Return `BackendUnavailable("backend_unavailable: netcdf backend required for read_gridded_dataset")` or `unsupported_layout`/`ambiguous_axes`.
- `variable=None` is valid only for a single variable. Multiple variables without selection raise `ambiguous_variable`.
- `subset_grid` supports raster-like spatial bounds by composing `window_from_bounds` and `read_raster_window`. Coordinate-array subsetting beyond affine grids returns `unsupported_layout` or `ambiguous_axes`.
- Missing CRS for spatial subset raises `missing_crs`.

Implementation steps:

- [ ] Add gridded metadata and subset helpers to `src/gis/domain.rs`.
- [ ] Reuse raster windowing and affine helpers.
- [ ] Add PyO3 functions and wrappers for L6 only.
- [ ] Update contract tests for L6 only.

Minimum tests:

- Single-band and multiband raster-like grid read, variable selection, ambiguous variable, spatial subset by bounds, missing CRS, unsupported NetCDF/HDF5 extension backend-unavailable, unsupported layout, ambiguous axes, and optional xarray/rioxarray reference skip tests.

## L7: Thematic/Domain Raster Helpers And CRS Estimation

APIs: `prepare_landcover_raster`, `prepare_population_raster`, `prepare_osm_scene`, `estimate_local_utm`.

Behavior:

- `prepare_landcover_raster` preserves categorical integer semantics, aligns to `target_info` with nearest-neighbor only, emits `categorical_resampling_warning` when any resampling/alignment occurs, returns class table/counts, and reports `unknown_class` for values outside `classes`.
- `prepare_population_raster` rejects negative population values with `invalid_argument`, preserves CRS/grid metadata, optionally aligns to `target_info`, and supports `normalization=None` or `"minmax"` through existing thematic primitives. Other normalization values raise `unsupported_option`.
- `prepare_osm_scene` composes `query_osm_features`, `parse_osm_features`, `load_context_vectors`, `load_building_footprints`, and `extract_building_heights`. It returns layer dicts for roads, water, buildings, landuse/context when present plus diagnostics. It must not render, style, label, route, or create MapScene objects.
- `estimate_local_utm` accepts WGS84 bounds tuple `(left, bottom, right, top)` or GeoJSON/read-vector-style geometry with CRS metadata. Non-WGS84 CRS must be transformed with existing CRS helpers or fail with `backend_unavailable: proj feature required`.
- UTM zone: `floor((lon_center + 180) / 6) + 1`, clamped to `1..60`; EPSG is `326xx` for center latitude >= 0 and `327xx` otherwise.
- Polar bounds above 84N or below 80S fail with `invalid_bounds` and low confidence metadata is not returned as success.
- Antimeridian inputs use the shortest wrapped center longitude and return `confidence="low"` plus an antimeridian metadata flag.

Implementation steps:

- [ ] Add thematic/domain raster helpers and UTM estimation to `src/gis/domain.rs` and `src/gis/crs.rs` where smaller.
- [ ] Reuse `normalize_raster`, `classify_raster`, `align_raster_grid`, `read_vector`, OSM helpers, building helpers, and CRS transforms.
- [ ] Add PyO3 functions and wrappers for L7 only.
- [ ] Update contract tests for L7 only.

Minimum tests:

- Landcover categorical alignment warning, unknown classes, class counts, population negative rejection, minmax normalization, CRS/grid preservation, OSM scene composition with mocked OSM/cache data, empty layers, height fallback propagation, UTM north/south zones, source CRS missing/invalid, polar failure, antimeridian low confidence, and unsupported CRS transform backend-unavailable.

## Contract And API Tests

- Update `tests/test_api_contracts.py` `EXPECTED_FUNCTIONS` in the phase that registers each API.
- Keep unimplemented future-phase functions in negative coverage until their phase lands.
- Update `python/forge3d/gis.py` `__all__` in the same phase as the wrapper.
- Update `python/forge3d/gis.pyi` stubs in the same phase as the wrapper.
- Add wrapper-surface tests asserting Python functions are callable and path-like arguments marshal through `os.fspath`.
- Add no-backend-Python-GIS-library tests that scan `python/forge3d/gis.py` and new GIS wrappers for forbidden runtime imports.
- Focused new files: `tests/test_gis_remote.py`, `tests/test_gis_cog_tiles.py`, `tests/test_gis_osm.py`, and `tests/test_gis_domain.py`.

## Test Plan

- `T-remote`: mocked URL fetch, cache hit, cache miss, stale-cache fallback, timeout, unsupported scheme/content type, checksum mismatch, malformed payload, atomic cache write, refresh behavior, no-live-network guard.
- `T-cog-tiles`: local COG read, remote COG backend-unavailable or mocked range-read path, window read, overview selection, invalid window/bounds, slippy tile index at known zooms, latitude/zoom validation, antimeridian handling.
- `T-osm`: mocked Overpass JSON, empty result, incomplete way, unsupported relation, tag filtering, cache behavior, CRS/bounds metadata, rate-limit diagnostic, malformed payload.
- `T-domain`: representative fixtures for DEM, landcover, population, building, Terrarium, gridded data, and OSM-scene helpers; missing input; CRS/grid mismatch; unsupported dtype/format; empty valid data; post-operation metadata validation.
- Re-run cross-bundle tests from `T-raster-meta`, `T-raster-read`, `T-crs`, `T-affine`, `T-align`, `T-vector-io`, `T-vector-crs`, `T-vector-geom`, and `T-thematic` whenever a helper composes those primitives.
- Optional reference checks may use `rasterio`, `geopandas`, `shapely`, `rioxarray`, `xarray`, `pyproj`, or `terra` only when installed and must skip cleanly.
- Reference tests compare metadata and numeric results with explicit tolerances; they must never be the only assertion for public behavior.

## Validation Plan

Run at minimum:

```powershell
git status --short
git diff --name-status
git diff --stat
git diff
git ls-files --others --exclude-standard
cargo fmt --check
cargo check
cargo check --features extension-module,weighted-oit,enable-tbn,enable-gpu-instancing,copc_laz
cargo check --features extension-module,weighted-oit,enable-tbn,enable-gpu-instancing,copc_laz,gis-remote
cargo check --features extension-module,weighted-oit,enable-tbn,enable-gpu-instancing,copc_laz,cog_streaming
python -m py_compile python/forge3d/gis.py
python -m pytest tests/test_api_contracts.py -v
python -m pytest tests/test_gis_remote.py -v
python -m pytest tests/test_gis_cog_tiles.py -v
python -m pytest tests/test_gis_osm.py -v
python -m pytest tests/test_gis_domain.py -v
```

Also run existing focused tests for any shared primitive touched:

```powershell
python -m pytest tests/test_gis_raster.py tests/test_gis_read_raster.py -v
python -m pytest tests/test_gis_crs_affine.py tests/test_gis_alignment_windowing.py tests/test_gis_resample_reproject.py -v
python -m pytest tests/test_gis_vector_io.py tests/test_gis_vector_crs.py tests/test_gis_vector_geom.py tests/test_gis_vector_overlay.py -v
python -m pytest tests/test_gis_rasterize_mask.py tests/test_gis_thematic.py -v
```

Broader cargo/Python suites are required only if touched files justify the runtime.

## Review Bundle Requirement

After each phase and after this planning change-set, create a temporary review bundle outside the repo containing:

- `git status --short --branch`
- `git diff --name-status`
- `git diff --stat`
- full `git diff`
- `git ls-files --others --exclude-standard`
- validation logs with command metadata and exit codes
- branch
- merge base
- current commit
- timestamp
- cwd

The review bundle is evidence only. Do not stage, commit, or include it in a PR.

## Backward Compatibility

Preserve all existing GIS APIs and behavior:

- `read_raster_info`, `read_raster`, `write_raster`, `RasterInfo`
- `parse_crs`, `inspect_crs`, `raster_crs`, `assign_crs`, `create_crs_transformer`, `transform_bounds`, `web_mercator_bounds`
- affine helpers, nodata/mask helpers, resampling/alignment/reprojection/window helpers, and compatibility aliases
- `VectorInfo`, `read_vector`, `geometry_type`, `vector_schema`, `feature_count`, `vector_crs`, `vector_bounds`, `reproject_vector`
- `validate_geometry`, `repair_geometry`, `geometry_measure`, `geometry_centroid`, `representative_point`, `interpolate_line`
- `union_geometries`, `dissolve_vector`, `buffer_geometry`, `clip_vector`, `intersect_vectors`, `simplify_geometry`, `load_boundary`
- `rasterize_vectors`, `geometry_mask`, `mask_raster`
- `normalize_raster`, `classify_raster`

## Open Questions

None block implementation. Conservative decisions are fixed above: no hidden default remote cache, no hidden Terrarium tile service, no GDAL/NetCDF/HDF5 first pass, and unsupported heavy formats fail with stable diagnostics.

## Self-Review

- Spec coverage: all 20 Later APIs are mapped to a phase, files, behavior, diagnostics, and tests.
- Placeholder scan: no deferred implementation placeholders are present.
- Type consistency: result shapes stay dict-based and reuse existing `RasterInfo`, `VectorInfo`, CRS, warning, and operation metadata conventions.

# Prompt: G-002 Later Domain And Remote Helpers Implementation Plan

You are working in `milos-agathon/forge3d`.

Task: Create the implementation plan for every API in `Contract Matrix: Later Domain And Remote Helpers` from `docs/carto-engine/rust-gis-implementation-plan.md`.

Context:
- G-002a1, G-002b, and G-002c C1-C6 are expected to already be implemented and merged to `main`.
- Treat the local checkout and GitHub/main history as authoritative, not memory.
- Before writing the plan, verify current `main` contains the C1-C6 public surfaces, tests, wrappers, stubs, and PyO3 exports.
- GitHub orientation from 2026-07-02: PR #93 added C1, PR #94 added C2, PR #95 added C3, PR #96 added C5, and PR #97 added C6. C4 exists in local main history. Re-verify; do not rely on this note.
- If the expected prerequisites are missing, do not invent assumptions. Stop after writing a short blocker note explaining exactly what is missing and which files/PRs prove it.

Canonical sources to inspect:
- Current repository `main` branch.
- Recent merged PRs/issues/commits related to G-002a1, G-002b, and G-002c C1-C6.
- `docs/carto-engine/rust-gis-implementation-plan.md`
- `docs/carto-engine/g-002c-implementation-plan.md`
- Existing G-002c phase plans under `docs/carto-engine/`
- `docs/carto-engine/gis-contract-evidence.md`
- `docs/carto-engine/gis-operation-api-crosswalk.md`, if present
- Existing implementation files:
  - `src/gis/mod.rs`
  - `src/gis/error.rs`
  - `src/gis/types.rs`
  - `src/gis/raster_info.rs`
  - `src/gis/raster_write.rs`
  - `src/gis/crs.rs`
  - `src/gis/affine.rs`
  - `src/gis/warp.rs`
  - `src/gis/vector.rs`
  - `src/gis/geometry.rs` and `src/gis/geometry/`
  - `src/gis/rasterize.rs`
  - `src/gis/thematic.rs`
  - `src/py_module/functions/gis.rs`
  - `src/py_module/classes.rs`
  - `python/forge3d/gis.py`
  - `python/forge3d/gis.pyi`
  - `tests/test_api_contracts.py`
  - existing `tests/test_gis_*.py`
  - `Cargo.toml` and `pyproject.toml`

Deliverable:
- Create one focused Markdown implementation plan for all Later Domain And Remote Helpers.
- Preferred path: `docs/carto-engine/g-002-later-domain-remote-helpers-implementation-plan.md`
- This is a planning change-set only. Do not implement Rust, Python wrappers, tests, examples, shaders, rendering behavior, visual goldens, or runtime behavior.

Scope:
Plan implementation of exactly these public APIs:
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

Required plan quality:
- Be accurate, concise, straightforward, systematic, surgically precise, and rigorous.
- State what exists today from G-002a1/G-002b/G-002c C1-C6, with file-level evidence.
- State exact Later deltas, file by file.
- Use checkbox implementation steps.
- Separate planning, implementation, tests, validation, and review-bundle requirements.
- Include explicit non-goals.
- Include backend dependency decisions and fallback behavior.
- Include diagnostics and expected error tokens.
- Include minimum test cases and reference-library comparison policy.
- Include open questions only if they block implementation; otherwise make conservative decisions.

Architecture constraints:
- GIS backend behavior belongs in Rust under `src/gis/`.
- New backend modules should be `src/gis/remote.rs`, `src/gis/tiles.rs`, `src/gis/osm.rs`, `src/gis/terrarium.rs`, and `src/gis/domain.rs` unless current code proves a smaller placement.
- Python wrappers in `python/forge3d/gis.py` stay thin: `os.fspath`/path normalization and argument marshaling only.
- Do not implement backend GIS behavior in Python.
- Do not use `rasterio`, `geopandas`, `shapely`, `rioxarray`, `xarray`, or `terra` as forge3d runtime backend behavior.
- Those Python GIS libraries may appear only in optional reference tests or docs examples, skipped cleanly when unavailable.
- Public functions must remain importable when optional backends are disabled.
- Backend-required behavior must raise `BackendUnavailable` with a message containing `backend_unavailable` and the required feature/backend name.
- Do not add GDAL, NetCDF/HDF5, or another heavy dependency unless the plan justifies the exact API that needs it and the fallback when absent.
- Prefer existing dependencies and features first. Inspect `Cargo.toml`; do not add a second HTTP/hash/image/GeoTIFF stack if existing `reqwest`, `sha2`, `image`, `tiff`, COG, CRS, raster, vector, or thematic code can be reused.

Phase split:
The plan may be one file, but it must split implementation into independently reviewable phases:
- L1 remote/cache primitives: `fetch_remote_geodata`, `cache_geodata`
- L2 COG and tile math: `read_cog`, `slippy_tile_index`
- L3 vector fetch and OSM payloads: `fetch_vector`, `query_osm_features`, `parse_osm_features`
- L4 local context/building vectors: `load_context_vectors`, `load_building_footprints`, `extract_building_heights`
- L5 DEM, Terrarium, and terrain derivatives: `prepare_dem`, `decode_terrarium_dem`, `build_terrarium_dem`, `prepare_terrain_derivatives`
- L6 gridded datasets: `read_gridded_dataset`, `subset_grid`
- L7 thematic/domain raster helpers and CRS estimation: `prepare_landcover_raster`, `prepare_population_raster`, `prepare_osm_scene`, `estimate_local_utm`

Shared output contracts:
- Reuse existing `RasterInfo`, `VectorInfo`, `CrsInfo`, and geometry/thematic result shapes where possible.
- Plan `RemoteDatasetInfo` with these keys unless existing code established a better compatible shape: `url`, `cache_path`, `status`, `content_type`, `byte_size`, `checksum`, `etag`, `last_modified`, `from_cache`, `warnings`.
- Plan a small domain operation metadata dict with: `name`, `source_kind`, `input_count`, `output_count`, `input_crs`, `output_crs`, `target_grid`, `changed`, `warnings`.
- Do not create large one-off result classes where dicts match existing GIS wrapper style.

Remote/cache behavior requirements:
- Remote fetching must be explicit; no helper may download implicitly unless its public API name or parameters say so.
- `cache=None` means no persistent cache unless the current code already defines a compatible default. If a default is used, the plan must name its path policy exactly.
- Cache writes must be atomic and must not leave partial files as cache hits.
- URL support must be explicit. Start with `http`/`https`; unsupported schemes fail with a stable diagnostic.
- Timeouts, checksum verification, malformed payloads, stale cache fallback, cache refresh, and content-type/driver checks must be testable without live network access.
- Tests must use a local mock/server or injectable transport; no test may depend on public internet availability.

COG/tile behavior requirements:
- `read_cog` must reuse existing COG/raster read code where possible.
- Local COG behavior must work without network features.
- Remote COG behavior must route through explicit remote/cache/range-read backend behavior and fail with `backend_unavailable` when unavailable.
- `slippy_tile_index` must define zoom validation, latitude clamp/rejection, antimeridian handling, x/y/z ordering, tile bounds, and Web Mercator bounds.
- Do not add map rendering, XYZ tile image fetching, recipe manifests, gallery goldens, or MapScene behavior.

OSM behavior requirements:
- `query_osm_features` must be a thin Rust-backed query/cache wrapper with mocked tests only.
- `parse_osm_features` must handle nodes, ways, closed ways, incomplete ways, unsupported relations, tag filtering, skipped counts, CRS/bounds metadata, and malformed payloads.
- `prepare_osm_scene` must compose smaller OSM/vector/building helpers; it must not introduce rendering, styling, label placement, MapScene, or network routing.

Domain behavior requirements:
- Domain helpers are wrappers over already shipped raster/vector/CRS/thematic primitives; do not duplicate primitive behavior.
- `prepare_dem` must define accepted sources, nodata/valid-mask policy, target grid alignment, dtype conversion, scale metadata, and empty-raster behavior.
- `prepare_terrain_derivatives` must define slope/hillshade formulas or backend requirement, units, missing-resolution behavior, and geographic-units warnings.
- `read_gridded_dataset` and `subset_grid` must define the supported first-pass layout. If NetCDF/OPeNDAP support needs a new backend, plan `BackendUnavailable` instead of fake support.
- `decode_terrarium_dem` must specify the Terrarium RGB formula, dtype/shape validation, nodata policy, and min/max metadata.
- `build_terrarium_dem` must compose tile index, remote/cache, decode, mosaic, partial-tile diagnostics, and `RasterInfo`.
- `prepare_landcover_raster` must preserve categorical semantics and require categorical resampling warnings.
- `prepare_population_raster` must reject or diagnose negative population, handle normalization through existing thematic primitives, and preserve CRS/grid metadata.
- `load_building_footprints` must define GeoJSON/GPKG/CityJSON first-pass support, CRS behavior, invalid rings, empty feature sets, and height attribute schema.
- `extract_building_heights` must define accepted attributes, units, levels-to-height defaults, fallback counts, invalid unit, negative/zero height, and missing-attribute diagnostics.
- `estimate_local_utm` must define WGS84/source CRS requirements, zone calculation, polar/antimeridian behavior, confidence metadata, and failure modes.

Stable diagnostics:
Use existing `GisError` variants where possible and include these lowercase diagnostic tokens in messages or warning codes when relevant:
- `network_timeout`
- `cache_miss`
- `cache_stale`
- `checksum_mismatch`
- `malformed_payload`
- `metadata_unavailable`
- `missing_crs`
- `invalid_crs`
- `crs_mismatch`
- `grid_mismatch`
- `missing_transform`
- `invalid_transform`
- `categorical_resampling_warning`
- `unsupported_dtype`
- `unsupported_driver`
- `unsupported_option`
- `unsupported_geometry_type`
- `missing_layer`
- `empty_feature_set`
- `empty_raster`
- `invalid_argument`
- `invalid_bounds`
- `shape_mismatch`
- `backend_unavailable`
- `unsupported_scheme`
- `rate_limited`
- `incomplete_way`
- `unsupported_relation`
- `ambiguous_variable`
- `unsupported_layout`
- `ambiguous_axes`
- `missing_tile`
- `partial_mosaic`
- `unknown_class`
- `invalid_height`
- `height_fallback`

Strict phase hygiene:
Do not implement or plan unrelated behavior outside the Later matrix.
Do not add:
- routing/travel-time/network analysis
- vector writes
- rendering, shaders, visual goldens, gallery goldens, examples, or recipe manifests
- MapScene recipe-family work
- UI/viewer behavior
- new C1-C6 redesigns beyond reuse needed by Later helpers
- Python GIS backend behavior

Backward compatibility:
The plan must explicitly preserve G-002a1/G-002b/G-002c C1-C6 and all existing GIS APIs, including:
- `read_raster_info`, `read_raster`, `write_raster`, `RasterInfo`
- `parse_crs`, `inspect_crs`, `raster_crs`, `assign_crs`, `create_crs_transformer`, `transform_bounds`, `web_mercator_bounds`
- affine helpers, nodata/mask helpers, resampling/alignment/reprojection/window helpers, and compatibility aliases
- `VectorInfo`, `read_vector`, `geometry_type`, `vector_schema`, `feature_count`, `vector_crs`, `vector_bounds`, `reproject_vector`
- `validate_geometry`, `repair_geometry`, `geometry_measure`, `geometry_centroid`, `representative_point`, `interpolate_line`
- `union_geometries`, `dissolve_vector`, `buffer_geometry`, `clip_vector`, `intersect_vectors`, `simplify_geometry`, `load_boundary`
- `rasterize_vectors`, `geometry_mask`, `mask_raster`
- `normalize_raster`, `classify_raster`

Testing plan:
Add focused test sections and exact fixture requirements:
- `T-remote`: mocked URL fetch, cache hit, cache miss, stale-cache fallback, timeout, unsupported scheme/content type, checksum mismatch, malformed payload, atomic cache write, refresh behavior, no-live-network guard.
- `T-cog-tiles`: local COG read, remote COG backend-unavailable or mocked range-read path, window read, overview selection, invalid window/bounds, slippy tile index at known zooms, latitude/zoom validation, antimeridian handling.
- `T-osm`: mocked Overpass JSON, empty result, incomplete way, unsupported relation, tag filtering, cache behavior, CRS/bounds metadata, rate-limit diagnostic, malformed payload.
- `T-domain`: representative fixtures for DEM, landcover, population, building, Terrarium, gridded data, and OSM-scene helpers; missing input; CRS/grid mismatch; unsupported dtype/format; empty valid data; post-operation metadata validation.
- Cross-bundle tests from `T-raster-meta`, `T-raster-read`, `T-crs`, `T-affine`, `T-align`, `T-vector-io`, `T-vector-crs`, `T-vector-geom`, and `T-thematic` whenever a helper composes those primitives.
- Optional `rasterio`, `geopandas`, `shapely`, `rioxarray`, `xarray`, `pyproj`, or `terra` reference checks only when installed; skip cleanly otherwise.

Contract/API tests:
Plan updates for:
- `tests/test_api_contracts.py` expected functions for Later APIs only in the phase that intentionally registers each API.
- `python/forge3d/gis.py` `__all__`
- `python/forge3d/gis.pyi` stubs
- wrapper-surface tests
- no-backend-Python-GIS-library tests
- focused files such as `tests/test_gis_remote.py`, `tests/test_gis_cog_tiles.py`, `tests/test_gis_osm.py`, and `tests/test_gis_domain.py`

Validation plan:
The Later plan must require, at minimum:
- `git status --short`
- `git diff --name-status`
- `git diff --stat`
- `git diff`
- `git ls-files --others --exclude-standard`
- `cargo fmt --check`
- `cargo check`
- `cargo check --features extension-module,weighted-oit,enable-tbn,enable-gpu-instancing,copc_laz`
- any additional cargo feature checks required by planned remote/COG/PROJ/topology features
- `python -m py_compile python/forge3d/gis.py`
- `python -m pytest tests/test_api_contracts.py -v`
- `python -m pytest tests/test_gis_remote.py -v`
- `python -m pytest tests/test_gis_cog_tiles.py -v`
- `python -m pytest tests/test_gis_osm.py -v`
- `python -m pytest tests/test_gis_domain.py -v`
- existing raster/vector/CRS/thematic focused tests for any shared primitive touched
- broader cargo/python tests only if touched files justify them

Review bundle rule:
After the planning change-set, generate a fresh temporary review bundle containing:
- git status
- diff name-status
- diff stat
- full diff
- untracked files
- validation logs
- command metadata and exit codes
- branch
- merge base
- current commit
- timestamp
- cwd

The review bundle is temporary evidence only. It must never be staged, committed, or included in the PR.

Required output from this Codex task:
1. The created Later Domain And Remote Helpers implementation-plan Markdown file.
2. A concise final summary listing:
   - files changed
   - prerequisite verification result
   - exact Later APIs planned
   - phase split
   - explicit non-goals
   - validation commands run and results
   - review bundle path
3. No runtime implementation code.

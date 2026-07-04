# Prompt: Fully Implement G-002 Later Domain And Remote Helpers

You are working in `milos-agathon/forge3d`.

Task: Fully implement every requirement in `docs/carto-engine/g-002-later-domain-remote-helpers-implementation-plan.md`.

This is an implementation task, not another planning task. Do not create a replacement plan. Use the plan as the contract and ship the code, wrappers, stubs, tests, validation, and review evidence it requires.

Required operating mode:
- Use `superpowers:subagent-driven-development` if available; otherwise use `superpowers:executing-plans`.
- Execute the plan task-by-task across L1 through L7.
- Do not stop after one phase unless blocked by a real technical blocker that prevents all useful progress.
- Keep each phase independently testable, but the final deliverable must include all 20 Later APIs.
- Be accurate, concise, straightforward, systematic, surgically precise, and rigorous.

Authoritative sources:
- `docs/carto-engine/g-002-later-domain-remote-helpers-implementation-plan.md`
- `docs/carto-engine/rust-gis-implementation-plan.md`
- Existing GIS implementation under `src/gis/`
- Existing Python surface in `python/forge3d/gis.py` and `python/forge3d/gis.pyi`
- Existing API contract tests in `tests/test_api_contracts.py`
- Existing focused GIS tests under `tests/test_gis_*.py`
- Current local checkout and GitHub `main`; do not rely on memory.

Pre-flight:
- Read the full Later implementation plan before editing.
- Verify the working tree with `git status --short`.
- Identify unrelated dirty files and leave them alone.
- Verify G-002a1, G-002b, and G-002c C1-C6 surfaces still exist.
- Inspect `Cargo.toml` before adding any dependency or feature.
- If the plan file is missing, incomplete, or contradicted by current `main`, stop with a short blocker note naming exact files and evidence.

Implement exactly these APIs:
- `fetch_remote_geodata`
- `cache_geodata`
- `fetch_vector`
- `read_cog`
- `slippy_tile_index`
- `query_osm_features`
- `parse_osm_features`
- `load_context_vectors`
- `prepare_osm_scene`
- `prepare_dem`
- `prepare_terrain_derivatives`
- `read_gridded_dataset`
- `subset_grid`
- `decode_terrarium_dem`
- `build_terrarium_dem`
- `prepare_landcover_raster`
- `prepare_population_raster`
- `load_building_footprints`
- `extract_building_heights`
- `estimate_local_utm`

Implementation requirements:
- Keep backend behavior in Rust under `src/gis/`.
- Add or use `src/gis/remote.rs`, `src/gis/tiles.rs`, `src/gis/osm.rs`, `src/gis/terrarium.rs`, and `src/gis/domain.rs` as planned.
- Keep Python wrappers thin: `os.fspath`, path normalization, keyword forwarding, and native dispatch only.
- Update PyO3 registration in `src/py_module/functions/gis.rs`.
- Update `python/forge3d/gis.py` `__all__`.
- Update `python/forge3d/gis.pyi` with exact stubs.
- Update `tests/test_api_contracts.py` so all 20 APIs are expected only after implementation.
- Add focused tests: `tests/test_gis_remote.py`, `tests/test_gis_cog_tiles.py`, `tests/test_gis_osm.py`, and `tests/test_gis_domain.py`.
- Reuse existing raster, CRS, affine, vector, rasterization, geometry, thematic, COG, `sha2`, `image`, `tiff`, `serde_json`, `ndarray`, optional `reqwest`, and optional `tokio` code where possible.
- Do not add GDAL, NetCDF, HDF5, a new HTTP stack, a new hash stack, a new image stack, or Python GIS runtime dependencies.
- Public functions must remain importable when optional backends are disabled.
- Backend-required behavior must raise `BackendUnavailable` with `backend_unavailable` and the missing feature/backend name.

Phase order:
1. L1 remote/cache primitives: `fetch_remote_geodata`, `cache_geodata`
2. L2 COG and tile math: `read_cog`, `slippy_tile_index`
3. L3 vector fetch and OSM payloads: `fetch_vector`, `query_osm_features`, `parse_osm_features`
4. L4 local context/building vectors: `load_context_vectors`, `load_building_footprints`, `extract_building_heights`
5. L5 DEM, Terrarium, terrain derivatives: `prepare_dem`, `decode_terrarium_dem`, `build_terrarium_dem`, `prepare_terrain_derivatives`
6. L6 gridded datasets: `read_gridded_dataset`, `subset_grid`
7. L7 thematic/domain raster helpers and CRS estimation: `prepare_landcover_raster`, `prepare_population_raster`, `prepare_osm_scene`, `estimate_local_utm`

Testing rules:
- Write failing focused tests before each phase implementation.
- Use mocked/local transport for remote and OSM tests. No test may depend on public internet availability.
- Optional reference-library checks must skip cleanly when unavailable.
- Never use `rasterio`, `geopandas`, `shapely`, `rioxarray`, `xarray`, or `terra` as runtime backend behavior.
- Add no-backend-Python-GIS-library tests for wrapper/runtime files.
- Preserve all existing G-002a1/G-002b/G-002c C1-C6 tests and API contracts.

Strict non-goals:
- No routing, travel-time, network analysis, graph cost modeling, or pathfinding.
- No vector writes.
- No rendering, shaders, UI, viewer, MapScene, gallery goldens, visual goldens, examples, or recipe manifests.
- No redesign of shipped GIS APIs except the smallest reuse needed for Later helpers.
- No hidden default downloads, hidden default remote cache, or hidden public Terrarium tile service.

Validation:
Run at minimum after the full implementation:

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
python -m pytest tests/test_gis_raster.py tests/test_gis_read_raster.py -v
python -m pytest tests/test_gis_crs_affine.py tests/test_gis_alignment_windowing.py tests/test_gis_resample_reproject.py -v
python -m pytest tests/test_gis_vector_io.py tests/test_gis_vector_crs.py tests/test_gis_vector_geom.py tests/test_gis_vector_overlay.py -v
python -m pytest tests/test_gis_rasterize_mask.py tests/test_gis_thematic.py -v
```

If a command fails, debug to root cause and fix it. Do not mark the task complete with failing focused tests unless the failure is a documented unrelated environment blocker with exact evidence.

Review bundle:
- Create a fresh temporary review bundle outside the repo after implementation and validation.
- Include `git status --short --branch`, diff name-status, diff stat, full diff, untracked files, validation logs, command metadata and exit codes, branch, merge base, current commit, timestamp, and cwd.
- Never stage, commit, or include the review bundle.

Final response must include:
- Files changed.
- APIs implemented.
- Backend/features added or reused.
- Explicit non-goals preserved.
- Validation commands and pass/fail results.
- Review bundle path.
- Any remaining blockers or test gaps.

Do not stage, commit, push, or open a PR unless the user explicitly asks.

# G-002c C6 Thematic Raster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement only `normalize_raster` and `classify_raster` for G-002c C6 thematic raster operations.

**Architecture:** Keep GIS backend behavior in Rust under `src/gis/`, with a new focused `src/gis/thematic.rs`. Python wrappers stay thin: path normalization, keyword forwarding, and `__all__` exposure only.

**Tech Stack:** Rust, PyO3, NumPy at the Python boundary, existing `RasterArray`/`RasterData`/`RasterDType`/`RasterInfo` helpers, and NumPy-only reference tests.

---

## Current State

Synced `main` is at merge commit `1189ad1b06227012db3a1f794a8d30e301945e19`, merged from PR #96. C5 is present on current `main`; do not write a blocker note.

| Surface | Evidence |
|---|---|
| G-002a1 raster read/write: `RasterInfo`, `read_raster_info`, `read_raster`, `write_raster` | `src/gis/mod.rs:27`, `src/gis/mod.rs:28`, `src/gis/raster_info.rs:38`, `src/gis/raster_info.rs:359`, `src/gis/raster_write.rs:287`, `python/forge3d/gis.py:44`, `python/forge3d/gis.py:49`, `python/forge3d/gis.py:304`, `tests/test_gis_raster.py:50` |
| G-002b CRS, affine, nodata/mask, resampling, alignment, reprojection, windows | `docs/carto-engine/g-002b-support-matrix.md:14`, `docs/carto-engine/g-002b-support-matrix.md:17`, `docs/carto-engine/g-002b-support-matrix.md:18`, `docs/carto-engine/g-002b-support-matrix.md:19`, `docs/carto-engine/g-002b-support-matrix.md:20`, `docs/carto-engine/g-002b-support-matrix.md:21`, `src/gis/mod.rs:31`, `src/gis/mod.rs:435`, `src/gis/mod.rs:473`, `src/gis/warp.rs:88` |
| G-002c C1 vector metadata/read: `VectorInfo`, `read_vector`, `geometry_type`, `vector_schema`, `feature_count`, `vector_crs`, `vector_bounds` | `src/gis/mod.rs:39`, `src/py_module/functions/gis.rs:8`, `src/py_module/functions/gis.rs:10`, `src/py_module/functions/gis.rs:11`, `src/py_module/functions/gis.rs:12`, `src/py_module/functions/gis.rs:13`, `src/py_module/functions/gis.rs:14`, `python/forge3d/gis.py:69`, `python/forge3d/gis.py:96`, `python/forge3d/gis.py:101`, `python/forge3d/gis.py:106`, `python/forge3d/gis.py:111`, `python/forge3d/gis.py:116`, `python/forge3d/gis.pyi:47`, `python/forge3d/gis.pyi:133` |
| G-002c C2: `reproject_vector` | `src/py_module/functions/gis.rs:9`, `python/forge3d/gis.py:87`, `python/forge3d/gis.pyi:143`, `tests/test_gis_vector_crs.py:1` |
| G-002c C3 geometry: `validate_geometry`, `repair_geometry`, `geometry_measure`, `geometry_centroid`, `representative_point`, `interpolate_line` | `src/gis/mod.rs:17`, `src/py_module/functions/gis.rs:15`, `src/py_module/functions/gis.rs:16`, `src/py_module/functions/gis.rs:17`, `src/py_module/functions/gis.rs:18`, `src/py_module/functions/gis.rs:19`, `src/py_module/functions/gis.rs:20`, `python/forge3d/gis.py:125`, `python/forge3d/gis.py:130`, `python/forge3d/gis.py:135`, `python/forge3d/gis.py:144`, `python/forge3d/gis.py:149`, `python/forge3d/gis.py:154`, `tests/test_gis_vector_geom.py:1` |
| G-002c C4 overlay/topology: `union_geometries`, `dissolve_vector`, `buffer_geometry`, `clip_vector`, `intersect_vectors`, `simplify_geometry`, `load_boundary` | `src/gis/mod.rs:22`, `src/py_module/functions/gis.rs:21`, `src/py_module/functions/gis.rs:22`, `src/py_module/functions/gis.rs:23`, `src/py_module/functions/gis.rs:24`, `src/py_module/functions/gis.rs:25`, `src/py_module/functions/gis.rs:26`, `src/py_module/functions/gis.rs:27`, `python/forge3d/gis.py:168`, `python/forge3d/gis.py:173`, `python/forge3d/gis.py:178`, `python/forge3d/gis.py:183`, `python/forge3d/gis.py:197`, `python/forge3d/gis.py:211`, `python/forge3d/gis.py:225`, `tests/test_gis_vector_overlay.py:25` |
| G-002c C5 rasterization/masks: `rasterize_vectors`, `geometry_mask`, `mask_raster` | `src/gis/mod.rs:9`, `src/gis/mod.rs:31`, `src/gis/mod.rs:37`, `src/gis/rasterize.rs:183`, `src/gis/rasterize.rs:209`, `src/gis/rasterize.rs:243`, `src/py_module/functions/gis.rs:28`, `src/py_module/functions/gis.rs:29`, `src/py_module/functions/gis.rs:30`, `python/forge3d/gis.py:239`, `python/forge3d/gis.py:261`, `python/forge3d/gis.py:279`, `python/forge3d/gis.pyi:266`, `python/forge3d/gis.pyi:278`, `python/forge3d/gis.pyi:288`, `tests/test_gis_rasterize_mask.py:96` |
| C6 is still absent from runtime/export surfaces | `rg normalize_raster classify_raster python/forge3d/gis.py python/forge3d/gis.pyi src/py_module/functions/gis.rs src/gis/mod.rs src/gis/rasterize.rs` returns no runtime hits; `tests/test_api_contracts.py:186` keeps both names in `LATER_GIS_FUNCTIONS`; `tests/test_gis_vector_geom.py:376` bans both names |

## C6 Public APIs

Plan exactly these functions:

```python
def normalize_raster(
    source,
    *,
    method="minmax",
    valid_mask=None,
    nodata=None,
    clip=None,
) -> dict[str, Any]: ...

def classify_raster(
    source,
    *,
    bins=None,
    labels=None,
    right=False,
    valid_mask=None,
    nodata=None,
    dtype="uint16",
) -> dict[str, Any]: ...
```

Both return a ThematicResult-like dict with these keys:

- `array`
- `info`
- `method`
- `valid_count`
- `nodata_count`
- `min`
- `max`
- `mean`
- `std`
- `class_table`
- `warnings`

## Behavior Decisions

- `normalize_raster` supports only `method="minmax"` in C6.
- Unsupported `method` values fail with `GisError::InvalidArgument` containing `unsupported_option`.
- Stats use only cells where source value is finite, not nodata, and `valid_mask` is not false.
- `valid_mask` is true-valid: false cells are excluded from stats and assigned to the invalid/nodata output class.
- `nodata_count` means all cells excluded from stats: explicit nodata, NaN, or `valid_mask=False`.
- Float `NaN` values are invalid even without explicit `nodata`; `nodata=np.nan` also matches NaN deliberately.
- `clip=(min, max)` is supported as a finite ordered pre-normalization clamp. Invalid clip values fail with `invalid_argument`.
- All invalid or all nodata input fails with `empty_raster`.
- Normalized output dtype is `float32`; invalid/nodata cells are `NaN`.
- `normalize_raster` returns `method="minmax"` and `class_table=None`.
- `classify_raster` supports explicit `bins` only.
- `classify_raster` returns `method="explicit_bins"`.
- `bins=None` fails with `invalid_argument`; C6 has no inferred classification.
- `bins` is a finite, strictly increasing sequence. Empty, non-finite, unsorted, or duplicate bins fail with `invalid_argument`.
- `right` follows `numpy.digitize(values, bins, right=right)`.
- Class ID `0` is reserved for nodata/invalid pixels. Valid classes start at `1`.
- Number of valid classes is `len(bins) + 1`.
- `labels`, when supplied, must have exactly `len(bins) + 1` entries.
- `class_table` contains one nodata row plus one row per valid class. Each row includes `class_id`, `label`, `left`, `right`, `right_inclusive`, `count`, and `nodata`.
- Output `dtype` supports integer class arrays only: `uint8`, `uint16`, `int16`, `uint32`, `int32`. Reject floats and unsupported names with `unsupported_dtype`.
- Output dtype must represent class IDs `0..len(bins)+1`; otherwise fail with `unsupported_dtype`.

## Backend Dependency Decision

- Use existing Rust raster, NumPy/PyO3, and ndarray-adjacent helpers already in the repository.
- Do not add `rasterio`, `geopandas`, `shapely`, `rioxarray`, `xarray`, `terra`, GDAL, or a new runtime dependency.
- Do not introduce `BackendUnavailable`; C6 is pure Rust/NumPy-compatible behavior. Use `BackendUnavailable` only if a real optional backend is introduced, which this plan forbids.
- Accept sources as local raster path, `RasterInfo`, NumPy array, or `read_raster`-style dict containing `array` and optional `info`. Do not add remote/cache/fetch behavior.

## File-Level Deltas

| File | C6 delta |
|---|---|
| Create `src/gis/thematic.rs` | Rust implementation for `normalize_raster`, `classify_raster`, shared source extraction, valid-mask/nodata filtering, stats, minmax normalization, explicit-bin classification, class table creation, result metadata, and PyO3 wrappers. |
| Modify `src/gis/mod.rs` | Add `pub mod thematic;` and re-export `normalize_raster_py` and `classify_raster_py` behind `extension-module`. Reuse existing private helpers from descendant module code where possible. |
| Modify `src/py_module/functions/gis.rs` | Register `normalize_raster_py` and `classify_raster_py` immediately after C5 rasterization/mask registrations. |
| Modify `python/forge3d/gis.py` | Add thin wrappers and add both names to `__all__`. Wrapper work is limited to `os.fspath`/`_path_or_self` normalization and keyword forwarding. |
| Modify `python/forge3d/gis.pyi` | Add exact stubs for both C6 signatures returning `dict[str, Any]`. |
| Modify `tests/test_api_contracts.py` | Move only `normalize_raster` and `classify_raster` from `LATER_GIS_FUNCTIONS` to `EXPECTED_FUNCTIONS`. |
| Modify `tests/test_gis_raster.py` | Update wrapper `__all__` and callable assertions if that test still hard-codes GIS public functions. |
| Create `tests/test_gis_thematic.py` | Focused T-thematic behavior tests, NumPy reference comparisons, wrapper-surface checks, and no-runtime-Python-GIS-backend assertions. |
| Do not modify `Cargo.toml` or `pyproject.toml` | No new runtime dependency or feature gate is planned for C6. |

## Diagnostics

Use existing `GisError` variants from `src/gis/error.rs`.

| Token | Variant | Use |
|---|---|---|
| `empty_raster` | `InvalidArgument` | No valid cells remain after nodata, NaN, and mask filtering. |
| `unsupported_dtype` | `UnsupportedDType` | Unsupported source/output dtype, float classification dtype, or dtype too small for class IDs. |
| `invalid_argument` | `InvalidArgument` | Bad bins, label mismatch, bad clip, invalid method argument shape, or malformed source dict. |
| `shape_mismatch` | `ShapeMismatch` | `valid_mask` shape does not match raster array shape. |
| `invalid_nodata` | `InvalidNodata` | Nodata list length mismatch, nonnumeric nodata, or non-finite nodata other than `NaN` for float sources. |
| `unsupported_option` | `InvalidArgument` | Unsupported normalization method or future classification method request. |

## Planning Steps

- [ ] Confirm working tree state with `git status --short` before implementation starts.
- [ ] Confirm current branch is synced to `origin/main` at or after `1189ad1b06227012db3a1f794a8d30e301945e19`.
- [ ] Run `rg normalize_raster classify_raster python/forge3d/gis.py python/forge3d/gis.pyi src/py_module/functions/gis.rs src/gis/mod.rs` and confirm C6 is absent before editing.
- [ ] Keep implementation to the files listed in "File-Level Deltas" unless `cargo check` proves a helper must move.

## Implementation Steps

- [ ] Create `tests/test_gis_thematic.py` with failing tests for the T-thematic cases listed below.
- [ ] Add `src/gis/thematic.rs` with a small `ThematicResult` struct and shared stats struct.
- [ ] Add source extraction in `src/gis/thematic.rs`: path or `RasterInfo` via `raster_info::read_raster`, dict via `array` plus optional `info`, and direct NumPy array via existing `extract_raster_array`.
- [ ] Add synthetic `RasterInfo` for bare NumPy arrays with `driver="memory"`, dtype from `RasterArray::dtype()`, dimensions from the array, and no CRS/transform.
- [ ] Validate source info shape against array shape and return `shape_mismatch` on mismatch.
- [ ] Parse `nodata` as scalar or per-band list and include `invalid_nodata` in all nodata validation failures.
- [ ] Parse `valid_mask` as bool NumPy array and accept either exact `(bands, height, width)` or 2D `(height, width)` broadcast across bands; reject anything else with `shape_mismatch`.
- [ ] Build the valid-cell mask once from finite values, nodata, and `valid_mask`.
- [ ] Compute `valid_count`, `nodata_count`, `min`, `max`, `mean`, and population `std` from valid cells only; fail with `empty_raster` if `valid_count == 0`.
- [ ] Implement `normalize_raster` minmax: clamp valid values when `clip` is supplied, normalize `(value - min) / (max - min)`, output `0.0` for valid cells when `min == max`, and output `NaN` for invalid cells.
- [ ] Implement `classify_raster` using explicit bins and NumPy digitize-compatible `right` semantics; assign invalid cells to class ID `0`.
- [ ] Build `class_table` with nodata row first and valid class rows in class ID order.
- [ ] Convert output arrays through existing Rust raster array to Python ndarray conversion helpers.
- [ ] Return `warnings` as an empty list unless an existing raster warning is carried from source metadata.
- [ ] Add PyO3 wrappers in `src/gis/thematic.rs` with exact Python-visible names and keyword-only signatures.
- [ ] Register wrappers in `src/gis/mod.rs` and `src/py_module/functions/gis.rs`.
- [ ] Add thin Python wrappers and `__all__` entries in `python/forge3d/gis.py`.
- [ ] Add `.pyi` stubs in `python/forge3d/gis.pyi`.
- [ ] Move C6 names into `EXPECTED_FUNCTIONS` in `tests/test_api_contracts.py`.
- [ ] Update any hard-coded public GIS callable lists in `tests/test_gis_raster.py`.

## Tests

Add a T-thematic section in `tests/test_gis_thematic.py` with these minimum cases:

- [ ] Minmax normalization on a tiny array.
- [ ] Minmax normalization with `clip`.
- [ ] Nodata excluded from stats.
- [ ] `valid_mask=False` excluded from stats.
- [ ] NaN handling for float arrays.
- [ ] All invalid/all nodata input raises `empty_raster`.
- [ ] Unsupported normalization method raises `unsupported_option`.
- [ ] Unsupported classification dtype raises `unsupported_dtype`.
- [ ] `valid_mask` shape mismatch raises `shape_mismatch`.
- [ ] Classification with explicit bins.
- [ ] Classification labels.
- [ ] `right=False` boundary behavior matches `np.digitize(..., right=False)`.
- [ ] `right=True` boundary behavior matches `np.digitize(..., right=True)`.
- [ ] Bad bins: empty, non-finite, unsorted, duplicate.
- [ ] Label count mismatch.
- [ ] Output dtype too small.
- [ ] `class_table` counts sum correctly.
- [ ] Nodata/invalid pixels use class ID `0`.
- [ ] Source as NumPy array.
- [ ] Source as `read_raster`-style result dict.
- [ ] No runtime Python GIS backend library use: assert implementation does not import or require `rasterio`, `geopandas`, `shapely`, `rioxarray`, `xarray`, or `terra`.

NumPy reference policy:

- [ ] Use NumPy only as the test oracle.
- [ ] For normalization, compare valid cells with `np.testing.assert_allclose`; assert invalid cells are `np.nan`.
- [ ] For classification, compare class IDs to `np.digitize` plus the reserved `0` invalid class policy.
- [ ] Do not use rasterio, GDAL, GeoPandas, Shapely, rioxarray, xarray, or terra in C6 tests.

Contract/API tests:

- [ ] Update `tests/test_api_contracts.py` expected functions only for `normalize_raster` and `classify_raster`.
- [ ] Update `python/forge3d/gis.py` `__all__`.
- [ ] Update `python/forge3d/gis.pyi` stubs.
- [ ] Add wrapper-surface tests proving both names are callable through `forge3d.gis`.
- [ ] Add no-backend-Python-GIS-library tests.

## Validation For C6 Implementation

Run at minimum:

- [ ] `git status --short`
- [ ] `git diff --name-status`
- [ ] `git diff --stat`
- [ ] `git diff`
- [ ] `git ls-files --others --exclude-standard`
- [ ] `cargo fmt --check`
- [ ] `cargo check`
- [ ] `python -m py_compile python/forge3d/gis.py`
- [ ] `python -m pytest tests/test_gis_thematic.py -q`
- [ ] `python -m pytest tests/test_api_contracts.py tests/test_gis_raster.py -q`
- [ ] `python -m pytest tests/test_gis_rasterize_mask.py -q`
- [ ] Broader raster/vector tests only if touched files justify them.

## Validation For This Planning Change-Set

Run after adding this Markdown file:

- [ ] `git status --short`
- [ ] `git diff --name-status`
- [ ] `git diff --stat`
- [ ] `git diff`
- [ ] `git ls-files --others --exclude-standard`
- [ ] `python -m pytest tests/test_api_contracts.py tests/test_gis_raster.py tests/test_gis_rasterize_mask.py -q`

## Review Bundle Requirement

After the planning change-set validation, generate a fresh temporary review bundle outside the repo. Include:

- [ ] `git status`
- [ ] diff name-status
- [ ] diff stat
- [ ] full diff
- [ ] untracked files
- [ ] validation logs
- [ ] command metadata and exit codes
- [ ] branch
- [ ] merge base
- [ ] current commit
- [ ] timestamp
- [ ] cwd

The review bundle is temporary evidence only. Never stage it, commit it, or include it in the PR.

## Non-Goals

- No runtime implementation in this planning change-set.
- No C7 or later API planning beyond explicitly deferring it.
- No color mapping, palettes, colormaps, rendering, shaders, examples, visual goldens, gallery goldens, or recipe manifests.
- No quantile, natural breaks, Jenks, domain classification, landcover, DEM, population, building, Terrarium, OSM, slippy tile, remote/cache/fetch, MapScene, or named-domain helpers.
- No vector writes.
- No Python GIS backend behavior.
- No new runtime backend dependency and no GDAL addition.

## Backward Compatibility

- Preserve all G-002a1/G-002b/G-002c C1-C5 APIs and behavior.
- Do not alter raster read/write, CRS, affine, nodata/mask, resampling, alignment, reprojection, or window helper contracts.
- Do not alter vector IO, vector CRS, geometry operation, overlay/topology, rasterization, geometry mask, or raster mask contracts.
- Keep public functions importable and keep existing contract tests passing.

## Open Questions

None block C6. Conservative decisions above are binding for first implementation.

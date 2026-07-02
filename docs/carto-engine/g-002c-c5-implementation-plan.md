# G-002c C5 Rasterization and Masks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add G-002c C5 rasterization and mask APIs on top of the committed C1-C4 vector and G-002a1/G-002b raster foundations.

**Architecture:** GIS backend behavior stays in Rust under `src/gis/`. Python wrappers in `python/forge3d/gis.py` stay thin: `os.fspath` path normalization and direct PyO3 argument marshaling only. C5 uses an explicit target raster grid and never infers shape, transform, bounds, or CRS from vector bounds.

**Tech Stack:** Rust, PyO3, NumPy at the Python boundary, existing `RasterInfo`/`RasterArray`/`RasterDType` raster helpers, existing affine and CRS helpers, existing GeoJSON-like vector/geometry parsing, and optional reference tests using rasterio only when installed.

---

## Global Constraints

- Plan exactly these public APIs: `rasterize_vectors`, `geometry_mask`, and `mask_raster`.
- Do not implement runtime code in this planning change-set.
- Backend GIS behavior belongs in Rust under `src/gis/`.
- Python wrappers stay thin: `os.fspath`/path normalization and argument marshaling only.
- Do not implement rasterization or masking behavior in Python.
- Do not use `rasterio`, `geopandas`, `shapely`, `rioxarray`, `xarray`, or `terra` as forge3d runtime backend behavior.
- Reference libraries may appear only in optional tests or docs examples and must skip cleanly when unavailable.
- Target grid must be explicit. Do not infer a grid from vector bounds.
- CRS mismatch must fail. Missing CRS must be reported, not guessed.
- Bounds order remains `(left, bottom, right, top)`.
- Mask polarity must be explicit and stable.
- `mask_raster` consumes an explicit mask. It must not secretly rasterize vectors.
- Do not add vector writes, rendering, shaders, examples, visual goldens, gallery goldens, remote/cache/fetch, OSM, Terrarium, slippy tiles, domain helpers, MapScene work, or C6 thematic behavior.

## Preflight Result

- Current branch: `main`.
- Current commit before this plan: `816661669a3ba00c28129add3df247e17240558b` (`Implement G-002c C4.7 load boundary`).
- C4.7 is committed in local history; it is not an uncommitted review diff.
- The older C4.7 review bundle at `C:\Users\milos\AppData\Local\Temp\forge3d-g002c-c47-review-bundle-20260702T062549Z` captured the pre-commit review diff for `src/gis/geometry/topology.rs`, `src/gis/vector.rs`, and `tests/test_gis_vector_overlay.py`.
- The only pre-existing dirty path before this C5 plan was `logs/.9a9fb3a1aee4abd4cecd121a192f7ef77e352837-audit.json`; it is unrelated to C4.7 source files and is left untouched.
- `python -m pytest tests/test_api_contracts.py -q` passed: `286 passed`.
- Direct local import probe with `sys.path.insert(0, "python")` found no missing C1-C4 public functions and no leaked C5/C6 functions.

## Current C1-C4 Evidence

| Phase | Public surface | Evidence |
|---|---|---|
| G-002a1 raster read/write | `RasterInfo`, `read_raster_info`, `read_raster`, `write_raster` | `src/gis/types.rs:230`, `src/gis/raster_info.rs:38`, `src/gis/raster_info.rs:359`, `src/gis/raster_write.rs:287`, `python/forge3d/gis.py:43`, `python/forge3d/gis.py:48`, `python/forge3d/gis.py:242`, `tests/test_gis_raster.py:50` |
| G-002b CRS/affine/nodata/window/resample/reproject | CRS helpers, affine helpers, `apply_nodata`, `read_raster_mask`, alignment, reprojection, windowing | `src/gis/affine.rs:49`, `src/gis/affine.rs:70`, `src/gis/affine.rs:129`, `src/gis/affine.rs:187`, `src/gis/warp.rs:49`, `src/gis/mod.rs:436`, `src/gis/mod.rs:465`, `tests/test_gis_crs_affine.py:1`, `tests/test_gis_alignment_windowing.py:1`, `tests/test_gis_resample_reproject.py:1` |
| G-002c C1 vector metadata/read | `VectorInfo`, `read_vector`, `geometry_type`, `vector_schema`, `feature_count`, `vector_crs`, `vector_bounds` | `src/gis/vector.rs:35`, `src/gis/vector.rs:89`, `src/gis/vector.rs:2069`, `src/py_module/functions/gis.rs:8`, `python/forge3d/gis.py:68`, `python/forge3d/gis.pyi:133`, `tests/test_gis_vector_io.py:86` |
| G-002c C2 vector reprojection | `reproject_vector` | `src/gis/vector.rs:186`, `src/gis/vector.rs:2174`, `src/py_module/functions/gis.rs:9`, `python/forge3d/gis.py:86`, `python/forge3d/gis.pyi:143`, `tests/test_gis_vector_crs.py:1` |
| G-002c C3 geometry | `validate_geometry`, `repair_geometry`, `geometry_measure`, `geometry_centroid`, `representative_point`, `interpolate_line` | `src/gis/geometry.rs`, `src/gis/geometry/parse.rs:8`, `src/gis/geometry/model.rs:5`, `src/gis/geometry/py.rs`, `python/forge3d/gis.py:124`, `python/forge3d/gis.pyi:185`, `tests/test_gis_vector_geom.py:1` |
| G-002c C4 overlay | `union_geometries`, `dissolve_vector`, `buffer_geometry`, `clip_vector`, `intersect_vectors`, `simplify_geometry`, `load_boundary` | `src/gis/geometry.rs:189`, `src/gis/vector.rs:255`, `src/gis/vector.rs:328`, `src/gis/vector.rs:375`, `src/gis/vector.rs:430`, `src/gis/geometry/topology.rs:5`, `src/py_module/functions/gis.rs:21`, `python/forge3d/gis.py:167`, `python/forge3d/gis.pyi:216`, `tests/test_gis_vector_overlay.py:25` |
| Contract lock | C1-C4 expected, C5/C6 absent | `tests/test_api_contracts.py:61`, `tests/test_api_contracts.py:109`, `tests/test_api_contracts.py:184` |

## Current C5/C6 Absence

`rasterize_vectors`, `geometry_mask`, `mask_raster`, `normalize_raster`, and `classify_raster` are absent from:

- `python/forge3d/gis.py` wrappers and `__all__`.
- `python/forge3d/gis.pyi` stubs.
- `src/py_module/functions/gis.rs` PyO3 function registration.
- `src/gis/mod.rs` re-exports.

They are intentionally listed only as later GIS functions in `tests/test_api_contracts.py:184`.

## C5 APIs Planned

```python
def rasterize_vectors(
    vectors,
    target_info,
    *,
    value=1,
    attribute=None,
    dtype="uint8",
    fill=0,
    all_touched=False,
) -> dict[str, Any]: ...

def geometry_mask(
    geometries,
    target_info,
    *,
    invert=False,
    all_touched=False,
    mask_polarity="true_inside",
) -> dict[str, Any]: ...

def mask_raster(
    source,
    mask,
    *,
    mask_polarity,
    crop=False,
    fill=None,
    nodata=None,
) -> dict[str, Any]: ...
```

## Backend Decision

- Do not add a new runtime dependency for C5.
- Implement the default C5 path in Rust in `src/gis/rasterize.rs`.
- Reuse existing `RasterInfo`, `RasterArray`, `RasterData`, `RasterDType`, affine helpers, CRS helpers, and GeoJSON-like geometry parsing.
- Do not require `geos-topology`; current `Cargo.toml:57` gates C4 topology with `geo`, and current `pyproject.toml` maturin features do not include `geos-topology`.
- Do not add GDAL for C5. If a later backend is added, it must be feature-gated, keep functions importable, and raise `BackendUnavailable` with `backend_unavailable` when absent.
- Optional reference checks may use `rasterio.features.rasterize`, `rasterio.features.geometry_mask`, or `rasterio.mask.mask` only inside tests guarded by `pytest.importorskip("rasterio")`.

## Behavior Contracts

### Target Grid

- `target_info` accepts a native `RasterInfo` or serialized `RasterInfo` dict.
- Required fields: `width`, `height`, `transform`, and CRS metadata (`crs_authority` or `crs_wkt`).
- Optional fields copied through output metadata: `path`, `driver`, `band_count`, `dtype_per_band`, `nodata_per_band`, `bounds`, `resolution`, `warnings`.
- Missing width/height/transform is `invalid_argument` or `missing_transform`.
- Missing target CRS is `missing_crs`.
- Bounds order remains `(left, bottom, right, top)`.

### Vector And Geometry Inputs

- Accepted C5 inputs are the same GeoJSON-like dict shapes already used by C1-C4: Geometry, Feature, FeatureCollection, read-vector result dicts, and local vector paths where the API says `vectors`.
- Source CRS is required for rasterization/mask geometry. Missing source CRS is `missing_crs`; incompatible source/target CRS is `crs_mismatch`.
- First C5 semantic support is polygonal rasterization: `Polygon`, `MultiPolygon`, and FeatureCollections containing polygonal geometries.
- `Point`, `LineString`, `MultiPoint`, `MultiLineString`, unsupported GeoJSON types, and malformed geometry raise `unsupported_geometry_type` or `invalid_geometry`.
- Empty geometry raises `empty_geometry`; an empty vector source raises `empty_feature_set`.

### Rasterization Semantics

- `rasterize_vectors` returns:
  - `array`: NumPy array shaped `(height, width)`.
  - `info`: serialized `RasterInfo` dict for the target grid and output dtype.
  - `target_shape`: `(height, width)`.
  - `target_transform`: `(a, b, c, d, e, f)`.
  - `target_bounds`: `(left, bottom, right, top)`.
  - `dtype`, `fill`, `burned_pixels`, `all_touched`, and `warnings`.
- Default `all_touched=False` burns a pixel when its center point is inside the polygon and not inside a hole.
- `all_touched=True` burns a pixel when the pixel cell touches polygon area: center inside, any cell corner inside, any polygon vertex inside the cell, or any polygon edge intersects the cell.
- Attribute burn reads `feature["properties"][attribute]`; missing or non-numeric values fail with `invalid_argument`.
- Constant burn uses `value`.
- `fill`, `value`, and attribute values must fit the output dtype or fail with `unsupported_dtype`/`invalid_argument`.

### Geometry Mask Semantics

- `geometry_mask` internally uses the same Rust rasterization kernel to create a boolean mask; it does not call Python GIS libraries.
- Accepted `mask_polarity` values are `true_inside` and `true_outside`.
- With `mask_polarity="true_inside"` and `invert=False`, mask values are true for pixels inside geometry.
- With `invert=True`, values are flipped and the returned `mask_polarity` is flipped between `true_inside` and `true_outside`.
- The result returns `mask`, `info`, `mask_polarity`, `true_count`, `false_count`, `crop_window=None`, and `warnings`.

### Raster Mask Semantics

- `mask_raster` accepts a local raster path, a `read_raster` result dict, or a NumPy array source.
- `mask` must be an explicit boolean NumPy-compatible array.
- Accepted mask shapes:
  - `(height, width)`: applies to every band.
  - `(1, height, width)`: applies to every band.
  - `(bands, height, width)`: per-band mask.
- Any other mask shape is `shape_mismatch`.
- `mask_polarity` has no default. Missing, `None`, or unknown values raise `invalid_argument` with `mask_polarity_explicit`.
- Accepted `mask_polarity` values are `true_valid`, `true_inside`, and `true_outside`. In all cases, true cells are the cells retained by `mask_raster`.
- `fill=None` uses `nodata` when supplied, otherwise uses existing source nodata when available, otherwise leaves masked pixels unchanged but returns the output mask.
- `fill` writes masked-out cells to the fill value after validating dtype compatibility.
- `nodata` may be scalar, per-band list, or `None`; invalid values raise `invalid_nodata`.
- `crop=True` crops to the minimal retained-pixel bounding window and updates `info.width`, `info.height`, `info.transform`, `info.bounds`, and `crop_window`.
- If `crop=True` and the mask retains no pixels, return `empty_raster`.

## Stable Diagnostics

C5 implementations must include these lowercase tokens in exception messages or warning codes:

- `missing_crs`
- `crs_mismatch`
- `mask_polarity_explicit`
- `unsupported_dtype`
- `unsupported_geometry_type`
- `invalid_geometry`
- `invalid_argument`
- `empty_geometry`
- `empty_feature_set`
- `empty_raster`
- `shape_mismatch`
- `invalid_nodata`
- `backend_unavailable`

Prefer current `GisError` variants in `src/gis/error.rs:6`:

- `MissingCrs` for `missing_crs`.
- `CrsMismatch` for `crs_mismatch`.
- `UnsupportedDType` for `unsupported_dtype`.
- `InvalidGeometry` for `unsupported_geometry_type`, `invalid_geometry`, and `empty_geometry`.
- `InvalidArgument` for `invalid_argument` and `mask_polarity_explicit`.
- `ShapeMismatch` for `shape_mismatch`.
- `InvalidNodata` for `invalid_nodata`.
- `BackendUnavailable` for `backend_unavailable`.

## File-Level Deltas

| File | C5 delta |
|---|---|
| Create `src/gis/rasterize.rs` | Rust implementation for `rasterize_vectors`, `geometry_mask`, `mask_raster`, target-grid extraction, CRS compatibility, dtype parsing, mask polarity parsing, crop-window calculation, and result metadata. |
| Modify `src/gis/mod.rs` | Add `pub mod rasterize;`, re-export the three Rust functions, and re-export the three PyO3 functions behind `extension-module`. |
| Modify `src/py_module/functions/gis.rs` | Register `rasterize_vectors_py`, `geometry_mask_py`, and `mask_raster_py` after C4 GIS registrations and before raster CRS/affine registrations. |
| Modify `python/forge3d/gis.py` | Add thin wrappers only. Use `_path_or_self(vectors)` and `_path_or_self(source)` where path-like inputs are accepted. Add the three names to `__all__`. |
| Modify `python/forge3d/gis.pyi` | Add stubs for the three exact C5 signatures and `dict[str, Any]` return types. Do not advertise rasterio/Shapely/GeoPandas objects. |
| Modify `tests/test_api_contracts.py` | Move only `rasterize_vectors`, `geometry_mask`, and `mask_raster` from `LATER_GIS_FUNCTIONS` to `EXPECTED_FUNCTIONS`. Leave `normalize_raster` and `classify_raster` absent for C6. |
| Modify `tests/test_gis_raster.py` | Update wrapper `__all__` and callable assertions. Keep stub/export parity checks. |
| Create `tests/test_gis_rasterize_mask.py` | Focused T-rasterize-mask behavior tests, backend-library absence checks, and optional rasterio references. |
| Do not modify `Cargo.toml` or `pyproject.toml` | No new runtime backend dependency is needed for first C5. |

## Planning Steps

- [ ] Confirm `git status --short` does not contain uncommitted C4 source/test diffs before implementation starts.
- [ ] Run `python -m pytest tests/test_api_contracts.py -q` and confirm C1-C4 are present and C5/C6 are absent before editing.
- [ ] Keep this C5 work to the files listed in "File-Level Deltas" unless a compiler error proves a shared helper must move.
- [ ] Do not implement C6 names in C5.

## Implementation Steps

### Task C5.0: Public Surface And Backend-Free Skeleton

**Files:**
- Create: `src/gis/rasterize.rs`
- Modify: `src/gis/mod.rs`
- Modify: `src/py_module/functions/gis.rs`
- Modify: `python/forge3d/gis.py`
- Modify: `python/forge3d/gis.pyi`
- Modify: `tests/test_api_contracts.py`
- Modify: `tests/test_gis_raster.py`
- Create: `tests/test_gis_rasterize_mask.py`

- [ ] Add failing contract tests for the three C5 functions.
- [ ] Add thin Python wrappers and stubs with the exact C5 signatures.
- [ ] Add PyO3 functions that marshal arguments and initially return `backend_unavailable: C5 skeleton incomplete` only inside this first local slice.
- [ ] Run `python -m pytest tests/test_api_contracts.py tests/test_gis_raster.py -q`.
- [ ] Replace the skeleton in later C5 tasks before completion; no final C5 implementation may leave the skeleton error path for supported inputs.

### Task C5.1: Target Grid, CRS, DType, And Polarity Helpers

**Files:**
- Modify: `src/gis/rasterize.rs`
- Test: `tests/test_gis_rasterize_mask.py`

- [ ] Implement `TargetGrid` extraction from `RasterInfo` and serialized dict.
- [ ] Validate positive shape, required transform, required CRS, and `(left, bottom, right, top)` bounds.
- [ ] Parse `dtype` into existing `RasterDType`.
- [ ] Validate scalar values against dtype using `RasterDType::nodata_fits`.
- [ ] Parse mask polarity with exact accepted spellings.
- [ ] Add tests for unsupported dtype, missing target CRS, missing source CRS, CRS mismatch, and missing/invalid mask polarity.

### Task C5.2: `rasterize_vectors`

**Files:**
- Modify: `src/gis/rasterize.rs`
- Test: `tests/test_gis_rasterize_mask.py`

- [ ] Resolve `vectors` from local path, read-vector result, FeatureCollection, Feature, or geometry dict.
- [ ] Require vector CRS and target CRS compatibility.
- [ ] Reject empty vector sources with `empty_feature_set`.
- [ ] Reject empty or malformed geometry with `empty_geometry` or `invalid_geometry`.
- [ ] Implement polygonal point-in-polygon center burn.
- [ ] Implement `all_touched=True` cell-touch burn.
- [ ] Implement constant burn and attribute burn.
- [ ] Return output metadata fields listed in "Rasterization Semantics".

### Task C5.3: `geometry_mask`

**Files:**
- Modify: `src/gis/rasterize.rs`
- Test: `tests/test_gis_rasterize_mask.py`

- [ ] Reuse the rasterization kernel to build a bool array.
- [ ] Support `true_inside`, `true_outside`, and `invert`.
- [ ] Return true/false counts and `mask_polarity_explicit` warning metadata.
- [ ] Test true-inside, true-outside, and inverted masks on the same small target grid.

### Task C5.4: `mask_raster`

**Files:**
- Modify: `src/gis/rasterize.rs`
- Test: `tests/test_gis_rasterize_mask.py`

- [ ] Resolve source raster from path, `read_raster` result dict, or NumPy array.
- [ ] Validate mask shape and broadcast 2D/one-band masks to every band.
- [ ] Apply fill/nodata handling with dtype validation.
- [ ] Preserve or synthesize output `info` without CRS guessing.
- [ ] Implement crop window and transform/bounds updates with existing affine helpers.
- [ ] Ensure this function never accepts vectors or calls rasterization internally.

## Test Plan

Add `tests/test_gis_rasterize_mask.py` with these required groups:

- [ ] Polygon constant burn into an explicit small `RasterInfo` grid.
- [ ] Attribute burn from feature properties.
- [ ] Fill value and dtype handling.
- [ ] `all_touched=True` and `all_touched=False` behavior.
- [ ] `geometry_mask` `true_inside` behavior.
- [ ] `geometry_mask` `invert=True` behavior.
- [ ] Explicit `mask_polarity` requirements and bad-polarity errors.
- [ ] `mask_raster` with `true_valid` mask.
- [ ] `mask_raster` crop behavior and `crop_window`.
- [ ] `fill` and `nodata` handling, including invalid nodata.
- [ ] Shape mismatch.
- [ ] CRS mismatch.
- [ ] Missing CRS.
- [ ] Empty geometry and empty source.
- [ ] Unsupported geometry type.
- [ ] Unsupported dtype.
- [ ] Optional rasterio reference checks, skipped when rasterio is unavailable.

Contract/API tests:

- [ ] `tests/test_api_contracts.py`: add only C5 APIs to `EXPECTED_FUNCTIONS`.
- [ ] `tests/test_api_contracts.py`: keep `normalize_raster` and `classify_raster` in later/absent assertions.
- [ ] `python/forge3d/gis.py`: update `__all__`.
- [ ] `python/forge3d/gis.pyi`: add stubs.
- [ ] `tests/test_gis_raster.py`: update wrapper-surface and stub parity tests.
- [ ] Add a no-backend-Python-GIS-library test that scans `python/forge3d/gis.py` and `src/gis/rasterize.rs` for forbidden runtime imports/usages of `rasterio`, `geopandas`, `shapely`, `rioxarray`, `xarray`, and `terra`.

Optional reference-library policy:

- Optional rasterio tests must use `pytest.importorskip("rasterio")`.
- Optional reference checks compare small arrays only.
- Optional checks must not be the only coverage for any required behavior.
- No mandatory test may require rasterio, geopandas, shapely, rioxarray, xarray, or terra.

## Validation Plan For C5 Implementation

Run at minimum after implementation:

```text
cargo fmt --check
cargo check
python -m py_compile python/forge3d/gis.py
python -m pytest tests/test_api_contracts.py tests/test_gis_raster.py tests/test_gis_rasterize_mask.py -q
python -m pytest tests/test_gis_raster.py tests/test_gis_crs_affine.py tests/test_gis_alignment_windowing.py tests/test_gis_resample_reproject.py -q
python -m pytest tests/test_gis_vector_io.py tests/test_gis_vector_crs.py tests/test_gis_vector_geom.py tests/test_gis_vector_overlay.py -q
git status --short
git diff --name-status
git diff --stat
git diff
git ls-files --others --exclude-standard
```

If C5 implementation touches only Rust/Python GIS API files and focused tests, do not run GPU/rendering suites unless a change crosses into rendering code.

## Validation Plan For This Planning Change-Set

Because this change-set is documentation/planning only, run:

```text
git status --short
git diff --name-status
git diff --stat
git diff
git ls-files --others --exclude-standard
python -c "import sphinx"
python -m sphinx -b html docs docs/_build/html
```

If Sphinx is unavailable, record the docs build as skipped because the dependency is missing. No Rust/Python runtime tests are required for this planning-only diff beyond the preflight contract test already run.

## Review Bundle Requirement

After this planning change-set, create a fresh temporary review bundle outside the repo. Do not stage or commit it.

The bundle must include:

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
%TEMP%/forge3d-g002c-c5-plan-review-bundle-<timestamp>/
```

## Explicit Non-Goals

- No C6 `normalize_raster` or `classify_raster`.
- No vector writes.
- No rasterization or masking behavior in Python.
- No runtime rasterio, geopandas, shapely, rioxarray, xarray, terra, or GDAL backend.
- No rendering, shaders, visual goldens, gallery goldens, or examples.
- No remote/cache/fetch, OSM, Terrarium, slippy tiles, domain helpers, or MapScene work.
- No implicit target grid inference from vector bounds.
- No hidden vector rasterization inside `mask_raster`.

## Open Questions

None block implementation. Conservative decisions are locked above: pure-Rust default C5 backend, polygonal rasterization first, explicit target grid, strict CRS checks, explicit mask polarity, and C6 remains absent.

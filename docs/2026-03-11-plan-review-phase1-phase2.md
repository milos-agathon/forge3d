# Rigorous Review: Phase 1 & Phase 2 Implementation Plan

**Date:** 2026-03-11
**Scope:** Implementation plan (`2026-03-10-developer-platform-plan.md`) and design spec (`2026-03-10-developer-platform-design.md`), cross-checked against the actual forge3d codebase.

---

## Severity Legend

- **BLOCKER** — Will cause the task to fail or produce wrong output. Must fix before execution.
- **ERROR** — Factually wrong. An executing agent will waste time or produce broken code.
- **WARNING** — Suboptimal or misleading. Won't break execution but degrades quality.
- **STALE** — Content that contradicts the "single rendering pathway" directive.
- **NOTE** — Observation for awareness, no action required.

---

## Phase 1 Findings

### Task 0.5: Remove legacy render functions

**ERROR — File Map says "Delete" but Task says "Modify" `render.py`.** The file map (line 46) says `Delete | python/forge3d/render.py` but Task 0.5 Step 2 (line 111) says "In `python/forge3d/render.py`, remove [functions]..." This implies modifying the file, not deleting it. The file is ~2000 lines; `render_raster` and helpers are large but there may be other functions in it (e.g., `render_offscreen_rgba`, which IS exported from `__init__.py`). The plan needs to clarify: delete the entire file, or surgically remove only the three named functions? If `render_offscreen_rgba` or other non-legacy utilities live in `render.py`, deleting the whole file will break imports.

**ERROR — Missing from removal list: `__init__.py` exports `RenderView`.** The codebase's `__init__.py` currently exports `RenderView` and `ViewerWidget` from `widgets.py`. The plan's Task 9 creates a new `widgets.py` that only has `ViewerWidget`, but Task 0.5 doesn't mention cleaning up the `RenderView` export. Meanwhile, the existing `widgets.py` already exports `["RenderView", "ViewerWidget", "widgets_available"]`. The plan needs to either: (a) delete the existing `widgets.py` entirely in Task 0.5 and recreate it in Task 9, or (b) add `RenderView` removal to the Task 0.5 cleanup.

**WARNING — Step 4 lists specific test files to update, but the list may be incomplete.** The plan names 6 test files. A `grep` for `render_raster` in the actual codebase should be run at execution time — the plan should note this explicitly rather than presenting a possibly-incomplete list.

### Task 1: Python version floor bump

**ERROR — Task 1 modifies `render.py` but Task 0.5 deletes/modifies it first.** Task 1 Step 4 says "In `python/forge3d/render.py`, remove lines 25-28" (compat shims). But Task 0.5 comes before Task 1 and fundamentally alters or deletes `render.py`. If the whole file is deleted in Task 0.5, these line references are wrong. If it's surgically modified, the line numbers shift. Either way, Task 1's render.py edits need to be reconciled with Task 0.5's outcome.

**NOTE — pyproject.toml already has `requires-python = ">=3.10"`.** The codebase already has the 3.10 floor, correct classifiers, and correct URLs. Task 1's Step 3 and Task 2's URL update steps are already done. The plan should note "verify these are correct" rather than "change these" — otherwise an agent will try to edit something that's already correct and either no-op or introduce confusion.

### Task 2: Fix stale metadata

**NOTE — Already done.** `docs/conf.py` already reads version from `pyproject.toml` via regex (line 18), copyright is already `'2025-2026, forge3d contributors'`, and project URLs are already set to `forge3d.dev`, `github.com/forge3d/forge3d`, etc. This task is a no-op. The plan should state "Verify the following are correct" rather than "Replace with."

### Task 3: Add Linux aarch64 wheel build

**NOTE — Already done.** The CI file already has `linux-arm` (aarch64-unknown-linux-gnu) in the build matrix and uses `maturin-action`. The plan should verify this instead of treating it as new work.

### Task 4: PyPI publish workflow

**WARNING — Smoke test matrix doesn't include aarch64.** The publish workflow's `smoke-test` job (lines 489-520) tests Windows, Linux x86, and macOS — but not Linux aarch64. Given that aarch64 is one of 4 wheel targets and the highest-risk build, it should be included in the publish smoke test matrix (even if via QEMU).

**WARNING — No version tag validation.** The publish workflow triggers on `v*` tags but doesn't verify the tag version matches `pyproject.toml` version. A mismatched tag (e.g., `v1.14.0` but code says `1.12.2`) will publish a confusingly-versioned package. Add a step: `python -c "import tomllib; v=tomllib.load(open('pyproject.toml','rb'))['project']['version']; tag='${GITHUB_REF_NAME}'.lstrip('v'); assert v==tag, f'{v}!={tag}'"`.

### Task 5: Smoke tests

**ERROR — Tier 2 test references non-existent API.** The test calls `renderer = forge3d.Renderer(width=64, height=64)` then `renderer.render_triangle_rgba()`. Checking the actual codebase: `Renderer` IS exported, but its constructor signature and whether it has a `render_triangle_rgba()` method needs verification. The `Renderer` class is a CPU fallback renderer — it may not have this exact method. If this API doesn't exist, the test will fail at the wrong level (AttributeError instead of testing rendering). The plan should verify the actual Renderer API or use the viewer snapshot approach instead.

### Task 6: Pro boundary decision log

**NOTE — Clean.** No issues found.

---

## Phase 2 Findings

### Task 7: Documentation site restructure

**STALE — Architecture page still says "two workflows."** Line 793 in the plan's architecture.md content reads: "It provides **two workflows** for rendering, sharing the same Rust GPU engine." This directly contradicts the "single rendering pathway" directive. The viewer + headless mode should be described as one workflow (same binary, same IPC), not two.

**WARNING — Architecture page references `forge3d.memory_metrics()`.** Verify this function actually exists and is exported. It IS in `__init__.py` exports, so this is fine, but the 512 MiB budget claim should be verified against the actual code.

**WARNING — Toctree includes files that may not exist.** The index.rst references `user/installation`, `user/pbr_materials`, `user/shadows_overview`, `user/path_tracing`, `memory/index`, `offscreen/index`, `integration/matplotlib`, `integration/cartopy`, `ingest/rasterio_tiles`, `ingest/xarray`, `user/troubleshooting_visuals`. Do ALL of these currently exist? If any are missing, the Sphinx build will fail. The plan should list which files already exist and which need placeholder stubs.

### Task 8: Sample datasets module

**ERROR — Plan creates `datasets.py` and `data/` directory, but they already exist.** The codebase already has `python/forge3d/datasets.py` (10,962 bytes) with a full implementation including `mini_dem()`, `sample_boundaries()`, `fetch_dem()`, `list_datasets()`, and remote dataset registry entries for rainier, fuji, swiss, luxembourg, etc. The `python/forge3d/data/` directory likely already has the bundled files. This task is almost entirely a no-op.

The plan should be: "Verify `datasets.py` has the required API surface and data files are bundled correctly" — not "Create `python/forge3d/datasets.py`." An executing agent that follows this literally will overwrite a working, more complete implementation with the plan's simpler version.

**ERROR — Plan's datasets API differs from actual API.** The existing `datasets.py` exports different function names than the plan assumes. The actual exports are `available`, `bundled`, `remote`, `dataset_info`, `fetch`, `fetch_cityjson`, `fetch_copc`, `fetch_dem`, `list_datasets`, `mini_dem`, `mini_dem_path`, `sample_boundaries`, `sample_boundaries_path`. The plan's version has a simpler API. Again — the plan would regress the codebase.

### Task 8.5: Upload sample datasets to GitHub Releases

**WARNING — May already be done (partially).** The existing `datasets.py` already has registry entries with URLs for rainier, fuji, swiss, luxembourg, and more. Verify whether these URLs are live before treating this as new work.

### Task 9: Jupyter widget — ViewerWidget

**ERROR — Plan creates `widgets.py` but it already exists.** The codebase has `python/forge3d/widgets.py` (20,907 bytes) that already exports `ViewerWidget`, `RenderView`, and `widgets_available`. The plan would overwrite this with a much simpler implementation. This should be: "Modify `python/forge3d/widgets.py` — remove `RenderView`, keep `ViewerWidget`, verify API surface."

**ERROR — `ViewerWidget.send_ipc()` calls `self._handle._send_command()` but actual ViewerHandle method is `send_ipc()`.** In the plan's widget code (line 1443), the `send_ipc` method calls `self._handle._send_command(cmd)`. But the actual `ViewerHandle` has a public `send_ipc(cmd)` method (line 174 in viewer.py), not a private `_send_command()`. The plan should use `self._handle.send_ipc(cmd)`.

**ERROR — Tutorials also use `v._send_command()` instead of `v.send_ipc()`.** Multiple tutorial code blocks call `v._send_command({"cmd": ...})` — this is calling a private method. The actual public API is `v.send_ipc({"cmd": ...})`. Every tutorial code block that uses `_send_command` should use `send_ipc` instead. Affected locations: GIS Tutorial 1 (lines 1569, 1785, 1802, 1904, 1907), GIS Tutorial 2 (line 1594), GIS Tutorial 4 (line 1681), Python Tutorial 2 (lines 1844, 1866, 1876), Python Tutorial 3 (lines 1904, 1907).

**WARNING — Plan's ViewerWidget doesn't match actual ViewerHandle parameter names.** The plan uses `set_orbit_camera(phi, theta, distance)` in the preamble (line 15) but the actual signature is `set_orbit_camera(phi_deg, theta_deg, radius, fov_deg=None)`. The parameter name `distance` should be `radius`. Similarly, `set_sun(azimuth, elevation)` should be `set_sun(azimuth_deg, elevation_deg)`. These mismatches appear in the preamble Key API facts.

### Task 10: GIS tutorials

**ERROR — Tutorial 1 calls `open_viewer_async(terrain_path="my_dem.tif")` but `terrain_path` only accepts paths that the viewer binary can load.** The viewer uses `load_terrain` IPC which may require specific formats. Verify whether the viewer binary can directly load GeoTIFFs or only .npy files. If GeoTIFF support requires rasterio on the Python side to convert first, the tutorial is misleading.

**WARNING — GIS Tutorial 4 calls `layer.to_overlay_arrays()` but `BuildingLayer` doesn't have this method.** The actual `BuildingLayer` class has `buildings` (list), `building_count`, `total_vertices`, `total_triangles`, `bounds()` — but no `to_overlay_arrays()`. The tutorial invents an API that doesn't exist. The actual pipeline would need to iterate over `layer.buildings`, collect their `positions` and `indices` arrays, and send them individually or concatenated.

### Task 11: Python tutorials

**ERROR — Tutorial 4 (scene bundles) uses `forge3d.Scene()` which doesn't exist as a user-facing class.** The code block creates `forge3d.Scene(width=800, height=600)` then calls `.set_terrain()`, `.set_camera()`, `.set_lighting()`. But `Scene` in the actual codebase is a native type from PyO3 bindings — it's not a high-level composition API with these methods. The tutorial's code is fictional.

**WARNING — The bundle API in the tutorial doesn't match the actual API.** The plan shows `save_bundle(scene, "my_scene.forge3d")` and `load_bundle("my_scene.forge3d")` as standalone functions. The actual codebase has `save_bundle` and `load_bundle` in `viewer_ipc.py` that work with socket connections, not Scene objects. The `forge3d` package exports `save_bundle` and `load_bundle` but their signatures need verification — they likely require a viewer socket connection, not a Scene.

### Task 12: Gallery

**WARNING — Gallery Step 3 says "Run existing examples to produce full-res images."** This assumes all 10 gallery scenarios have existing example scripts. Only some do. The gallery entries for shadow comparison, SVG export, and composed map plate may not have ready-made scripts. The plan should identify which gallery entries map to existing examples and which need new scripts.

### Task 13: Notebooks

**WARNING — Task 13 Step 2 references `TerrainExplorer` widget.** Line 2100 in an older version referenced this, but the current version should reference `ViewerWidget`. Verify the notebook descriptions don't reference non-existent widgets.

### Task 14: API reference

**NOTE — Clean after corrections.** The task now correctly targets `viewer.py`, `viewer_ipc.py`, and `animation.py` instead of `render.py`.

---

## Cross-Cutting Issues

### 1. Plan Creates Files That Already Exist

The most serious systemic issue: **Tasks 8, 9, and partially Tasks 1-3 describe creating or modifying files that already exist in the codebase with working implementations.** An executing agent that follows the plan literally will overwrite working code with simpler, potentially broken versions. Each of these tasks should be reframed as "verify and update" rather than "create from scratch."

Affected existing files:
- `python/forge3d/datasets.py` — exists, 10,962 bytes, full implementation
- `python/forge3d/widgets.py` — exists, 20,907 bytes, has ViewerWidget + RenderView
- `python/forge3d/data/` — likely exists with bundled data files
- `pyproject.toml` — already has 3.10 floor, correct URLs, correct classifiers
- `docs/conf.py` — already has dynamic version, correct copyright
- `.github/workflows/ci.yml` — already has aarch64 target

### 2. Private API Usage in Tutorials (`_send_command` vs `send_ipc`)

Almost every tutorial code block uses `v._send_command({"cmd": ...})`. The underscore prefix signals a private method. The actual `ViewerHandle` has a public `send_ipc()` method that should be used instead. Tutorial code should exclusively use public API methods. This affects ~15 code blocks across Tasks 10 and 11.

### 3. Invented APIs in Tutorials

Several tutorial code blocks use APIs that don't exist in the codebase:
- `BuildingLayer.to_overlay_arrays()` — doesn't exist
- `forge3d.Scene(width, height)` with `.set_terrain()`, `.set_camera()`, `.set_lighting()` — doesn't exist as shown
- `loaded.render()` for bundles — doesn't exist

### 4. Design Spec Still Has Stale "16 targets" Reference

Line 16 of the design spec says "Pre-built wheels for 16 targets (4 platforms × 4 Python versions)". This should be "4 abi3 wheels (one per platform)." The abi3 wheel strategy means one wheel per platform covers all Python 3.10+ versions — not 16 separate wheels.

### 5. Architecture Page Contradiction

The architecture.md content in Task 7 still describes "two workflows" (interactive viewer + headless rendering) as if they're separate paths. Under the single-pathway directive, these should be presented as "one workflow, two modes" — the same viewer binary running windowed or offscreen.

---

## Summary

| Severity | Count | Key Theme |
|----------|-------|-----------|
| BLOCKER  | 0     | — |
| ERROR    | 10    | Overwrites existing code, wrong API names, fictional APIs |
| WARNING  | 10    | Stale references, incomplete lists, missing validations |
| STALE    | 1     | "Two workflows" in architecture page |
| NOTE     | 4     | Already-done tasks, observations |

The plan's biggest risks are: (1) overwriting existing working code with simpler stub implementations, (2) tutorial code blocks that reference non-existent or private APIs, and (3) a few remaining contradictions with the "single rendering pathway" directive. None are blockers — all are fixable — but an executing agent that follows the plan verbatim will hit friction at Tasks 1-3 (already done), Task 8 (datasets exist), Task 9 (widgets exist), and the tutorial tasks (wrong API calls).

# forge3d `src/` Structural Refactoring Plan

**Date:** 2026-03-15
**Status:** Re-baselined against the live tree after the terrain renderer second pass
**Scope:** Structural refactors in `src/` only. No intentional behavior, API, or shader changes.
**Primary source of truth:** `docs/notes/codebase-refactor.md`
**Supporting source:** `AGENTS.md`

This revision replaces stale task sizing, removes infeasible split strategies, and turns the plan into an execution-ready campaign.

## 0. Intent

The goal is to reduce structural complexity in `src/` while preserving:

- Rust public surfaces
- Python import and runtime behavior
- WGSL bind group and uniform layout compatibility

This is a refactor campaign, not a feature campaign. Any proposed API redesign, feature-flag change, shader edit, or behavioral cleanup belongs in a separate plan.

## 1. Governing Rules

### 1.1 Repo guardrails

These come directly from `docs/notes/codebase-refactor.md` and `AGENTS.md`:

- Tests define behavior. Prefer Python tests as the acceptance gate.
- Work in small steps. Extract helpers before introducing new structure.
- Keep Rust, Python, and WGSL in sync when a change crosses those boundaries.
- Preserve public API unless a separate expand/contract plan exists.
- Respect feature gates and PyO3 registration requirements.
- Prefer targeted validation during work; reserve full-suite validation for task completion and merge readiness.

### 1.2 Campaign conventions

These are plan-level goals for this campaign:

- Target `<=300` LOC for ordinary Rust files.
- Treat `<=300` as a campaign objective, not as a repo-wide rule baked into the playbook.
- If a PyO3 wrapper file cannot reach `<=300` without a separate build-configuration decision, keep the wrapper thin and record the exception explicitly.

### 1.3 Hard constraints discovered during re-assessment

1. `Scene` cannot be split into dozens of separate `#[pymethods]` blocks under the current crate configuration.
   `Cargo.toml` enables `pyo3` macros but not `multiple-pymethods`, so the safe default is one `#[pymethods]` block per class.

2. Splitting a single large method across two files is not a pure move.
   `Scene::new`, `Scene::render_png`, `Scene::render_rgba`, and `TerrainRenderParams` accessors must first be decomposed into private helpers with explicit inputs and outputs.

3. The old plan mixed exact line maps with stale file versions.
   That level of detail is only valid after re-reading the live file immediately before the split.

4. Phase tables must reflect the current tree exactly.
   Carrying forward old rows makes scheduling and metrics misleading.

## 2. Current Inventory

Inventory below is from the live tree on 2026-03-15.

- Total Rust files in `src/`: `750`
- Total LOC in `src/`: `144,807`
- Files `>300` LOC: `137`
- Files `>600` LOC: `29`
- Files `>1000` LOC: `0`
- Files `301-600` LOC: `108`
- Largest file: `src/path_tracing/compute.rs` at `935` LOC

### 2.1 Files over 1,000 LOC

None.

Phase 1's structural objective is now closed: the live tree has zero Rust files above 1,000 LOC. The remaining queue is entirely in the 601-1000 and 301-600 bands.

### 2.2 Files in the 601-1000 LOC band

These are the exact current Phase 2 candidates.

| File | LOC |
|------|-----|
| `path_tracing/compute.rs` | 935 |
| `vector/point/renderer.rs` | 867 |
| `core/clouds/renderer.rs` | 827 |
| `vector/api.rs` | 820 |
| `path_tracing/hybrid_compute.rs` | 818 |
| `core/memory_tracker.rs` | 817 |
| `terrain/page_table.rs` | 806 |
| `style/expressions.rs` | 772 |
| `viewer/event_loop/stdin_reader.rs` | 771 |
| `viewer/terrain/render/screen.rs` | 768 |
| `core/water_surface.rs` | 756 |
| `core/dual_source_oit.rs` | 742 |
| `path_tracing/wavefront/queues.rs` | 728 |
| `viewer/render/main_loop/geometry.rs` | 724 |
| `viewer/terrain/render/offscreen.rs` | 699 |
| `core/screen_space_effects/ssr/constructor.rs` | 692 |
| `viewer/terrain/dof.rs` | 672 |
| `core/screen_space_effects/ssgi/constructor.rs` | 668 |
| `viewer/viewer_enums.rs` | 666 |
| `render/material_set.rs` | 657 |
| `core/bloom.rs` | 649 |
| `import/cityjson.rs` | 638 |
| `offscreen/brdf_tile.rs` | 635 |
| `render/params/config/tests.rs` | 632 |
| `viewer/terrain/overlay.rs` | 626 |
| `core/compressed_textures.rs` | 620 |
| `viewer/state/viewer_helpers.rs` | 618 |
| `viewer/terrain/shader_pbr.rs` | 613 |
| `viewer/pointcloud.rs` | 609 |

### 2.3 Files explicitly not in Phase 2 anymore

These appeared in the old draft but are now `<=600` LOC and belong in Phase 3 only:

- `core/virtual_texture.rs` - 590
- `accel/cpu_bvh.rs` - 557
- `terrain/bloom_processor.rs` - 580
- `core/texture_upload.rs` - 556
- `core/tile_cache.rs` - 528
- `vector/indirect.rs` - 541
- `core/scene_graph.rs` - 522
- `viewer/terrain/pbr_renderer.rs` - 575
- `terrain/renderer.rs` - 56
- `terrain/renderer/draw.rs` - 523
- `terrain/renderer/resources.rs` - 551
- `terrain/renderer/bind_groups.rs` - 6
- `terrain/renderer/shadows.rs` - 5

## 3. Validation Strategy

### 3.1 Per-commit fast gate

Run after every atomic move:

```powershell
cargo fmt -- --check
cargo check -q --all-features
```

Then run the smallest relevant Python test slice for the touched domain.

Examples:

```powershell
pytest -q tests/test_p5*.py
pytest -q tests/test_terrain*.py tests/test_t*.py
pytest -q tests/test_b*.py tests/test_lighting*.py
pytest -q tests/test_api_contracts.py tests/smoke_test.py
```

### 3.2 PyO3 gate

Run only when touching `#[pyclass]`, `#[pymethods]`, `#[pyfunction]`, or `_forge3d` registration:

```powershell
maturin develop --release
python -c "import forge3d; print(forge3d.__version__)"
pytest -q tests/test_api_contracts.py tests/smoke_test.py
```

### 3.3 Feature-gate checks

Run when a task touches `#[cfg(feature = ...)]` code, `lib.rs` registration, or maturin-facing surfaces:

```powershell
cargo check --all-features
cargo check --no-default-features
```

### 3.4 High/critical task completion gate

Run after finishing a Phase 1 task and before merge:

```powershell
cargo build --all-features
python -m compileall -q python/
pytest -q tests/
```

Use `python examples/terrain_demo.py` only for terrain-facing tasks.

## 4. Execution Pattern

Every large-file split follows the same shape:

1. Re-read the live file and regenerate the task-local symbol map.
2. Add the destination module skeleton (`mod.rs` and empty leaf files).
3. Move leaf constants, POD structs, and pure helpers first.
4. Extract large method bodies into private helpers.
5. Move public wrappers last.
6. Re-export from the existing module so downstream callers do not change.
7. Validate after each commit with the smallest credible gate.

Do not start by moving entire public API blocks blindly.

## 5. Phase 0 - Baseline

Before any refactor work:

```powershell
Get-Location
git rev-parse --show-toplevel
git rev-parse HEAD
git status --porcelain
```

Capture the live file inventory:

```powershell
Get-ChildItem -Recurse -Filter *.rs src |
  ForEach-Object {
    [PSCustomObject]@{
      Lines = (Get-Content $_.FullName | Measure-Object -Line).Lines
      File = $_.FullName
    }
  } |
  Sort-Object Lines -Descending
```

If the worktree is clean, create a branch for the campaign. If the worktree is dirty, snapshot state and coordinate with the existing branch rather than forcing a new branch.

## 6. Phase 1 - Files over 1,000 LOC

### 6.1 `src/scene/mod.rs` - 6438 LOC - Critical

**Current reality**

- One `Scene` pyclass
- One large `#[pymethods] impl Scene`
- One trailing plain `impl Scene`
- Large single methods for construction and readback paths

**Required strategy**

- Keep one `#[pymethods]` block unless a separate prerequisite explicitly enables `pyo3/multiple-pymethods`.
- Move implementation bodies into private helpers first.
- Keep the Python-facing wrappers thin.

**Target shape**

- `scene/mod.rs` - orchestrator and re-exports
- `scene/types.rs` - `SceneGlobals`, small shared types/constants
- `scene/ssao.rs` - `SsaoResources`, SSAO uniform types, SSAO helpers
- `scene/texture_helpers.rs` - texture creation helpers
- `scene/core.rs` - `Scene` struct and constructor-side private helpers
- `scene/render_paths.rs` - common render-path helpers shared by PNG/RGBA
- `scene/postfx_cpu.rs` - CPU-side post-fx helpers
- `scene/stats.rs` - stats collection/formatting helpers
- `scene/private_impl.rs` - plain `impl Scene` internals
- `scene/py_api.rs` - the sole `#[pymethods] impl Scene`

**Execution note**

The first pass is to make `py_api.rs` thin, not to force a 37-file layout from a stale line map.

**Validation**

- PyO3 gate
- `pytest -q tests/test_api_contracts.py tests/smoke_test.py`
- Scene-facing tests such as `tests/test_aov.py`, `tests/test_dof.py`, `tests/test_bloom_effect.py`, and `tests/test_ssgi_ssr_wiring.py`

### 6.2 `src/terrain/renderer.rs` - 5605 LOC - Critical

**Current reality**

- `TerrainScene` is the real core type
- `TerrainRenderer` is the PyO3 wrapper
- Large concentrations of uniforms, pipeline cache, shadow resources, water reflection, AOV, and upload/draw logic

**Target shape**

- `terrain/renderer/mod.rs` - orchestrator and re-exports
- `terrain/renderer/core.rs` - `TerrainScene`, `ViewerTerrainData`
- `terrain/renderer/py_api.rs` - `TerrainRenderer` wrapper and `#[pymethods]`
- `terrain/renderer/uniforms.rs`
- `terrain/renderer/msaa.rs`
- `terrain/renderer/pipeline_cache.rs`
- `terrain/renderer/bind_groups.rs`
- `terrain/renderer/shadows.rs`
- `terrain/renderer/water_reflection.rs`
- `terrain/renderer/aov.rs`
- `terrain/renderer/height_ao.rs`
- `terrain/renderer/upload.rs`
- `terrain/renderer/draw.rs`

**Validation**

- `pytest -q tests/test_terrain*.py tests/test_t*.py`
- `python examples/terrain_demo.py`

**Live status (2026-03-15)**

- Parent `terrain/renderer.rs` is now a 56 LOC orchestrator.
- `terrain/renderer/draw.rs` was added for the main render path.
- `terrain/renderer/bind_groups.rs` and `terrain/renderer/shadows.rs` were split into submodules and are now thin stubs.
- Remaining terrain-renderer cleanup is now Phase 3-band work (`draw.rs` 523 LOC, `resources.rs` 551 LOC, `height_ao.rs` 495 LOC, `water_reflection.rs` 443 LOC).

### 6.3 `src/lib.rs` - 4623 LOC - Critical

**Current reality**

The old draft materially under-scoped this file. `lib.rs` contains:

- multiple `#[pyclass]` types
- many `#[pyfunction]` exports across vector, path tracing, viewer, diagnostics, BRDF, CSM, and point-cloud surfaces
- the `_forge3d` module registration function

**Target shape**

- `lib.rs` - crate root plus `_forge3d` entry point
- `py_types/frame.rs`
- `py_types/screen_space_gi.rs`
- `py_types/picking.rs`
- `py_types/styles.rs`
- `py_types/pointcloud.rs`
- `py_types/aov.rs`
- `py_functions/viewer.rs`
- `py_functions/vector.rs`
- `py_functions/path_tracing.rs`
- `py_functions/csm.rs`
- `py_functions/diagnostics.rs`
- `py_functions/brdf.rs`
- `py_functions/pointcloud.rs`
- `py_module/classes.rs`
- `py_module/functions.rs`
- `py_module/mod.rs`

**Execution note**

Before moving code, inventory every `#[pyclass]` and `#[pyfunction]` in the live file. Do not rely on the old 16-file estimate.

**Validation**

- PyO3 gate
- `pytest -q tests/test_api_contracts.py tests/smoke_test.py`

### 6.4 `src/core/screen_space_effects.rs` - 4460 LOC - Critical

**Current reality**

This file contains four major subsystems:

- `HzbPyramid`
- SSAO
- SSGI
- SSR
- `ScreenSpaceEffectsManager`

The old draft omitted the manager, which made its split map incomplete.

**Target shape**

- `core/screen_space/mod.rs`
- `core/screen_space/hzb.rs`
- `core/screen_space/settings.rs`
- `core/screen_space/ssao.rs`
- `core/screen_space/ssgi.rs`
- `core/screen_space/ssr.rs`
- `core/screen_space/manager.rs`
- `core/screen_space/shared_blur.rs`
- `core/screen_space/uniforms.rs`

Use additional leaf files only after the first structural split compiles cleanly.

**Validation**

- `pytest -q tests/test_p5*.py`

### 6.5 `src/terrain/mod.rs` - 2695 LOC - High

Split according to actual live symbols, not generic terrain labels:

- lights (`PointLight`, `SpotLight`)
- `ColormapLUT`
- `TerrainUniforms`
- `Globals`
- `TerrainSpike` wrapper
- mesh/build helpers

Keep `TerrainSpike` PyO3 methods isolated from pure terrain helpers.

### 6.6 `src/terrain/render_params.rs` - 2326 LOC - High

**Current reality**

- many native settings structs and `Default` impls
- one large `TerrainRenderParams` `#[pymethods]` block

**Target shape**

- `terrain/render_params/mod.rs`
- `terrain/render_params/parse.rs`
- `terrain/render_params/native_lighting.rs`
- `terrain/render_params/native_material.rs`
- `terrain/render_params/native_effects.rs`
- `terrain/render_params/native_overlays.rs`
- `terrain/render_params/native_postfx.rs`
- `terrain/render_params/core.rs`
- `terrain/render_params/private_impl.rs`
- `terrain/render_params/py_api.rs`

Apply the same PyO3 rule as `Scene`: thin `py_api.rs`, helper-heavy internals.

### 6.7 `src/core/ibl.rs` - 1671 LOC - High

Split by actual responsibilities:

- renderer/types
- prefilter
- irradiance
- BRDF LUT
- cache/image IO
- shared image utilities

Do not rename `IBLRenderer` to a different conceptual type as part of the split.

### 6.8 `src/pipeline/pbr.rs` - 1570 LOC - High

Split by actual live responsibilities:

- tone mapping
- textures
- materials
- uniforms
- IBL fallback resources
- `PbrPipelineWithShadows`

Do not plan around a nonexistent `PbrPipeline` rename.

### 6.9 `src/lighting/py_bindings.rs` - 1263 LOC - High

Split by current wrapper families:

- `light.rs`
- `material.rs`
- `shadow.rs`
- `gi.rs`
- `atmosphere.rs`
- `screen_space.rs`
- `sky.rs`
- `volumetrics.rs`
- `sun_position.rs`
- `mod.rs`

This file is a good candidate for a clean per-class split because it already has many separate `#[pyclass]` surfaces.

### 6.10 Remaining Phase 1 files

These require live mini-plans before execution:

| File | LOC |
|------|-----|
| `viewer/terrain/render.rs` | 2518 |
| `viewer/terrain/scene.rs` | 2032 |
| `viewer/cmd/handler.rs` | 1788 |
| `viewer/render/main_loop.rs` | 1712 |
| `viewer/terrain/vector_overlay.rs` | 1488 |
| `viewer/ipc/protocol.rs` | 1363 |
| `path_tracing/wavefront/pipeline.rs` | 1207 |
| `path_tracing/wavefront/mod.rs` | 1141 |
| `accel/lbvh_gpu.rs` | 1001 |

Rule for all of them: write a task-local move map from the live file immediately before moving code.

## 7. Phase 2 - Exact 601-1000 LOC Queue

Split these only after Phase 1 stabilizes:

| File | LOC | First split boundary |
|------|-----|----------------------|
| `path_tracing/compute.rs` | 935 | pipeline vs dispatch vs uniforms |
| `vector/point/renderer.rs` | 867 | init vs upload vs draw |
| `core/clouds/renderer.rs` | 827 | init vs render vs uniforms |
| `vector/api.rs` | 820 | public API by vector operation family |
| `path_tracing/hybrid_compute.rs` | 818 | setup vs dispatch |
| `core/memory_tracker.rs` | 817 | registry vs reporting vs policy-free helpers |
| `terrain/page_table.rs` | 806 | table bookkeeping vs GPU upload |
| `style/expressions.rs` | 772 | parser vs evaluator |
| `viewer/event_loop/stdin_reader.rs` | 771 | I/O vs parsing |
| `viewer/terrain/render/screen.rs` | 768 | camera/setup vs terrain pass vs overlay/effects chain |
| `core/water_surface.rs` | 756 | init vs simulation vs render |
| `core/dual_source_oit.rs` | 742 | pipeline vs pass execution |
| `path_tracing/wavefront/queues.rs` | 728 | queue types vs queue management |
| `viewer/render/main_loop/geometry.rs` | 724 | geometry pass vs fog path vs debug/fallback helpers |
| `viewer/terrain/render/offscreen.rs` | 699 | snapshot scene pass vs snapshot OIT vs post-process chain |
| `core/screen_space_effects/ssr/constructor.rs` | 692 | bind group layouts vs pipeline creation vs texture/resource init |
| `viewer/terrain/dof.rs` | 672 | setup vs execution |
| `core/screen_space_effects/ssgi/constructor.rs` | 668 | bind group layouts vs pipeline creation vs texture/history init |
| `viewer/viewer_enums.rs` | 666 | split by domain |
| `render/material_set.rs` | 657 | host-side set management vs GPU ops |
| `core/bloom.rs` | 649 | pipelines vs passes |
| `import/cityjson.rs` | 638 | parsing vs geometry generation |
| `offscreen/brdf_tile.rs` | 635 | rendering vs helpers/tests |
| `render/params/config/tests.rs` | 632 | fixtures vs assertions |
| `viewer/terrain/overlay.rs` | 626 | types vs render path |
| `core/compressed_textures.rs` | 620 | formats vs upload |
| `viewer/state/viewer_helpers.rs` | 618 | helper families |
| `viewer/terrain/shader_pbr.rs` | 613 | shader generation vs pipeline glue |
| `viewer/pointcloud.rs` | 609 | load vs render |

## 8. Phase 3 - 301-600 LOC Cleanup

There are currently `90` files in this band.

Phase 3 policy:

- split opportunistically when touching a file for other work
- absorb the eight files that fell out of Phase 2
- keep Phase 3 out of the critical-path campaign until Phase 1 and Phase 2 are stable

## 9. Sequencing and Parallelism

### 9.1 Recommended order

1. Phase 0 baseline
2. `scene/mod.rs`
3. `lib.rs`
4. `core/screen_space_effects.rs`
5. `terrain/render_params.rs`
6. `lighting/py_bindings.rs`
7. `terrain/renderer.rs`
8. `terrain/mod.rs`
9. remaining Phase 1 files
10. Phase 2 exact queue
11. Phase 3 opportunistic cleanup

### 9.2 Parallelism limits

Do not run these concurrently:

- `scene/mod.rs`
- `lib.rs`
- `terrain/render_params.rs`
- `lighting/py_bindings.rs`

Reason: they are PyO3-heavy and have a high chance of stepping on shared Python-facing validation and registration surfaces.

Safe parallel lanes after the first critical tasks stabilize:

- terrain lane: `terrain/renderer.rs`, `terrain/mod.rs`
- viewer lane: viewer-only files
- path tracing lane: wavefront and compute files
- rendering-core lane: `core/ibl.rs`, `pipeline/pbr.rs`

## 10. Definition of Done

A task is done when all of the following are true:

1. The live task inventory was refreshed before work started.
2. Public Rust and Python surfaces are unchanged unless explicitly approved elsewhere.
3. All impacted sync pairs remain consistent.
4. `cargo fmt -- --check` passes.
5. `cargo check -q --all-features` passes.
6. Relevant domain tests pass.
7. PyO3 tasks also pass `maturin develop --release` plus API-contract smoke checks.
8. High/critical tasks also pass full `pytest -q tests/` before merge.
9. The refactor result reaches the `<=300` target for ordinary Rust files.
10. Any remaining PyO3 wrapper exception is thin, justified, and recorded explicitly.
11. `git status --porcelain` shows only intended changes.

## 11. Stop Conditions

Stop and re-plan if any of the following happen:

1. The split requires a public API change.
2. The task needs a new Cargo feature or PyO3 feature toggle that was not planned.
3. A helper extraction changes shader content or bind group layout.
4. A file thought to be "structure only" also contains feature-gated registration or platform-specific logic.
5. The task-local symbol map does not match the plan's expected structure.
6. Validation failures spread outside the touched domain.

## 12. Plan Maintenance Rules

To keep this plan accurate while the tree evolves:

- Regenerate the inventory at the start of every phase.
- When a file drops below a threshold, move it to the lower phase immediately.
- When a task uncovers a new hard constraint, add the rule here before continuing.
- Do not add speculative target files unless they are grounded in the live source.

This document is now the execution baseline. Any future edits should preserve that standard.

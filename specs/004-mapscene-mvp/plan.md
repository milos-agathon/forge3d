# Implementation Plan: MapScene MVP

**Branch**: `004-mapscene-mvp` | **Date**: 2026-05-15 | **Spec**: `specs/004-mapscene-mvp/spec.md`
**Input**: Feature specification from `specs/004-mapscene-mvp/spec.md`

## Summary

Introduce the typed `MapScene` / `SceneRecipe` MVP workflow that validates a complete offline 3D map scene, compiles deterministic labels, renders PNG output, and saves a reproducible review bundle for supported layer types. The implementation wraps existing forge3d substrate in `python/forge3d/helpers/offscreen.py`, `python/forge3d/bundle.py`, `python/forge3d/terrain_params.py`, `style.py`, `buildings.py`, `pointcloud.py`, `crs.py`, `map_plate.py`, and native `Scene.render_png`/`render_rgba` where available. The selected public product API path is `python/forge3d/map_scene.py`, exported through `python/forge3d/__init__.py` and `python/forge3d/__init__.pyi`.

## Technical Context

**Language/Version**: Python 3.x public API with Rust/PyO3 render substrate.
**Primary Dependencies**: features `001` diagnostics, `002` label truth, `003` LabelPlan, terrain/offscreen/viewer/bundle/style/building/point-cloud/CRS helpers.
**Storage**: PNG output files and deterministic review bundle metadata.
**Testing**: pytest for typed construction, validation, render policy, bundle save, examples; native tests only when Rust render or binding code changes.
**Target Platform**: Offline Python map rendering.
**Project Type**: Product-layer library API.
**Performance Goals**: MVP scenes render reproducibly under fixed profile; validation cheap enough to run before render.
**Constraints**: No raw IPC required for canonical MVP examples; no implicit CRS transforms; warnings continue by default; exact PNG comparison unless a numeric tolerance is recorded before verification.
**Scale/Scope**: P0-R4, Section 15, Section 21 MVP, Milestone 3.

## Constitution Check

- [x] PRD traceability: P0-R4-AC1 through P0-R4-AC8 and Section 21 MVP must-include list.
- [x] Offline map scope: typed offline 3D map scene, PNG, bundle; no web/live globe scope.
- [x] API truthfulness: validate/render/save_bundle produce real report/output/bundle or typed diagnostics.
- [x] Determinism: recipe serialization, validation ordering, label plans, bundles, examples, and PNG fixtures fixed by reproducibility profile.
- [x] Diagnostics first: validation covers CRS, glyphs, style, support status, memory, fallback/pro-gated paths before render.
- [x] Typed product contract: centers `MapScene`, `SceneRecipe`, `LabelPlan`, `ValidationReport`, `Diagnostic`, and `Bundle`.
- [x] Evidence plan: acceptance mapping below.
- [x] Support wording: P1/P2/unavailable capabilities represented as intent plus diagnostics, not support claims.
- [x] Compatibility: wraps existing substrate; no raw IPC in MVP user workflow.
- [x] Continuity: implementation updates matrix, ledger, context pack.

## Project Structure

```text
specs/004-mapscene-mvp/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── mapscene-contract.md
└── tasks.md
```

Existing paths to inspect during tasks:

```text
python/forge3d/__init__.py
python/forge3d/__init__.pyi
python/forge3d/viewer.py
python/forge3d/helpers/offscreen.py
python/forge3d/terrain_params.py
python/forge3d/style.py
python/forge3d/buildings.py
python/forge3d/pointcloud.py
python/forge3d/bundle.py
python/forge3d/crs.py
python/forge3d/map_plate.py
src/scene/
src/labels/
src/import/
src/tiles3d/
src/pointcloud/
src/bundle/
examples/
tests/
docs/
```

**Structure Decision**: `MapScene` should be a top-level product contract that wraps existing `Scene`/viewer/offscreen substrate internally. Tasks must decide exact files after inspecting exports. For B-004, default render path should prefer the native/offscreen PNG path for MVP; viewer snapshot may remain optional/diagnosed if not required for canonical examples.

## Test Strategy First

1. Typed construction tests for `SceneRecipe`, `TerrainSource`, `RasterOverlay`, `VectorOverlay`, `LabelLayer`, `PointCloudLayer`, `BuildingLayer` intent, `MapFurnitureLayer`, camera, lighting, and `OutputSpec`.
2. Validation report tests for structured report, CRS mismatch, missing glyphs, unsupported style fields/types, pro-gated/fallback paths, incomplete 3D Tiles, VT unsupported families, memory estimate, and label rejection summary.
3. Render policy tests: warnings continue by default, fail-on-warning blocks warnings, error/fatal block always.
4. PNG render integration test for a supported deterministic fixture with fixed reproducibility profile and exact comparison unless tolerance is recorded first.
5. Bundle save tests comparing deterministic manifest/recipe/diagnostics/label-plan metadata for supported layer types.
6. Negative no-op tests for `validate`, `render`, and `save_bundle`.
7. Canonical example smoke tests for terrain+raster, terrain+vector+labels, and terrain+buildings+labels or honest building diagnostics.
8. Docs audit for MVP quickstart and support matrices.

## Implementation Strategy

1. Define typed recipe/data models and validation rules before render wiring.
2. Build validation adapters around existing CRS, style, glyph/label, memory, building, point-cloud, VT, and bundle metadata.
3. Integrate `LabelPlan.compile()` for point/polygon label layers and map furniture keepouts.
4. Wire MVP PNG rendering through the inspected native/offscreen path; if a layer cannot render honestly, validation must block or diagnose before output success.
5. Extend bundle save path or create bundle-ready metadata sidecar through existing `bundle.py` after inspecting schema and Pro gating.
6. Add canonical examples using `MapScene` only; no direct `viewer_ipc`.
7. Update docs and support matrices with exact support classifications.

## Acceptance Mapping

| PRD AC | Components | Tests | Docs | Diagnostics |
|---|---|---|---|---|
| P0-R4-AC1 | `MapScene.validate`, `ValidationReport` | validation contract tests | quickstart | all MVP codes |
| P0-R4-AC2 | `MapScene.render` PNG | PNG fixture test | quickstart | render-blocking diagnostics |
| P0-R4-AC3 | `save_bundle` | deterministic bundle tests | bundle docs | serialized diagnostics |
| P0-R4-AC4 | typed recipe objects | construction tests | API docs | unsupported layer intent |
| P0-R4-AC5 | CRS validation | CRS mismatch tests | CRS note | `crs_mismatch` |
| P0-R4-AC6 | memory estimate | memory estimate tests | diagnostics ref | `estimated_gpu_memory` |
| P0-R4-AC7 | support status adapters | negative diagnostics tests | support matrices | pro/fallback/experimental/unsupported |
| P0-R4-AC8 | examples | smoke tests | three examples | example diagnostics |
| Section 21 | end-to-end workflow | MVP workflow review | Offline Map Quickstart | glyph/style/CRS diagnostics |

## Migration And Compatibility

This is additive. Existing low-level terrain/viewer/bundle/style/building/point-cloud APIs remain. `MapScene` must not hide Pro gating or fallbacks. Older bundles without diagnostics should still load through existing APIs; `MapScene` bundles must include deterministic diagnostics and recipe intent.

## Verification Matrix And Ledger

P0-R4 and Section 21 rows cannot advance until typed construction, validation, PNG, bundle, examples, docs, diagnostics, and no-raw-IPC evidence exist. Exact commands and artifact paths must be recorded in `docs/superpowers/state/requirements-verification-matrix.md`, the ledger, and context pack.

## Rollback And Safe Failure

If render preparation encounters unsupported/pro-gated/fallback features, render must return/block with diagnostics rather than write success. If bundle save cannot preserve required state, it must fail with diagnostics and avoid a misleading review artifact. If deterministic PNG comparison is not stable, record numeric tolerance before verification or keep the row unverified.

# Tasks: MapScene MVP

**Input**: `specs/004-mapscene-mvp/plan.md`, `spec.md`, `research.md`, `data-model.md`, `quickstart.md`, and `contracts/mapscene-contract.md`

**Scope**: P0-R4, PRD Section 15, Section 21 MVP, Milestone 3, typed recipe validation/render/bundle workflow, canonical examples, and required continuity state updates.

**Required task format**: Every task below includes ID, title, PRD tags, files, action, verification, done evidence, and dependencies. The first checklist line preserves the Spec Kit task ID and `[P]` marker where work can be safely parallelized.

**Execution rule**: Tests must be written before or alongside source changes. `validate()`, `render()`, and `save_bundle()` must produce real reports, PNGs, bundles, or typed diagnostics; no raw IPC is allowed in canonical MVP examples.

## Phase 1: Product API Path and Typed Recipe

- [X] T001 [PRD: P0-R4-AC1, P0-R4-AC4, P0-R4-AC7] Inspect MapScene ownership paths and P0 dependencies.
  - ID: T001
  - Title: Inspect MapScene ownership paths
  - PRD tags: [P0-R4-AC1] [P0-R4-AC4] [P0-R4-AC7]
  - Files: expected `python/forge3d/map_scene.py`, expected `python/forge3d/diagnostics.py`, expected `python/forge3d/label_plan.py`, `python/forge3d/__init__.py`, `python/forge3d/__init__.pyi`, `python/forge3d/helpers/offscreen.py`, `python/forge3d/bundle.py`, `docs/superpowers/state/implementation-ledger.md`
  - Action: Confirm product API path, public exports, diagnostics/LabelPlan prerequisites, and native/offscreen render path before implementation; record any missing P0 dependency as a blocker rather than duplicating contracts.
  - Verification: `rg -n "class (MapScene|SceneRecipe|ValidationReport|LabelPlan)|def render_png|render_rgba|save_bundle|viewer_ipc" python src tests docs/superpowers/state`
  - Done evidence: Ledger records chosen source paths, render path, bundle path, and P0 dependency status.
  - Dependencies: None

- [X] T002 [P] [PRD: P0-R4-AC4] Add typed recipe construction tests.
  - ID: T002
  - Title: Add MapScene recipe construction tests
  - PRD tags: [P0-R4-AC4]
  - Files: `tests/test_mapscene_recipe_contract.py`, expected `python/forge3d/map_scene.py`, `python/forge3d/__init__.pyi`
  - Action: Add tests constructing `MapScene`, `SceneRecipe`, `TerrainSource`, `RasterOverlay`, `VectorOverlay`, `LabelLayer`, `BuildingLayer`, `PointCloudLayer`, `MapFurnitureLayer`, camera, lighting, and `OutputSpec`.
  - Verification: `pytest tests/test_mapscene_recipe_contract.py -q`
  - Done evidence: Tests fail until every required recipe component is public and typed.
  - Dependencies: T001

- [X] T003 [PRD: P0-R4-AC4] Implement typed recipe models and exports.
  - ID: T003
  - Title: Implement typed MapScene recipe models
  - PRD tags: [P0-R4-AC4]
  - Files: `python/forge3d/map_scene.py`, `python/forge3d/__init__.py`, `python/forge3d/__init__.pyi`
  - Action: Implement the typed recipe classes, deterministic serialization-ready fields, and public exports without wiring render success for unsupported layers.
  - Verification: `pytest tests/test_mapscene_recipe_contract.py -q`
  - Done evidence: Construction tests pass and all required layer/camera/lighting/output types are represented.
  - Dependencies: T002

## Phase 2: Validation and Diagnostics

- [ ] T004 [P] [PRD: P0-R4-AC1, P0-R4-AC5, P0-R4-AC6, P0-R4-AC7] Add MapScene validation report tests.
  - ID: T004
  - Title: Add MapScene validation tests
  - PRD tags: [P0-R4-AC1] [P0-R4-AC5] [P0-R4-AC6] [P0-R4-AC7]
  - Files: `tests/test_mapscene_validation.py`, expected `python/forge3d/map_scene.py`, expected `python/forge3d/diagnostics.py`, `python/forge3d/crs.py`, `python/forge3d/style.py`, `python/forge3d/mem.py`
  - Action: Add tests for structured `ValidationReport` before render, CRS mismatch without implicit transforms, missing glyphs, unsupported style fields/types, memory estimate, unsupported layer intent, and warning/error/fatal blocking status.
  - Verification: `pytest tests/test_mapscene_validation.py -q`
  - Done evidence: Tests fail until validation produces structured reports and typed diagnostics before render.
  - Dependencies: T003

- [ ] T005 [P] [PRD: P0-R4-AC7] Add unsupported/pro-gated/fallback/no-op validation tests.
  - ID: T005
  - Title: Add MapScene support-status negative tests
  - PRD tags: [P0-R4-AC7]
  - Files: `tests/test_mapscene_support_status.py`, expected `python/forge3d/map_scene.py`, `python/forge3d/buildings.py`, `python/forge3d/tiles3d.py`, `python/forge3d/terrain_params.py`
  - Action: Add negative tests for unsupported, experimental, Pro-gated, placeholder/fallback, VT unsupported family, incomplete public 3D Tiles, building fallback, and no-op validate/render/save paths.
  - Verification: `pytest tests/test_mapscene_support_status.py -q`
  - Done evidence: Tests fail if unavailable layers are silently ignored or marked renderable.
  - Dependencies: T003

- [ ] T006 [PRD: P0-R4-AC1, P0-R4-AC5, P0-R4-AC6, P0-R4-AC7] Implement MapScene validation adapters.
  - ID: T006
  - Title: Implement MapScene validation
  - PRD tags: [P0-R4-AC1] [P0-R4-AC5] [P0-R4-AC6] [P0-R4-AC7]
  - Files: `python/forge3d/map_scene.py`, expected `python/forge3d/diagnostics.py`, `python/forge3d/crs.py`, `python/forge3d/style.py`, `python/forge3d/buildings.py`, `python/forge3d/pointcloud.py`, `python/forge3d/tiles3d.py`, `python/forge3d/terrain_params.py`, `python/forge3d/mem.py`
  - Action: Implement pre-render validation for recipe structure, CRS, glyphs, style support, memory estimate, layer support status, P1/P2 intent diagnostics, and deterministic report ordering.
  - Verification: `pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py -q`
  - Done evidence: Validation tests pass and unsupported/pro-gated/fallback paths return typed diagnostics.
  - Dependencies: T004, T005

- [ ] T007 [P] [PRD: P0-R4-AC1, P0-R4-AC4, P0-R4-AC7] Add LabelPlan integration tests.
  - ID: T007
  - Title: Add MapScene LabelPlan integration tests
  - PRD tags: [P0-R4-AC1] [P0-R4-AC4] [P0-R4-AC7]
  - Files: `tests/test_mapscene_label_plan_integration.py`, expected `python/forge3d/map_scene.py`, expected `python/forge3d/label_plan.py`
  - Action: Add tests proving `LabelLayer` integrates deterministic `LabelPlan.compile()` for point/polygon labels, map furniture keepouts feed the plan, and label rejection summaries surface in validation.
  - Verification: `pytest tests/test_mapscene_label_plan_integration.py -q`
  - Done evidence: Tests fail until label plans are compiled or typed diagnostics are returned.
  - Dependencies: T003

- [ ] T008 [PRD: P0-R4-AC1, P0-R4-AC4, P0-R4-AC7] Implement LabelPlan and map furniture integration.
  - ID: T008
  - Title: Implement MapScene LabelPlan integration
  - PRD tags: [P0-R4-AC1] [P0-R4-AC4] [P0-R4-AC7]
  - Files: `python/forge3d/map_scene.py`, expected `python/forge3d/label_plan.py`, expected `python/forge3d/diagnostics.py`, `python/forge3d/map_plate.py`
  - Action: Wire `LabelLayer` and `MapFurnitureLayer` to deterministic label planning, preserving compiled plan data and diagnostics for validation, render prep, and bundle save.
  - Verification: `pytest tests/test_mapscene_label_plan_integration.py tests/test_mapscene_validation.py -q`
  - Done evidence: Label integration tests pass and rejected labels are visible in reports.
  - Dependencies: T006, T007

## Phase 3: PNG Render and Bundle Save

- [ ] T009 [P] [PRD: P0-R4-AC2, P0-R4-AC7] Add deterministic PNG render tests.
  - ID: T009
  - Title: Add MapScene PNG render tests
  - PRD tags: [P0-R4-AC2] [P0-R4-AC7]
  - Files: `tests/test_mapscene_render_png.py`, expected `python/forge3d/map_scene.py`, `python/forge3d/helpers/offscreen.py`, `tests/fixtures/`
  - Action: Add a supported fixture render test with fixed recipe inputs, camera, lighting, output size, seed, and exact PNG comparison unless a numeric tolerance is recorded before verification.
  - Verification: `pytest tests/test_mapscene_render_png.py -q`
  - Done evidence: Tests fail until `render()` writes PNG only for non-blocked supported scenes and returns diagnostics otherwise.
  - Dependencies: T006, T008

- [ ] T010 [P] [PRD: P0-R4-AC2, P0-R4-AC7] Add render policy and render no-op tests.
  - ID: T010
  - Title: Add MapScene render policy tests
  - PRD tags: [P0-R4-AC2] [P0-R4-AC7]
  - Files: `tests/test_mapscene_render_policy.py`, expected `python/forge3d/map_scene.py`, expected `python/forge3d/diagnostics.py`
  - Action: Add tests proving render performs validation if needed, warnings continue by default, fail-on-warning blocks warnings, errors/fatals block always, and blocked render does not write misleading PNG output.
  - Verification: `pytest tests/test_mapscene_render_policy.py -q`
  - Done evidence: Tests fail on render success after blocking diagnostics or no-op PNG writes.
  - Dependencies: T006

- [ ] T011 [PRD: P0-R4-AC2, P0-R4-AC7] Implement PNG render path.
  - ID: T011
  - Title: Implement MapScene PNG render
  - PRD tags: [P0-R4-AC2] [P0-R4-AC7]
  - Files: `python/forge3d/map_scene.py`, `python/forge3d/helpers/offscreen.py`, inspect `src/scene/`, inspect `python/forge3d/viewer.py`
  - Action: Wire `MapScene.render(path)` through the inspected native/offscreen PNG path for supported MVP scenes, applying validation and render policy before writing output.
  - Verification: `pytest tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py -q`
  - Done evidence: PNG and policy tests pass with no raw IPC requirement.
  - Dependencies: T009, T010

- [ ] T012 [P] [PRD: P0-R4-AC3, P0-R4-AC7] Add deterministic bundle save tests.
  - ID: T012
  - Title: Add MapScene bundle save tests
  - PRD tags: [P0-R4-AC3] [P0-R4-AC7]
  - Files: `tests/test_mapscene_save_bundle.py`, expected `python/forge3d/map_scene.py`, `python/forge3d/bundle.py`
  - Action: Add tests comparing deterministic review bundle manifest/recipe/diagnostics/compiled label plan/source references/camera/lighting/output/layer metadata for supported layer types.
  - Verification: `pytest tests/test_mapscene_save_bundle.py -q`
  - Done evidence: Tests fail on missing reproducibility data, lost diagnostics, or misleading bundle writes after blocking diagnostics.
  - Dependencies: T006, T008

- [ ] T013 [PRD: P0-R4-AC3, P0-R4-AC7] Implement reproducible bundle save.
  - ID: T013
  - Title: Implement MapScene save_bundle
  - PRD tags: [P0-R4-AC3] [P0-R4-AC7]
  - Files: `python/forge3d/map_scene.py`, `python/forge3d/bundle.py`, expected `python/forge3d/diagnostics.py`
  - Action: Implement `save_bundle(path)` for supported MVP layer metadata and diagnostics, preserving deterministic scene intent and refusing misleading review artifacts for unsupported states.
  - Verification: `pytest tests/test_mapscene_save_bundle.py -q`
  - Done evidence: Bundle save tests pass with stable serialized intent.
  - Dependencies: T012

## Phase 4: Canonical Examples and Docs

- [ ] T014 [P] [PRD: P0-R4-AC8, P0-R4-AC1, P0-R4-AC2, P0-R4-AC3] Add canonical example smoke tests.
  - ID: T014
  - Title: Add MapScene canonical example tests
  - PRD tags: [P0-R4-AC8] [P0-R4-AC1] [P0-R4-AC2] [P0-R4-AC3]
  - Files: `tests/test_mapscene_examples.py`, expected `examples/mapscene_terrain_raster.py`, expected `examples/mapscene_vector_labels.py`, expected `examples/mapscene_buildings_labels.py`
  - Action: Add smoke tests for terrain+raster, terrain+vector+labels, and terrain+buildings+labels examples; assert validation, PNG render or typed diagnostics, bundle save, and no direct `viewer_ipc`.
  - Verification: `pytest tests/test_mapscene_examples.py -q`
  - Done evidence: Example tests fail until all three examples exist and avoid raw IPC.
  - Dependencies: T011, T013

- [ ] T015 [PRD: P0-R4-AC8, P0-R4-AC1, P0-R4-AC2, P0-R4-AC3, P0-R4-AC7] Implement canonical MapScene examples.
  - ID: T015
  - Title: Implement MapScene examples
  - PRD tags: [P0-R4-AC8] [P0-R4-AC1] [P0-R4-AC2] [P0-R4-AC3] [P0-R4-AC7]
  - Files: `examples/mapscene_terrain_raster.py`, `examples/mapscene_vector_labels.py`, `examples/mapscene_buildings_labels.py`
  - Action: Create examples using `MapScene` only, with terrain+raster render/bundle, vector+labels deterministic plan, and buildings+labels render or honest Pro-gated/placeholder/unsupported diagnostics.
  - Verification: `pytest tests/test_mapscene_examples.py -q`
  - Done evidence: Smoke tests pass and `rg -n "viewer_ipc" examples/mapscene_terrain_raster.py examples/mapscene_vector_labels.py examples/mapscene_buildings_labels.py` finds no matches.
  - Dependencies: T014

- [ ] T016 [P] [PRD: P0-R4-AC8, P0-R4-AC7] Add MVP docs/support audit tests.
  - ID: T016
  - Title: Add MapScene docs audit tests
  - PRD tags: [P0-R4-AC8] [P0-R4-AC7]
  - Files: `tests/test_mapscene_docs.py`, `docs/guides/offline_3d_map_rendering.md`, `docs/guides/diagnostics_reference.md`, `docs/guides/style_support_matrix.md`, `docs/guides/building_support_matrix.md`, `docs/guides/tiles3d_support_matrix.md`, `docs/guides/virtual_texturing_support_matrix.md`, `docs/guides/competitive_positioning.md`
  - Action: Add docs audit tests for Offline 3D Map Rendering Quickstart, LabelPlan references, support matrices, diagnostics reference, competitive positioning, no overclaim wording, and MVP workflow evidence.
  - Verification: `pytest tests/test_mapscene_docs.py -q`
  - Done evidence: Docs audit fails until required MVP docs and support matrices are present and honest.
  - Dependencies: T001

- [ ] T017 [PRD: P0-R4-AC8, P0-R4-AC7, P0-R4-AC1] Update MVP docs and support matrices.
  - ID: T017
  - Title: Update MapScene MVP docs
  - PRD tags: [P0-R4-AC8] [P0-R4-AC7] [P0-R4-AC1]
  - Files: `docs/guides/offline_3d_map_rendering.md`, `docs/guides/diagnostics_reference.md`, `docs/guides/style_support_matrix.md`, `docs/guides/building_support_matrix.md`, `docs/guides/tiles3d_support_matrix.md`, `docs/guides/virtual_texturing_support_matrix.md`, `docs/guides/competitive_positioning.md`, `docs/index.rst`
  - Action: Document typed scene recipe workflow, validation, PNG render, bundle save, examples, diagnostics, support classifications, and explicit non-goals.
  - Verification: `pytest tests/test_mapscene_docs.py -q`
  - Done evidence: Docs audit passes and docs avoid overclaiming MVT, 3D Tiles, textured PBR, VT normal/mask, or browser delivery.
  - Dependencies: T016

## Phase 5: Verification and Continuity

- [ ] T018 [P] [PRD: P0-R4-AC1, P0-R4-AC2, P0-R4-AC3, P0-R4-AC8] Add quickstart validation tests.
  - ID: T018
  - Title: Add MapScene quickstart validation tests
  - PRD tags: [P0-R4-AC1] [P0-R4-AC2] [P0-R4-AC3] [P0-R4-AC8]
  - Files: `tests/test_mapscene_quickstart.py`, `specs/004-mapscene-mvp/quickstart.md`, expected `python/forge3d/map_scene.py`
  - Action: Add automated checks for terrain+raster, terrain+vector+labels, terrain+buildings+labels or honest diagnostics, CRS mismatch, unsupported style fields, missing glyphs, and bundle diagnostics.
  - Verification: `pytest tests/test_mapscene_quickstart.py -q`
  - Done evidence: Quickstart scenarios pass or produce typed diagnostics for intentionally unavailable paths.
  - Dependencies: T015, T017

- [ ] T019 [PRD: P0-R4-AC1, P0-R4-AC2, P0-R4-AC3, P0-R4-AC4, P0-R4-AC5, P0-R4-AC6, P0-R4-AC7, P0-R4-AC8] Run full MapScene MVP verification.
  - ID: T019
  - Title: Run full MapScene MVP verification
  - PRD tags: [P0-R4-AC1] [P0-R4-AC2] [P0-R4-AC3] [P0-R4-AC4] [P0-R4-AC5] [P0-R4-AC6] [P0-R4-AC7] [P0-R4-AC8]
  - Files: `tests/test_mapscene_recipe_contract.py`, `tests/test_mapscene_validation.py`, `tests/test_mapscene_support_status.py`, `tests/test_mapscene_label_plan_integration.py`, `tests/test_mapscene_render_png.py`, `tests/test_mapscene_render_policy.py`, `tests/test_mapscene_save_bundle.py`, `tests/test_mapscene_examples.py`, `tests/test_mapscene_docs.py`, `tests/test_mapscene_quickstart.py`
  - Action: Run the full MapScene feature test set and record the result summary.
  - Verification: `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q`
  - Done evidence: All feature tests pass or blockers are recorded with exact failing commands.
  - Dependencies: T018

- [ ] T020 [PRD: P0-R4-AC1, P0-R4-AC2, P0-R4-AC3, P0-R4-AC4, P0-R4-AC5, P0-R4-AC6, P0-R4-AC7, P0-R4-AC8] Update requirements verification matrix.
  - ID: T020
  - Title: Update requirements verification matrix
  - PRD tags: [P0-R4-AC1] [P0-R4-AC2] [P0-R4-AC3] [P0-R4-AC4] [P0-R4-AC5] [P0-R4-AC6] [P0-R4-AC7] [P0-R4-AC8]
  - Files: `docs/superpowers/state/requirements-verification-matrix.md`
  - Action: Update P0-R4 rows with exact source, tests, docs, diagnostics, examples, PNG, bundle, and command evidence; do not mark `Verified` without successful verification output.
  - Verification: `rg -n "P0-R4-AC[1-8]|004-mapscene-mvp|tests/test_mapscene|examples/mapscene" docs/superpowers/state/requirements-verification-matrix.md`
  - Done evidence: Matrix rows reference concrete artifacts and statuses match evidence.
  - Dependencies: T019

- [ ] T021 [PRD: P0-R4-AC1, P0-R4-AC3, P0-R4-AC7, P0-R4-AC8] Update implementation ledger and current context pack.
  - ID: T021
  - Title: Update continuity state
  - PRD tags: [P0-R4-AC1] [P0-R4-AC3] [P0-R4-AC7] [P0-R4-AC8]
  - Files: `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/current-context-pack.md`
  - Action: Record completed tasks, verification commands, validation/render/bundle evidence, example evidence, support decisions, diagnostics, and residual blockers.
  - Verification: `rg -n "004-mapscene-mvp|P0-R4|MapScene|ValidationReport|save_bundle|viewer_ipc" docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md`
  - Done evidence: Ledger and context pack contain current outcomes and no unverified success claims.
  - Dependencies: T020

## Dependencies and Execution Order

- T001 blocks source implementation.
- T002, T004, T005, T007, T009, T010, T012, T014, T016, and T018 can be authored in parallel after their listed prerequisites because they target separate test/docs files.
- Implementation follows tests: T003 after T002, T006 after T004/T005, T008 after T007, T011 after T009/T010, T013 after T012, T015 after T014, and T017 after T016.
- T020 and T021 happen after verification evidence exists.

## PRD Coverage Check

- P0-R4-AC1: T001, T004, T006, T007, T008, T014, T015, T017, T018, T019, T020, T021
- P0-R4-AC2: T009, T010, T011, T014, T015, T018, T019, T020
- P0-R4-AC3: T012, T013, T014, T015, T018, T019, T020, T021
- P0-R4-AC4: T001, T002, T003, T007, T008, T019, T020
- P0-R4-AC5: T004, T006, T019, T020
- P0-R4-AC6: T004, T006, T019, T020
- P0-R4-AC7: T001, T004, T005, T006, T007, T008, T009, T010, T011, T012, T013, T015, T016, T017, T019, T020, T021
- P0-R4-AC8: T014, T015, T016, T017, T018, T019, T020, T021

## Detailed Pre-Implementation Coverage Audit Table

| PRD requirement | Acceptance criterion | Task IDs covering it | Test task IDs | Docs task IDs | Verification command | Risk if omitted |
|---|---|---|---|---|---|---|
| P0-R4 MapScene MVP | P0-R4-AC1 structured `validate()` report | T001, T004, T006, T007, T008, T014, T015, T017, T018, T019, T020, T021 | T004, T007, T014, T018 | T017, T020, T021 | `pytest tests/test_mapscene_validation.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q` | Users cannot diagnose scene issues before render. |
| P0-R4 MapScene MVP | P0-R4-AC2 PNG output | T009, T010, T011, T014, T015, T018, T019, T020 | T009, T010, T014, T018 | T017, T020 | `pytest tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q` | MVP lacks a real reproducible render output path. |
| P0-R4 MapScene MVP | P0-R4-AC3 reproducible review bundle | T012, T013, T014, T015, T018, T019, T020, T021 | T012, T014, T018 | T017, T020, T021 | `pytest tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q` | Reviewers cannot reproduce scene configuration or diagnostics. |
| P0-R4 MapScene MVP | P0-R4-AC4 typed recipe coverage | T001, T002, T003, T007, T008, T019, T020 | T002, T007 | T017, T020 | `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_label_plan_integration.py -q` | Users still stitch terrain, overlays, labels, camera, lighting, output, and assets manually. |
| P0-R4 MapScene MVP | P0-R4-AC5 CRS mismatch detection | T004, T006, T019, T020 | T004 | T017, T020 | `pytest tests/test_mapscene_validation.py -q` | Layers can render in the wrong coordinate space without pre-render failure. |
| P0-R4 MapScene MVP | P0-R4-AC6 GPU memory estimate | T004, T006, T019, T020 | T004 | T017, T020 | `pytest tests/test_mapscene_validation.py -q` | Oversized scenes have no pre-render resource-risk evidence. |
| P0-R4 MapScene MVP | P0-R4-AC7 unsupported features typed diagnostics | T001, T004, T005, T006, T007, T008, T009, T010, T011, T012, T013, T015, T016, T017, T019, T020, T021 | T004, T005, T007, T009, T010, T012, T016 | T016, T017, T020, T021 | `pytest tests/test_mapscene_support_status.py tests/test_mapscene_validation.py tests/test_mapscene_render_policy.py tests/test_mapscene_docs.py -q` | Unsupported, Pro-gated, placeholder, or experimental features can silently fall back. |
| P0-R4 MapScene MVP | P0-R4-AC8 three canonical examples | T014, T015, T016, T017, T018, T019, T020, T021 | T014, T016, T018 | T016, T017, T020, T021 | `pytest tests/test_mapscene_examples.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q` | MVP cannot prove the typed workflow works for required scene recipes. |
| Section 21 MVP must include | `MapScene` typed API | T001, T002, T003, T019, T020 | T002 | T017, T020 | `pytest tests/test_mapscene_recipe_contract.py -q` | MVP remains a scattered low-level API set. |
| Section 21 MVP must include | Terrain source | T002, T003, T014, T015, T018, T019, T020 | T002, T014, T018 | T017, T020 | `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q` | The canonical offline terrain workflow is not represented. |
| Section 21 MVP must include | Raster overlay | T002, T003, T014, T015, T018, T019, T020 | T002, T014, T018 | T017, T020 | `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q` | Terrain+raster scenes cannot be authored through the typed recipe. |
| Section 21 MVP must include | Vector overlay | T002, T003, T014, T015, T018, T019, T020 | T002, T014, T018 | T017, T020 | `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q` | Vector styling stays outside the MVP scene contract. |
| Section 21 MVP must include | Basic label layer | T002, T003, T007, T008, T014, T015, T018, T019, T020 | T002, T007, T014, T018 | T017, T020 | `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_examples.py -q` | Label-bearing MVP examples cannot use typed map-scene state. |
| Section 21 MVP must include | Deterministic `LabelPlan` for point and polygon labels | T007, T008, T014, T015, T018, T019, T020 | T007, T014, T018 | T017, T020 | `pytest tests/test_mapscene_label_plan_integration.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q` | MapScene can bypass deterministic label planning. |
| Section 21 MVP must include | Camera and output spec | T002, T003, T009, T014, T015, T018, T019, T020 | T002, T009, T014, T018 | T017, T020 | `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_render_png.py tests/test_mapscene_examples.py -q` | Reproducible rendering cannot be fixed by recipe inputs. |
| Section 21 MVP must include | Validation report | T004, T006, T018, T019, T020, T021 | T004, T018 | T017, T020, T021 | `pytest tests/test_mapscene_validation.py tests/test_mapscene_quickstart.py -q` | Validation evidence is unavailable before render. |
| Section 21 MVP must include | PNG render output | T009, T010, T011, T014, T015, T018, T019, T020 | T009, T010, T014, T018 | T017, T020 | `pytest tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_examples.py -q` | MVP cannot produce a real output artifact. |
| Section 21 MVP must include | Bundle save | T012, T013, T014, T015, T018, T019, T020, T021 | T012, T014, T018 | T017, T020, T021 | `pytest tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q` | Scene review/reproduction evidence is lost. |
| Section 21 MVP must include | Style support diagnostics | T004, T005, T006, T016, T017, T018, T019, T020 | T004, T005, T016, T018 | T016, T017, T020 | `pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q` | Unsupported style fields or layer types can be silently ignored. |
| Section 21 MVP must include | Missing-glyph diagnostics | T004, T006, T007, T008, T018, T019, T020 | T004, T007, T018 | T017, T020 | `pytest tests/test_mapscene_validation.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_quickstart.py -q` | Glyph problems are found only after render or not at all. |
| Section 21 MVP must include | CRS mismatch diagnostics | T004, T006, T018, T019, T020 | T004, T018 | T017, T020 | `pytest tests/test_mapscene_validation.py tests/test_mapscene_quickstart.py -q` | Incorrect CRS combinations can proceed as apparently valid scenes. |

## Audit Rerun Result

- PASS: Every P0-R4 acceptance criterion and every Section 21 MVP must-include item has task coverage, test IDs, docs/support IDs, and a verification command.
- PASS: `validate()`, `render()`, and `save_bundle()` tasks require real reports, PNGs, bundles, or typed diagnostics; no no-op success path is permitted.
- PASS: Public API and example tasks are paired with support-level documentation and raw-IPC avoidance checks.

## Extension Hooks

**Optional Hook**: git  
Command: `/speckit.git.commit`  
Description: Auto-commit after task generation  

Prompt: Commit task changes?  
To execute: `/speckit.git.commit`

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

- [X] T004 [P] [PRD: P0-R4-AC1, P0-R4-AC5, P0-R4-AC6, P0-R4-AC7] Add MapScene validation report tests.
  - ID: T004
  - Title: Add MapScene validation tests
  - PRD tags: [P0-R4-AC1] [P0-R4-AC5] [P0-R4-AC6] [P0-R4-AC7]
  - Files: `tests/test_mapscene_validation.py`, expected `python/forge3d/map_scene.py`, expected `python/forge3d/diagnostics.py`, `python/forge3d/crs.py`, `python/forge3d/style.py`, `python/forge3d/mem.py`
  - Action: Add tests for structured `ValidationReport` before render, CRS mismatch without implicit transforms, missing glyphs, unsupported style fields/types, memory estimate, unsupported layer intent, and warning/error/fatal blocking status.
  - Verification: `pytest tests/test_mapscene_validation.py -q`
  - Done evidence: Tests fail until validation produces structured reports and typed diagnostics before render.
  - Dependencies: T003

- [X] T005 [P] [PRD: P0-R4-AC7] Add unsupported/pro-gated/fallback/no-op validation tests.
  - ID: T005
  - Title: Add MapScene support-status negative tests
  - PRD tags: [P0-R4-AC7]
  - Files: `tests/test_mapscene_support_status.py`, expected `python/forge3d/map_scene.py`, `python/forge3d/buildings.py`, `python/forge3d/tiles3d.py`, `python/forge3d/terrain_params.py`
  - Action: Add negative tests for unsupported, experimental, Pro-gated, placeholder/fallback, VT unsupported family, incomplete public 3D Tiles, building fallback, and no-op validate/render/save paths.
  - Verification: `pytest tests/test_mapscene_support_status.py -q`
  - Done evidence: Tests fail if unavailable layers are silently ignored or marked renderable.
  - Dependencies: T003

- [X] T006 [PRD: P0-R4-AC1, P0-R4-AC5, P0-R4-AC6, P0-R4-AC7] Implement MapScene validation adapters.
  - ID: T006
  - Title: Implement MapScene validation
  - PRD tags: [P0-R4-AC1] [P0-R4-AC5] [P0-R4-AC6] [P0-R4-AC7]
  - Files: `python/forge3d/map_scene.py`, expected `python/forge3d/diagnostics.py`, `python/forge3d/crs.py`, `python/forge3d/style.py`, `python/forge3d/buildings.py`, `python/forge3d/pointcloud.py`, `python/forge3d/tiles3d.py`, `python/forge3d/terrain_params.py`, `python/forge3d/mem.py`
  - Action: Implement pre-render validation for recipe structure, CRS, glyphs, style support, memory estimate, layer support status, P1/P2 intent diagnostics, and deterministic report ordering.
  - Verification: `pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py -q`
  - Done evidence: Validation tests pass and unsupported/pro-gated/fallback paths return typed diagnostics.
  - Dependencies: T004, T005

- [X] T006A [PRD: P0-R4-AC1, P0-R4-AC4, P0-R4-AC7] Add and implement source-data validation remediation.
  - ID: T006A
  - Title: Add MapScene source identity and renderable-data diagnostics
  - PRD tags: [P0-R4-AC1] [P0-R4-AC4] [P0-R4-AC7]
  - Files: `tests/test_mapscene_validation.py`, `python/forge3d/map_scene.py`
  - Action: Add tests and implementation for missing terrain/raster/point-cloud source identity plus vector/label layers without renderable data.
  - Verification: `pytest tests/test_mapscene_validation.py -q`
  - Done evidence: Selected red/green run proved the tests initially failed with status `ok`; selected green run reported `22 passed`, and broader selected run reported `25 passed`.
  - Dependencies: T006

- [X] T006B [PRD: P0-R4-AC1, P0-R4-AC3, P0-R4-AC7, Section 21] Add missing/unsupported external asset diagnostics tests.
  - ID: T006B
  - Title: Add MapScene external asset diagnostics tests
  - PRD tags: [P0-R4-AC1] [P0-R4-AC3] [P0-R4-AC7] [Section 21]
  - Files: `tests/test_mapscene_validation.py`, `tests/test_mapscene_save_bundle.py`
  - Action: Add red tests proving terrain, raster, and point-cloud paths that reference missing local assets or unsupported asset formats emit typed diagnostics before render, and that bundle save preserves those diagnostics without claiming renderability.
  - Verification: `pytest tests/test_mapscene_validation.py tests/test_mapscene_save_bundle.py -q`
  - Done evidence: Red run reported `3 failed, 10 passed` because missing local assets and unsupported suffixes returned `ok`; final selected command `pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_save_bundle.py -q` reported `17 passed`.
  - Dependencies: T006A

- [X] T006C [PRD: P0-R4-AC1, P0-R4-AC3, P0-R4-AC7, Section 21] Implement missing/unsupported external asset diagnostics.
  - ID: T006C
  - Title: Implement MapScene external asset diagnostics
  - PRD tags: [P0-R4-AC1] [P0-R4-AC3] [P0-R4-AC7] [Section 21]
  - Files: `python/forge3d/map_scene.py`, `tests/test_mapscene_validation.py`, `tests/test_mapscene_support_status.py`, `tests/test_mapscene_label_plan_integration.py`, `tests/test_mapscene_render_png.py`, `tests/test_mapscene_render_policy.py`, `tests/test_mapscene_save_bundle.py`, `tests/test_mapscene_quickstart.py`, `examples/mapscene_terrain_raster.py`, `examples/mapscene_vector_labels.py`, `examples/mapscene_buildings_labels.py`
  - Action: Add `missing_external_asset` and `unsupported_asset_format` validation for terrain, raster, and point-cloud assets; make symbolic test/example fixture references explicit with `asset_status: fixture`; keep blocked bundles non-renderable.
  - Verification: `pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_save_bundle.py -q`
  - Done evidence: Focused selected command reported `17 passed`; combined selected MapScene command reported `37 passed`; `python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_validation.py tests\test_mapscene_save_bundle.py` exited 0.
  - Dependencies: T006B

- [X] T006D [PRD: P0-R4-AC1, P0-R4-AC5, P0-R4-AC7, Section 21] Add terrain CRS and building source truth red tests.
  - ID: T006D
  - Title: Add MapScene terrain CRS and building source diagnostics tests
  - PRD tags: [P0-R4-AC1] [P0-R4-AC5] [P0-R4-AC7] [Section 21]
  - Files: `tests/test_mapscene_validation.py`
  - Action: Add red tests proving terrain with missing CRS metadata and supported building layers with missing or unsupported source assets cannot validate as `ok`.
  - Verification: `pytest tests/test_mapscene_validation.py tests/test_mapscene_docs.py -q`
  - Done evidence: Red command `python -m pytest tests/test_mapscene_validation.py tests/test_mapscene_docs.py -q -p no:cacheprovider` reported `3 failed, 12 passed`, proving missing terrain CRS, supported building source asset failures, and diagnostics-reference inventory gaps were accepted/undocumented.
  - Dependencies: T006C

- [X] T006E [PRD: P0-R4-AC1, P0-R4-AC5, P0-R4-AC7, Section 21] Implement terrain CRS and building source diagnostics.
  - ID: T006E
  - Title: Implement MapScene terrain CRS and building source diagnostics
  - PRD tags: [P0-R4-AC1] [P0-R4-AC5] [P0-R4-AC7] [Section 21]
  - Files: `python/forge3d/map_scene.py`, `tests/test_mapscene_validation.py`
  - Action: Emit typed `missing_crs`, `missing_external_asset`, and `unsupported_asset_format` diagnostics for terrain and building source inputs before render, preserving deterministic layer summaries and support levels.
  - Verification: `pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py -q`
  - Done evidence: Implemented typed `missing_crs`, `missing_external_asset`, and `unsupported_asset_format` coverage for missing terrain CRS and supported building source paths. Focused validation/docs command reported `15 passed`; validation/support/docs command reported `19 passed`; combined selected regression reported `26 passed`; `py_compile` exited 0.
  - Dependencies: T006D

- [X] T007 [P] [PRD: P0-R4-AC1, P0-R4-AC4, P0-R4-AC7] Add LabelPlan integration tests.
  - ID: T007
  - Title: Add MapScene LabelPlan integration tests
  - PRD tags: [P0-R4-AC1] [P0-R4-AC4] [P0-R4-AC7]
  - Files: `tests/test_mapscene_label_plan_integration.py`, expected `python/forge3d/map_scene.py`, expected `python/forge3d/label_plan.py`
  - Action: Add tests proving `LabelLayer` integrates deterministic `LabelPlan.compile()` for point/polygon labels, map furniture keepouts feed the plan, and label rejection summaries surface in validation.
  - Verification: `pytest tests/test_mapscene_label_plan_integration.py -q`
  - Done evidence: Tests fail until label plans are compiled or typed diagnostics are returned.
  - Dependencies: T003

- [X] T008 [PRD: P0-R4-AC1, P0-R4-AC4, P0-R4-AC7] Implement LabelPlan and map furniture integration.
  - ID: T008
  - Title: Implement MapScene LabelPlan integration
  - PRD tags: [P0-R4-AC1] [P0-R4-AC4] [P0-R4-AC7]
  - Files: `python/forge3d/map_scene.py`, expected `python/forge3d/label_plan.py`, expected `python/forge3d/diagnostics.py`, `python/forge3d/map_plate.py`
  - Action: Wire `LabelLayer` and `MapFurnitureLayer` to deterministic label planning, preserving compiled plan data and diagnostics for validation, render prep, and bundle save.
  - Verification: `pytest tests/test_mapscene_label_plan_integration.py tests/test_mapscene_validation.py -q`
  - Done evidence: Label integration tests pass and rejected labels are visible in reports.
  - Dependencies: T006, T007

## Phase 3: PNG Render and Bundle Save

- [X] T009 [P] [PRD: P0-R4-AC2, P0-R4-AC7] Add deterministic PNG render tests.
  - ID: T009
  - Title: Add MapScene PNG render tests
  - PRD tags: [P0-R4-AC2] [P0-R4-AC7]
  - Files: `tests/test_mapscene_render_png.py`, expected `python/forge3d/map_scene.py`, `python/forge3d/helpers/offscreen.py`, `tests/fixtures/`
  - Action: Add a supported fixture render test with fixed recipe inputs, camera, lighting, output size, seed, and exact PNG comparison unless a numeric tolerance is recorded before verification.
  - Verification: `pytest tests/test_mapscene_render_png.py -q`
  - Done evidence: `tests/test_mapscene_render_png.py` was added; red run failed on the unwired `render()` method; selected green run reported `22 passed`, and broader selected run reported `25 passed`.
  - Dependencies: T006, T008

- [X] T010 [P] [PRD: P0-R4-AC2, P0-R4-AC7] Add render policy and render no-op tests.
  - ID: T010
  - Title: Add MapScene render policy tests
  - PRD tags: [P0-R4-AC2] [P0-R4-AC7]
  - Files: `tests/test_mapscene_render_policy.py`, expected `python/forge3d/map_scene.py`, expected `python/forge3d/diagnostics.py`
  - Action: Add tests proving render performs validation if needed, warnings continue by default, fail-on-warning blocks warnings, errors/fatals block always, and blocked render does not write misleading PNG output.
  - Verification: `pytest tests/test_mapscene_render_policy.py -q`
  - Done evidence: `tests/test_mapscene_render_policy.py` was added; red run failed on unwired render behavior; selected green run reported `22 passed`, and broader selected run reported `25 passed`.
  - Dependencies: T006

- [X] T011 [PRD: P0-R4-AC2, P0-R4-AC7] Implement PNG render path.
  - ID: T011
  - Title: Implement MapScene PNG render
  - PRD tags: [P0-R4-AC2] [P0-R4-AC7]
  - Files: `python/forge3d/map_scene.py`, `python/forge3d/helpers/offscreen.py`, inspect `src/scene/`, inspect `python/forge3d/viewer.py`
  - Action: Wire `MapScene.render(path)` through the inspected native/offscreen PNG path for supported MVP scenes, applying validation and render policy before writing output.
  - Verification: `pytest tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py -q`
  - Done evidence: `MapScene.render()` now validates, enforces warning/error/fatal policy, writes deterministic PNG via `python/forge3d/helpers/offscreen.py`, and selected commands reported `22 passed` / `25 passed`.
  - Dependencies: T009, T010

- [X] T011A [PRD: P0-R4-AC2, P0-R4-AC7, Constitution III] Add recipe-sensitive render regression test.
  - ID: T011A
  - Title: Add MapScene recipe-sensitive render regression test
  - PRD tags: [P0-R4-AC2] [P0-R4-AC7] [Constitution III]
  - Files: `tests/test_mapscene_render_png.py`
  - Action: Add a regression test proving different supported terrain/raster/vector recipe data cannot produce byte-identical PNG output under the same output size and seed.
  - Verification: `pytest tests/test_mapscene_render_png.py -q`
  - Done evidence: Red run failed because two different supported recipes produced byte-identical PNG bytes; green run reported `3 passed`.
  - Dependencies: T011

- [X] T011B [PRD: P0-R4-AC2, P0-R4-AC7, Constitution III] Implement recipe-sensitive deterministic MVP renderer.
  - ID: T011B
  - Title: Implement recipe-sensitive MapScene render output
  - PRD tags: [P0-R4-AC2] [P0-R4-AC7] [Constitution III]
  - Files: `python/forge3d/map_scene.py`
  - Action: Replace the data-insensitive fallback PNG path with a deterministic MVP compositor whose bytes depend on terrain, raster, vector, label plan, point-cloud, camera, lighting, output, and reproducibility recipe data.
  - Verification: `pytest tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py -q`
  - Done evidence: Focused render command reported `3 passed`; selected policy/render command is part of the final verification checkpoint.
  - Dependencies: T011A

- [X] T012 [P] [PRD: P0-R4-AC3, P0-R4-AC7] Add deterministic bundle save tests.
  - ID: T012
  - Title: Add MapScene bundle save tests
  - PRD tags: [P0-R4-AC3] [P0-R4-AC7]
  - Files: `tests/test_mapscene_save_bundle.py`, expected `python/forge3d/map_scene.py`, `python/forge3d/bundle.py`
  - Action: Add tests comparing deterministic review bundle manifest/recipe/diagnostics/compiled label plan/source references/camera/lighting/output/layer metadata for supported layer types.
  - Verification: `pytest tests/test_mapscene_save_bundle.py -q`
  - Done evidence: `tests/test_mapscene_save_bundle.py` was added; red run failed on unwired `save_bundle()`; selected green run reported `22 passed`, and broader selected run reported `25 passed`.
  - Dependencies: T006, T008

- [X] T013 [PRD: P0-R4-AC3, P0-R4-AC7] Implement reproducible bundle save.
  - ID: T013
  - Title: Implement MapScene save_bundle
  - PRD tags: [P0-R4-AC3] [P0-R4-AC7]
  - Files: `python/forge3d/map_scene.py`, `python/forge3d/bundle.py`, expected `python/forge3d/diagnostics.py`
  - Action: Implement `save_bundle(path)` for supported MVP layer metadata and diagnostics, preserving deterministic scene intent and refusing misleading review artifacts for unsupported states.
  - Verification: `pytest tests/test_mapscene_save_bundle.py -q`
  - Done evidence: `MapScene.save_bundle()` now writes deterministic review metadata, recipe intent, validation report, label plans, and label source references; blocked scenes are recorded as non-renderable. Selected commands reported `22 passed` / `25 passed`.
  - Dependencies: T012

- [X] T013A [PRD: P0-R4-AC3, Section 21, Constitution VII] Add non-label bundle source metadata tests.
  - ID: T013A
  - Title: Add MapScene bundle source metadata tests
  - PRD tags: [P0-R4-AC3] [Section 21] [Constitution VII]
  - Files: `tests/test_mapscene_save_bundle.py`
  - Action: Extend bundle tests to require deterministic terrain, raster, vector, point-cloud, and label source metadata payloads plus source-layer IDs in the review bundle.
  - Verification: `pytest tests/test_mapscene_save_bundle.py -q`
  - Done evidence: Red run failed on missing `scene/layer_sources/terrain.json`; green run reported `2 passed`.
  - Dependencies: T013

- [X] T013B [PRD: P0-R4-AC3, Section 21, Constitution VII] Implement bundle source metadata payloads.
  - ID: T013B
  - Title: Implement MapScene bundle layer source metadata
  - PRD tags: [P0-R4-AC3] [Section 21] [Constitution VII]
  - Files: `python/forge3d/map_scene.py`
  - Action: Write deterministic `scene/layer_sources/*.json` payloads for terrain and supported recipe layers, include stable source references and source hashes, and list source layer IDs in review metadata.
  - Verification: `pytest tests/test_mapscene_save_bundle.py -q`
  - Done evidence: Focused bundle command reported `2 passed`.
  - Dependencies: T013A

## Phase 4: Canonical Examples and Docs

- [X] T014 [P] [PRD: P0-R4-AC8, P0-R4-AC1, P0-R4-AC2, P0-R4-AC3] Add canonical example smoke tests.
  - ID: T014
  - Title: Add MapScene canonical example tests
  - PRD tags: [P0-R4-AC8] [P0-R4-AC1] [P0-R4-AC2] [P0-R4-AC3]
  - Files: `tests/test_mapscene_examples.py`, expected `examples/mapscene_terrain_raster.py`, expected `examples/mapscene_vector_labels.py`, expected `examples/mapscene_buildings_labels.py`
  - Action: Add smoke tests for terrain+raster, terrain+vector+labels, and terrain+buildings+labels examples; assert validation, PNG render or typed diagnostics, bundle save, and no direct `viewer_ipc`.
  - Verification: `pytest tests/test_mapscene_examples.py -q`
  - Done evidence: Red run failed because all three `examples/mapscene_*.py` files were missing; green run `pytest tests/test_mapscene_examples.py -q` reported `4 passed`.
  - Dependencies: T011, T013

- [X] T015 [PRD: P0-R4-AC8, P0-R4-AC1, P0-R4-AC2, P0-R4-AC3, P0-R4-AC7] Implement canonical MapScene examples.
  - ID: T015
  - Title: Implement MapScene examples
  - PRD tags: [P0-R4-AC8] [P0-R4-AC1] [P0-R4-AC2] [P0-R4-AC3] [P0-R4-AC7]
  - Files: `examples/mapscene_terrain_raster.py`, `examples/mapscene_vector_labels.py`, `examples/mapscene_buildings_labels.py`
  - Action: Create examples using `MapScene` only, with terrain+raster render/bundle, vector+labels deterministic plan, and buildings+labels render or honest Pro-gated/placeholder/unsupported diagnostics.
  - Verification: `pytest tests/test_mapscene_examples.py -q`
  - Done evidence: Added `examples/mapscene_terrain_raster.py`, `examples/mapscene_vector_labels.py`, and `examples/mapscene_buildings_labels.py`; smoke tests pass and the selected raw-IPC audit command is part of final verification.
  - Dependencies: T014

- [X] T016 [P] [PRD: P0-R4-AC8, P0-R4-AC7] Add MVP docs/support audit tests.
  - ID: T016
  - Title: Add MapScene docs audit tests
  - PRD tags: [P0-R4-AC8] [P0-R4-AC7]
  - Files: `tests/test_mapscene_docs.py`, `docs/guides/offline_3d_map_rendering.md`, `docs/guides/diagnostics_reference.md`, `docs/guides/style_support_matrix.md`, `docs/guides/building_support_matrix.md`, `docs/guides/tiles3d_support_matrix.md`, `docs/guides/virtual_texturing_support_matrix.md`, `docs/guides/competitive_positioning.md`
  - Action: Add docs audit tests for Offline 3D Map Rendering Quickstart, LabelPlan references, support matrices, diagnostics reference, competitive positioning, no overclaim wording, and MVP workflow evidence.
  - Verification: `pytest tests/test_mapscene_docs.py -q`
  - Done evidence: Added broad docs/support audit tests for canonical example links, support-guide links, support-matrix taxonomy, current MapScene diagnostics/ownership, competitive-positioning boundaries, and overclaim avoidance. Red `python -m pytest tests/test_mapscene_docs.py -q -p no:cacheprovider` reported `4 failed, 3 passed` after parser correction because docs were missing required anchors and honest support rows.
  - Dependencies: T001

- [X] T017 [PRD: P0-R4-AC8, P0-R4-AC7, P0-R4-AC1] Update MVP docs and support matrices.
  - ID: T017
  - Title: Update MapScene MVP docs
  - PRD tags: [P0-R4-AC8] [P0-R4-AC7] [P0-R4-AC1]
  - Files: `docs/guides/offline_3d_map_rendering.md`, `docs/guides/diagnostics_reference.md`, `docs/guides/style_support_matrix.md`, `docs/guides/building_support_matrix.md`, `docs/guides/tiles3d_support_matrix.md`, `docs/guides/virtual_texturing_support_matrix.md`, `docs/guides/competitive_positioning.md`, `docs/index.rst`
  - Action: Document typed scene recipe workflow, validation, PNG render, bundle save, examples, diagnostics, support classifications, and explicit non-goals.
  - Verification: `pytest tests/test_mapscene_docs.py -q`
  - Done evidence: Updated offline MapScene guide, style/building/3D Tiles/VT support matrices, and competitive positioning. Green focused docs command reported `7 passed`; selected docs/examples/quickstart command reported `15 passed`; validation/support/docs command reported `23 passed`; `py_compile` exited 0.
  - Dependencies: T016

- [X] T016A [P] [PRD: P0-R4-AC1, P0-R4-AC2, P0-R4-AC3, P0-R4-AC7] Add MapScene docs drift remediation tests.
  - ID: T016A
  - Title: Add MapScene API/docs drift tests
  - PRD tags: [P0-R4-AC1] [P0-R4-AC2] [P0-R4-AC3] [P0-R4-AC7]
  - Files: `tests/test_mapscene_docs.py`, `docs/guides/offline_3d_map_rendering.md`, `docs/api/api_reference.rst`, `specs/004-mapscene-mvp/plan.md`, `specs/004-mapscene-mvp/contracts/mapscene-contract.md`
  - Action: Add targeted docs tests for public `forge3d.map_scene` API documentation, truthful support levels, and stale feature-004 module-path placeholders.
  - Verification: `pytest tests/test_mapscene_docs.py -q`
  - Done evidence: Red run failed on missing `forge3d.map_scene` API docs and stale `TBD`; selected green run reported `22 passed`, and broader selected run reported `25 passed`.
  - Dependencies: T013

- [X] T017A [PRD: P0-R4-AC1, P0-R4-AC2, P0-R4-AC3, P0-R4-AC7] Update MapScene API docs and stale plan/contract wording.
  - ID: T017A
  - Title: Update scoped MapScene docs drift
  - PRD tags: [P0-R4-AC1] [P0-R4-AC2] [P0-R4-AC3] [P0-R4-AC7]
  - Files: `docs/guides/offline_3d_map_rendering.md`, `docs/api/api_reference.rst`, `specs/004-mapscene-mvp/plan.md`, `specs/004-mapscene-mvp/contracts/mapscene-contract.md`
  - Action: Document the public module path, validation/render/bundle behavior, support classifications, and remove stale feature-004 `TBD`/later-owned wording without closing the broader canonical examples/quickstart docs tasks.
  - Verification: `pytest tests/test_mapscene_docs.py -q`
  - Done evidence: Docs drift tests pass as part of the selected `22 passed` and broader selected `25 passed` runs.
  - Dependencies: T016A

- [X] T016B [P] [PRD: P0-R4-AC8, P0-R4-AC7, Section 21] Add scoped MapScene diagnostics docs inventory test.
  - ID: T016B
  - Title: Add MapScene diagnostics inventory docs test
  - PRD tags: [P0-R4-AC8] [P0-R4-AC7] [Section 21]
  - Files: `tests/test_mapscene_docs.py`, `docs/guides/diagnostics_reference.md`
  - Action: Add a focused docs audit requiring the diagnostics reference to document the MapScene validation codes exercised by feature `004`.
  - Verification: `pytest tests/test_mapscene_docs.py -q`
  - Done evidence: Red command `python -m pytest tests/test_mapscene_validation.py tests/test_mapscene_docs.py -q -p no:cacheprovider` failed because `missing_crs` and other required MapScene diagnostic codes were absent from `docs/guides/diagnostics_reference.md`.
  - Dependencies: T006E

- [X] T017B [PRD: P0-R4-AC8, P0-R4-AC7, Section 21] Update scoped MapScene diagnostics docs inventory.
  - ID: T017B
  - Title: Update MapScene diagnostics reference inventory
  - PRD tags: [P0-R4-AC8] [P0-R4-AC7] [Section 21]
  - Files: `docs/guides/diagnostics_reference.md`
  - Action: Document the missing CRS, missing source, missing renderable data, external asset, unsupported format, unsupported output, unsupported layer type, and unsupported feature diagnostics without closing the broader T016/T017 docs/support audit.
  - Verification: `pytest tests/test_mapscene_docs.py -q`
  - Done evidence: Diagnostics reference now documents the scoped MapScene validation inventory; `python -m pytest tests/test_mapscene_validation.py tests/test_mapscene_docs.py -q -p no:cacheprovider` reported `15 passed`.
  - Dependencies: T016B

## Phase 5: Verification and Continuity

- [X] T018 [P] [PRD: P0-R4-AC1, P0-R4-AC2, P0-R4-AC3, P0-R4-AC8] Add quickstart validation tests.
  - ID: T018
  - Title: Add MapScene quickstart validation tests
  - PRD tags: [P0-R4-AC1] [P0-R4-AC2] [P0-R4-AC3] [P0-R4-AC8]
  - Files: `tests/test_mapscene_quickstart.py`, `specs/004-mapscene-mvp/quickstart.md`, expected `python/forge3d/map_scene.py`
  - Action: Add automated checks for terrain+raster, terrain+vector+labels, terrain+buildings+labels or honest diagnostics, CRS mismatch, unsupported style fields, missing glyphs, and bundle diagnostics.
  - Verification: `pytest tests/test_mapscene_quickstart.py -q`
  - Done evidence: Red run failed because quickstart did not link canonical examples and example modules were missing; green run `pytest tests/test_mapscene_quickstart.py -q` reported `4 passed`.
  - Dependencies: T015

- [X] T020R [PRD: Constitution I, Constitution VII] Correct P0 deferred-status honesty for prior label rows.
  - ID: T020R
  - Title: Correct P0 deferred-status release-gate honesty
  - PRD tags: [Constitution I] [Constitution VII]
  - Files: `docs/superpowers/state/requirements-verification-matrix.md`, `docs/superpowers/state/open-blockers.md`
  - Action: Stop treating P0 diagnostic deferrals as reclassified release gates without the constitution-required human decision record; record a blocker or honest status for missing decisions.
  - Verification: `rg -n "P0-R1-AC2|P0-R1-AC3|P0-R2-AC4|P0-R2-AC6|human decision|P0 deferral" docs/superpowers/state/requirements-verification-matrix.md docs/superpowers/state/open-blockers.md`
  - Done evidence: Matrix and blockers updated during the red-remediation checkpoint.
  - Dependencies: T013B

- [X] T021R [PRD: Constitution X] Update red-remediation continuity state.
  - ID: T021R
  - Title: Update red-remediation continuity state
  - PRD tags: [Constitution X]
  - Files: `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/current-context-pack.md`
  - Action: Record selected red task IDs, verification commands/results, remaining open tasks, and the next exact prompt.
  - Verification: `rg -n "T011A|T011B|T013A|T013B|T014|T015|T018|T020R|T021R|next exact prompt" docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md`
  - Done evidence: Ledger and current context pack updated during the red-remediation checkpoint.
  - Dependencies: T020R

- [X] T020S [PRD: P0-R4-AC1, P0-R4-AC3, P0-R4-AC7, Constitution VII] Update matrix for external asset diagnostics remediation.
  - ID: T020S
  - Title: Update external asset diagnostics matrix state
  - PRD tags: [P0-R4-AC1] [P0-R4-AC3] [P0-R4-AC7] [Constitution VII]
  - Files: `docs/superpowers/state/requirements-verification-matrix.md`
  - Action: Record the new missing/unsupported external asset diagnostic evidence without claiming full docs/support audit or final feature verification.
  - Verification: `rg -n "T006B|T006C|missing_external_asset|unsupported_asset_format|P0-R4-AC7" docs/superpowers/state/requirements-verification-matrix.md`
  - Done evidence: Matrix evidence log and P0-R4 rows now cite `missing_external_asset`, `unsupported_asset_format`, focused `17 passed`, and combined `37 passed` selected verification.
  - Dependencies: T006C

- [X] T021S [PRD: P0-R4-AC1, P0-R4-AC3, P0-R4-AC7, Constitution X] Update continuity for external asset diagnostics remediation.
  - ID: T021S
  - Title: Update external asset diagnostics continuity state
  - PRD tags: [P0-R4-AC1] [P0-R4-AC3] [P0-R4-AC7] [Constitution X]
  - Files: `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/current-context-pack.md`
  - Action: Record selected task IDs, red/green verification evidence, files changed, remaining open tasks, and the next exact prompt for the next clean checkpoint.
  - Verification: `rg -n "T006B|T006C|T020S|T021S|missing_external_asset|unsupported_asset_format|Next Exact Prompt" docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md`
  - Done evidence: Ledger and current context pack updated during the external asset diagnostics checkpoint.
  - Dependencies: T020S

- [X] T020T [PRD: P0-R4-AC1, P0-R4-AC5, P0-R4-AC7, P0-R4-AC8, Constitution VII] Update matrix for terrain CRS/building source diagnostics.
  - ID: T020T
  - Title: Update terrain CRS and building source diagnostics matrix state
  - PRD tags: [P0-R4-AC1] [P0-R4-AC5] [P0-R4-AC7] [P0-R4-AC8] [Constitution VII]
  - Files: `docs/superpowers/state/requirements-verification-matrix.md`
  - Action: Record selected red/green evidence for missing terrain CRS, building source asset diagnostics, and scoped diagnostics-reference coverage without claiming full MVP verification.
  - Verification: `rg -n "T006D|T006E|T016B|T017B|T020T|missing_crs|missing_external_asset|unsupported_asset_format|P0-R4-AC5|P0-R4-AC7" docs/superpowers/state/requirements-verification-matrix.md`
  - Done evidence: Matrix evidence log, P0-R4 rows, and diagnostic-inventory rows updated for T006D/T006E/T016B/T017B without claiming full MVP verification.
  - Dependencies: T017B

- [X] T021T [PRD: P0-R4-AC1, P0-R4-AC5, P0-R4-AC7, P0-R4-AC8, Constitution X] Update continuity for terrain CRS/building source diagnostics.
  - ID: T021T
  - Title: Update terrain CRS and building source diagnostics continuity state
  - PRD tags: [P0-R4-AC1] [P0-R4-AC5] [P0-R4-AC7] [P0-R4-AC8] [Constitution X]
  - Files: `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/current-context-pack.md`
  - Action: Record selected task IDs, red/green verification evidence, files changed, remaining red blockers, and the next exact prompt for the next clean checkpoint.
  - Verification: `rg -n "T006D|T006E|T016B|T017B|T020T|T021T|missing_crs|missing_external_asset|unsupported_asset_format|Next Exact Prompt" docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md`
  - Done evidence: Ledger and current context pack updated with selected task IDs, red/green evidence, remaining blockers, and next exact prompt.
  - Dependencies: T020T

- [X] T019 [PRD: P0-R4-AC1, P0-R4-AC2, P0-R4-AC3, P0-R4-AC4, P0-R4-AC5, P0-R4-AC6, P0-R4-AC7, P0-R4-AC8] Run full MapScene MVP verification.
  - ID: T019
  - Title: Run full MapScene MVP verification
  - PRD tags: [P0-R4-AC1] [P0-R4-AC2] [P0-R4-AC3] [P0-R4-AC4] [P0-R4-AC5] [P0-R4-AC6] [P0-R4-AC7] [P0-R4-AC8]
  - Files: `tests/test_mapscene_recipe_contract.py`, `tests/test_mapscene_validation.py`, `tests/test_mapscene_support_status.py`, `tests/test_mapscene_label_plan_integration.py`, `tests/test_mapscene_render_png.py`, `tests/test_mapscene_render_policy.py`, `tests/test_mapscene_save_bundle.py`, `tests/test_mapscene_examples.py`, `tests/test_mapscene_docs.py`, `tests/test_mapscene_quickstart.py`
  - Action: Run the full MapScene feature test set and record the result summary.
  - Verification: `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q`
  - Done evidence: `python -m pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q -p no:cacheprovider` reported `44 passed in 0.84s`.
  - Dependencies: T018

- [X] T020 [PRD: P0-R4-AC1, P0-R4-AC2, P0-R4-AC3, P0-R4-AC4, P0-R4-AC5, P0-R4-AC6, P0-R4-AC7, P0-R4-AC8] Update requirements verification matrix.
  - ID: T020
  - Title: Update requirements verification matrix
  - PRD tags: [P0-R4-AC1] [P0-R4-AC2] [P0-R4-AC3] [P0-R4-AC4] [P0-R4-AC5] [P0-R4-AC6] [P0-R4-AC7] [P0-R4-AC8]
  - Files: `docs/superpowers/state/requirements-verification-matrix.md`
  - Action: Update P0-R4 rows with exact source, tests, docs, diagnostics, examples, PNG, bundle, and command evidence; do not mark `Verified` without successful verification output.
  - Verification: `rg -n "P0-R4-AC[1-8]|004-mapscene-mvp|tests/test_mapscene|examples/mapscene" docs/superpowers/state/requirements-verification-matrix.md`
  - Done evidence: Historical closure evidence recorded T019-T021 and the full `44 passed` command; T022-T025 later blocked the synthetic P0-R4-AC2 evidence, and T026-T029 then superseded that blocker with source-derived PNG render evidence.
  - Dependencies: T019

- [X] T021 [PRD: P0-R4-AC1, P0-R4-AC3, P0-R4-AC7, P0-R4-AC8] Update implementation ledger and current context pack.
  - ID: T021
  - Title: Update continuity state
  - PRD tags: [P0-R4-AC1] [P0-R4-AC3] [P0-R4-AC7] [P0-R4-AC8]
  - Files: `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/current-context-pack.md`
  - Action: Record completed tasks, verification commands, validation/render/bundle evidence, example evidence, support decisions, diagnostics, and residual blockers.
  - Verification: `rg -n "004-mapscene-mvp|P0-R4|MapScene|ValidationReport|save_bundle|viewer_ipc" docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md`
  - Done evidence: Historical closure evidence recorded T019-T021 outcomes; T022-T025 later updated the ledger/context pack with current full verification evidence, residual blockers, and the next exact prompt.
  - Dependencies: T020

## Phase 6: RED/YELLOW Audit Remediation

- [X] T022 [PRD: P0-R4-AC2, P0-R4-AC7, Constitution III, Constitution IV] Add render truth gate tests.
  - ID: T022
  - Title: Add MapScene render truth gate tests
  - PRD tags: [P0-R4-AC2] [P0-R4-AC7] [Constitution III] [Constitution IV]
  - Files: `tests/test_mapscene_render_png.py`, `tests/test_mapscene_examples.py`, `tests/test_mapscene_quickstart.py`, `tests/test_mapscene_save_bundle.py`
  - Action: Add red tests proving the current recipe-sensitive compositor is diagnosed as `placeholder_fallback` and cannot write a successful PNG when no real terrain/raster/vector render adapter is proven.
  - Verification: `$env:PYTHONPATH='python'; python -m pytest tests/test_mapscene_render_png.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_save_bundle.py -q -p no:cacheprovider`
  - Done evidence: Red focused command reported `11 failed, 22 passed`, proving the current synthetic compositor still wrote PNG output and omitted renderability diagnostics before implementation; green focused command reported `33 passed`.
  - Dependencies: T021

- [X] T023 [PRD: P0-R4-AC1, P0-R4-AC7, Constitution III, Constitution V] Implement layer renderability diagnostics.
  - ID: T023
  - Title: Implement MapScene renderability diagnostics
  - PRD tags: [P0-R4-AC1] [P0-R4-AC7] [Constitution III] [Constitution V]
  - Files: `python/forge3d/map_scene.py`, `tests/test_mapscene_validation.py`, `tests/test_mapscene_support_status.py`, `tests/test_mapscene_render_policy.py`
  - Action: Emit typed diagnostics for the unproven render backend, vector path-only overlays without loaded features, point-cloud MapScene rendering, and supported building intent without a proven MapScene render adapter; block render success instead of writing placeholder output.
  - Verification: `$env:PYTHONPATH='python'; python -m pytest tests/test_mapscene_render_png.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_save_bundle.py -q -p no:cacheprovider`
  - Done evidence: `python/forge3d/map_scene.py` now emits `placeholder_fallback` for the unproven MapScene render backend, vector path-only loader, point-cloud MapScene render path, and supported building render adapter. Green focused command reported `33 passed`; full MapScene suite reported `47 passed`.
  - Dependencies: T022

- [X] T024 [PRD: P0-R4-AC2, P0-R4-AC3, P0-R4-AC8, Constitution VIII] Update docs and scope wording.
  - ID: T024
  - Title: Update MapScene render and bundle scope wording
  - PRD tags: [P0-R4-AC2] [P0-R4-AC3] [P0-R4-AC8] [Constitution VIII]
  - Files: `docs/guides/offline_3d_map_rendering.md`, `docs/api/api_reference.rst`, `specs/004-mapscene-mvp/quickstart.md`, `specs/004-mapscene-mvp/contracts/mapscene-contract.md`, `tests/test_mapscene_docs.py`
  - Action: Stop calling synthetic fixture rendering verified PNG output; keep bundle scope limited to deterministic review metadata, diagnostics, and source references until a real load/render round trip exists.
  - Verification: `$env:PYTHONPATH='python'; python -m pytest tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q -p no:cacheprovider`
  - Done evidence: Docs audit red command reported `1 failed, 10 passed` before wording updates; after updating the Offline 3D Map Rendering guide, API reference, quickstart, and contract, `tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py` reported `11 passed`.
  - Dependencies: T023

- [X] T025 [PRD: P0-R4-AC2, P0-R4-AC3, P0-R4-AC7, P0-R4-AC8, Constitution X] Update RED/YELLOW remediation continuity state.
  - ID: T025
  - Title: Update RED/YELLOW remediation state
  - PRD tags: [P0-R4-AC2] [P0-R4-AC3] [P0-R4-AC7] [P0-R4-AC8] [Constitution X]
  - Files: `docs/superpowers/state/requirements-verification-matrix.md`, `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/current-context-pack.md`, `docs/superpowers/state/open-blockers.md`, `specs/004-mapscene-mvp/tasks.md`
  - Action: Record selected RED/YELLOW task IDs, verification commands/results, revised P0-R4 status, remaining P0 blockers, and the next exact prompt for a new chat.
  - Verification: `rg -n "T022|T023|T024|T025|render_backend|placeholder_fallback|P0-R4-AC2|Next Exact Prompt" docs/superpowers/state specs/004-mapscene-mvp/tasks.md`
  - Done evidence: Matrix, ledger, context pack, blockers, and task list updated with T022-T025 evidence, revised P0-R4-AC2 blocked status, verification commands, and the next exact prompt.
  - Dependencies: T024

## Phase 7: RED/YELLOW Source-Derived Render Completion

- [X] T026 [PRD: P0-R4-AC2, P0-R4-AC7, P0-R4-AC8, Constitution III, Constitution IV] Add source-derived render tests.
  - ID: T026
  - Title: Add source-derived MapScene render evidence tests
  - PRD tags: [P0-R4-AC2] [P0-R4-AC7] [P0-R4-AC8] [Constitution III] [Constitution IV]
  - Files: `tests/test_mapscene_render_png.py`, `tests/test_mapscene_render_policy.py`, `tests/test_mapscene_examples.py`, `tests/test_mapscene_quickstart.py`, `tests/test_mapscene_docs.py`
  - Action: Add RED tests proving supported MVP terrain/raster/vector/label recipes write PNG output, render bytes change when source data changes, examples produce real PNG artifacts, and docs describe source-derived output instead of a blocked placeholder backend.
  - Verification: `$env:PYTHONPATH='python'; python -m pytest tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_docs.py -q -p no:cacheprovider`
  - Done evidence: Focused RED command reported `8 failed, 14 passed`, proving the prior T022-T025 blocked backend still failed the source-derived PNG acceptance path and docs/examples were stale.
  - Dependencies: T025

- [X] T027 [PRD: P0-R4-AC2, P0-R4-AC7, Constitution III, Constitution V] Implement source-derived MapScene PNG adapter.
  - ID: T027
  - Title: Implement source-derived deterministic PNG render path
  - PRD tags: [P0-R4-AC2] [P0-R4-AC7] [Constitution III] [Constitution V]
  - Files: `python/forge3d/map_scene.py`
  - Action: Replace the global `mapscene.render_backend` placeholder block with deterministic source-derived RGBA generation for supported non-blocked MVP scenes, keep validation-before-render policy, write PNG output, and preserve typed diagnostics for unsupported vector path-loader, point-cloud, building, and other fallback paths.
  - Verification: `$env:PYTHONPATH='python'; python -m pytest tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_docs.py -q -p no:cacheprovider`
  - Done evidence: `python/forge3d/map_scene.py` now uses `_render_source_derived_rgba(...)`, records `mapscene.render_backend` and `mapscene.render_png` as supported after a successful write, and no longer injects the global render-backend `placeholder_fallback` into supported scenes. Focused GREEN command reported `22 passed`.
  - Dependencies: T026

- [X] T028 [PRD: P0-R4-AC2, P0-R4-AC8, Constitution VIII] Update render docs, examples, quickstart, and contract wording.
  - ID: T028
  - Title: Update public source-derived render wording
  - PRD tags: [P0-R4-AC2] [P0-R4-AC8] [Constitution VIII]
  - Files: `docs/guides/offline_3d_map_rendering.md`, `docs/api/api_reference.rst`, `specs/004-mapscene-mvp/quickstart.md`, `specs/004-mapscene-mvp/contracts/mapscene-contract.md`, `examples/mapscene_terrain_raster.py`, `examples/mapscene_vector_labels.py`
  - Action: Reword public docs and executable examples so supported terrain/raster and vector/label scenes report PNG output, while still documenting diagnostic-bearing unsupported paths.
  - Verification: `$env:PYTHONPATH='python'; python -m pytest tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_docs.py -q -p no:cacheprovider`
  - Done evidence: Public docs now classify `MapScene.render` PNG path as `supported`, API/contract/quickstart wording says source-derived PNG output is written for supported MVP recipes, terrain/raster and vector/label examples create PNG artifacts, and the focused GREEN command reported `22 passed`.
  - Dependencies: T027

- [X] T029 [PRD: P0-R4-AC2, P0-R4-AC7, P0-R4-AC8, Constitution X] Update source-derived render continuity state.
  - ID: T029
  - Title: Update source-derived render state
  - PRD tags: [P0-R4-AC2] [P0-R4-AC7] [P0-R4-AC8] [Constitution X]
  - Files: `docs/superpowers/state/requirements-verification-matrix.md`, `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/current-context-pack.md`, `docs/superpowers/state/open-blockers.md`, `specs/004-mapscene-mvp/tasks.md`
  - Action: Record selected RED/YELLOW task IDs, focused and full verification results, revised P0-R4 status, resolved R-026 status, remaining P0 label blockers, and the next exact prompt for a new chat.
  - Verification: `rg -n "T026|T027|T028|T029|P0-R4-AC2|source-derived|Next Exact Prompt|R-026" specs/004-mapscene-mvp/tasks.md docs/superpowers/state/requirements-verification-matrix.md docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md docs/superpowers/state/open-blockers.md`
  - Done evidence: Matrix, ledger, context pack, blockers, and task list updated with T026-T029 evidence, revised P0-R4-AC2 verified status, verification commands, and the next exact prompt.
  - Dependencies: T028

## Dependencies and Execution Order

- T001 blocks source implementation.
- T002, T004, T005, T007, T009, T010, T012, T014, T016, and T018 can be authored in parallel after their listed prerequisites because they target separate test/docs files.
- Implementation follows tests: T003 after T002, T006 after T004/T005, T008 after T007, T011 after T009/T010, T013 after T012, T015 after T014, and T017 after T016.
- T020 and T021 happen after verification evidence exists.
- T026-T029 supersede T022-T025 for P0-R4-AC2: T026 proves the blocked-backend gap, T027 implements source-derived PNG output, T028 updates public wording/examples, and T029 updates continuity state.

## PRD Coverage Check

- P0-R4-AC1: T001, T004, T006, T006A, T006B, T006C, T006D, T006E, T007, T008, T014, T015, T016A, T017, T017A, T018, T019, T020, T020S, T020T, T021, T021S, T021T, T023, T025
- P0-R4-AC2: T009, T010, T011, T014, T015, T016A, T017A, T018, T019, T020, T022, T023, T024, T025, T026, T027, T028, T029
- P0-R4-AC3: T006B, T006C, T012, T013, T014, T015, T016A, T017A, T018, T019, T020, T020S, T021, T021S, T024, T025
- P0-R4-AC4: T001, T002, T003, T006A, T007, T008, T019, T020
- P0-R4-AC5: T004, T006, T006D, T006E, T019, T020, T020T, T021T
- P0-R4-AC6: T004, T006, T019, T020
- P0-R4-AC7: T001, T004, T005, T006, T006A, T006B, T006C, T006D, T006E, T007, T008, T009, T010, T011, T012, T013, T015, T016, T016A, T016B, T017, T017A, T017B, T019, T020, T020S, T020T, T021, T021S, T021T, T022, T023, T025, T026, T027, T029
- P0-R4-AC8: T014, T015, T016, T016B, T017, T017B, T018, T019, T020, T020T, T021, T021T, T024, T025, T026, T028, T029

## Detailed Pre-Implementation Coverage Audit Table

| PRD requirement | Acceptance criterion | Task IDs covering it | Test task IDs | Docs task IDs | Verification command | Risk if omitted |
|---|---|---|---|---|---|---|
| P0-R4 MapScene MVP | P0-R4-AC1 structured `validate()` report | T001, T004, T006, T007, T008, T014, T015, T017, T018, T019, T020, T021 | T004, T007, T014, T018 | T017, T020, T021 | `pytest tests/test_mapscene_validation.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q` | Users cannot diagnose scene issues before render. |
| P0-R4 MapScene MVP | P0-R4-AC2 PNG output | T009, T010, T011, T014, T015, T018, T019, T020, T022, T023, T024, T025, T026, T027, T028, T029 | T009, T010, T014, T018, T022, T026 | T017, T020, T024, T028, T029 | `pytest tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_docs.py -q` | Source-derived PNG adapter or public wording could regress to synthetic/no-op output. |
| P0-R4 MapScene MVP | P0-R4-AC3 reproducible review bundle | T012, T013, T014, T015, T018, T019, T020, T021 | T012, T014, T018 | T017, T020, T021 | `pytest tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q` | Reviewers cannot reproduce scene configuration or diagnostics. |
| P0-R4 MapScene MVP | P0-R4-AC4 typed recipe coverage | T001, T002, T003, T007, T008, T019, T020 | T002, T007 | T017, T020 | `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_label_plan_integration.py -q` | Users still stitch terrain, overlays, labels, camera, lighting, output, and assets manually. |
| P0-R4 MapScene MVP | P0-R4-AC5 CRS mismatch detection | T004, T006, T006D, T006E, T019, T020, T020T | T004, T006D | T017, T020, T020T | `pytest tests/test_mapscene_validation.py -q` | Layers or terrain can render in the wrong coordinate space without pre-render failure. |
| P0-R4 MapScene MVP | P0-R4-AC6 GPU memory estimate | T004, T006, T019, T020 | T004 | T017, T020 | `pytest tests/test_mapscene_validation.py -q` | Oversized scenes have no pre-render resource-risk evidence. |
| P0-R4 MapScene MVP | P0-R4-AC7 unsupported features typed diagnostics | T001, T004, T005, T006, T006B, T006C, T006D, T006E, T007, T008, T009, T010, T011, T012, T013, T015, T016, T016B, T017, T017B, T019, T020, T020S, T020T, T021, T021S, T021T, T022, T023, T025, T026, T027, T029 | T004, T005, T006B, T006D, T007, T009, T010, T012, T016, T016B, T022, T026 | T016, T016B, T017, T017B, T020, T020T, T021, T021T, T024, T029 | `pytest tests/test_mapscene_support_status.py tests/test_mapscene_validation.py tests/test_mapscene_render_policy.py tests/test_mapscene_docs.py -q` | Unsupported, Pro-gated, placeholder, experimental, missing-asset, missing-CRS, or unsupported-format paths can silently fall back. |
| P0-R4 MapScene MVP | P0-R4-AC8 three canonical examples | T014, T015, T016, T016B, T017, T017B, T018, T019, T020, T020T, T021, T021T, T024, T025, T026, T028, T029 | T014, T016, T016B, T018, T026 | T016, T016B, T017, T017B, T020, T020T, T021, T021T, T024, T028, T029 | `pytest tests/test_mapscene_examples.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q` | MVP cannot prove the typed workflow works for required scene recipes. |
| Section 21 MVP must include | `MapScene` typed API | T001, T002, T003, T019, T020 | T002 | T017, T020 | `pytest tests/test_mapscene_recipe_contract.py -q` | MVP remains a scattered low-level API set. |
| Section 21 MVP must include | Terrain source | T002, T003, T014, T015, T018, T019, T020 | T002, T014, T018 | T017, T020 | `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q` | The canonical offline terrain workflow is not represented. |
| Section 21 MVP must include | Raster overlay | T002, T003, T014, T015, T018, T019, T020 | T002, T014, T018 | T017, T020 | `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q` | Terrain+raster scenes cannot be authored through the typed recipe. |
| Section 21 MVP must include | Vector overlay | T002, T003, T014, T015, T018, T019, T020 | T002, T014, T018 | T017, T020 | `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q` | Vector styling stays outside the MVP scene contract. |
| Section 21 MVP must include | Basic label layer | T002, T003, T007, T008, T014, T015, T018, T019, T020 | T002, T007, T014, T018 | T017, T020 | `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_examples.py -q` | Label-bearing MVP examples cannot use typed map-scene state. |
| Section 21 MVP must include | Deterministic `LabelPlan` for point and polygon labels | T007, T008, T014, T015, T018, T019, T020 | T007, T014, T018 | T017, T020 | `pytest tests/test_mapscene_label_plan_integration.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q` | MapScene can bypass deterministic label planning. |
| Section 21 MVP must include | Camera and output spec | T002, T003, T009, T014, T015, T018, T019, T020 | T002, T009, T014, T018 | T017, T020 | `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_render_png.py tests/test_mapscene_examples.py -q` | Reproducible rendering cannot be fixed by recipe inputs. |
| Section 21 MVP must include | Validation report | T004, T006, T018, T019, T020, T021 | T004, T018 | T017, T020, T021 | `pytest tests/test_mapscene_validation.py tests/test_mapscene_quickstart.py -q` | Validation evidence is unavailable before render. |
| Section 21 MVP must include | PNG render output | T009, T010, T011, T014, T015, T018, T019, T020, T022, T023, T024, T025, T026, T027, T028, T029 | T009, T010, T014, T018, T022, T026 | T017, T020, T024, T028, T029 | `pytest tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_examples.py -q` | MVP cannot produce a real output artifact. |
| Section 21 MVP must include | Bundle save | T012, T013, T014, T015, T018, T019, T020, T021 | T012, T014, T018 | T017, T020, T021 | `pytest tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q` | Scene review/reproduction evidence is lost. |
| Section 21 MVP must include | Style support diagnostics | T004, T005, T006, T016, T017, T018, T019, T020 | T004, T005, T016, T018 | T016, T017, T020 | `pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q` | Unsupported style fields or layer types can be silently ignored. |
| Section 21 MVP must include | Missing-glyph diagnostics | T004, T006, T007, T008, T018, T019, T020 | T004, T007, T018 | T017, T020 | `pytest tests/test_mapscene_validation.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_quickstart.py -q` | Glyph problems are found only after render or not at all. |
| Section 21 MVP must include | CRS mismatch diagnostics | T004, T006, T018, T019, T020 | T004, T018 | T017, T020 | `pytest tests/test_mapscene_validation.py tests/test_mapscene_quickstart.py -q` | Incorrect CRS combinations can proceed as apparently valid scenes. |

## Audit Rerun Result

- PASS: Every P0-R4 acceptance criterion and every Section 21 MVP must-include item has task coverage, test IDs, docs/support IDs, and a verification command.
- PASS: `validate()`, `render()`, and `save_bundle()` tasks require real reports, PNGs, bundles, or typed diagnostics; no no-op success path is permitted.
- PASS: T026-T029 replace the T022-T025 blocked synthetic render state with deterministic source-derived PNG output for supported MVP terrain/raster/vector/label recipes while preserving typed diagnostics for unsupported paths.
- PASS: Public API and example tasks are paired with support-level documentation and raw-IPC avoidance checks.

## Extension Hooks

**Optional Hook**: git
Command: `/speckit.git.commit`
Description: Auto-commit after task generation

Prompt: Commit task changes?
To execute: `/speckit.git.commit`

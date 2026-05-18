# Tasks: Material, VT, and Large-Scene P2

**Input**: `specs/006-material-vt-large-scene-p2/plan.md`, `spec.md`, `research.md`, `data-model.md`, `quickstart.md`, and `contracts/p2-material-vt-large-scene-contract.md`

**Scope**: P2-R1 through P2-R4, Milestone 5, P2 diagnostics/support honesty, deferred-path documentation, and required continuity state updates.

**Required task format**: Every task below includes ID, title, PRD tags, files, action, verification, done evidence, and dependencies. The first checklist line preserves the Spec Kit task ID and `[P]` marker where work can be safely parallelized.

**Execution rule**: This P2 feature extends P0/P1 product APIs. If required P0/P1 contracts are missing, record a typed blocker/diagnostic and stop that slice instead of creating duplicate `MapScene`, `LabelPlan`, or diagnostics contracts.

## Phase 1: P0/P1 Dependency and Diagnostic Gates

- [X] T001 [PRD: P2-R1-AC1, P2-R2-AC1, P2-R3-AC1, P2-R4-AC1] Inspect P0/P1 product API dependencies before source edits.
  - ID: T001
  - Title: Inspect P2 dependency paths
  - PRD tags: [P2-R1-AC1] [P2-R2-AC1] [P2-R3-AC1] [P2-R4-AC1]
  - Files: expected `python/forge3d/map_scene.py`, expected `python/forge3d/diagnostics.py`, expected `python/forge3d/label_plan.py`, `python/forge3d/terrain_params.py`, `src/terrain/render_params/native_vt.rs`, `src/terrain/renderer/virtual_texture.rs`, `python/forge3d/buildings.py`, `python/forge3d/textures.py`, `python/forge3d/materials.py`, `python/forge3d/pointcloud.py`, `python/forge3d/tiles3d.py`, `python/forge3d/geometry.py`, `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/open-blockers.md`
  - Action: Confirm actual extension points for VT families, textured building materials, advanced labels, large-scene summaries, diagnostics, `MapScene`, and `LabelPlan`; record blockers for missing P0/P1 foundations instead of duplicating them.
  - Verification: `rg -n "MapScene|Diagnostic|LabelPlan|virtual texture|normal|mask|texture|uv|cache|lod|instancing|ResourceSummary" python src docs/superpowers/state`
  - Done evidence: Ledger records exact extension paths and any `p0_dependency_missing` or `p1_dependency_missing` blocker entries.
  - Completion evidence (2026-05-18): Required inspection command exited `0`. P0/P1 owner paths exist and should be extended, not duplicated: `python/forge3d/map_scene.py` (`MapScene`, `BuildingLayer`, `Tiles3DLayer`, validation, bundle load/save), `python/forge3d/diagnostics.py` (`Diagnostic`, `ValidationReport`, `vt_unsupported_family_diagnostic`, `estimated_gpu_memory_diagnostic`), and `python/forge3d/label_plan.py` (`LabelPlan.compile`). VT extension paths are `python/forge3d/terrain_params.py`, `src/terrain/render_params/native_vt.rs`, and `src/terrain/renderer/virtual_texture.rs`; inspection confirms Python accepts `normal`/`mask` while native runtime pages only `albedo`. Building texture extension paths are `python/forge3d/buildings.py`, `python/forge3d/textures.py`, `python/forge3d/materials.py`, `src/io/gltf_read.rs`, and `src/io/obj_read.rs`; inspection confirms scalar building material support and general texture/UV substrate but no P2 typed building texture diagnostics yet. Large-scene extension paths are `python/forge3d/pointcloud.py`, `python/forge3d/tiles3d.py`, `python/forge3d/geometry.py`, `src/pointcloud/renderer.rs`, `src/tiles3d/renderer.rs`, `src/scene/py_api/instanced_mesh.rs`, and `src/terrain/renderer/py_api.rs`; inspection confirms fragmented cache/LOD/memory/instancing stats but no unified `LargeSceneResourceSummary`. No `p0_dependency_missing` or `p1_dependency_missing` blocker was opened.
  - Dependencies: None

- [X] T002 [P] [PRD: P2-R1-AC2, P2-R2-AC2, P2-R2-AC3, P2-R2-AC4, P2-R4-AC2, P2-R4-AC3] Add P2 feature-local diagnostic code tests.
  - ID: T002
  - Title: Add P2 diagnostic code tests
  - PRD tags: [P2-R1-AC2] [P2-R2-AC2] [P2-R2-AC3] [P2-R2-AC4] [P2-R4-AC2] [P2-R4-AC3]
  - Files: `tests/test_p2_diagnostics_contract.py`, expected `python/forge3d/diagnostics.py`, expected `python/forge3d/map_scene.py`
  - Action: Add tests for structured, serializable `missing_texture_path`, `missing_uvs`, `unsupported_texture_format`, `unavailable_cache_lod_stats`, `unsupported_instancing_path`, `vt_unsupported_family`, `placeholder_fallback`, `pro_gated_path`, and `experimental_feature` diagnostics.
  - Verification: `pytest tests/test_p2_diagnostics_contract.py -q`
  - Done evidence: Tests fail until P2 diagnostics include severity, remediation, support level, affected IDs, and deterministic serialization.
  - Completion evidence (2026-05-18): Added `tests/test_p2_diagnostics_contract.py` with diagnostic factory serialization tests and MapScene pre-render adapter tests for `missing_texture_path`, `missing_uvs`, `unsupported_texture_format`, `unavailable_cache_lod_stats`, `unsupported_instancing_path`, existing `vt_unsupported_family`, `placeholder_fallback`, `pro_gated_path`, and `experimental_feature`. RED verification `PYTHONPATH=python python -m pytest tests\test_p2_diagnostics_contract.py -q` reported `4 failed` before T003 because `P2_FEATURE_DIAGNOSTIC_CODES`/factory imports were missing and MapScene validation returned `ok` or omitted P2 diagnostics.
  - Dependencies: T001

- [X] T003 [PRD: P2-R1-AC2, P2-R2-AC2, P2-R2-AC3, P2-R2-AC4, P2-R4-AC2, P2-R4-AC3] Implement P2 diagnostic adapters.
  - ID: T003
  - Title: Implement P2 diagnostic adapters
  - PRD tags: [P2-R1-AC2] [P2-R2-AC2] [P2-R2-AC3] [P2-R2-AC4] [P2-R4-AC2] [P2-R4-AC3]
  - Files: expected `python/forge3d/diagnostics.py`, expected `python/forge3d/map_scene.py`, `python/forge3d/buildings.py`, `python/forge3d/terrain_params.py`
  - Action: Add P2 diagnostic helpers through the shared diagnostics contract and wire availability metadata without classifying diagnosed unavailable paths as supported.
  - Verification: `pytest tests/test_p2_diagnostics_contract.py -q`
  - Done evidence: P2 diagnostic contract tests pass or blockers record missing P0/P1 dependencies.
  - Completion evidence (2026-05-18): Implemented `P2_FEATURE_DIAGNOSTIC_CODES` and P2 factories in `python/forge3d/diagnostics.py`; exported them through `python/forge3d/__init__.py` and `python/forge3d/__init__.pyi`; added MapScene metadata adapters in `python/forge3d/map_scene.py` for building textured-material intent and cache/LOD/instancing availability. GREEN verification `PYTHONPATH=python python -m pytest tests\test_p2_diagnostics_contract.py -q` reported `4 passed`; focused regression `PYTHONPATH=python python -m pytest tests\test_p1_diagnostics_contract.py tests\test_p1_building_workflow_support.py tests\test_p1_tiles3d_support.py -q` reported `15 passed`; `py_compile` for updated product modules and T002 tests exited `0`.
  - Dependencies: T002

## Phase 2: Virtual Texture Families

- [X] T004 [P] [PRD: P2-R1-AC1, P2-R1-AC2, P2-R1-AC3] Add VT family validation and no-silent-skip tests.
  - ID: T004
  - Title: Add VT family validation tests
  - PRD tags: [P2-R1-AC1] [P2-R1-AC2] [P2-R1-AC3]
  - Files: `tests/test_p2_vt_family_validation.py`, expected `python/forge3d/map_scene.py`, `python/forge3d/terrain_params.py`, `src/terrain/render_params/native_vt.rs`, `src/terrain/renderer/virtual_texture.rs`
  - Action: Add tests for albedo-only scenes, albedo+normal/mask requests, deterministic family ordering, `vt_unsupported_family` diagnostics when native runtime cannot render a family, and render blocking/no-silent-skip behavior.
  - Verification: `pytest tests/test_p2_vt_family_validation.py -q`
  - Done evidence: Tests fail if normal/mask requests are accepted then silently skipped.
  - Completion evidence (2026-05-18): Added `tests/test_p2_vt_family_validation.py` covering albedo-only validation/render, deterministic albedo+normal/mask reports, `vt_unsupported_family` affected IDs, unsupported feature classification, and render blocking/no-silent-skip behavior. RED verification `PYTHONPATH=python python -m pytest tests\test_p2_vt_family_validation.py -q` reported `2 failed, 1 passed` before T005 because non-albedo VT diagnostics had no `object_id`.
  - Dependencies: T003

- [X] T005 [PRD: P2-R1-AC1, P2-R1-AC2, P2-R1-AC3] Implement VT family validation or native runtime support.
  - ID: T005
  - Title: Implement VT family support or diagnostics
  - PRD tags: [P2-R1-AC1] [P2-R1-AC2] [P2-R1-AC3]
  - Files: expected `python/forge3d/map_scene.py`, `python/forge3d/terrain_params.py`, `src/terrain/render_params/native_vt.rs`, `src/terrain/renderer/virtual_texture.rs`, inspect shader/resource files referenced by VT runtime
  - Action: Either render requested normal/mask families end to end through native runtime or emit deterministic pre-render `vt_unsupported_family` diagnostics that block successful render where required.
  - Verification: `pytest tests/test_p2_vt_family_validation.py -q`
  - Done evidence: VT tests pass with real support or explicit unsupported diagnostics.
  - Completion evidence (2026-05-18): Updated `python/forge3d/terrain_params.py` so `validate_terrain_vt_support()` emits deterministic `object_id` values (`vt.normal`, `vt.mask`) for unsupported non-albedo families and classifies `vt.normal`/`vt.mask` as `unsupported` in the support map. GREEN verification `PYTHONPATH=python python -m pytest tests\test_p2_vt_family_validation.py -q` reported `3 passed`.
  - Dependencies: T004

- [X] T006 [P] [PRD: P2-R1-AC4] Add VT support matrix docs audit tests.
  - ID: T006
  - Title: Add VT docs audit tests
  - PRD tags: [P2-R1-AC4]
  - Files: `tests/test_p2_vt_docs.py`, `docs/guides/virtual_texturing_support_matrix.md`
  - Action: Add docs audit tests requiring exact albedo/normal/mask runtime support status, no silent-skip wording, and no support claim for diagnosed unsupported families.
  - Verification: `pytest tests/test_p2_vt_docs.py -q`
  - Done evidence: Docs audit fails until VT support status is exact.
  - Completion evidence (2026-05-18): Added `tests/test_p2_vt_docs.py`. RED docs group reported VT docs failures before T007 because normal/mask were documented as `missing` and no no-silent-skip wording existed.
  - Dependencies: T001

- [X] T007 [PRD: P2-R1-AC4] Update VT support matrix.
  - ID: T007
  - Title: Update VT support documentation
  - PRD tags: [P2-R1-AC4]
  - Files: `docs/guides/virtual_texturing_support_matrix.md`, `docs/index.rst`
  - Action: Document albedo, normal, and mask family status, runtime limitations, diagnostics, and remediation without overclaiming implemented support.
  - Verification: `pytest tests/test_p2_vt_docs.py -q`
  - Done evidence: VT docs audit passes and matrix states exact runtime behavior.
  - Completion evidence (2026-05-18): Updated `docs/guides/virtual_texturing_support_matrix.md` and `docs/index.rst`; full P2 command reported `35 passed`.
  - Dependencies: T006

## Phase 3: Textured Building Materials

- [X] T008 [P] [PRD: P2-R2-AC1, P2-R2-AC5] Add textured building MapScene fixture tests.
  - ID: T008
  - Title: Add textured building fixture tests
  - PRD tags: [P2-R2-AC1] [P2-R2-AC5]
  - Files: `tests/test_p2_textured_building_mapscene.py`, expected `python/forge3d/map_scene.py`, `python/forge3d/buildings.py`, `python/forge3d/textures.py`, `python/forge3d/materials.py`, `tests/fixtures/`
  - Action: Add tests for a building layer with albedo texture and UVs that either renders through `MapScene` or emits typed diagnostics explaining unavailable texture support.
  - Verification: `pytest tests/test_p2_textured_building_mapscene.py -q`
  - Done evidence: Tests fail if scalar fallback is presented as textured PBR success.
  - Completion evidence (2026-05-18): Added `tests/test_p2_textured_building_mapscene.py`. RED verification reported valid texture+UV intent incorrectly validated/rendered without blocking diagnostics before T010.
  - Dependencies: T003

- [X] T009 [P] [PRD: P2-R2-AC2, P2-R2-AC3, P2-R2-AC4] Add building UV, texture path, format, and fallback tests.
  - ID: T009
  - Title: Add building texture diagnostic tests
  - PRD tags: [P2-R2-AC2] [P2-R2-AC3] [P2-R2-AC4]
  - Files: `tests/test_p2_building_texture_diagnostics.py`, `python/forge3d/buildings.py`, `python/forge3d/textures.py`, expected `python/forge3d/map_scene.py`
  - Action: Add tests for UV presence/absence, missing texture paths, unreadable paths, unsupported texture formats, scalar/material fallback diagnostics, affected layer/object IDs, and Pro-gated native paths.
  - Verification: `pytest tests/test_p2_building_texture_diagnostics.py -q`
  - Done evidence: Tests fail until missing UV/path/format/fallback cases are reported before render.
  - Completion evidence (2026-05-18): Added `tests/test_p2_building_texture_diagnostics.py`. RED verification failed on missing `textured_material_status` and missing explicit unsupported textured-PBR diagnostic for Pro-gated textured intent.
  - Dependencies: T003

- [X] T010 [PRD: P2-R2-AC1, P2-R2-AC2, P2-R2-AC3, P2-R2-AC4, P2-R2-AC5] Implement textured building material support or diagnostics.
  - ID: T010
  - Title: Implement textured building materials or diagnostics
  - PRD tags: [P2-R2-AC1] [P2-R2-AC2] [P2-R2-AC3] [P2-R2-AC4] [P2-R2-AC5]
  - Files: expected `python/forge3d/map_scene.py`, `python/forge3d/buildings.py`, `python/forge3d/textures.py`, `python/forge3d/materials.py`, inspect `src/import/`, inspect `src/io/gltf_read.rs`, inspect `src/io/obj_read.rs`
  - Action: Support albedo texture material intent where geometry and UVs are available, or emit missing UV/path/format/Pro-gated/fallback diagnostics before render with deterministic layer summaries.
  - Verification: `pytest tests/test_p2_textured_building_mapscene.py tests/test_p2_building_texture_diagnostics.py -q`
  - Done evidence: Building material tests pass with real texture support or explicit diagnostics.
  - Completion evidence (2026-05-18): Updated `python/forge3d/map_scene.py` to classify textured PBR render intent as explicit `unsupported_feature` when texture+UV inputs exist but no end-to-end render path exists, while preserving missing path/UV/format/scalar fallback diagnostics. Focused building/P2/P1 command reported `14 passed`.
  - Dependencies: T008, T009

- [X] T011 [P] [PRD: P2-R2-AC4] Add building texture docs audit tests.
  - ID: T011
  - Title: Add building texture docs audit tests
  - PRD tags: [P2-R2-AC4]
  - Files: `tests/test_p2_building_texture_docs.py`, `docs/guides/building_support_matrix.md`
  - Action: Add docs audit tests requiring material fallback status, albedo texture prerequisites, UV requirements, Pro-gated/fallback classifications, and no textured PBR overclaims.
  - Verification: `pytest tests/test_p2_building_texture_docs.py -q`
  - Done evidence: Docs audit fails until fallback and prerequisites are explicit.
  - Completion evidence (2026-05-18): Added `tests/test_p2_building_texture_docs.py`. RED docs group failed until textured PBR status, UV/path prerequisites, and scalar fallback wording were explicit.
  - Dependencies: T001

- [X] T012 [PRD: P2-R2-AC4, P2-R2-AC5] Update building texture support docs.
  - ID: T012
  - Title: Update building texture documentation
  - PRD tags: [P2-R2-AC4] [P2-R2-AC5]
  - Files: `docs/guides/building_support_matrix.md`, `docs/guides/offline_3d_map_rendering.md`, `docs/index.rst`
  - Action: Document textured building material status, fixture behavior, UV/texture prerequisites, fallback diagnostics, and `MapScene` verification path.
  - Verification: `pytest tests/test_p2_building_texture_docs.py -q`
  - Done evidence: Docs audit passes and no fallback path is described as textured support.
  - Completion evidence (2026-05-18): Updated `docs/guides/building_support_matrix.md`, `docs/guides/offline_3d_map_rendering.md`, and `docs/index.rst`; full P2 command reported `35 passed`.
  - Dependencies: T011

## Phase 4: Advanced Static Labels

- [X] T013 [P] [PRD: P2-R3-AC1, P2-R3-AC2] Add repeated and curved advanced label tests.
  - ID: T013
  - Title: Add repeated and curved label tests
  - PRD tags: [P2-R3-AC1] [P2-R3-AC2]
  - Files: `tests/test_p2_advanced_labels_repeated_curved.py`, expected `python/forge3d/label_plan.py`, `src/labels/line_label.rs`, `src/labels/curved.rs`
  - Action: Add deterministic tests for repeated labels along long lines with configurable repeat distance, curved text along paths where supported, and experimental/unsupported diagnostics where not supported.
  - Verification: `pytest tests/test_p2_advanced_labels_repeated_curved.py -q`
  - Done evidence: Tests fail on nondeterministic repeats or unqualified curved text success.
  - Completion evidence (2026-05-18): Added `tests/test_p2_advanced_labels_repeated_curved.py`. RED verification failed because `LabelPlan` rejected all line geometry and curved text had no `experimental_feature` diagnostic.
  - Dependencies: T003

- [X] T014 [PRD: P2-R3-AC1, P2-R3-AC2] Implement repeated and curved label behavior or diagnostics.
  - ID: T014
  - Title: Implement repeated and curved label modes
  - PRD tags: [P2-R3-AC1] [P2-R3-AC2]
  - Files: expected `python/forge3d/label_plan.py`, `src/labels/line_label.rs`, `src/labels/curved.rs`
  - Action: Implement deterministic repeated line placement and curved text where real placement is available; otherwise return explicit `experimental_feature` or unsupported diagnostics.
  - Verification: `pytest tests/test_p2_advanced_labels_repeated_curved.py -q`
  - Done evidence: Repeated/curved tests pass with real output or typed diagnostics.
  - Completion evidence (2026-05-18): Updated `python/forge3d/label_plan.py` with deterministic repeated `LineString` label candidates and explicit `experimental_feature` diagnostics for curved text. Focused label/P2 regression command reported `10 passed`.
  - Dependencies: T013

- [X] T015 [P] [PRD: P2-R3-AC3, P2-R3-AC4, P2-R3-AC5] Add road/river, leader-line, and priority preset tests.
  - ID: T015
  - Title: Add advanced label rule tests
  - PRD tags: [P2-R3-AC3] [P2-R3-AC4] [P2-R3-AC5]
  - Files: `tests/test_p2_advanced_label_rules.py`, expected `python/forge3d/label_plan.py`, inspect `python/forge3d/map_plate.py`
  - Action: Add tests for road/river placement presets, landmark/callout leader-line placement, and priority presets for capitals, cities, rivers, peaks, roads, and annotations with deterministic conflict resolution.
  - Verification: `pytest tests/test_p2_advanced_label_rules.py -q`
  - Done evidence: Tests fail until rules produce deterministic accepted/rejected outputs or typed unsupported diagnostics.
  - Completion evidence (2026-05-18): Added `tests/test_p2_advanced_label_rules.py`. RED verification failed because `priority_rules="cartographic"` was not accepted and leader-line candidate details were absent.
  - Dependencies: T003

- [X] T016 [PRD: P2-R3-AC3, P2-R3-AC4, P2-R3-AC5] Implement advanced label rules or diagnostics.
  - ID: T016
  - Title: Implement advanced label rules
  - PRD tags: [P2-R3-AC3] [P2-R3-AC4] [P2-R3-AC5]
  - Files: expected `python/forge3d/label_plan.py`, inspect `python/forge3d/map_plate.py`
  - Action: Implement road/river presets, leader-line placement, and multi-class priority presets where available; otherwise emit typed diagnostics and deterministic rejected-label reasons.
  - Verification: `pytest tests/test_p2_advanced_label_rules.py -q`
  - Done evidence: Advanced rule tests pass without silent no-op placement.
  - Completion evidence (2026-05-18): Added cartographic priority preset handling, line placement presets, and deterministic leader-line candidate details in `python/forge3d/label_plan.py`; focused label/P2 regression command reported `10 passed`.
  - Dependencies: T015

- [X] T017 [P] [PRD: P2-R3-AC6] Add complex-script shaping decision and diagnostic tests.
  - ID: T017
  - Title: Add shaping decision tests
  - PRD tags: [P2-R3-AC6]
  - Files: `tests/test_p2_complex_shaping_decision.py`, expected `python/forge3d/label_plan.py`, `docs/guides/label_support_matrix.md`
  - Action: Add tests requiring either a prioritized HarfBuzz-compatible shaping path with behavior tests or documented non-blocking deferral plus diagnostics for labels requiring that path.
  - Verification: `pytest tests/test_p2_complex_shaping_decision.py -q`
  - Done evidence: Tests fail if complex-script shaping is silently treated as supported or becomes an undocumented MVP blocker.
  - Completion evidence (2026-05-18): Added `tests/test_p2_complex_shaping_decision.py`. RED verification failed because complex-script text was accepted as a normal point label without shaping diagnostics.
  - Dependencies: T003

- [X] T018 [PRD: P2-R3-AC6] Implement shaping deferral or prioritized path.
  - ID: T018
  - Title: Implement shaping decision handling
  - PRD tags: [P2-R3-AC6]
  - Files: expected `python/forge3d/label_plan.py`, `docs/guides/label_support_matrix.md`, inspect `src/labels/typography.rs`
  - Action: Implement the documented product decision: either wire a real shaping path with tests, or return honest non-blocking diagnostics and update support docs.
  - Verification: `pytest tests/test_p2_complex_shaping_decision.py -q`
  - Done evidence: Shaping tests pass and support docs match the product decision.
  - Completion evidence (2026-05-18): `LabelPlan.compile()` now emits `experimental_feature` for complex-script shaping and rejects it deterministically as diagnostic-bearing; `docs/guides/label_support_matrix.md` records the non-MVP-blocking HarfBuzz-compatible shaping deferral. Full P2 command reported `35 passed`.
  - Dependencies: T017

## Phase 5: Large-Scene Resource Diagnostics

- [X] T019 [P] [PRD: P2-R4-AC1] Add large-scene memory budget tests.
  - ID: T019
  - Title: Add large-scene memory tests
  - PRD tags: [P2-R4-AC1]
  - Files: `tests/test_p2_large_scene_memory.py`, expected `python/forge3d/map_scene.py`, `python/forge3d/mem.py`, `python/forge3d/pointcloud.py`, `python/forge3d/tiles3d.py`, `python/forge3d/buildings.py`
  - Action: Add tests for memory budget estimates from terrain, raster, point cloud, building, and tile metadata, including budget exceedance diagnostics and unavailable metadata reporting.
  - Verification: `pytest tests/test_p2_large_scene_memory.py -q`
  - Done evidence: Tests fail if estimates are absent where metadata exists or precision is invented where metadata is unavailable.
  - Completion evidence (2026-05-18): Added `tests/test_p2_large_scene_memory.py`. RED verification failed because no unified `large_scene.resources` summary existed.
  - Dependencies: T003

- [X] T020 [P] [PRD: P2-R4-AC2, P2-R4-AC3] Add cache/LOD and instancing status tests.
  - ID: T020
  - Title: Add cache LOD and instancing tests
  - PRD tags: [P2-R4-AC2] [P2-R4-AC3]
  - Files: `tests/test_p2_large_scene_cache_lod_instancing.py`, expected `python/forge3d/map_scene.py`, `python/forge3d/pointcloud.py`, `python/forge3d/tiles3d.py`, `python/forge3d/geometry.py`, `src/scene/py_api/instanced_mesh.rs`, `src/terrain/renderer/py_api.rs`
  - Action: Add tests for terrain, point cloud, building, and tile cache/LOD stats where available, `unavailable_cache_lod_stats` diagnostics where unavailable, and `unsupported_instancing_path` diagnostics for unsupported map-scene instancing workflows.
  - Verification: `pytest tests/test_p2_large_scene_cache_lod_instancing.py -q`
  - Done evidence: Tests distinguish available, unavailable, and unsupported stats by layer type.
  - Completion evidence (2026-05-18): Added `tests/test_p2_large_scene_cache_lod_instancing.py`. RED verification failed on absent scene-level cache/LOD/instancing normalization.
  - Dependencies: T003

- [X] T021 [P] [PRD: P2-R4-AC4] Add bottleneck layer diagnostics tests.
  - ID: T021
  - Title: Add bottleneck diagnostics tests
  - PRD tags: [P2-R4-AC4]
  - Files: `tests/test_p2_large_scene_bottlenecks.py`, expected `python/forge3d/map_scene.py`, expected `python/forge3d/diagnostics.py`
  - Action: Add tests for render diagnostics that identify bottleneck layer types from timing, memory, count, cache, or LOD data where available, with deterministic ordering.
  - Verification: `pytest tests/test_p2_large_scene_bottlenecks.py -q`
  - Done evidence: Tests fail if bottleneck summaries are nondeterministic or invented without data.
  - Completion evidence (2026-05-18): Added `tests/test_p2_large_scene_bottlenecks.py`. RED verification failed because bottleneck layer-type summaries were absent.
  - Dependencies: T003

- [X] T022 [PRD: P2-R4-AC1, P2-R4-AC2, P2-R4-AC3, P2-R4-AC4] Implement large-scene resource summaries.
  - ID: T022
  - Title: Implement large-scene resource summaries
  - PRD tags: [P2-R4-AC1] [P2-R4-AC2] [P2-R4-AC3] [P2-R4-AC4]
  - Files: expected `python/forge3d/map_scene.py`, `python/forge3d/mem.py`, `python/forge3d/pointcloud.py`, `python/forge3d/tiles3d.py`, `python/forge3d/buildings.py`, `python/forge3d/geometry.py`, inspect `src/pointcloud/`, inspect `src/tiles3d/`, inspect `src/scene/py_api/instanced_mesh.rs`
  - Action: Normalize memory estimates, cache/LOD stats, instancing status, unavailable-stat diagnostics, and bottleneck layer summaries into deterministic `LargeSceneResourceSummary` validation output.
  - Verification: `pytest tests/test_p2_large_scene_memory.py tests/test_p2_large_scene_cache_lod_instancing.py tests/test_p2_large_scene_bottlenecks.py -q`
  - Done evidence: Large-scene diagnostics tests pass without live-globe or hosted-streaming scope expansion.
  - Completion evidence (2026-05-18): Implemented opt-in `large_scene.resources` summaries in `python/forge3d/map_scene.py` via `diagnostics_policy={"large_scene_summary": True}`, normalizing memory estimates, unavailable stats, cache/LOD details, instancing status, and deterministic bottleneck layer types. Focused large-scene plus MapScene regression command reported `23 passed`.
  - Dependencies: T019, T020, T021

- [X] T023 [P] [PRD: P2-R4-AC5] Add large-scene docs audit tests.
  - ID: T023
  - Title: Add large-scene docs audit tests
  - PRD tags: [P2-R4-AC5]
  - Files: `tests/test_p2_large_scene_docs.py`, `docs/guides/offline_3d_map_rendering.md`, `docs/guides/large_scene_support.md`, `docs/guides/competitive_positioning.md`
  - Action: Add docs audit tests differentiating offline large-scene rendering from live globe streaming, hosted tile-provider parity, general DCC, game/editor, and non-map rendering.
  - Verification: `pytest tests/test_p2_large_scene_docs.py -q`
  - Done evidence: Docs audit fails on live globe, hosted streaming, or general engine overclaims.
  - Completion evidence (2026-05-18): Added `tests/test_p2_large_scene_docs.py`. RED docs group failed because `docs/guides/large_scene_support.md` was missing.
  - Dependencies: T001

- [X] T024 [PRD: P2-R4-AC5, P2-R4-AC1, P2-R4-AC2, P2-R4-AC3, P2-R4-AC4] Update large-scene diagnostics docs.
  - ID: T024
  - Title: Update large-scene documentation
  - PRD tags: [P2-R4-AC5] [P2-R4-AC1] [P2-R4-AC2] [P2-R4-AC3] [P2-R4-AC4]
  - Files: `docs/guides/large_scene_support.md`, `docs/guides/offline_3d_map_rendering.md`, `docs/guides/competitive_positioning.md`, `docs/index.rst`
  - Action: Document memory estimates, cache/LOD stat availability, instancing status, bottleneck diagnostics, unavailable-stat diagnostics, and offline-first scope boundaries.
  - Verification: `pytest tests/test_p2_large_scene_docs.py -q`
  - Done evidence: Docs audit passes and support wording preserves offline map scope.
  - Completion evidence (2026-05-18): Added `docs/guides/large_scene_support.md` and linked it from `docs/index.rst`; updated competitive/offline docs to keep live globe, hosted tile-provider, DCC, and game/editor parity as non-goals. Full P2 command reported `35 passed`.
  - Dependencies: T023

## Phase 6: Integrated Verification, Docs, and Continuity

- [X] T025 [P] [PRD: P2-R1-AC2, P2-R2-AC4, P2-R3-AC1, P2-R4-AC4] Add P2 determinism and no-op success tests.
  - ID: T025
  - Title: Add P2 determinism and no-op tests
  - PRD tags: [P2-R1-AC2] [P2-R2-AC4] [P2-R3-AC1] [P2-R4-AC4]
  - Files: `tests/test_p2_determinism_noop.py`, expected `python/forge3d/map_scene.py`, expected `python/forge3d/label_plan.py`, expected `python/forge3d/diagnostics.py`
  - Action: Add repeated validation comparisons for VT, building texture, advanced labels, and large-scene summaries, plus negative tests proving skipped families, scalar fallback, unsupported advanced labels, unavailable stats, and no-op paths cannot report success.
  - Verification: `pytest tests/test_p2_determinism_noop.py -q`
  - Done evidence: Tests fail on nondeterministic reports or silent no-op success.
  - Completion evidence (2026-05-18): Added `tests/test_p2_determinism_noop.py`; focused command with quickstart tests reported `5 passed`.
  - Dependencies: T005, T010, T014, T016, T018, T022

- [X] T026 [P] [PRD: P2-R1-AC4, P2-R2-AC4, P2-R3-AC6, P2-R4-AC5] Add P2 support docs audit tests.
  - ID: T026
  - Title: Add P2 support docs audit tests
  - PRD tags: [P2-R1-AC4] [P2-R2-AC4] [P2-R3-AC6] [P2-R4-AC5]
  - Files: `tests/test_p2_support_docs.py`, `docs/guides/virtual_texturing_support_matrix.md`, `docs/guides/building_support_matrix.md`, `docs/guides/label_support_matrix.md`, `docs/guides/large_scene_support.md`, `docs/guides/competitive_positioning.md`
  - Action: Add an integrated docs audit requiring exact PRD support-level terms and explicit non-MVP-blocking deferral text for diagnosed P2 gaps.
  - Verification: `pytest tests/test_p2_support_docs.py -q`
  - Done evidence: Docs audit fails on overclaiming or missing P2 deferral wording.
  - Completion evidence (2026-05-18): Added `tests/test_p2_support_docs.py`. RED docs group failed until integrated support docs used exact support terms, non-MVP-blocking deferral wording, and no parity overclaims.
  - Dependencies: T007, T012, T018, T024

- [X] T027 [PRD: P2-R1-AC4, P2-R2-AC4, P2-R3-AC6, P2-R4-AC5] Update integrated P2 support docs.
  - ID: T027
  - Title: Update integrated P2 support docs
  - PRD tags: [P2-R1-AC4] [P2-R2-AC4] [P2-R3-AC6] [P2-R4-AC5]
  - Files: `docs/guides/virtual_texturing_support_matrix.md`, `docs/guides/building_support_matrix.md`, `docs/guides/label_support_matrix.md`, `docs/guides/large_scene_support.md`, `docs/guides/competitive_positioning.md`, `docs/index.rst`
  - Action: Ensure all P2 support matrices and docs use exact support classifications, document implemented vs diagnosed paths, and state deferred P2 gaps do not block P0 MVP readiness.
  - Verification: `pytest tests/test_p2_support_docs.py -q`
  - Done evidence: Integrated P2 docs audit passes.
  - Completion evidence (2026-05-18): Updated VT, building, label, large-scene, offline, competitive-positioning docs and `docs/index.rst`; docs audit command reported `8 passed`, and full P2 command reported `35 passed`.
  - Dependencies: T026

- [X] T028 [P] [PRD: P2-R1-AC1, P2-R2-AC5, P2-R3-AC1, P2-R4-AC1] Add quickstart validation tests.
  - ID: T028
  - Title: Add P2 quickstart validation tests
  - PRD tags: [P2-R1-AC1] [P2-R2-AC5] [P2-R3-AC1] [P2-R4-AC1]
  - Files: `tests/test_p2_quickstart.py`, `specs/006-material-vt-large-scene-p2/quickstart.md`, expected `python/forge3d/map_scene.py`
  - Action: Add automated quickstart checks for VT family validation, textured buildings, advanced labels, and large-scene diagnostics.
  - Verification: `pytest tests/test_p2_quickstart.py -q`
  - Done evidence: Quickstart scenarios pass or emit typed diagnostics for unavailable P2 paths.
  - Completion evidence (2026-05-18): Added `tests/test_p2_quickstart.py`; focused command with determinism tests reported `5 passed`.
  - Dependencies: T025, T027

- [X] T029 [PRD: P2-R1-AC1, P2-R1-AC2, P2-R1-AC3, P2-R1-AC4, P2-R2-AC1, P2-R2-AC2, P2-R2-AC3, P2-R2-AC4, P2-R2-AC5, P2-R3-AC1, P2-R3-AC2, P2-R3-AC3, P2-R3-AC4, P2-R3-AC5, P2-R3-AC6, P2-R4-AC1, P2-R4-AC2, P2-R4-AC3, P2-R4-AC4, P2-R4-AC5] Run full P2 verification.
  - ID: T029
  - Title: Run full P2 verification
  - PRD tags: [P2-R1-AC1] [P2-R1-AC2] [P2-R1-AC3] [P2-R1-AC4] [P2-R2-AC1] [P2-R2-AC2] [P2-R2-AC3] [P2-R2-AC4] [P2-R2-AC5] [P2-R3-AC1] [P2-R3-AC2] [P2-R3-AC3] [P2-R3-AC4] [P2-R3-AC5] [P2-R3-AC6] [P2-R4-AC1] [P2-R4-AC2] [P2-R4-AC3] [P2-R4-AC4] [P2-R4-AC5]
  - Files: `tests/test_p2_diagnostics_contract.py`, `tests/test_p2_vt_family_validation.py`, `tests/test_p2_vt_docs.py`, `tests/test_p2_textured_building_mapscene.py`, `tests/test_p2_building_texture_diagnostics.py`, `tests/test_p2_building_texture_docs.py`, `tests/test_p2_advanced_labels_repeated_curved.py`, `tests/test_p2_advanced_label_rules.py`, `tests/test_p2_complex_shaping_decision.py`, `tests/test_p2_large_scene_memory.py`, `tests/test_p2_large_scene_cache_lod_instancing.py`, `tests/test_p2_large_scene_bottlenecks.py`, `tests/test_p2_large_scene_docs.py`, `tests/test_p2_determinism_noop.py`, `tests/test_p2_support_docs.py`, `tests/test_p2_quickstart.py`
  - Action: Run the full P2 feature test set and record the result summary.
  - Verification: `pytest tests/test_p2_diagnostics_contract.py tests/test_p2_vt_family_validation.py tests/test_p2_vt_docs.py tests/test_p2_textured_building_mapscene.py tests/test_p2_building_texture_diagnostics.py tests/test_p2_building_texture_docs.py tests/test_p2_advanced_labels_repeated_curved.py tests/test_p2_advanced_label_rules.py tests/test_p2_complex_shaping_decision.py tests/test_p2_large_scene_memory.py tests/test_p2_large_scene_cache_lod_instancing.py tests/test_p2_large_scene_bottlenecks.py tests/test_p2_large_scene_docs.py tests/test_p2_determinism_noop.py tests/test_p2_support_docs.py tests/test_p2_quickstart.py -q`
  - Done evidence: All feature tests pass or blockers are recorded with exact failing commands.
  - Completion evidence (2026-05-18): Full P2 verification command with `PYTHONPATH=python` reported `35 passed in 0.93s`; `py_compile` for changed modules and new P2 tests exited `0`.
  - Dependencies: T028

- [X] T030 [PRD: P2-R1-AC1, P2-R1-AC2, P2-R1-AC3, P2-R1-AC4, P2-R2-AC1, P2-R2-AC2, P2-R2-AC3, P2-R2-AC4, P2-R2-AC5, P2-R3-AC1, P2-R3-AC2, P2-R3-AC3, P2-R3-AC4, P2-R3-AC5, P2-R3-AC6, P2-R4-AC1, P2-R4-AC2, P2-R4-AC3, P2-R4-AC4, P2-R4-AC5] Update requirements verification matrix.
  - ID: T030
  - Title: Update requirements verification matrix
  - PRD tags: [P2-R1-AC1] [P2-R1-AC2] [P2-R1-AC3] [P2-R1-AC4] [P2-R2-AC1] [P2-R2-AC2] [P2-R2-AC3] [P2-R2-AC4] [P2-R2-AC5] [P2-R3-AC1] [P2-R3-AC2] [P2-R3-AC3] [P2-R3-AC4] [P2-R3-AC5] [P2-R3-AC6] [P2-R4-AC1] [P2-R4-AC2] [P2-R4-AC3] [P2-R4-AC4] [P2-R4-AC5]
  - Files: `docs/superpowers/state/requirements-verification-matrix.md`
  - Action: Update P2 rows with exact source, tests, docs, diagnostics, and command evidence; use `Deferred with diagnostic` only when typed diagnostics, docs, and negative tests exist.
  - Verification: `rg -n "P2-R[1-4]-AC[0-9]+|006-material-vt-large-scene-p2|tests/test_p2_" docs/superpowers/state/requirements-verification-matrix.md`
  - Done evidence: Matrix rows reference concrete artifacts and statuses match actual evidence.
  - Completion evidence (2026-05-18): Updated `docs/superpowers/state/requirements-verification-matrix.md` with P2-R1 through P2-R4 statuses, exact test/docs evidence, deferred-with-diagnostic rows for textured PBR runtime, curved labels, and complex-script shaping, plus diagnostic inventory updates for VT, textured building, large-scene memory/cache/LOD, and instancing. Verification `rg -n "P2-R[1-4]-AC[0-9]+|006-material-vt-large-scene-p2|tests/test_p2_" docs\superpowers\state\requirements-verification-matrix.md` exited `0`.
  - Dependencies: T029

- [X] T031 [PRD: P2-R1-AC2, P2-R2-AC4, P2-R3-AC6, P2-R4-AC5] Update implementation ledger and current context pack.
  - ID: T031
  - Title: Update continuity state
  - PRD tags: [P2-R1-AC2] [P2-R2-AC4] [P2-R3-AC6] [P2-R4-AC5]
  - Files: `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/current-context-pack.md`, `docs/superpowers/state/open-blockers.md`
  - Action: Record completed tasks, verification commands, dependency blockers, implemented vs diagnosed P2 paths, non-MVP-blocking deferrals, docs evidence, and residual risks.
  - Verification: `rg -n "006-material-vt-large-scene-p2|P2-R|vt_unsupported_family|missing_texture_path|unsupported_instancing_path|non-MVP-blocking" docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md docs/superpowers/state/open-blockers.md`
  - Done evidence: Ledger and context pack contain current outcomes and no unverified success claims.
  - Completion evidence (2026-05-18): Updated `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/current-context-pack.md`, and `docs/superpowers/state/open-blockers.md` with feature `006` Milestone 5 outcomes, non-MVP-blocking P2 deferrals, verification commands, and mitigated blocker status. Verification `rg -n "006-material-vt-large-scene-p2|P2-R|vt_unsupported_family|missing_texture_path|unsupported_instancing_path|non-MVP-blocking" docs\superpowers\state\implementation-ledger.md docs\superpowers\state\current-context-pack.md docs\superpowers\state\open-blockers.md` exited `0`.
  - Dependencies: T030

## Dependencies and Execution Order

- T001 blocks all source implementation.
- T002, T004, T006, T008, T009, T011, T013, T015, T017, T019, T020, T021, T023, T025, T026, and T028 can be authored in parallel where prerequisites are satisfied because they target separate tests/docs.
- Implementation follows tests: T003 after T002, T005 after T004, T007 after T006, T010 after T008/T009, T012 after T011, T014 after T013, T016 after T015, T018 after T017, T022 after T019/T020/T021, T024 after T023, and T027 after T026.
- T030 and T031 happen only after verification evidence exists.

## PRD Coverage Check

- P2-R1-AC1: T001, T004, T005, T028, T029, T030
- P2-R1-AC2: T002, T003, T004, T005, T025, T029, T030, T031
- P2-R1-AC3: T004, T005, T029, T030
- P2-R1-AC4: T006, T007, T026, T027, T029, T030
- P2-R2-AC1: T001, T008, T010, T029, T030
- P2-R2-AC2: T002, T003, T009, T010, T029, T030
- P2-R2-AC3: T002, T003, T009, T010, T029, T030
- P2-R2-AC4: T002, T003, T009, T010, T011, T012, T025, T026, T027, T029, T030, T031
- P2-R2-AC5: T008, T010, T012, T028, T029, T030
- P2-R3-AC1: T001, T013, T014, T025, T028, T029, T030
- P2-R3-AC2: T013, T014, T029, T030
- P2-R3-AC3: T015, T016, T029, T030
- P2-R3-AC4: T015, T016, T029, T030
- P2-R3-AC5: T015, T016, T029, T030
- P2-R3-AC6: T017, T018, T026, T027, T029, T030, T031
- P2-R4-AC1: T001, T019, T022, T024, T028, T029, T030
- P2-R4-AC2: T002, T003, T020, T022, T024, T029, T030
- P2-R4-AC3: T002, T003, T020, T022, T024, T029, T030
- P2-R4-AC4: T021, T022, T024, T025, T029, T030
- P2-R4-AC5: T023, T024, T026, T027, T029, T030, T031

## Detailed Pre-Implementation Coverage Audit Table

| PRD requirement | Acceptance criterion | Task IDs covering it | Test task IDs | Docs task IDs | Verification command | Risk if omitted |
|---|---|---|---|---|---|---|
| P2-R1 Virtual Texture Normal/Mask | P2-R1-AC1 normal/mask rendered or validation unsupported | T001, T004, T005, T028, T029, T030 | T004, T028 | T007, T030 | `pytest tests/test_p2_vt_family_validation.py tests/test_p2_quickstart.py -q` | Non-albedo VT requests can look accepted while runtime cannot render them. |
| P2-R1 Virtual Texture Normal/Mask | P2-R1-AC2 no silent non-albedo skip | T002, T003, T004, T005, T025, T029, T030, T031 | T002, T004, T025 | T007, T026, T027, T030, T031 | `pytest tests/test_p2_diagnostics_contract.py tests/test_p2_vt_family_validation.py tests/test_p2_determinism_noop.py -q` | Normal/mask families can be skipped without structured diagnostics. |
| P2-R1 Virtual Texture Normal/Mask | P2-R1-AC3 albedo-only and albedo+normal/mask tests | T004, T005, T029, T030 | T004 | T030 | `pytest tests/test_p2_vt_family_validation.py -q` | Test coverage misses the configuration that exposes the runtime gap. |
| P2-R1 Virtual Texture Normal/Mask | P2-R1-AC4 exact support docs | T006, T007, T026, T027, T029, T030 | T006, T026 | T006, T007, T026, T027, T030 | `pytest tests/test_p2_vt_docs.py tests/test_p2_support_docs.py -q` | VT docs can overclaim normal/mask runtime support. |
| P2-R2 Textured PBR Buildings | P2-R2-AC1 albedo texture support or diagnostic | T001, T008, T010, T029, T030 | T008 | T012, T030 | `pytest tests/test_p2_textured_building_mapscene.py -q` | Scalar material fallback can be mistaken for textured PBR support. |
| P2-R2 Textured PBR Buildings | P2-R2-AC2 UV diagnostics | T002, T003, T009, T010, T029, T030 | T002, T009 | T012, T030 | `pytest tests/test_p2_diagnostics_contract.py tests/test_p2_building_texture_diagnostics.py -q` | Building texture failures lack actionable UV evidence. |
| P2-R2 Textured PBR Buildings | P2-R2-AC3 missing texture path diagnostics | T002, T003, T009, T010, T029, T030 | T002, T009 | T012, T030 | `pytest tests/test_p2_diagnostics_contract.py tests/test_p2_building_texture_diagnostics.py -q` | Missing assets are discovered after render or silently downgraded. |
| P2-R2 Textured PBR Buildings | P2-R2-AC4 explicit material fallback and docs | T002, T003, T009, T010, T011, T012, T025, T026, T027, T029, T030, T031 | T002, T009, T011, T025, T026 | T011, T012, T026, T027, T030, T031 | `pytest tests/test_p2_building_texture_diagnostics.py tests/test_p2_building_texture_docs.py tests/test_p2_determinism_noop.py tests/test_p2_support_docs.py -q` | Fallback materials can be presented as successful textured workflows. |
| P2-R2 Textured PBR Buildings | P2-R2-AC5 textured building fixture through `MapScene` | T008, T010, T012, T028, T029, T030 | T008, T028 | T012, T030 | `pytest tests/test_p2_textured_building_mapscene.py tests/test_p2_quickstart.py -q` | End-to-end textured building behavior is unproven. |
| P2-R3 Advanced Static Labels | P2-R3-AC1 repeated labels along long lines | T001, T013, T014, T025, T028, T029, T030 | T013, T025, T028 | T027, T030 | `pytest tests/test_p2_advanced_labels_repeated_curved.py tests/test_p2_determinism_noop.py tests/test_p2_quickstart.py -q` | Repeated label placement can be nondeterministic or a no-op. |
| P2-R3 Advanced Static Labels | P2-R3-AC2 curved text along paths | T013, T014, T029, T030 | T013 | T027, T030 | `pytest tests/test_p2_advanced_labels_repeated_curved.py -q` | Curved text can be overclaimed without renderable behavior or diagnostics. |
| P2-R3 Advanced Static Labels | P2-R3-AC3 road and river rules | T015, T016, T029, T030 | T015 | T027, T030 | `pytest tests/test_p2_advanced_label_rules.py -q` | Road/river presets remain undocumented placeholders. |
| P2-R3 Advanced Static Labels | P2-R3-AC4 landmark and callout leader-line placement | T015, T016, T029, T030 | T015 | T027, T030 | `pytest tests/test_p2_advanced_label_rules.py -q` | Callout or leader-line workflows can silently fail. |
| P2-R3 Advanced Static Labels | P2-R3-AC5 multi-class priority presets | T015, T016, T029, T030 | T015 | T027, T030 | `pytest tests/test_p2_advanced_label_rules.py -q` | Advanced label conflicts lack deterministic priority behavior. |
| P2-R3 Advanced Static Labels | P2-R3-AC6 optional complex-script shaping decision | T017, T018, T026, T027, T029, T030, T031 | T017, T026 | T017, T018, T026, T027, T030, T031 | `pytest tests/test_p2_complex_shaping_decision.py tests/test_p2_support_docs.py -q` | Complex-script shaping can become an accidental hidden blocker or overclaim. |
| P2-R4 Large-Scene Maturity | P2-R4-AC1 memory budget estimates | T001, T019, T022, T024, T028, T029, T030 | T019, T028 | T023, T024, T030 | `pytest tests/test_p2_large_scene_memory.py tests/test_p2_quickstart.py -q` | Large scenes have no pre-render memory-risk signal. |
| P2-R4 Large-Scene Maturity | P2-R4-AC2 cache/LOD stats where available | T002, T003, T020, T022, T024, T029, T030 | T002, T020 | T023, T024, T030 | `pytest tests/test_p2_diagnostics_contract.py tests/test_p2_large_scene_cache_lod_instancing.py -q` | Cache or LOD availability remains fragmented or invented. |
| P2-R4 Large-Scene Maturity | P2-R4-AC3 instancing workflows surfaced or diagnosed | T002, T003, T020, T022, T024, T029, T030 | T002, T020 | T023, T024, T030 | `pytest tests/test_p2_diagnostics_contract.py tests/test_p2_large_scene_cache_lod_instancing.py -q` | Instancing paths can become no-op map-scene settings. |
| P2-R4 Large-Scene Maturity | P2-R4-AC4 bottleneck layer diagnostics | T021, T022, T024, T025, T029, T030 | T021, T025 | T023, T024, T030 | `pytest tests/test_p2_large_scene_bottlenecks.py tests/test_p2_determinism_noop.py -q` | Render bottlenecks remain invisible or nondeterministic. |
| P2-R4 Large-Scene Maturity | P2-R4-AC5 offline scope docs | T023, T024, T026, T027, T029, T030, T031 | T023, T026 | T023, T024, T026, T027, T030, T031 | `pytest tests/test_p2_large_scene_docs.py tests/test_p2_support_docs.py -q` | Large-scene docs can imply live-globe, hosted tile, DCC, or game-engine parity. |

## Audit Rerun Result

- PASS: Every P2-R1 through P2-R4 acceptance criterion has task coverage, test IDs, docs/support IDs where needed, and verification commands.
- PASS: P2 deferrals are explicitly non-MVP-blocking and require typed diagnostics, support docs, and blocker records when P0/P1 foundations are missing.
- PASS: VT, textured building, advanced label, and large-scene diagnostic requirements are structured and serializable, not printed-text-only.

## Extension Hooks

**Optional Hook**: git  
Command: `/speckit.git.commit`  
Description: Auto-commit after task generation  

Prompt: Commit task changes?  
To execute: `/speckit.git.commit`

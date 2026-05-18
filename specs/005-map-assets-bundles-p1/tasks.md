# Tasks: Map Assets and Bundle Round-Trip P1

**Input**: `specs/005-map-assets-bundles-p1/plan.md`, `spec.md`, `research.md`, `data-model.md`, `quickstart.md`, and `contracts/p1-assets-bundle-contract.md`

**Scope**: P1-R1 through P1-R5, Milestone 4, P0-R5/P0-R6 diagnostic and support-honesty dependencies, and required continuity state updates.

**Required task format**: Every task below includes ID, title, PRD tags, files, action, verification, done evidence, and dependencies. The first checklist line also preserves the Spec Kit task ID and `[P]` parallel marker where the file set is independent.

**Execution rule**: Write or update tests before or alongside implementation. Do not implement a parallel P1-only `Diagnostic`, `ValidationReport`, `LabelPlan`, or `MapScene` if the P0 feature-owned contract is missing. Record a blocker and stop that implementation slice.

## Phase 1: Prerequisite Gates and Shared Contracts

- [X] T001 [PRD: P0-R5-AC1, P0-R5-AC6, P0-R6-AC5, P1-R1-AC5, P1-R5-AC2] Add prerequisite public-contract tests.
  - ID: T001
  - Title: Add P0/P1 prerequisite public-contract tests
  - PRD tags: [P0-R5-AC1] [P0-R5-AC6] [P0-R6-AC5] [P1-R1-AC5] [P1-R5-AC2]
  - Files: `tests/test_p1_prerequisite_contracts.py`, inspect `python/forge3d/__init__.py`, `python/forge3d/__init__.pyi`, expected `python/forge3d/diagnostics.py`, expected `python/forge3d/label_plan.py`, expected `python/forge3d/map_scene.py`
  - Action: Add tests asserting `Diagnostic`, `ValidationReport`, `LabelPlan`, `MapScene`, `LabelLayer`, `BuildingLayer`, `Tiles3DLayer`, `MapScene.save_bundle`, and `MapScene.load_bundle` are public only through the feature-owned product API paths.
  - Verification: `pytest tests/test_p1_prerequisite_contracts.py -q`
  - Done evidence: Tests fail with explicit missing-contract assertions before P0 dependencies exist, then pass when feature 001-004 contracts are available.
  - Session evidence: RED `python -m pytest tests\test_p1_prerequisite_contracts.py -q` reported `2 failed, 1 passed` on missing `Tiles3DLayer` and P1 API methods; GREEN reported `3 passed`.
  - Dependencies: None

- [X] T002 [PRD: P0-R5-AC1, P0-R5-AC6, P0-R6-AC5, P1-R1-AC5, P1-R5-AC2] Inspect prerequisite implementation paths before any P1 source edits.
  - ID: T002
  - Title: Inspect prerequisite implementation paths
  - PRD tags: [P0-R5-AC1] [P0-R5-AC6] [P0-R6-AC5] [P1-R1-AC5] [P1-R5-AC2]
  - Files: `python/forge3d/`, `src/labels/`, `src/bundle/`, `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/requirements-verification-matrix.md`, `docs/superpowers/state/open-blockers.md`
  - Action: Run the inspection command, identify the actual owner files for diagnostics, validation reports, label planning, and `MapScene`; if any P0 owner contract is missing, record a typed diagnostic/blocker entry with code `p0_dependency_missing`, severity `fatal`, affected PRD IDs, remediation, and owner feature instead of creating duplicate P1 contracts.
  - Verification: `rg -n "class (Diagnostic|ValidationReport|LabelPlan|MapScene)|def save_bundle|def load_bundle" python src docs/superpowers/state`
  - Done evidence: Ledger records exact source paths to extend, or `docs/superpowers/state/open-blockers.md` records the typed `p0_dependency_missing` blocker tied to features 001 through 004.
  - Session evidence: P0 product owner paths are `python/forge3d/diagnostics.py`, `python/forge3d/label_plan.py`, and `python/forge3d/map_scene.py`; no `p0_dependency_missing` blocker was needed. Verification `rg -n "class Diagnostic|class ValidationReport|class LabelPlan|class MapScene|def save_bundle|def load_bundle" python src docs\superpowers\state` found those paths.
  - Dependencies: T001

- [X] T003 [P] [PRD: P0-R5-AC1, P0-R5-AC2, P0-R5-AC3, P0-R5-AC4, P0-R5-AC5, P0-R5-AC6] Add P1 diagnostic serialization and no-silent-ignore tests.
  - ID: T003
  - Title: Add diagnostic contract tests for P1 workflows
  - PRD tags: [P0-R5-AC1] [P0-R5-AC2] [P0-R5-AC3] [P0-R5-AC4] [P0-R5-AC5] [P0-R5-AC6]
  - Files: `tests/test_p1_diagnostics_contract.py`, inspect actual diagnostic module path from T002
  - Action: Add tests for structured fields, severity values, affected layer/object IDs where knowable, bundle serialization, fail-on-warning behavior, and unsupported/pro-gated/placeholder no-op prevention.
  - Verification: `pytest tests/test_p1_diagnostics_contract.py -q`
  - Done evidence: Tests cover `crs_mismatch`, `missing_glyphs`, `pro_gated_path`, `placeholder_fallback`, `experimental_feature`, `python_public_3dtiles_incomplete`, `missing_label_field`, `unicode_coverage_gap`, `unsupported_tile_format`, `unsupported_tile_feature`, `missing_external_asset`, and `unavailable_terrain_sampler`.
  - Session evidence: RED `python -m pytest tests\test_p1_diagnostics_contract.py -q` reported `2 failed` on missing feature-local diagnostic factories; GREEN reported `2 passed`.
  - red/yellow remediation evidence: `tests/test_p1_diagnostics_contract.py::test_prd_p0_diagnostic_inventory_is_structured_serializable_and_actionable` now covers the required P0 diagnostic inventory: `crs_mismatch`, `missing_glyphs`, `unsupported_style_field`, `unsupported_style_layer_type`, `pro_gated_path`, `placeholder_fallback`, `experimental_feature`, `vt_unsupported_family`, `python_public_3dtiles_incomplete`, `estimated_gpu_memory`, and `label_rejection_summary`, plus deterministic `ValidationReport` serialization. Focused remediation command reported `18 passed`.
  - Dependencies: T001

- [X] T004 [P] [PRD: P0-R6-AC1, P0-R6-AC2, P0-R6-AC3, P0-R6-AC4, P0-R6-AC5, P0-R6-AC6, P1-R4-AC6] Add support-matrix wording audit tests.
  - ID: T004
  - Title: Add support-matrix wording audit tests
  - PRD tags: [P0-R6-AC1] [P0-R6-AC2] [P0-R6-AC3] [P0-R6-AC4] [P0-R6-AC5] [P0-R6-AC6] [P1-R4-AC6]
  - Files: `tests/test_p1_docs_support_matrix.py`, `docs/guides/feature_map.md`, `docs/guides/data_and_scene_workflows.md`, `docs/api/api_reference.rst`
  - Action: Add docs-audit tests that require local/provided style scope, supported layer-type listings, unsupported style diagnostics, style output feedability for labels, and no full Mapbox or Cesium runtime parity claims.
  - Verification: `pytest tests/test_p1_docs_support_matrix.py -q`
  - Done evidence: Audit tests fail on missing or overclaiming support text and pass only when exact PRD support-level terms are present.
  - Session evidence: red/yellow remediation added `tests/test_p1_docs_support_matrix.py` and docs updates in `docs/guides/data_and_scene_workflows.md`, `docs/guides/feature_map.md`, and `docs/api/api_reference.rst`. The audit locks local/provided style scope, supported `fill`, `line`, and `circle` wording, unsupported style diagnostics, no full Mapbox GL or Cesium runtime parity claims, P1 asset API support boundaries, and the top-level building compatibility decision. Focused remediation command reported `18 passed`.
  - Dependencies: T001

- [X] T005 [P] [PRD: P1-R1-AC1, P1-R2-AC1, P1-R3-AC2, P1-R4-AC1, P1-R5-AC5] Create deterministic P1 fixture inventory tests.
  - ID: T005
  - Title: Create P1 fixture inventory tests
  - PRD tags: [P1-R1-AC1] [P1-R2-AC1] [P1-R3-AC2] [P1-R4-AC1] [P1-R5-AC5]
  - Files: `tests/test_p1_fixture_inventory.py`, expected `tests/fixtures/p1_labels.geojson`, expected `tests/fixtures/p1_tileset/tileset.json`, `assets/geojson/sample_buildings.city.json`, `assets/geojson/mount_fuji_buildings.geojson`, `assets/fonts/default_atlas.json`, `assets/fonts/default_atlas.png`
  - Action: Add fixture inventory tests for labels, fonts, buildings, 3D Tiles, and bundles; synthesize temporary fixtures inside tests where checked-in binary fixtures are intentionally absent.
  - Verification: `pytest tests/test_p1_fixture_inventory.py -q`
  - Done evidence: Tests prove fixture availability or explicit synthetic fixture generation without network access.
  - Session evidence: `python -m pytest tests\test_p1_fixture_inventory.py -q` reported `3 passed`, covering synthetic label and tileset fixtures plus checked-in default atlas/building fixtures.
  - red/yellow remediation evidence: `tests/test_p1_fixture_inventory.py::test_p1_bundle_fixture_manifest_can_be_synthesized_deterministically` adds bundle fixture manifest coverage without network or binary assets, including deterministic `scene/mapscene_recipe.json`, `scene/mapscene_review.json`, `scene/state.json`, and sorted source layer IDs. Focused remediation command reported `18 passed`.
  - Dependencies: T001

- [X] T006 [PRD: P0-R5-AC6, P1-R1-AC5, P1-R3-AC1, P1-R4-AC1, P1-R5-AC2] Wire P1 implementations only to existing P0 product contracts.
  - ID: T006
  - Title: Wire to existing P0 product contracts
  - PRD tags: [P0-R5-AC6] [P1-R1-AC5] [P1-R3-AC1] [P1-R4-AC1] [P1-R5-AC2]
  - Files: actual paths discovered by T002; expected `python/forge3d/labels.py`, expected `python/forge3d/map_scene.py`, expected `python/forge3d/diagnostics.py`, `python/forge3d/__init__.py`, `python/forge3d/__init__.pyi`, `docs/superpowers/state/open-blockers.md`
  - Action: Add or extend only the P0-owned product modules so P1 classes use shared diagnostics, validation reports, label plans, and map-scene state; if a P0 module does not exist, stop, emit or record typed diagnostic `p0_dependency_missing`, and update the blocker record before any P1 implementation.
  - Verification: `pytest tests/test_p1_prerequisite_contracts.py tests/test_p1_diagnostics_contract.py -q`
  - Done evidence: Public P1 APIs are exported from product paths and do not create duplicate contracts, or the typed P0 dependency diagnostic and blocker record exist.
  - Session evidence: Implemented P1 API wiring in existing P0 product modules only: `Diagnostic`/`ValidationReport` in `python/forge3d/diagnostics.py`, `LabelPlan` in `python/forge3d/label_plan.py`, and `MapScene`/asset layer contracts in `python/forge3d/map_scene.py`. Combined P1 command reported `8 passed`; MapScene regression subset reported `34 passed`; `py_compile` exited `0`.
  - red/yellow remediation evidence: compatibility decision is explicit in `tests/test_p1_prerequisite_contracts.py`: product `BuildingLayer.from_geojson`, `from_cityjson`, and `from_mesh` are owned by `forge3d.map_scene.BuildingLayer` and exposed at top level as `MapSceneBuildingLayer`, while legacy `forge3d.BuildingLayer` remains the `forge3d.buildings.BuildingLayer` class for backward compatibility. Focused remediation command reported `18 passed`.
  - Dependencies: T001, T002, T003

## Phase 2: User Story 1 - Ingest Geospatial Labels Into MapScene

**Goal**: Data-driven point, line, and polygon label sources compile into deterministic label plans or typed diagnostics.

**Independent test**: Run the User Story 1 tests and verify accepted/rejected label output is stable for repeated inputs.

- [X] T007 [P] [US1] [PRD: P1-R1-AC1, P0-R5-AC6] Add LabelLayer geometry ingestion tests.
  - ID: T007
  - Title: Add geometry ingestion tests for LabelLayer
  - PRD tags: [P1-R1-AC1] [P0-R5-AC6]
  - Files: `tests/test_p1_label_layer_geometry.py`, expected `python/forge3d/labels.py`
  - Action: Add tests for `LabelLayer.from_features`, `from_geodataframe` when GeoPandas is available, `from_style_layer`, point/line/polygon candidate creation, invalid geometry diagnostics, and unsupported geometry negative cases.
  - Verification: `pytest tests/test_p1_label_layer_geometry.py -q`
  - Done evidence: Tests distinguish accepted candidates from rejected geometry with typed diagnostics.
  - Session evidence: Added `tests/test_p1_label_layer_geometry.py` covering deterministic point, line, and polygon candidate ingestion; invalid/empty geometry diagnostics; unsupported geometry diagnostics with affected feature IDs; GeoDataFrame-like ingestion; and style-layer text field ingestion. RED command reported `4 failed`; GREEN command reported `4 passed`.
  - Dependencies: T001, T005

- [X] T008 [US1] [PRD: P1-R1-AC1, P0-R5-AC6] Implement LabelLayer geometry ingestion.
  - ID: T008
  - Title: Implement LabelLayer geometry ingestion
  - PRD tags: [P1-R1-AC1] [P0-R5-AC6]
  - Files: `python/forge3d/labels.py`, `python/forge3d/__init__.py`, `python/forge3d/__init__.pyi`
  - Action: Implement `LabelLayer.from_features`, `from_geodataframe`, and `from_style_layer` to normalize point, line, and polygon sources into deterministic label candidates or typed diagnostics.
  - Verification: `pytest tests/test_p1_label_layer_geometry.py -q`
  - Done evidence: Geometry tests pass without treating unsupported or invalid geometry as successful ingestion.
  - Session evidence: Implemented geometry validation in `python/forge3d/map_scene.py` for `Point`, `LineString`, and `Polygon`; invalid geometry emits `placeholder_fallback` with object IDs; unsupported geometry emits `unsupported_feature` with object IDs; accepted label payloads now carry `geometry_type` and `placement_kind`; accepted label ordering is deterministic. Verification `python -m pytest tests\test_p1_label_layer_geometry.py -q` reported `4 passed`.
  - Dependencies: T006, T007

- [X] T009 [P] [US1] [PRD: P1-R1-AC2, P1-R1-AC3, P0-R5-AC3] Add CRS and terrain-sampling tests.
  - ID: T009
  - Title: Add LabelLayer CRS and terrain-sampling tests
  - PRD tags: [P1-R1-AC2] [P1-R1-AC3] [P0-R5-AC3]
  - Files: `tests/test_p1_label_layer_crs_terrain.py`, `python/forge3d/crs.py`, expected `python/forge3d/labels.py`
  - Action: Add tests for CRS transform through forge3d utilities, CRS mismatch diagnostics, terrain sampler success, unavailable terrain sampler diagnostics, and affected layer/object ID propagation.
  - Verification: `pytest tests/test_p1_label_layer_crs_terrain.py -q`
  - Done evidence: Tests cover both available and unavailable CRS/terrain paths.
  - Dependencies: T001, T005

- [X] T010 [US1] [PRD: P1-R1-AC2, P1-R1-AC3, P0-R5-AC3] Implement CRS transform and terrain policy.
  - ID: T010
  - Title: Implement LabelLayer CRS and terrain policy
  - PRD tags: [P1-R1-AC2] [P1-R1-AC3] [P0-R5-AC3]
  - Files: `python/forge3d/labels.py`, `python/forge3d/crs.py`
  - Action: Wire label ingestion to `transform_coords` or diagnosed fallback behavior, add terrain-height sampling hooks, and report `crs_mismatch` or `unavailable_terrain_sampler` before render.
  - Verification: `pytest tests/test_p1_label_layer_crs_terrain.py -q`
  - Done evidence: CRS and terrain tests pass with structured diagnostics on unavailable paths.
  - Dependencies: T008, T009

- [X] T011 [P] [US1] [PRD: P1-R1-AC4, P0-R5-AC6] Add label expression tests.
  - ID: T011
  - Title: Add label expression tests
  - PRD tags: [P1-R1-AC4] [P0-R5-AC6]
  - Files: `tests/test_p1_label_expressions.py`, `python/forge3d/style_expressions.py`, expected `python/forge3d/labels.py`
  - Action: Add tests for `{name}`, `get`, `concat`, `coalesce`, `upcase`, `downcase`, stable expression ordering, and `missing_label_field` diagnostics when properties are absent.
  - Verification: `pytest tests/test_p1_label_expressions.py -q`
  - Done evidence: Missing fields do not silently produce empty accepted labels.
  - Dependencies: T001, T005

- [X] T012 [US1] [PRD: P1-R1-AC4, P0-R5-AC6] Implement label text expressions.
  - ID: T012
  - Title: Implement label text expressions
  - PRD tags: [P1-R1-AC4] [P0-R5-AC6]
  - Files: `python/forge3d/labels.py`, `python/forge3d/style_expressions.py`
  - Action: Implement simple text-expression evaluation for label ingestion and convert missing fields into `missing_label_field` diagnostics with remediation.
  - Verification: `pytest tests/test_p1_label_expressions.py -q`
  - Done evidence: Expression tests pass for supported forms and missing-field diagnostics.
  - Dependencies: T008, T011

- [X] T013 [P] [US1] [PRD: P1-R1-AC5, P0-R3-AC1] Add LabelLayer to LabelPlan integration tests.
  - ID: T013
  - Title: Add LabelLayer to LabelPlan integration tests
  - PRD tags: [P1-R1-AC5] [P0-R3-AC1]
  - Files: `tests/test_p1_label_plan_integration.py`, expected `python/forge3d/labels.py`, expected `python/forge3d/label_plan.py`
  - Action: Add tests proving `label_layer.compile_labels(camera, viewport, terrain)` feeds the feature 003 `LabelPlan.compile()` contract and returns deterministic accepted/rejected labels for repeated fixed inputs.
  - Verification: `pytest tests/test_p1_label_plan_integration.py -q`
  - Done evidence: Tests compare stable serialized label-plan output exactly.
  - Dependencies: T001, T005

- [X] T014 [US1] [PRD: P1-R1-AC5, P0-R3-AC1] Implement compile_labels integration.
  - ID: T014
  - Title: Implement LabelLayer compile_labels integration
  - PRD tags: [P1-R1-AC5] [P0-R3-AC1]
  - Files: `python/forge3d/labels.py`, expected `python/forge3d/label_plan.py`
  - Action: Implement `LabelLayer.compile_labels(camera, viewport, terrain)` as an adapter to the feature 003 `LabelPlan` compiler and preserve accepted/rejected output ordering.
  - Verification: `pytest tests/test_p1_label_plan_integration.py -q`
  - Done evidence: Integration tests pass without P1-local reimplementation of `LabelPlan`.
  - Dependencies: T006, T013

- [X] T015 [P] [US1] [PRD: P1-R1-AC6, P1-R2-AC4, P0-R5-AC6] Add missing-field and missing-glyph pre-render tests.
  - ID: T015
  - Title: Add pre-render label diagnostic tests
  - PRD tags: [P1-R1-AC6] [P1-R2-AC4] [P0-R5-AC6]
  - Files: `tests/test_p1_label_diagnostics.py`, expected `python/forge3d/labels.py`, expected `python/forge3d/typography.py`
  - Action: Add tests for missing text fields, missing glyphs, Unicode coverage gaps, and pre-render validation diagnostics.
  - Verification: `pytest tests/test_p1_label_diagnostics.py -q`
  - Done evidence: Tests fail on no-op success and pass only with typed diagnostics before render.
  - Dependencies: T001, T005

- [X] T016 [US1] [PRD: P1-R1-AC6, P1-R2-AC4, P0-R5-AC6] Implement pre-render label validation.
  - ID: T016
  - Title: Implement LabelLayer pre-render diagnostics
  - PRD tags: [P1-R1-AC6] [P1-R2-AC4] [P0-R5-AC6]
  - Files: `python/forge3d/labels.py`, expected `python/forge3d/typography.py`
  - Action: Add validation that reports `missing_label_field`, `missing_glyphs`, and `unicode_coverage_gap` before render and includes affected label or layer identifiers where possible.
  - Verification: `pytest tests/test_p1_label_diagnostics.py -q`
  - Done evidence: Diagnostic tests pass and no missing glyph path is treated as successful render preparation.
  - Dependencies: T012, T014, T015

## Phase 3: User Story 2 - Use Production Typography Basics

**Goal**: Default Latin atlas, font fallback declarations, Unicode coverage diagnostics, metrics, multiline labels, and callout typography are real or honestly diagnosed.

**Independent test**: Run the typography test files and verify unsupported generation, shaping, and layout paths produce typed diagnostics.

- [X] T017 [P] [US2] [PRD: P1-R2-AC1, P1-R1-AC6] Add default Latin atlas coverage tests.
  - ID: T017
  - Title: Add default Latin atlas tests
  - PRD tags: [P1-R2-AC1] [P1-R1-AC6]
  - Files: `tests/test_p1_font_atlas_default.py`, `assets/fonts/default_atlas.json`, `assets/fonts/default_atlas.png`, expected `python/forge3d/typography.py`
  - Action: Add tests proving `FontAtlas.default_latin()` loads the bundled atlas, reports deterministic coverage ranges, and detects absent glyphs.
  - Verification: `pytest tests/test_p1_font_atlas_default.py -q`
  - Done evidence: Tests prove the default atlas artifact and coverage metadata are present.
  - Dependencies: T001, T005

- [X] T018 [US2] [PRD: P1-R2-AC1, P1-R1-AC6] Implement default Latin FontAtlas.
  - ID: T018
  - Title: Implement default Latin FontAtlas
  - PRD tags: [P1-R2-AC1] [P1-R1-AC6]
  - Files: `python/forge3d/typography.py`, `python/forge3d/__init__.py`, `python/forge3d/__init__.pyi`, `assets/fonts/default_atlas.json`, `assets/fonts/default_atlas.png`
  - Action: Implement `FontAtlas.default_latin()` and public exports without silently substituting undocumented glyph coverage.
  - Verification: `pytest tests/test_p1_font_atlas_default.py -q`
  - Done evidence: Default atlas tests pass and docs can reference exact glyph coverage.
  - Dependencies: T006, T017

- [X] T019 [P] [US2] [PRD: P1-R2-AC2, P0-R5-AC6] Add TTF/OTF atlas generation or diagnostic tests.
  - ID: T019
  - Title: Add font-generation tests
  - PRD tags: [P1-R2-AC2] [P0-R5-AC6]
  - Files: `tests/test_p1_font_atlas_generation.py`, expected `python/forge3d/typography.py`
  - Action: Add tests for `FontAtlas.from_font(path, ranges=...)` using an available local TTF/OTF fixture or a temporary invalid font, requiring typed diagnostics instead of usable-looking placeholder atlases when generation cannot complete.
  - Verification: `pytest tests/test_p1_font_atlas_generation.py -q`
  - Done evidence: Tests cover success where real generation exists or structured failure where unavailable.
  - Dependencies: T001, T005

- [X] T020 [US2] [PRD: P1-R2-AC2, P0-R5-AC6] Implement FontAtlas.from_font behavior.
  - ID: T020
  - Title: Implement font atlas generation or diagnostics
  - PRD tags: [P1-R2-AC2] [P0-R5-AC6]
  - Files: `python/forge3d/typography.py`, inspect `src/labels/atlas.rs`, inspect `src/labels/typography.rs`
  - Action: Implement real TTF/OTF atlas generation only if substrate exists; otherwise return typed generation diagnostics and prevent placeholder atlas success.
  - Verification: `pytest tests/test_p1_font_atlas_generation.py -q`
  - Done evidence: Generation tests pass for supported or diagnosed paths.
  - Dependencies: T018, T019

- [X] T021 [P] [US2] [PRD: P1-R2-AC3, P0-R5-AC6] Add font fallback range tests.
  - ID: T021
  - Title: Add font fallback range tests
  - PRD tags: [P1-R2-AC3] [P0-R5-AC6]
  - Files: `tests/test_p1_font_fallback_ranges.py`, expected `python/forge3d/typography.py`
  - Action: Add tests for `FontFallbackRange` declarations, deterministic priority ordering, overlapping ranges, and unsupported range diagnostics.
  - Verification: `pytest tests/test_p1_font_fallback_ranges.py -q`
  - Done evidence: Tests prove fallback selection is deterministic for repeated inputs.
  - Dependencies: T001

- [X] T022 [US2] [PRD: P1-R2-AC3, P0-R5-AC6] Implement deterministic font fallback ranges.
  - ID: T022
  - Title: Implement font fallback ranges
  - PRD tags: [P1-R2-AC3] [P0-R5-AC6]
  - Files: `python/forge3d/typography.py`, `python/forge3d/__init__.pyi`
  - Action: Implement `FontFallbackRange` and deterministic selection logic for configured ranges, including diagnostics for invalid or unsupported fallback declarations.
  - Verification: `pytest tests/test_p1_font_fallback_ranges.py -q`
  - Done evidence: Fallback tests pass with stable ordered results.
  - Dependencies: T018, T021

- [X] T023 [P] [US2] [PRD: P1-R2-AC4, P1-R1-AC6] Add Unicode coverage diagnostic tests.
  - ID: T023
  - Title: Add Unicode coverage tests
  - PRD tags: [P1-R2-AC4] [P1-R1-AC6]
  - Files: `tests/test_p1_unicode_coverage.py`, expected `python/forge3d/typography.py`, expected `python/forge3d/labels.py`
  - Action: Add tests for supported Latin text, unsupported ranges, `unicode_coverage_gap`, and integration with LabelLayer pre-render validation.
  - Verification: `pytest tests/test_p1_unicode_coverage.py -q`
  - Done evidence: Tests report Unicode coverage gaps before render.
  - Dependencies: T017, T021

- [X] T024 [US2] [PRD: P1-R2-AC4, P1-R1-AC6] Implement Unicode coverage validation.
  - ID: T024
  - Title: Implement Unicode coverage validation
  - PRD tags: [P1-R2-AC4] [P1-R1-AC6]
  - Files: `python/forge3d/typography.py`, `python/forge3d/labels.py`
  - Action: Implement text coverage checks using active atlas and fallback ranges, returning `unicode_coverage_gap` diagnostics without claiming complex-script shaping support.
  - Verification: `pytest tests/test_p1_unicode_coverage.py tests/test_p1_label_diagnostics.py -q`
  - Done evidence: Unicode and label diagnostic tests pass.
  - Dependencies: T016, T022, T023

- [X] T025 [P] [US2] [PRD: P1-R2-AC5, P0-R5-AC6] Add typography metric tests.
  - ID: T025
  - Title: Add kerning, tracking, and line-height tests
  - PRD tags: [P1-R2-AC5] [P0-R5-AC6]
  - Files: `tests/test_p1_typography_metrics.py`, expected `python/forge3d/typography.py`
  - Action: Add tests for kerning, tracking, line-height, and typed unsupported diagnostics where any exposed control cannot affect layout metrics.
  - Verification: `pytest tests/test_p1_typography_metrics.py -q`
  - Done evidence: Tests prove controls either affect measurable layout state or return unsupported diagnostics.
  - Dependencies: T001

- [X] T026 [US2] [PRD: P1-R2-AC5, P0-R5-AC6] Implement TypographySettings metrics.
  - ID: T026
  - Title: Implement typography metric handling
  - PRD tags: [P1-R2-AC5] [P0-R5-AC6]
  - Files: `python/forge3d/typography.py`, inspect `src/labels/typography.rs`, inspect `src/labels/py_bindings.rs`
  - Action: Implement `TypographySettings` metric application where substrate exists, or explicit unsupported diagnostics where unavailable.
  - Verification: `pytest tests/test_p1_typography_metrics.py -q`
  - Done evidence: Metric tests pass without no-op success.
  - Dependencies: T018, T025

- [X] T027 [P] [US2] [PRD: P1-R2-AC6, P0-R5-AC6] Add multiline and callout typography tests.
  - ID: T027
  - Title: Add multiline and callout tests
  - PRD tags: [P1-R2-AC6] [P0-R5-AC6]
  - Files: `tests/test_p1_multiline_callouts.py`, expected `python/forge3d/labels.py`, expected `python/forge3d/typography.py`
  - Action: Add tests for multiline label layout, callout text handling, leader/callout unsupported diagnostics, and interaction with active typography settings.
  - Verification: `pytest tests/test_p1_multiline_callouts.py -q`
  - Done evidence: Tests cover supported or diagnosed multiline and callout paths.
  - Dependencies: T001

- [X] T028 [US2] [PRD: P1-R2-AC6, P0-R5-AC6] Implement multiline and callout typography behavior.
  - ID: T028
  - Title: Implement multiline and callout typography behavior
  - PRD tags: [P1-R2-AC6] [P0-R5-AC6]
  - Files: `python/forge3d/labels.py`, `python/forge3d/typography.py`, inspect `src/labels/callout.rs`, inspect `src/labels/leader.rs`
  - Action: Implement multiline and callout behavior where available, or typed diagnostics for unsupported layout/render cases.
  - Verification: `pytest tests/test_p1_multiline_callouts.py -q`
  - Done evidence: Multiline/callout tests pass without silently dropping text or leader behavior.
  - Dependencies: T024, T026, T027

- [X] T029 [P] [US2] [PRD: P1-R2-AC7, P1-R2-AC4, P0-R6-AC6] Add complex-script shaping deferral docs audit.
  - ID: T029
  - Title: Add complex-script shaping deferral audit
  - PRD tags: [P1-R2-AC7] [P1-R2-AC4] [P0-R6-AC6]
  - Files: `tests/test_p1_typography_docs.py`, `docs/guides/data_and_scene_workflows.md`, `docs/api/api_reference.rst`
  - Action: Add docs-audit tests requiring optional non-Latin shaping to be marked deferred, experimental, or unsupported unless a product decision explicitly makes it P1-blocking.
  - Verification: `pytest tests/test_p1_typography_docs.py -q`
  - Done evidence: Docs audit blocks overclaiming complex-script shaping support.
  - Dependencies: T004

## Phase 4: User Story 3 - Integrate Building Layers Honestly

**Goal**: Building paths expose actual support level, diagnostics, summaries, and MapScene render preparation without mistaking Pro-gated or zero-geometry fallback behavior for support.

**Independent test**: Run building P1 tests with and without the Pro fixture path where applicable.

- [X] T030 [P] [US3] [PRD: P1-R3-AC1, P0-R5-AC6] Add building support-status validation tests.
  - ID: T030
  - Title: Add building support-status tests
  - PRD tags: [P1-R3-AC1] [P0-R5-AC6]
  - Files: `tests/test_p1_building_support_status.py`, `python/forge3d/buildings.py`, expected `python/forge3d/map_scene.py`
  - Action: Add tests for native available path, Pro-gated path, placeholder/fallback path, unsupported path, and `BuildingLayer.from_geojson`, `from_cityjson`, and `from_mesh` validation.
  - Verification: `pytest tests/test_p1_building_support_status.py -q`
  - Done evidence: Tests fail if `MapScene.validate()` collapses distinct support statuses.
  - Dependencies: T001, T005

- [X] T031 [US3] [PRD: P1-R3-AC1, P0-R5-AC6] Implement BuildingLayer product adapter support statuses.
  - ID: T031
  - Title: Implement BuildingLayer support-status adapter
  - PRD tags: [P1-R3-AC1] [P0-R5-AC6]
  - Files: `python/forge3d/buildings.py`, expected `python/forge3d/map_scene.py`, `python/forge3d/__init__.py`, `python/forge3d/__init__.pyi`
  - Action: Extend or wrap existing building loaders with `from_geojson`, `from_cityjson`, and `from_mesh` product adapters that preserve exact support status and diagnostics.
  - Verification: `pytest tests/test_p1_building_support_status.py -q`
  - Done evidence: Support-status tests pass for native, Pro-gated, placeholder/fallback, and unsupported cases.
  - Dependencies: T006, T030

- [X] T032 [P] [US3] [PRD: P1-R3-AC2, P0-R5-AC6] Add CityJSON and GeoJSON MapScene render-prep tests.
  - ID: T032
  - Title: Add building render-prep tests
  - PRD tags: [P1-R3-AC2] [P0-R5-AC6]
  - Files: `tests/test_p1_building_mapscene_render.py`, `assets/geojson/sample_buildings.city.json`, `assets/geojson/mount_fuji_buildings.geojson`, expected `python/forge3d/map_scene.py`
  - Action: Add tests for CityJSON and GeoJSON fixtures through `MapScene` where native support exists, and typed non-success diagnostics where unavailable.
  - Verification: `pytest tests/test_p1_building_mapscene_render.py -q`
  - Done evidence: Tests never require unavailable native support to pass as supported.
  - Dependencies: T001, T005

- [X] T033 [US3] [PRD: P1-R3-AC2, P0-R5-AC6] Implement building MapScene render preparation.
  - ID: T033
  - Title: Implement building render preparation
  - PRD tags: [P1-R3-AC2] [P0-R5-AC6]
  - Files: `python/forge3d/buildings.py`, expected `python/forge3d/map_scene.py`
  - Action: Integrate building layers into `MapScene.validate()` and render preparation, gated by real native availability and exact diagnostics when not renderable.
  - Verification: `pytest tests/test_p1_building_mapscene_render.py -q`
  - Done evidence: Building render-prep tests pass for supported or diagnosed paths.
  - Dependencies: T031, T032

- [X] T034 [P] [US3] [PRD: P1-R3-AC3, P0-R5-AC3] Add building summary diagnostics tests.
  - ID: T034
  - Title: Add building summary diagnostics tests
  - PRD tags: [P1-R3-AC3] [P0-R5-AC3]
  - Files: `tests/test_p1_building_summaries.py`, `python/forge3d/buildings.py`, expected `python/forge3d/map_scene.py`
  - Action: Add tests for geometry count, total vertices/triangles, bounding boxes, source format, and affected layer identifiers in validation reports.
  - Verification: `pytest tests/test_p1_building_summaries.py -q`
  - Done evidence: Tests prove inspectable building data appears in layer summaries or diagnostics.
  - Dependencies: T001

- [X] T035 [US3] [PRD: P1-R3-AC3, P0-R5-AC3] Implement building layer summaries.
  - ID: T035
  - Title: Implement building summaries
  - PRD tags: [P1-R3-AC3] [P0-R5-AC3]
  - Files: `python/forge3d/buildings.py`, expected `python/forge3d/map_scene.py`
  - Action: Add deterministic building layer summaries with geometry counts, bounds, source format, and affected IDs where possible.
  - Verification: `pytest tests/test_p1_building_summaries.py -q`
  - Done evidence: Building summary diagnostics tests pass.
  - Dependencies: T031, T034

- [X] T036 [P] [US3] [PRD: P1-R3-AC4, P0-R6-AC2, P0-R6-AC6] Add scalar PBR building docs audit.
  - ID: T036
  - Title: Add scalar PBR building docs audit
  - PRD tags: [P1-R3-AC4] [P0-R6-AC2] [P0-R6-AC6]
  - Files: `tests/test_p1_building_docs.py`, `docs/guides/data_and_scene_workflows.md`, `docs/api/api_reference.rst`, `docs/tutorials/gis-track/04-3d-buildings.md`
  - Action: Add docs-audit tests requiring exact scalar PBR support status and no overclaiming of building material support.
  - Verification: `pytest tests/test_p1_building_docs.py -q`
  - Done evidence: Audit tests fail until scalar PBR status is documented with PRD support-level terms.
  - Dependencies: T004

- [X] T037 [P] [US3] [PRD: P1-R3-AC5, P0-R5-AC6] Add textured PBR support or unsupported tests.
  - ID: T037
  - Title: Add textured PBR building tests
  - PRD tags: [P1-R3-AC5] [P0-R5-AC6]
  - Files: `tests/test_p1_building_textured_pbr.py`, `python/forge3d/buildings.py`, expected `python/forge3d/map_scene.py`
  - Action: Add tests that either exercise end-to-end textured PBR support or require typed `unsupported` diagnostics and documentation when textured PBR is unavailable.
  - Verification: `pytest tests/test_p1_building_textured_pbr.py -q`
  - Done evidence: Tests prevent textured PBR from being implied by scalar material support.
  - Dependencies: T001

- [X] T038 [US3] [PRD: P1-R3-AC5, P0-R5-AC6] Implement textured PBR support status.
  - ID: T038
  - Title: Implement textured PBR support status
  - PRD tags: [P1-R3-AC5] [P0-R5-AC6]
  - Files: `python/forge3d/buildings.py`, expected `python/forge3d/map_scene.py`
  - Action: Implement textured PBR end to end only if available; otherwise add explicit unsupported diagnostics and material status in layer summaries.
  - Verification: `pytest tests/test_p1_building_textured_pbr.py -q`
  - Done evidence: Textured PBR tests pass for implemented or diagnosed behavior.
  - Dependencies: T031, T037

- [X] T039 [P] [US3] [PRD: P1-R3-AC6, P0-R5-AC6] Add zero-geometry fallback negative tests.
  - ID: T039
  - Title: Add building zero-geometry fallback tests
  - PRD tags: [P1-R3-AC6] [P0-R5-AC6]
  - Files: `tests/test_p1_building_placeholder_fallback.py`, `python/forge3d/buildings.py`
  - Action: Add negative tests proving the existing zero-geometry GeoJSON and CityJSON fallbacks emit `placeholder_fallback` diagnostics and do not count as successful ingestion.
  - Verification: `pytest tests/test_p1_building_placeholder_fallback.py -q`
  - Done evidence: Tests catch empty-position fallback success.
  - Dependencies: T001

- [X] T040 [US3] [PRD: P1-R3-AC6, P0-R5-AC6] Implement placeholder fallback diagnostics.
  - ID: T040
  - Title: Implement building placeholder fallback diagnostics
  - PRD tags: [P1-R3-AC6] [P0-R5-AC6]
  - Files: `python/forge3d/buildings.py`, expected `python/forge3d/map_scene.py`
  - Action: Convert zero-geometry fallback paths into explicit `placeholder_fallback` diagnostics and mark the layer not renderable unless real geometry exists.
  - Verification: `pytest tests/test_p1_building_placeholder_fallback.py tests/test_p1_building_support_status.py -q`
  - Done evidence: Placeholder fallback and support-status tests pass.
  - Dependencies: T031, T039

## Phase 5: User Story 4 - Load Supported 3D Tiles Into MapScene

**Goal**: Public Python `Tiles3DLayer` workflows load supported local fixtures, distinguish unsupported formats/features, expose available cache/LOD diagnostics, and avoid Cesium parity claims.

**Independent test**: Run all P1 3D Tiles tests against local synthetic fixtures.

- [X] T041 [P] [US4] [PRD: P1-R4-AC1, P0-R5-AC6] Add Tiles3DLayer public API load tests.
  - ID: T041
  - Title: Add Tiles3DLayer public load tests
  - PRD tags: [P1-R4-AC1] [P0-R5-AC6]
  - Files: `tests/test_p1_tiles3d_public_api.py`, `python/forge3d/tiles3d.py`, expected `python/forge3d/map_scene.py`
  - Action: Add tests for `Tiles3DLayer.from_tileset_json`, `Tiles3DLayer.from_b3dm`, `MapScene` layer addition, supported local tileset fixture loading, and public exports.
  - Verification: `pytest tests/test_p1_tiles3d_public_api.py -q`
  - Done evidence: Tests prove typed public API use without raw IPC.
  - Dependencies: T001, T005

- [X] T042 [US4] [PRD: P1-R4-AC1, P0-R5-AC6] Implement Tiles3DLayer public API.
  - ID: T042
  - Title: Implement Tiles3DLayer public API
  - PRD tags: [P1-R4-AC1] [P0-R5-AC6]
  - Files: `python/forge3d/tiles3d.py`, expected `python/forge3d/map_scene.py`, `python/forge3d/__init__.py`, `python/forge3d/__init__.pyi`
  - Action: Implement `Tiles3DLayer.from_tileset_json` and `from_b3dm` adapters scoped to offline local fixtures and `MapScene` validation/render preparation.
  - Verification: `pytest tests/test_p1_tiles3d_public_api.py -q`
  - Done evidence: Public API tests pass without claiming full runtime parity.
  - Dependencies: T006, T041

- [X] T043 [P] [US4] [PRD: P1-R4-AC2, P0-R5-AC6] Add supported and unsupported tile format tests.
  - ID: T043
  - Title: Add tile format validation tests
  - PRD tags: [P1-R4-AC2] [P0-R5-AC6]
  - Files: `tests/test_p1_tiles3d_format_validation.py`, `python/forge3d/tiles3d.py`
  - Action: Add tests for valid `tileset.json`, unsupported or malformed tilesets, unsupported content extensions, and `unsupported_tile_format` diagnostics.
  - Verification: `pytest tests/test_p1_tiles3d_format_validation.py -q`
  - Done evidence: Tests distinguish supported metadata from unsupported formats.
  - Dependencies: T001, T005

- [X] T044 [US4] [PRD: P1-R4-AC2, P0-R5-AC6] Implement 3D Tiles format validation diagnostics.
  - ID: T044
  - Title: Implement 3D Tiles format diagnostics
  - PRD tags: [P1-R4-AC2] [P0-R5-AC6]
  - Files: `python/forge3d/tiles3d.py`, expected `python/forge3d/map_scene.py`
  - Action: Implement supported-format checks and typed `unsupported_tile_format` diagnostics without falling back to metadata-only success.
  - Verification: `pytest tests/test_p1_tiles3d_format_validation.py -q`
  - Done evidence: Format validation tests pass.
  - Dependencies: T042, T043

- [X] T045 [P] [US4] [PRD: P1-R4-AC3, P0-R5-AC3] Add 3D Tiles cache diagnostics tests.
  - ID: T045
  - Title: Add 3D Tiles cache diagnostics tests
  - PRD tags: [P1-R4-AC3] [P0-R5-AC3]
  - Files: `tests/test_p1_tiles3d_cache_stats.py`, `python/forge3d/tiles3d.py`
  - Action: Add tests for cache stats in diagnostics or layer summaries where available, and typed unavailable diagnostics where the renderer cannot expose stats.
  - Verification: `pytest tests/test_p1_tiles3d_cache_stats.py -q`
  - Done evidence: Tests cover cache hit/miss/entry summaries or explicit unsupported diagnostics.
  - Dependencies: T001

- [X] T046 [US4] [PRD: P1-R4-AC3, P0-R5-AC3] Implement 3D Tiles cache summary adapter.
  - ID: T046
  - Title: Implement 3D Tiles cache summaries
  - PRD tags: [P1-R4-AC3] [P0-R5-AC3]
  - Files: `python/forge3d/tiles3d.py`, expected `python/forge3d/map_scene.py`, inspect `src/tiles3d/renderer.rs`
  - Action: Expose `Tiles3DLayer` cache statistics in layer summaries or diagnostics using existing renderer data when available.
  - Verification: `pytest tests/test_p1_tiles3d_cache_stats.py -q`
  - Done evidence: Cache diagnostics tests pass.
  - Dependencies: T042, T045

- [X] T047 [P] [US4] [PRD: P1-R4-AC4, P0-R5-AC6] Add LOD and screen-space-error config tests.
  - ID: T047
  - Title: Add LOD and SSE config tests
  - PRD tags: [P1-R4-AC4] [P0-R5-AC6]
  - Files: `tests/test_p1_tiles3d_lod_config.py`, `python/forge3d/tiles3d.py`, `tests/test_3dtiles_sse.py`
  - Action: Add tests for `lod` or screen-space-error configuration, deterministic visible tile selection where exposed, and typed diagnostics where unavailable through `MapScene`.
  - Verification: `pytest tests/test_p1_tiles3d_lod_config.py -q`
  - Done evidence: Tests prove configuration changes behavior or reports unsupported status.
  - Dependencies: T001

- [X] T048 [US4] [PRD: P1-R4-AC4, P0-R5-AC6] Implement LOD/SSE product adapter.
  - ID: T048
  - Title: Implement LOD/SSE adapter
  - PRD tags: [P1-R4-AC4] [P0-R5-AC6]
  - Files: `python/forge3d/tiles3d.py`, expected `python/forge3d/map_scene.py`
  - Action: Wire public LOD/SSE configuration to existing `Tiles3dRenderer` behavior or return typed unsupported diagnostics when not available in `MapScene`.
  - Verification: `pytest tests/test_p1_tiles3d_lod_config.py tests/test_3dtiles_sse.py -q`
  - Done evidence: LOD tests pass without no-op configuration success.
  - Dependencies: T042, T047

- [X] T049 [P] [US4] [PRD: P1-R4-AC5, P0-R5-AC6] Add unsupported B3DM/GLB feature tests.
  - ID: T049
  - Title: Add unsupported B3DM/GLB feature tests
  - PRD tags: [P1-R4-AC5] [P0-R5-AC6]
  - Files: `tests/test_p1_tiles3d_unsupported_features.py`, `python/forge3d/tiles3d.py`
  - Action: Add negative tests for malformed B3DM, unsupported embedded GLB/glTF features, metadata-only success, and typed `unsupported_tile_feature` diagnostics.
  - Verification: `pytest tests/test_p1_tiles3d_unsupported_features.py -q`
  - Done evidence: Tests prove unsupported payloads cannot be mistaken for renderable tile content.
  - Dependencies: T001

- [X] T050 [US4] [PRD: P1-R4-AC5, P0-R5-AC6] Implement unsupported B3DM/GLB diagnostics.
  - ID: T050
  - Title: Implement unsupported tile feature diagnostics
  - PRD tags: [P1-R4-AC5] [P0-R5-AC6]
  - Files: `python/forge3d/tiles3d.py`, inspect `src/tiles3d/b3dm.rs`, inspect `src/tiles3d/mod.rs`
  - Action: Convert unsupported B3DM/GLB parsing and rendering gaps into structured `unsupported_tile_feature` diagnostics rather than raw exceptions or NotImplemented success paths.
  - Verification: `pytest tests/test_p1_tiles3d_unsupported_features.py -q`
  - Done evidence: Unsupported feature tests pass.
  - Dependencies: T042, T049

- [X] T051 [P] [US4] [PRD: P1-R4-AC6, P0-R6-AC6] Add 3D Tiles no-Cesium-parity docs audit.
  - ID: T051
  - Title: Add 3D Tiles support docs audit
  - PRD tags: [P1-R4-AC6] [P0-R6-AC6]
  - Files: `tests/test_p1_tiles3d_docs.py`, `docs/guides/data_and_scene_workflows.md`, `docs/guides/feature_map.md`, `docs/api/api_reference.rst`
  - Action: Add audit tests requiring 3D Tiles docs to state offline local-fixture scope and explicitly deny full Cesium runtime parity.
  - Verification: `pytest tests/test_p1_tiles3d_docs.py -q`
  - Done evidence: Docs audit fails on full Cesium parity overclaims.
  - Dependencies: T004

## Phase 6: User Story 5 - Round-Trip Map Scene Bundles

**Goal**: `MapScene.save_bundle()` and `MapScene.load_bundle()` preserve deterministic scene intent, source labels, compiled label plans where available, diagnostics, support statuses, missing asset diagnostics, and review metadata.

**Independent test**: Save, load, validate, and compare a P1 bundle with assets present and with one external asset missing.

- [X] T052 [P] [US5] [PRD: P1-R5-AC1, P0-R5-AC4] Add MapScene bundle manifest field tests.
  - ID: T052
  - Title: Add bundle manifest field tests
  - PRD tags: [P1-R5-AC1] [P0-R5-AC4]
  - Files: `tests/test_p1_mapscene_bundle_manifest.py`, `python/forge3d/bundle.py`, expected `python/forge3d/map_scene.py`
  - Action: Add tests proving bundles include terrain source metadata, layer metadata, camera, lighting, output spec, label sources, compiled label plans where available, diagnostics, layer summaries, and supported export settings.
  - Verification: `pytest tests/test_p1_mapscene_bundle_manifest.py -q`
  - Done evidence: Manifest tests compare deterministic JSON fields exactly.
  - Dependencies: T001, T005

- [X] T053 [US5] [PRD: P1-R5-AC1, P0-R5-AC4] Implement P1 bundle manifest schema.
  - ID: T053
  - Title: Implement P1 bundle manifest schema
  - PRD tags: [P1-R5-AC1] [P0-R5-AC4]
  - Files: `python/forge3d/bundle.py`, expected `python/forge3d/map_scene.py`
  - Action: Extend bundle save paths to persist required P1 scene fields deterministically while retaining compatibility with existing bundle behavior.
  - Verification: `pytest tests/test_p1_mapscene_bundle_manifest.py tests/test_bundle_roundtrip.py -q`
  - Done evidence: P1 and existing bundle tests pass.
  - Dependencies: T006, T052

- [X] T054 [P] [US5] [PRD: P1-R5-AC2, P1-R5-AC5] Add available-asset bundle reconstruction tests.
  - ID: T054
  - Title: Add bundle reconstruction tests
  - PRD tags: [P1-R5-AC2] [P1-R5-AC5]
  - Files: `tests/test_p1_mapscene_bundle_reconstruct.py`, expected `python/forge3d/map_scene.py`, `python/forge3d/bundle.py`
  - Action: Add tests that save and load a scene with assets present, reconstruct renderable scene intent where support status permits rendering, and validate stable output state.
  - Verification: `pytest tests/test_p1_mapscene_bundle_reconstruct.py -q`
  - Done evidence: Reconstruction tests pass without requiring unsupported paths to render.
  - Dependencies: T001, T005

- [X] T055 [US5] [PRD: P1-R5-AC2, P1-R5-AC5] Implement MapScene bundle reconstruction.
  - ID: T055
  - Title: Implement bundle reconstruction
  - PRD tags: [P1-R5-AC2] [P1-R5-AC5]
  - Files: `python/forge3d/bundle.py`, expected `python/forge3d/map_scene.py`
  - Action: Implement `MapScene.load_bundle(path)` reconstruction for assets that are available and support status permits, preserving diagnostics otherwise.
  - Verification: `pytest tests/test_p1_mapscene_bundle_reconstruct.py -q`
  - Done evidence: Available-asset reconstruction tests pass.
  - Dependencies: T053, T054

- [X] T056 [P] [US5] [PRD: P1-R5-AC3, P0-R5-AC1] Add missing external asset diagnostics tests.
  - ID: T056
  - Title: Add missing external asset tests
  - PRD tags: [P1-R5-AC3] [P0-R5-AC1]
  - Files: `tests/test_p1_bundle_missing_assets.py`, `python/forge3d/bundle.py`, expected `python/forge3d/map_scene.py`
  - Action: Add tests that remove or move one external bundle asset and require `missing_external_asset` diagnostics during load or validation without false render success.
  - Verification: `pytest tests/test_p1_bundle_missing_assets.py -q`
  - Done evidence: Missing asset tests pass only with structured diagnostics.
  - Dependencies: T001, T005

- [X] T057 [US5] [PRD: P1-R5-AC3, P0-R5-AC1] Implement missing external asset diagnostics.
  - ID: T057
  - Title: Implement missing asset diagnostics
  - PRD tags: [P1-R5-AC3] [P0-R5-AC1]
  - Files: `python/forge3d/bundle.py`, expected `python/forge3d/map_scene.py`
  - Action: Add load and validation diagnostics for missing external assets, preserving code, severity, remediation, support status, and affected layer/object where possible.
  - Verification: `pytest tests/test_p1_bundle_missing_assets.py -q`
  - Done evidence: Missing external asset tests pass.
  - Dependencies: T053, T056

- [X] T058 [P] [US5] [PRD: P1-R5-AC4, P0-R5-AC4] Add review metadata bundle tests.
  - ID: T058
  - Title: Add review metadata tests
  - PRD tags: [P1-R5-AC4] [P0-R5-AC4]
  - Files: `tests/test_p1_bundle_review_metadata.py`, `python/forge3d/bundle.py`, expected `python/forge3d/map_scene.py`
  - Action: Add tests for support status, diagnostics, layer summaries, scene intent, review layers, variants, and deterministic review metadata in bundle payloads.
  - Verification: `pytest tests/test_p1_bundle_review_metadata.py -q`
  - Done evidence: Review metadata tests pass with stable serialized output.
  - Dependencies: T001, T005

- [X] T059 [US5] [PRD: P1-R5-AC4, P0-R5-AC4] Implement bundle review metadata.
  - ID: T059
  - Title: Implement review metadata persistence
  - PRD tags: [P1-R5-AC4] [P0-R5-AC4]
  - Files: `python/forge3d/bundle.py`, expected `python/forge3d/map_scene.py`
  - Action: Persist review-state fields, layer summaries, diagnostics, and scene intent in deterministic order without breaking existing `SceneState` round-trip behavior.
  - Verification: `pytest tests/test_p1_bundle_review_metadata.py tests/test_bundle_roundtrip.py -q`
  - Done evidence: Review metadata and existing bundle tests pass.
  - Dependencies: T053, T058

- [X] T060 [P] [US5] [PRD: P1-R5-AC5, P1-R5-AC1, P1-R5-AC2, P0-R3-AC1] Add deterministic bundle round-trip automation tests.
  - ID: T060
  - Title: Add deterministic bundle round-trip tests
  - PRD tags: [P1-R5-AC5] [P1-R5-AC1] [P1-R5-AC2] [P0-R3-AC1]
  - Files: `tests/test_p1_bundle_deterministic_roundtrip.py`, `python/forge3d/bundle.py`, expected `python/forge3d/map_scene.py`
  - Action: Add exact-comparison tests for save/load/validate round trip, stable manifest ordering, source label persistence, compiled `LabelPlan` payload persistence where available, and replay diagnostics where unavailable.
  - Verification: `pytest tests/test_p1_bundle_deterministic_roundtrip.py -q`
  - Done evidence: Repeated bundle saves produce byte-stable or exactly equal normalized JSON outputs.
  - Dependencies: T001, T005

- [X] T061 [US5] [PRD: P1-R5-AC5, P1-R5-AC1, P1-R5-AC2, P0-R3-AC1] Implement deterministic P1 bundle round trip.
  - ID: T061
  - Title: Implement deterministic bundle round trip
  - PRD tags: [P1-R5-AC5] [P1-R5-AC1] [P1-R5-AC2] [P0-R3-AC1]
  - Files: `python/forge3d/bundle.py`, expected `python/forge3d/map_scene.py`, expected `python/forge3d/label_plan.py`
  - Action: Ensure P1 bundle save/load preserves source labels and compiled label plans where available, records replay gaps as diagnostics, and writes stable JSON ordering.
  - Verification: `pytest tests/test_p1_bundle_deterministic_roundtrip.py tests/test_bundle_roundtrip.py -q`
  - Done evidence: Deterministic round-trip tests pass with no tolerance unless explicitly recorded in `plan.md`.
  - Dependencies: T053, T055, T057, T059, T060

## Phase 7: Polish, Documentation, State, and Milestone Evidence

- [X] T062 [P] [PRD: P1-R1-AC5, P1-R3-AC1, P1-R4-AC1, P1-R5-AC5] Add Milestone 4 integrated workflow test.
  - ID: T062
  - Title: Add Milestone 4 integrated workflow test
  - PRD tags: [P1-R1-AC5] [P1-R3-AC1] [P1-R4-AC1] [P1-R5-AC5]
  - Files: `tests/test_p1_milestone4_integrated_assets.py`, expected `python/forge3d/map_scene.py`, expected `python/forge3d/labels.py`, `python/forge3d/buildings.py`, `python/forge3d/tiles3d.py`, `python/forge3d/bundle.py`
  - Action: Add an end-to-end local workflow test for labels, buildings, 3D Tiles validation, bundle save/load, diagnostics, and support status.
  - Verification: `pytest tests/test_p1_milestone4_integrated_assets.py -q`
  - Done evidence: Integrated workflow test produces stable evidence without raw IPC.
  - Dependencies: T014, T031, T042, T061

- [X] T063 [PRD: P1-R1-AC5, P1-R3-AC1, P1-R4-AC1, P1-R5-AC5] Implement Milestone 4 integrated workflow support.
  - ID: T063
  - Title: Implement Milestone 4 integration glue
  - PRD tags: [P1-R1-AC5] [P1-R3-AC1] [P1-R4-AC1] [P1-R5-AC5]
  - Files: expected `python/forge3d/map_scene.py`, expected `python/forge3d/labels.py`, `python/forge3d/buildings.py`, `python/forge3d/tiles3d.py`, `python/forge3d/bundle.py`
  - Action: Add only the integration glue needed for MapScene to validate labels, building layers, 3D Tiles layers, and bundle state together.
  - Verification: `pytest tests/test_p1_milestone4_integrated_assets.py -q`
  - Done evidence: Integrated workflow test passes.
  - Dependencies: T062

- [X] T064 [P] [PRD: P0-R5-AC6, P1-R1-AC1, P1-R2-AC2, P1-R3-AC6, P1-R4-AC5, P1-R5-AC3] Add cross-feature no-op success regression tests.
  - ID: T064
  - Title: Add no-op success regression tests
  - PRD tags: [P0-R5-AC6] [P1-R1-AC1] [P1-R2-AC2] [P1-R3-AC6] [P1-R4-AC5] [P1-R5-AC3]
  - Files: `tests/test_p1_no_op_success.py`, expected `python/forge3d/labels.py`, expected `python/forge3d/typography.py`, `python/forge3d/buildings.py`, `python/forge3d/tiles3d.py`, `python/forge3d/bundle.py`
  - Action: Add negative tests proving invalid label ingestion, unavailable atlas generation, zero-geometry building fallback, unsupported tile features, and missing bundle assets cannot return success without state or diagnostics.
  - Verification: `pytest tests/test_p1_no_op_success.py -q`
  - Done evidence: No-op regression tests pass.
  - Dependencies: T016, T020, T040, T050, T057

- [X] T065 [PRD: P1-R1-AC1, P1-R1-AC2, P1-R1-AC3, P1-R1-AC4, P1-R1-AC5, P1-R1-AC6, P1-R2-AC1, P1-R2-AC2, P1-R2-AC3, P1-R2-AC4, P1-R2-AC5, P1-R2-AC6, P1-R2-AC7, P1-R3-AC1, P1-R3-AC2, P1-R3-AC3, P1-R3-AC4, P1-R3-AC5, P1-R3-AC6, P1-R4-AC1, P1-R4-AC2, P1-R4-AC3, P1-R4-AC4, P1-R4-AC5, P1-R4-AC6, P1-R5-AC1, P1-R5-AC2, P1-R5-AC3, P1-R5-AC4, P1-R5-AC5, P0-R6-AC1, P0-R6-AC6] Update docs and support matrices.
  - ID: T065
  - Title: Update P1 docs and support matrices
  - PRD tags: [P1-R1-AC1] [P1-R1-AC2] [P1-R1-AC3] [P1-R1-AC4] [P1-R1-AC5] [P1-R1-AC6] [P1-R2-AC1] [P1-R2-AC2] [P1-R2-AC3] [P1-R2-AC4] [P1-R2-AC5] [P1-R2-AC6] [P1-R2-AC7] [P1-R3-AC1] [P1-R3-AC2] [P1-R3-AC3] [P1-R3-AC4] [P1-R3-AC5] [P1-R3-AC6] [P1-R4-AC1] [P1-R4-AC2] [P1-R4-AC3] [P1-R4-AC4] [P1-R4-AC5] [P1-R4-AC6] [P1-R5-AC1] [P1-R5-AC2] [P1-R5-AC3] [P1-R5-AC4] [P1-R5-AC5] [P0-R6-AC1] [P0-R6-AC6]
  - Files: `docs/guides/data_and_scene_workflows.md`, `docs/guides/feature_map.md`, `docs/api/api_reference.rst`, `docs/tutorials/gis-track/04-3d-buildings.md`, `docs/tutorials/python-track/04-scene-bundles.md`
  - Action: Document every public P1 API and support-level boundary from T008, T010, T012, T014, T018, T020, T022, T026, T028, T031, T033, T035, T038, T040, T042, T044, T046, T048, T050, T053, T055, T057, T059, T061, and T063, including deterministic behavior, failure behavior, diagnostics, bundle/export implications, exact PRD support-level wording, 3D Tiles no-Cesium-parity scope, bundle round-trip policy, and typed API examples.
  - Verification: `pytest tests/test_p1_docs_support_matrix.py tests/test_p1_typography_docs.py tests/test_p1_building_docs.py tests/test_p1_tiles3d_docs.py -q`
  - Done evidence: Docs audits pass and all P1 docs use exact PRD support-level terms.
  - Dependencies: T016, T028, T040, T050, T061, T063, T029, T036, T051

- [X] T066 [P] [PRD: P0-R6-AC5, P0-R6-AC6, P1-R1-AC1, P1-R4-AC1, P1-R5-AC5] Add canonical typed API example smoke tests.
  - ID: T066
  - Title: Add typed API example smoke tests
  - PRD tags: [P0-R6-AC5] [P0-R6-AC6] [P1-R1-AC1] [P1-R4-AC1] [P1-R5-AC5]
  - Files: `tests/test_p1_examples_typed_api.py`, expected `examples/p1_map_assets_bundle.py`, `docs/guides/data_and_scene_workflows.md`
  - Action: Add a smoke test for a canonical P1 example using `MapScene`, `LabelLayer`, `BuildingLayer`, `Tiles3DLayer`, and bundle APIs, with no direct `viewer_ipc` calls.
  - Verification: `pytest tests/test_p1_examples_typed_api.py -q`
  - Done evidence: Example smoke test passes and `rg -n "viewer_ipc" examples/p1_map_assets_bundle.py` finds no matches.
  - Dependencies: T063, T065

- [X] T067 [PRD: P1-R1-AC1, P1-R1-AC2, P1-R1-AC3, P1-R1-AC4, P1-R1-AC5, P1-R1-AC6, P1-R2-AC1, P1-R2-AC2, P1-R2-AC3, P1-R2-AC4, P1-R2-AC5, P1-R2-AC6, P1-R2-AC7, P1-R3-AC1, P1-R3-AC2, P1-R3-AC3, P1-R3-AC4, P1-R3-AC5, P1-R3-AC6, P1-R4-AC1, P1-R4-AC2, P1-R4-AC3, P1-R4-AC4, P1-R4-AC5, P1-R4-AC6, P1-R5-AC1, P1-R5-AC2, P1-R5-AC3, P1-R5-AC4, P1-R5-AC5] Update requirements verification matrix.
  - ID: T067
  - Title: Update requirements verification matrix
  - PRD tags: [P1-R1-AC1] [P1-R1-AC2] [P1-R1-AC3] [P1-R1-AC4] [P1-R1-AC5] [P1-R1-AC6] [P1-R2-AC1] [P1-R2-AC2] [P1-R2-AC3] [P1-R2-AC4] [P1-R2-AC5] [P1-R2-AC6] [P1-R2-AC7] [P1-R3-AC1] [P1-R3-AC2] [P1-R3-AC3] [P1-R3-AC4] [P1-R3-AC5] [P1-R3-AC6] [P1-R4-AC1] [P1-R4-AC2] [P1-R4-AC3] [P1-R4-AC4] [P1-R4-AC5] [P1-R4-AC6] [P1-R5-AC1] [P1-R5-AC2] [P1-R5-AC3] [P1-R5-AC4] [P1-R5-AC5]
  - Files: `docs/superpowers/state/requirements-verification-matrix.md`
  - Action: Update every feature 005 row with implementation/test/evidence paths and status based on completed tasks; do not mark `Verified` until commands have run and evidence is present.
  - Verification: `rg -n "P1-R[1-5]-AC[0-9]+.*005-map-assets-bundles-p1|tests/test_p1_" docs/superpowers/state/requirements-verification-matrix.md`
  - Done evidence: Matrix rows reference exact test/docs/source evidence for P1-R1 through P1-R5.
  - Dependencies: T064, T065, T066

- [X] T068 [PRD: P1-R5-AC5, P0-R5-AC4, P0-R5-AC6] Update implementation ledger and current context pack.
  - ID: T068
  - Title: Update continuity state
  - PRD tags: [P1-R5-AC5] [P0-R5-AC4] [P0-R5-AC6]
  - Files: `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/current-context-pack.md`
  - Action: Record completed tasks, verification commands, P0 dependency outcomes, unsupported/pro-gated decisions, docs evidence, and residual blockers for future chats.
  - Verification: `rg -n "005-map-assets-bundles-p1|P1-R5-AC5|unsupported_tile_feature|missing_external_asset|placeholder_fallback" docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md`
  - Done evidence: Ledger and context pack contain current task outcomes and no unverified success claims.
  - Dependencies: T067

- [X] T069 [PRD: P1-R1-AC5, P1-R2-AC4, P1-R3-AC1, P1-R4-AC5, P1-R5-AC5] Run quickstart validation and record evidence.
  - ID: T069
  - Title: Run quickstart validation
  - PRD tags: [P1-R1-AC5] [P1-R2-AC4] [P1-R3-AC1] [P1-R4-AC5] [P1-R5-AC5]
  - Files: `specs/005-map-assets-bundles-p1/quickstart.md`, `docs/superpowers/state/implementation-ledger.md`
  - Action: Execute the quickstart scenarios for labels, typography, buildings, 3D Tiles, and bundle round trip; record commands and observed results.
  - Verification: `pytest tests/test_p1_milestone4_integrated_assets.py tests/test_p1_bundle_deterministic_roundtrip.py tests/test_p1_no_op_success.py -q`
  - Done evidence: Ledger records quickstart command output summary and any diagnosed unsupported paths.
  - Dependencies: T063, T064, T068

## Dependencies and Execution Order

## 2026-05-18 Completion Evidence

- Session scope: completed T009-T069 to remediate RED/YELLOW findings R-030 through R-035 without adding new product scope.
- Tests added or extended: `tests/test_p1_label_layer_crs_terrain.py`, `tests/test_p1_label_expressions_plan_diagnostics.py`, `tests/test_p1_typography_font_support.py`, `tests/test_p1_building_workflow_support.py`, `tests/test_p1_tiles3d_support.py`, and `tests/test_p1_bundle_roundtrip.py`.
- Implementation evidence: `python/forge3d/map_scene.py` now covers CRS transform via `forge3d.crs.transform_coords`, terrain sampling hooks, pre-render glyph diagnostics, `FontAtlas`, `FontFallbackRange`, `TypographySettings`, textured PBR unsupported diagnostics, 3D Tiles cache/LOD metadata, unsupported tile feature summaries, and P1 bundle `supported_export_settings`.
- Verification: `$p1 = rg --files tests | Where-Object { $_ -match 'test_p1_.*\.py$' }; python -m pytest @p1 -q` reported `51 passed`.
- Guardrail verification: `python -m pytest tests\test_mapscene_validation.py tests\test_mapscene_save_bundle.py tests\test_mapscene_render_png.py tests\test_mapscene_docs.py tests\test_mapscene_examples.py tests\test_mapscene_quickstart.py tests\test_mapscene_recipe_contract.py tests\test_mapscene_render_policy.py tests\test_mapscene_support_status.py tests\test_mapscene_label_plan_integration.py @p1 -q` reported `101 passed`.
- Syntax verification: `python -m py_compile python\forge3d\map_scene.py python\forge3d\diagnostics.py python\forge3d\__init__.py tests\test_p1_label_layer_crs_terrain.py tests\test_p1_label_expressions_plan_diagnostics.py tests\test_p1_typography_font_support.py tests\test_p1_building_workflow_support.py tests\test_p1_tiles3d_support.py tests\test_p1_bundle_roundtrip.py` exited `0`.

### Phase Dependencies

- Phase 1 blocks source implementation for all user stories.
- User Story 1 depends on T001 through T006.
- User Story 2 depends on T001 through T006; typography integration with labels depends on T016 where noted.
- User Story 3 depends on T001 through T006 and can run in parallel with User Stories 1, 2, 4, and 5 after the P0 gate passes.
- User Story 4 depends on T001 through T006 and can run in parallel after the P0 gate passes.
- User Story 5 depends on T001 through T006 and later depends on story-specific source paths where bundle persistence needs their serialized state.
- Phase 7 depends on the specific user-story tasks named in each task.

### Parallel Opportunities

- T003, T004, and T005 can run in parallel after T001 because they create independent test files.
- T007, T009, T011, T013, and T015 can run in parallel after fixtures and prerequisites are known.
- T017, T019, T021, T023, T025, T027, and T029 can run in parallel because they target separate tests and docs audits.
- T030, T032, T034, T036, T037, and T039 can run in parallel because they target separate building tests or docs audits.
- T041, T043, T045, T047, T049, and T051 can run in parallel because they target separate 3D Tiles test files or docs audits.
- T052, T054, T056, T058, and T060 can run in parallel after prerequisites because they target separate bundle test files.
- T062, T064, and T066 can be authored in parallel after their listed story dependencies are available.

## Parallel Execution Examples

```powershell
# User Story 1 test authoring after Phase 1
pytest tests/test_p1_label_layer_geometry.py tests/test_p1_label_layer_crs_terrain.py tests/test_p1_label_expressions.py tests/test_p1_label_plan_integration.py tests/test_p1_label_diagnostics.py -q

# User Story 3 and User Story 4 negative tests can be authored independently
pytest tests/test_p1_building_placeholder_fallback.py tests/test_p1_tiles3d_unsupported_features.py -q

# Bundle test authoring before implementation
pytest tests/test_p1_mapscene_bundle_manifest.py tests/test_p1_bundle_missing_assets.py tests/test_p1_bundle_deterministic_roundtrip.py -q
```

## PRD Coverage Check

- P0-R5-AC1: T001, T003, T056, T057
- P0-R5-AC2: T003
- P0-R5-AC3: T003, T009, T010, T034, T035, T045, T046
- P0-R5-AC4: T003, T052, T053, T058, T059, T068
- P0-R5-AC5: T003
- P0-R5-AC6: T001, T003, T006, T007, T008, T011, T012, T015, T016, T019, T020, T021, T022, T025, T026, T027, T028, T030, T031, T032, T033, T037, T038, T039, T040, T041, T042, T043, T044, T047, T048, T049, T050, T064, T068
- P0-R6-AC1: T004, T065
- P0-R6-AC2: T004, T036, T065
- P0-R6-AC3: T004
- P0-R6-AC4: T004
- P0-R6-AC5: T001, T004, T066
- P0-R6-AC6: T004, T029, T036, T051, T065, T066
- P1-R1-AC1 through P1-R1-AC6: T005, T007 through T016, T064, T065, T066, T067
- P1-R2-AC1 through P1-R2-AC7: T017 through T029, T065, T067
- P1-R3-AC1 through P1-R3-AC6: T030 through T040, T062 through T065, T067
- P1-R4-AC1 through P1-R4-AC6: T041 through T051, T062 through T067
- P1-R5-AC1 through P1-R5-AC5: T052 through T061, T062 through T069

## Detailed Pre-Implementation Coverage Audit Table

| PRD requirement | Acceptance criterion | Task IDs covering it | Test task IDs | Docs task IDs | Verification command | Risk if omitted |
|---|---|---|---|---|---|---|
| P0-R5 Production Diagnostics | P0-R5-AC1 structured diagnostic objects | T001, T002, T003, T056, T057 | T003, T056 | T065 | `pytest tests/test_p1_diagnostics_contract.py tests/test_p1_bundle_missing_assets.py -q` | Unsupported, missing-asset, or deferred paths degrade to text or silent failure. |
| P0-R5 Production Diagnostics | P0-R5-AC2 severity values | T003 | T003 | T065 | `pytest tests/test_p1_diagnostics_contract.py -q` | Validation policy cannot distinguish warnings, errors, and blockers. |
| P0-R5 Production Diagnostics | P0-R5-AC3 affected layer/object IDs | T003, T009, T010, T034, T035, T045, T046, T056, T057 | T003, T009, T034, T045, T056 | T065 | `pytest tests/test_p1_diagnostics_contract.py tests/test_p1_label_layer_crs_terrain.py tests/test_p1_building_summaries.py tests/test_p1_tiles3d_cache_stats.py tests/test_p1_bundle_missing_assets.py -q` | Diagnostics become unactionable for feature, layer, and asset repair. |
| P0-R5 Production Diagnostics | P0-R5-AC4 bundle-serializable diagnostics | T003, T052, T053, T058, T059, T068 | T003, T052, T058 | T065, T068 | `pytest tests/test_p1_diagnostics_contract.py tests/test_p1_mapscene_bundle_manifest.py tests/test_p1_bundle_review_metadata.py -q` | Review bundles lose diagnostic evidence and cannot reproduce validation state. |
| P0-R5 Production Diagnostics | P0-R5-AC5 fail-on-warning policy | T003 | T003 | T065 | `pytest tests/test_p1_diagnostics_contract.py -q` | Render policy cannot enforce conservative validation behavior. |
| P0-R5 Production Diagnostics | P0-R5-AC6 unsupported never silently ignored | T003, T006, T007, T008, T019, T020, T030, T031, T037, T038, T039, T040, T041, T042, T043, T044, T047, T048, T049, T050, T064, T068 | T003, T064 | T065, T068 | `pytest tests/test_p1_diagnostics_contract.py tests/test_p1_no_op_success.py -q` | No-op success can pass as real ingestion, rendering, or bundle reconstruction. |
| P0-R6 Style Honesty | P0-R6-AC1 local/provided feature scope | T004, T065 | T004 | T065 | `pytest tests/test_p1_docs_support_matrix.py -q` | Docs may imply streamed MVT or full Mapbox support. |
| P0-R6 Style Honesty | P0-R6-AC2 supported layer list | T004, T036, T065 | T004, T036 | T065 | `pytest tests/test_p1_docs_support_matrix.py tests/test_p1_building_docs.py -q` | Users cannot tell which map layers are actually supported. |
| P0-R6 Style Honesty | P0-R6-AC3 supported fill/line/circle scope | T004 | T004 | T065 | `pytest tests/test_p1_docs_support_matrix.py -q` | Existing style capability can be overclaimed or underdocumented. |
| P0-R6 Style Honesty | P0-R6-AC4 unsupported style diagnostics | T004 | T004 | T065 | `pytest tests/test_p1_docs_support_matrix.py -q` | Unsupported style inputs may be silently dropped. |
| P0-R6 Style Honesty | P0-R6-AC5 style output feeds typed APIs | T001, T004, T066 | T001, T004, T066 | T065, T066 | `pytest tests/test_p1_prerequisite_contracts.py tests/test_p1_docs_support_matrix.py tests/test_p1_examples_typed_api.py -q` | Canonical workflows fall back to raw IPC or disconnected style paths. |
| P0-R6 Style Honesty | P0-R6-AC6 no full Mapbox/Cesium parity claims | T004, T029, T036, T051, T065, T066 | T004, T029, T036, T051, T066 | T029, T036, T051, T065, T066 | `pytest tests/test_p1_docs_support_matrix.py tests/test_p1_typography_docs.py tests/test_p1_building_docs.py tests/test_p1_tiles3d_docs.py tests/test_p1_examples_typed_api.py -q` | Public docs overclaim unsupported map-engine parity. |
| P1-R1 LabelLayer Data Ingestion | P1-R1-AC1 point, line, polygon geometry ingestion | T005, T007, T008, T064, T065, T066, T067 | T005, T007, T064, T066 | T065, T066 | `pytest tests/test_p1_fixture_inventory.py tests/test_p1_label_layer_geometry.py tests/test_p1_no_op_success.py tests/test_p1_examples_typed_api.py -q` | Label sources fail silently or unsupported geometry appears accepted. |
| P1-R1 LabelLayer Data Ingestion | P1-R1-AC2 CRS transform or mismatch diagnostic | T009, T010, T065, T067 | T009 | T065 | `pytest tests/test_p1_label_layer_crs_terrain.py -q` | Labels render in the wrong CRS or mismatch remediation is missing. |
| P1-R1 LabelLayer Data Ingestion | P1-R1-AC3 terrain sampling or unavailable diagnostic | T009, T010, T065, T067 | T009 | T065 | `pytest tests/test_p1_label_layer_crs_terrain.py -q` | Height placement can be wrong with no structured warning. |
| P1-R1 LabelLayer Data Ingestion | P1-R1-AC4 simple label expressions | T011, T012, T065, T067 | T011 | T065 | `pytest tests/test_p1_label_expressions.py -q` | Labels compile empty or nondeterministic text from real data. |
| P1-R1 LabelLayer Data Ingestion | P1-R1-AC5 deterministic LabelPlan output | T001, T006, T013, T014, T062, T063, T065, T067, T069 | T001, T013, T062, T069 | T065, T066, T067 | `pytest tests/test_p1_label_plan_integration.py tests/test_p1_milestone4_integrated_assets.py -q` | LabelLayer bypasses the deterministic P0 LabelPlan contract. |
| P1-R1 LabelLayer Data Ingestion | P1-R1-AC6 missing field and missing glyph validation | T015, T016, T017, T018, T023, T024, T065, T067 | T015, T017, T023 | T065 | `pytest tests/test_p1_label_diagnostics.py tests/test_p1_font_atlas_default.py tests/test_p1_unicode_coverage.py -q` | Missing data or glyphs appear only after render, or not at all. |
| P1-R2 Typography and Font Handling | P1-R2-AC1 default Latin atlas coverage | T005, T017, T018, T065, T067 | T005, T017 | T065 | `pytest tests/test_p1_fixture_inventory.py tests/test_p1_font_atlas_default.py -q` | Default atlas behavior is undocumented or fixture-dependent. |
| P1-R2 Typography and Font Handling | P1-R2-AC2 TTF/OTF generation or typed diagnostic | T019, T020, T064, T065, T067 | T019, T064 | T065 | `pytest tests/test_p1_font_atlas_generation.py tests/test_p1_no_op_success.py -q` | Font generation can return a placeholder atlas as success. |
| P1-R2 Typography and Font Handling | P1-R2-AC3 deterministic fallback ranges | T021, T022, T065, T067 | T021 | T065 | `pytest tests/test_p1_font_fallback_ranges.py -q` | Fallback behavior changes across runs or unsupported ranges are hidden. |
| P1-R2 Typography and Font Handling | P1-R2-AC4 Unicode coverage diagnostics | T015, T016, T023, T024, T029, T065, T067, T069 | T015, T023, T029 | T029, T065 | `pytest tests/test_p1_unicode_coverage.py tests/test_p1_label_diagnostics.py tests/test_p1_typography_docs.py -q` | Unsupported Unicode appears as missing text without remediation. |
| P1-R2 Typography and Font Handling | P1-R2-AC5 kerning, tracking, line-height effects or diagnostics | T025, T026, T065, T067 | T025 | T065 | `pytest tests/test_p1_typography_metrics.py -q` | Typography setters can become no-op public controls. |
| P1-R2 Typography and Font Handling | P1-R2-AC6 multiline labels and callouts or diagnostics | T027, T028, T065, T067 | T027 | T065 | `pytest tests/test_p1_multiline_callouts.py -q` | Multiline or callout text can be dropped or overclaimed. |
| P1-R2 Typography and Font Handling | P1-R2-AC7 optional non-Latin shaping non-blocking | T029, T065, T067 | T029 | T029, T065 | `pytest tests/test_p1_typography_docs.py -q` | Complex-script shaping can accidentally become a hidden P1 blocker or overclaim. |
| P1-R3 Integrated Building Workflow | P1-R3-AC1 building support status classification | T006, T030, T031, T062, T063, T065, T067, T069 | T030, T062 | T065 | `pytest tests/test_p1_building_support_status.py tests/test_p1_milestone4_integrated_assets.py -q` | Pro-gated or fallback building paths look supported. |
| P1-R3 Integrated Building Workflow | P1-R3-AC2 CityJSON/GeoJSON render prep where available | T005, T032, T033, T065, T067 | T005, T032 | T065 | `pytest tests/test_p1_fixture_inventory.py tests/test_p1_building_mapscene_render.py -q` | Available native paths are not exposed through MapScene. |
| P1-R3 Integrated Building Workflow | P1-R3-AC3 geometry count and bounds diagnostics | T034, T035, T065, T067 | T034 | T065 | `pytest tests/test_p1_building_summaries.py -q` | Building summaries cannot prove what was ingested. |
| P1-R3 Integrated Building Workflow | P1-R3-AC4 scalar PBR status docs | T036, T065, T067 | T036 | T036, T065 | `pytest tests/test_p1_building_docs.py -q` | Scalar material support is ambiguous or overclaimed. |
| P1-R3 Integrated Building Workflow | P1-R3-AC5 textured PBR implemented or unsupported | T037, T038, T065, T067 | T037 | T065 | `pytest tests/test_p1_building_textured_pbr.py -q` | Textured PBR can be implied without end-to-end support. |
| P1-R3 Integrated Building Workflow | P1-R3-AC6 zero-geometry fallback diagnostics | T039, T040, T064, T065, T067 | T039, T064 | T065 | `pytest tests/test_p1_building_placeholder_fallback.py tests/test_p1_no_op_success.py -q` | Placeholder geometry becomes indistinguishable from successful ingestion. |
| P1-R4 Public 3D Tiles Scene Integration | P1-R4-AC1 public Tiles3DLayer load workflow | T005, T006, T041, T042, T062, T063, T065, T066, T067 | T005, T041, T062, T066 | T065, T066 | `pytest tests/test_p1_tiles3d_public_api.py tests/test_p1_milestone4_integrated_assets.py tests/test_p1_examples_typed_api.py -q` | Users cannot exercise tiles through typed MapScene APIs. |
| P1-R4 Public 3D Tiles Scene Integration | P1-R4-AC2 supported vs unsupported formats | T043, T044, T065, T067 | T043 | T065 | `pytest tests/test_p1_tiles3d_format_validation.py -q` | Unsupported formats are parsed as metadata-only success. |
| P1-R4 Public 3D Tiles Scene Integration | P1-R4-AC3 cache stats or honest absence | T045, T046, T065, T067 | T045 | T065 | `pytest tests/test_p1_tiles3d_cache_stats.py -q` | Cache/LOD state is invented or absent without explanation. |
| P1-R4 Public 3D Tiles Scene Integration | P1-R4-AC4 LOD/SSE config or unsupported diagnostic | T047, T048, T065, T067 | T047 | T065 | `pytest tests/test_p1_tiles3d_lod_config.py tests/test_3dtiles_sse.py -q` | Configuration becomes a no-op or gives nondeterministic tile selection. |
| P1-R4 Public 3D Tiles Scene Integration | P1-R4-AC5 unsupported B3DM/GLB feature diagnostics | T049, T050, T064, T065, T067, T069 | T049, T064 | T065 | `pytest tests/test_p1_tiles3d_unsupported_features.py tests/test_p1_no_op_success.py -q` | Unsupported tile payloads can be mistaken for renderable content. |
| P1-R4 Public 3D Tiles Scene Integration | P1-R4-AC6 no full Cesium runtime parity docs | T004, T051, T065, T067 | T004, T051 | T051, T065 | `pytest tests/test_p1_docs_support_matrix.py tests/test_p1_tiles3d_docs.py -q` | Public docs overclaim 3D Tiles production/runtime parity. |
| P1-R5 Bundle Round-Trip | P1-R5-AC1 save required bundle fields | T052, T053, T060, T061, T065, T067 | T052, T060 | T065 | `pytest tests/test_p1_mapscene_bundle_manifest.py tests/test_p1_bundle_deterministic_roundtrip.py -q` | Bundles omit scene intent or diagnostics needed for review. |
| P1-R5 Bundle Round-Trip | P1-R5-AC2 load reconstructs available assets | T001, T006, T054, T055, T060, T061, T065, T067 | T001, T054, T060 | T065 | `pytest tests/test_p1_mapscene_bundle_reconstruct.py tests/test_p1_bundle_deterministic_roundtrip.py -q` | Loaded bundles cannot reproduce renderable scene intent. |
| P1-R5 Bundle Round-Trip | P1-R5-AC3 missing external asset diagnostics | T056, T057, T064, T065, T067 | T056, T064 | T065 | `pytest tests/test_p1_bundle_missing_assets.py tests/test_p1_no_op_success.py -q` | Missing assets are silently ignored or treated as renderable. |
| P1-R5 Bundle Round-Trip | P1-R5-AC4 review metadata and support status | T058, T059, T065, T067 | T058 | T065 | `pytest tests/test_p1_bundle_review_metadata.py -q` | Review bundles lose support status, summaries, or scene intent. |
| P1-R5 Bundle Round-Trip | P1-R5-AC5 deterministic round-trip automation | T005, T054, T055, T060, T061, T062, T063, T064, T065, T066, T067, T068, T069 | T005, T054, T060, T062, T064, T066, T069 | T065, T068 | `pytest tests/test_p1_bundle_deterministic_roundtrip.py tests/test_p1_milestone4_integrated_assets.py tests/test_p1_no_op_success.py -q` | Round trips drift or pass without stable artifact evidence. |
| Milestone 4 Integrated Geospatial Assets | Milestone 4 exit criteria for buildings, 3D Tiles, bundles, diagnostics | T062, T063, T065, T067, T068, T069 | T062, T064, T069 | T065, T067, T068 | `pytest tests/test_p1_milestone4_integrated_assets.py tests/test_p1_no_op_success.py -q` | The feature ships isolated APIs without integrated geospatial-asset evidence. |

## Audit Rerun Result

- PASS: Every P1-R1 through P1-R5 acceptance criterion, all P0 diagnostic/style dependencies in scope, and Milestone 4 exit criteria have task coverage, test IDs, docs/support IDs where needed, and verification commands.
- PASS: Public P1 API implementation tasks are paired with docs/support wording in T065 and typed example checks in T066.
- PASS: P0 dependency deferral requires typed `p0_dependency_missing` diagnostics plus blocker records rather than silent or narrative-only deferral.

## MVP and Incremental Strategy

- MVP for this P1 feature is User Story 1 plus prerequisite gates only.
- Building, 3D Tiles, and bundle work should remain independently testable and may be diagnosed as unsupported or Pro-gated where the renderer substrate does not support the path.
- Stop at each story checkpoint and run that story's tests before proceeding to cross-story integration.

## Extension Hooks

**Optional Hook**: git  
Command: `/speckit.git.commit`  
Description: Auto-commit after task generation  

Prompt: Commit task changes?  
To execute: `/speckit.git.commit`


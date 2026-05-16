# Tasks: Diagnostics and Support Matrices

**Input**: `specs/001-diagnostics-support-matrices/plan.md`, `spec.md`, `research.md`, `data-model.md`, `quickstart.md`, and `contracts/diagnostics-contract.md`

**Scope**: P0-R5, P0-R6, PRD Sections 13 and 16, Appendix B, Milestone 0, and required continuity state updates.

**Required task format**: Every task below includes ID, title, PRD tags, files, action, verification, done evidence, and dependencies. The first checklist line preserves the Spec Kit task ID and `[P]` marker where work can be safely parallelized.

**Execution rule**: Write or update tests before or alongside implementation. This feature defines diagnostics/support contracts only; it must not implement `LabelPlan`, `MapScene`, building rendering, 3D Tiles rendering, or VT normal/mask runtime.

## Phase 1: Contract and Path Gates

- [X] T001 [PRD: P0-R5-AC1, P0-R5-AC4, P0-R5-AC6] Inspect diagnostics ownership paths before source edits.
  - ID: T001
  - Title: Inspect diagnostics ownership paths
  - PRD tags: [P0-R5-AC1] [P0-R5-AC4] [P0-R5-AC6]
  - Files: `python/forge3d/__init__.py`, `python/forge3d/__init__.pyi`, expected `python/forge3d/diagnostics.py`, `tests/`, `docs/superpowers/state/implementation-ledger.md`
  - Action: Confirm the diagnostics module path, export conventions, and test naming; if a different path is required, update `specs/001-diagnostics-support-matrices/plan.md`, `contracts/diagnostics-contract.md`, and this `tasks.md` before implementation.
  - Verification: `rg -n "Diagnostic|ValidationReport|RenderFailurePolicy|SeverityPolicy" python tests docs/superpowers/state`
  - Done evidence: Ledger records the chosen source/test/docs paths and no implementation has started on an unrecorded path.
  - Dependencies: None

- [X] T002 [P] [PRD: P0-R5-AC1, P0-R5-AC2, P0-R5-AC3, P0-R5-AC5] Add diagnostics object and policy contract tests.
  - ID: T002
  - Title: Add diagnostics object and policy tests
  - PRD tags: [P0-R5-AC1] [P0-R5-AC2] [P0-R5-AC3] [P0-R5-AC5]
  - Files: `tests/test_diagnostics_contract.py`, expected `python/forge3d/diagnostics.py`
  - Action: Add tests for `Diagnostic`, `ValidationReport`, `LayerSummary`, `SupportMatrixEntry`, severity values, affected layer/object IDs, status calculation, unknown severity/support rejection, and warning/error/fatal render blocking policy.
  - Verification: `pytest tests/test_diagnostics_contract.py -q`
  - Done evidence: Tests fail before the contract exists and pass only with structured diagnostic objects and policy behavior.
  - Dependencies: T001

- [X] T003 [PRD: P0-R5-AC1, P0-R5-AC2, P0-R5-AC3, P0-R5-AC5] Implement diagnostics data models and exports.
  - ID: T003
  - Title: Implement diagnostics data models
  - PRD tags: [P0-R5-AC1] [P0-R5-AC2] [P0-R5-AC3] [P0-R5-AC5]
  - Files: `python/forge3d/diagnostics.py`, `python/forge3d/__init__.py`, `python/forge3d/__init__.pyi`
  - Action: Implement the public diagnostics classes, validation helpers, severity policy, render policy, deterministic ordering keys, and package exports.
  - Verification: `pytest tests/test_diagnostics_contract.py -q`
  - Done evidence: Contract tests pass and public imports work from `forge3d`.
  - Dependencies: T002

- [X] T004 [P] [PRD: P0-R5-AC4, P0-R5-AC1] Add bundle-ready serialization and determinism tests.
  - ID: T004
  - Title: Add diagnostic serialization tests
  - PRD tags: [P0-R5-AC4] [P0-R5-AC1]
  - Files: `tests/test_diagnostics_bundle_serialization.py`, expected `python/forge3d/diagnostics.py`, inspect `python/forge3d/bundle.py`
  - Action: Add tests for `to_dict()`, `from_dict()`, sorted JSON serialization, lossless bundle-ready payloads, deterministic diagnostic ordering, and repeated byte-for-byte serialization.
  - Verification: `pytest tests/test_diagnostics_bundle_serialization.py -q`
  - Done evidence: Fixed reports serialize identically across repeated runs and preserve code, severity, IDs, support level, remediation, and details.
  - Dependencies: T001

- [X] T005 [PRD: P0-R5-AC4, P0-R5-AC1] Implement deterministic diagnostic serialization.
  - ID: T005
  - Title: Implement diagnostic serialization
  - PRD tags: [P0-R5-AC4] [P0-R5-AC1]
  - Files: `python/forge3d/diagnostics.py`
  - Action: Implement deterministic `to_dict()` and `from_dict()` for diagnostic objects and reports with sorted details and stable list ordering.
  - Verification: `pytest tests/test_diagnostics_contract.py tests/test_diagnostics_bundle_serialization.py -q`
  - Done evidence: Contract and serialization tests pass with byte-stable payloads.
  - Dependencies: T003, T004

## Phase 2: Unsupported Path Diagnostics

- [X] T006 [P] [PRD: P0-R5-AC6, P0-R5-AC3, P0-R5-AC2] Add required diagnostic inventory negative tests.
  - ID: T006
  - Title: Add required diagnostic inventory tests
  - PRD tags: [P0-R5-AC6] [P0-R5-AC3] [P0-R5-AC2]
  - Files: `tests/test_diagnostics_support_paths.py`, inspect `python/forge3d/buildings.py`, `python/forge3d/tiles3d.py`, `python/forge3d/terrain_params.py`, `python/forge3d/viewer_ipc.py`, `src/terrain/renderer/virtual_texture.rs`, `src/viewer/cmd/labels_command.rs`
  - Action: Add negative tests for `crs_mismatch`, `missing_glyphs`, `pro_gated_path`, `placeholder_fallback`, `experimental_feature`, `vt_unsupported_family`, `python_public_3dtiles_incomplete`, `estimated_gpu_memory`, and `label_rejection_summary`, including affected IDs where knowable.
  - Verification: `pytest tests/test_diagnostics_support_paths.py -q`
  - Done evidence: Tests fail on silent ignore, log-only warnings, placeholder success, and missing affected-ID propagation.
  - Dependencies: T003

- [X] T007 [PRD: P0-R5-AC6, P0-R5-AC3, P0-R5-AC2] Implement diagnostic factories for known unsupported paths.
  - ID: T007
  - Title: Implement unsupported path diagnostic factories
  - PRD tags: [P0-R5-AC6] [P0-R5-AC3] [P0-R5-AC2]
  - Files: `python/forge3d/diagnostics.py`, inspect `python/forge3d/buildings.py`, `python/forge3d/tiles3d.py`, `python/forge3d/terrain_params.py`, `python/forge3d/mem.py`
  - Action: Add reusable diagnostic constructors or validation helpers for the required inventory without reclassifying unavailable render paths as supported.
  - Verification: `pytest tests/test_diagnostics_support_paths.py -q`
  - Done evidence: Required inventory diagnostics are structured, severity-bearing, and support-level classified.
  - Dependencies: T006

- [X] T008 [P] [PRD: P0-R5-AC6, P0-R5-AC5] Add no-op and render-policy regression tests.
  - ID: T008
  - Title: Add no-op and render-policy regression tests
  - PRD tags: [P0-R5-AC6] [P0-R5-AC5]
  - Files: `tests/test_diagnostics_no_op_policy.py`, expected `python/forge3d/diagnostics.py`
  - Action: Add tests proving warning diagnostics continue by default, fail under fail-on-warning, errors/fatals block always, and placeholder/fallback diagnostics cannot produce successful render-prep status.
  - Verification: `pytest tests/test_diagnostics_no_op_policy.py -q`
  - Done evidence: Tests demonstrate no unsupported, experimental, Pro-gated, placeholder/fallback, or no-op path can be marked successful by policy.
  - Dependencies: T003

- [X] T009 [PRD: P0-R5-AC5, P0-R5-AC6] Wire report blocking behavior to policies.
  - ID: T009
  - Title: Implement report blocking behavior
  - PRD tags: [P0-R5-AC5] [P0-R5-AC6]
  - Files: `python/forge3d/diagnostics.py`
  - Action: Implement `ValidationReport.render_blocked(...)`, `has_errors`, and status derivation so fail-on-warning and error/fatal behavior are deterministic.
  - Verification: `pytest tests/test_diagnostics_contract.py tests/test_diagnostics_no_op_policy.py -q`
  - Done evidence: Policy and no-op tests pass.
  - Dependencies: T008

## Phase 3: Style Honesty and Support Matrices

- [X] T010 [P] [PRD: P0-R6-AC1, P0-R6-AC2, P0-R6-AC3, P0-R6-AC4, P0-R6-AC5] Add style support diagnostic tests.
  - ID: T010
  - Title: Add style support diagnostic tests
  - PRD tags: [P0-R6-AC1] [P0-R6-AC2] [P0-R6-AC3] [P0-R6-AC4] [P0-R6-AC5]
  - Files: `tests/test_diagnostics_style_support.py`, `python/forge3d/style.py`, expected `python/forge3d/diagnostics.py`
  - Action: Add tests for local/provided feature style scope, audited `fill`, `line`, and `circle` support, unsupported layer types, unsupported paint/layout fields, and style output shape consumable by `VectorOverlay` and future `LabelLayer`.
  - Verification: `pytest tests/test_diagnostics_style_support.py -q`
  - Done evidence: Tests produce `unsupported_style_layer_type` and `unsupported_style_field` diagnostics instead of silently dropping unsupported inputs.
  - Dependencies: T003

- [X] T011 [PRD: P0-R6-AC3, P0-R6-AC4, P0-R6-AC5, P0-R5-AC6] Implement style support diagnostics.
  - ID: T011
  - Title: Implement style support diagnostics
  - PRD tags: [P0-R6-AC3] [P0-R6-AC4] [P0-R6-AC5] [P0-R5-AC6]
  - Files: `python/forge3d/style.py`, `python/forge3d/diagnostics.py`, `python/forge3d/__init__.pyi`
  - Action: Add style audit/validation helpers that preserve supported local feature styling while emitting structured diagnostics for unsupported style fields and layer types.
  - Verification: `pytest tests/test_diagnostics_style_support.py -q`
  - Done evidence: Style diagnostics tests pass and no streamed MVT support is implied by the API.
  - Dependencies: T010

- [X] T012 [P] [PRD: P0-R6-AC1, P0-R6-AC2, P0-R6-AC3, P0-R6-AC6] Add docs support-matrix audit tests.
  - ID: T012
  - Title: Add support-matrix docs audit tests
  - PRD tags: [P0-R6-AC1] [P0-R6-AC2] [P0-R6-AC3] [P0-R6-AC6]
  - Files: `tests/test_support_matrices_docs.py`, `docs/guides/style_support_matrix.md`, `docs/guides/label_support_matrix.md`, `docs/guides/building_support_matrix.md`, `docs/guides/tiles3d_support_matrix.md`, `docs/guides/virtual_texturing_support_matrix.md`, `docs/guides/diagnostics_reference.md`
  - Action: Add audit tests requiring exact PRD support-level vocabulary, supported style layer listings, local/provided feature scope, and no full Mapbox, Cesium, VT normal/mask, textured PBR, or production line/curved label overclaims.
  - Verification: `pytest tests/test_support_matrices_docs.py -q`
  - Done evidence: Docs audit fails until required matrices exist and forbidden wording is absent.
  - Dependencies: T001

- [X] T013 [PRD: P0-R6-AC1, P0-R6-AC2, P0-R6-AC3, P0-R6-AC4, P0-R6-AC5, P0-R6-AC6, P0-R5-AC1] Add diagnostics and support-matrix documentation.
  - ID: T013
  - Title: Add diagnostics and support-matrix docs
  - PRD tags: [P0-R6-AC1] [P0-R6-AC2] [P0-R6-AC3] [P0-R6-AC4] [P0-R6-AC5] [P0-R6-AC6] [P0-R5-AC1]
  - Files: `docs/guides/offline_3d_map_rendering.md`, `docs/guides/diagnostics_reference.md`, `docs/guides/style_support_matrix.md`, `docs/guides/label_support_matrix.md`, `docs/guides/building_support_matrix.md`, `docs/guides/tiles3d_support_matrix.md`, `docs/guides/virtual_texturing_support_matrix.md`, `docs/guides/competitive_positioning.md`, `docs/index.rst`
  - Action: Create or update docs with exact diagnostic codes, severity/remediation guidance, support classifications, local/provided style scope, and matrix entries for style, labels, buildings, 3D Tiles, VT, and competitive positioning.
  - Verification: `pytest tests/test_support_matrices_docs.py -q`
  - Done evidence: Docs audit passes and `docs/index.rst` references new source docs that should be built.
  - Dependencies: T012

## Phase 4: Validation, Continuity, and Coverage

- [X] T014 [P] [PRD: P0-R5-AC1, P0-R5-AC4, P0-R5-AC6] Add quickstart validation tests.
  - ID: T014
  - Title: Add diagnostics quickstart validation tests
  - PRD tags: [P0-R5-AC1] [P0-R5-AC4] [P0-R5-AC6]
  - Files: `tests/test_diagnostics_quickstart.py`, `specs/001-diagnostics-support-matrices/quickstart.md`, expected `python/forge3d/diagnostics.py`
  - Action: Add automated checks for the quickstart scenarios: style support validation, render warning policy, bundle-ready serialization, and negative path diagnostics.
  - Verification: `pytest tests/test_diagnostics_quickstart.py -q`
  - Done evidence: Quickstart scenarios are executable and produce structured diagnostics.
  - Dependencies: T005, T007, T009, T011

- [X] T015 [PRD: P0-R5-AC1, P0-R5-AC2, P0-R5-AC3, P0-R5-AC4, P0-R5-AC5, P0-R5-AC6, P0-R6-AC1, P0-R6-AC2, P0-R6-AC3, P0-R6-AC4, P0-R6-AC5, P0-R6-AC6] Run full feature verification.
  - ID: T015
  - Title: Run full diagnostics feature verification
  - PRD tags: [P0-R5-AC1] [P0-R5-AC2] [P0-R5-AC3] [P0-R5-AC4] [P0-R5-AC5] [P0-R5-AC6] [P0-R6-AC1] [P0-R6-AC2] [P0-R6-AC3] [P0-R6-AC4] [P0-R6-AC5] [P0-R6-AC6]
  - Files: `tests/test_diagnostics_contract.py`, `tests/test_diagnostics_bundle_serialization.py`, `tests/test_diagnostics_support_paths.py`, `tests/test_diagnostics_no_op_policy.py`, `tests/test_diagnostics_style_support.py`, `tests/test_support_matrices_docs.py`, `tests/test_diagnostics_quickstart.py`
  - Action: Run the full feature test set and record observed output summary without changing requirement status beyond available evidence.
  - Verification: `pytest tests/test_diagnostics_contract.py tests/test_diagnostics_bundle_serialization.py tests/test_diagnostics_support_paths.py tests/test_diagnostics_no_op_policy.py tests/test_diagnostics_style_support.py tests/test_support_matrices_docs.py tests/test_diagnostics_quickstart.py -q`
  - Done evidence: All feature tests pass or blockers are recorded with exact failing command output.
  - Dependencies: T014

- [X] T016 [PRD: P0-R5-AC1, P0-R5-AC2, P0-R5-AC3, P0-R5-AC4, P0-R5-AC5, P0-R5-AC6, P0-R6-AC1, P0-R6-AC2, P0-R6-AC3, P0-R6-AC4, P0-R6-AC5, P0-R6-AC6] Update requirements verification matrix.
  - ID: T016
  - Title: Update requirements verification matrix
  - PRD tags: [P0-R5-AC1] [P0-R5-AC2] [P0-R5-AC3] [P0-R5-AC4] [P0-R5-AC5] [P0-R5-AC6] [P0-R6-AC1] [P0-R6-AC2] [P0-R6-AC3] [P0-R6-AC4] [P0-R6-AC5] [P0-R6-AC6]
  - Files: `docs/superpowers/state/requirements-verification-matrix.md`
  - Action: Update rows for P0-R5 and P0-R6 with exact source, test, docs, diagnostics, and command evidence; do not mark `Verified` unless verification was run and evidence is present.
  - Verification: `rg -n "P0-R5-AC[1-6]|P0-R6-AC[1-6]|001-diagnostics-support-matrices|tests/test_diagnostics" docs/superpowers/state/requirements-verification-matrix.md`
  - Done evidence: Matrix rows reference concrete artifacts and statuses match actual evidence.
  - Dependencies: T015

- [X] T017 [PRD: P0-R5-AC4, P0-R5-AC6, P0-R6-AC6] Update implementation ledger and current context pack.
  - ID: T017
  - Title: Update continuity state
  - PRD tags: [P0-R5-AC4] [P0-R5-AC6] [P0-R6-AC6]
  - Files: `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/current-context-pack.md`
  - Action: Record completed diagnostics tasks, verification commands, docs/support-matrix evidence, unsupported-path decisions, and residual blockers for future sessions.
  - Verification: `rg -n "001-diagnostics-support-matrices|P0-R5|P0-R6|unsupported_style_field|vt_unsupported_family|placeholder_fallback" docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md`
  - Done evidence: Ledger and context pack contain current task outcomes and no unverified success claims.
  - Dependencies: T016

## Phase 5: Strict Audit Gap Remediation

- [X] T018 [PRD: P0-R5-AC4] Add scene-bundle diagnostic persistence tests.
  - ID: T018
  - Title: Add scene-bundle diagnostic persistence tests
  - PRD tags: [P0-R5-AC4]
  - Files: `tests/test_bundle_roundtrip.py`, `python/forge3d/bundle.py`, `python/forge3d/diagnostics.py`
  - Action: Add a failing test proving a `ValidationReport` can be saved in a scene bundle, loaded back, and preserved without losing diagnostic fields.
  - Verification: `pytest tests/test_bundle_roundtrip.py tests/test_diagnostics_bundle_serialization.py -q`
  - Done evidence: Test fails before bundle integration and passes after diagnostics are persisted in `scene/state.json`.
  - Dependencies: T017

- [X] T019 [PRD: P0-R5-AC4] Implement scene-bundle diagnostic persistence.
  - ID: T019
  - Title: Implement scene-bundle diagnostic persistence
  - PRD tags: [P0-R5-AC4]
  - Files: `python/forge3d/bundle.py`
  - Action: Add `ValidationReport` preservation to `SceneState`, `LoadedBundle`, `save_bundle()`, and `load_bundle()` without changing older bundle compatibility.
  - Verification: `pytest tests/test_bundle_roundtrip.py tests/test_diagnostics_bundle_serialization.py -q`
  - Done evidence: Bundle save/load tests preserve `ValidationReport.to_dict()` exactly.
  - Dependencies: T018

- [X] T020 [PRD: P0-R5-AC6, P0-R5-AC3] Add non-style workflow diagnostic wiring tests.
  - ID: T020
  - Title: Add non-style workflow diagnostic wiring tests
  - PRD tags: [P0-R5-AC6] [P0-R5-AC3]
  - Files: `tests/test_diagnostics_support_paths.py`, `python/forge3d/buildings.py`, `python/forge3d/tiles3d.py`, `python/forge3d/terrain_params.py`, `python/forge3d/diagnostics.py`
  - Action: Add failing tests proving building placeholder/fallback, public Python 3D Tiles incompleteness, VT non-albedo families, and experimental label paths expose public `ValidationReport` helpers.
  - Verification: `pytest tests/test_diagnostics_support_paths.py -q`
  - Done evidence: Non-style validators return structured diagnostics with affected IDs and blocking status.
  - Dependencies: T019

- [X] T021 [PRD: P0-R5-AC6, P0-R5-AC3] Implement non-style workflow diagnostic helpers.
  - ID: T021
  - Title: Implement non-style workflow diagnostic helpers
  - PRD tags: [P0-R5-AC6] [P0-R5-AC3]
  - Files: `python/forge3d/buildings.py`, `python/forge3d/tiles3d.py`, `python/forge3d/terrain_params.py`, `python/forge3d/diagnostics.py`, `python/forge3d/__init__.py`, `python/forge3d/__init__.pyi`
  - Action: Wire diagnostics into public validation helpers for known non-style unsupported, placeholder/fallback, experimental, and VT paths without claiming render support.
  - Verification: `pytest tests/test_diagnostics_support_paths.py -q`
  - Done evidence: Validators return typed diagnostics rather than requiring callers to instantiate factories manually.
  - Dependencies: T020

- [X] T022 [PRD: P0-R6-AC5] Add style output workflow compatibility tests.
  - ID: T022
  - Title: Add style output workflow compatibility tests
  - PRD tags: [P0-R6-AC5]
  - Files: `tests/test_diagnostics_style_support.py`, `python/forge3d/style.py`, `python/forge3d/terrain_params.py`
  - Action: Add failing tests proving supported style output can become `VectorOverlayConfig` payloads and symbol layers can produce future `LabelLayer` contract payloads without streamed MVT claims.
  - Verification: `pytest tests/test_diagnostics_style_support.py -q`
  - Done evidence: Tests prove style output is not just a support summary; it feeds typed vector and label-contract shapes.
  - Dependencies: T021

- [X] T023 [PRD: P0-R6-AC5] Implement style output workflow helpers.
  - ID: T023
  - Title: Implement style output workflow helpers
  - PRD tags: [P0-R6-AC5]
  - Files: `python/forge3d/style.py`, `python/forge3d/__init__.py`, `python/forge3d/__init__.pyi`
  - Action: Implement local/provided feature helpers that convert supported fill, line, and circle style output to `VectorOverlayConfig` and symbol layers to a future `LabelLayer` contract payload.
  - Verification: `pytest tests/test_diagnostics_style_support.py -q`
  - Done evidence: Style compatibility tests pass and no streamed MVT support is implied.
  - Dependencies: T022

- [X] T024 [PRD: P0-R5-AC4, P0-R5-AC6, P0-R6-AC5] Run strict audit remediation verification and update matrix.
  - ID: T024
  - Title: Run remediation verification and update matrix
  - PRD tags: [P0-R5-AC4] [P0-R5-AC6] [P0-R6-AC5]
  - Files: `docs/superpowers/state/requirements-verification-matrix.md`
  - Action: Run focused and full feature verification, then update matrix evidence for the remediated acceptance criteria only when command output supports it.
  - Verification: `pytest tests/test_bundle_roundtrip.py tests/test_diagnostics_bundle_serialization.py tests/test_diagnostics_support_paths.py tests/test_diagnostics_style_support.py tests/test_diagnostics_contract.py tests/test_diagnostics_no_op_policy.py tests/test_support_matrices_docs.py tests/test_diagnostics_quickstart.py -q`
  - Done evidence: Matrix rows cite bundle persistence, non-style validators, style workflow helpers, and passing command output.
  - Dependencies: T023

- [X] T025 [PRD: P0-R5-AC4, P0-R5-AC6, P0-R6-AC5] Update blockers, ledger, and context pack.
  - ID: T025
  - Title: Update continuity state after remediation
  - PRD tags: [P0-R5-AC4] [P0-R5-AC6] [P0-R6-AC5]
  - Files: `docs/superpowers/state/open-blockers.md`, `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/current-context-pack.md`, `specs/001-diagnostics-support-matrices/tasks.md`
  - Action: Mark T018-T025 complete as evidence supports them, close or update R-015 through R-017, and record commands/results without claiming full feature completion before analysis.
  - Verification: `rg -n "T018|T019|T020|T021|T022|T023|T024|T025|R-015|R-016|R-017|P0-R5-AC4|P0-R5-AC6|P0-R6-AC5" specs/001-diagnostics-support-matrices/tasks.md docs/superpowers/state/open-blockers.md docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md docs/superpowers/state/requirements-verification-matrix.md`
  - Done evidence: Continuity artifacts record remediated evidence, `/speckit.analyze` pass metrics, and any remaining downstream non-feature gaps.
  - Dependencies: T024

## Phase 6: Follow-Up Audit Gap Remediation

- [X] T026 [PRD: P0-R6-AC4] Add parsed-style unsupported field regression test.
  - ID: T026
  - Title: Add parsed-style unsupported field regression test
  - PRD tags: [P0-R6-AC4]
  - Files: `tests/test_diagnostics_style_support.py`, `python/forge3d/style.py`
  - Action: Add a failing regression test proving `validate_style_support(parse_style(raw_style))` still emits `unsupported_style_field` diagnostics for unsupported paint and layout keys.
  - Verification: `pytest tests/test_diagnostics_style_support.py -q`
  - Done evidence: Test fails before unsupported field metadata is retained on parsed `StyleLayer` objects and passes after implementation.
  - Dependencies: T025

- [X] T027 [PRD: P0-R6-AC4] Preserve unsupported style fields through parsing.
  - ID: T027
  - Title: Preserve unsupported style fields through parsing
  - PRD tags: [P0-R6-AC4]
  - Files: `python/forge3d/style.py`
  - Action: Preserve unsupported paint/layout field names on parsed style layers and have validation use that metadata when raw dictionaries are not available.
  - Verification: `pytest tests/test_diagnostics_style_support.py -q`
  - Done evidence: Parsed-style validation emits typed `unsupported_style_field` diagnostics without requiring the original raw style dict.
  - Dependencies: T026

- [X] T028 [PRD: P0-R6-AC6] Add public style API overclaim audit test.
  - ID: T028
  - Title: Add public style API overclaim audit test
  - PRD tags: [P0-R6-AC6]
  - Files: `tests/test_support_matrices_docs.py`, `python/forge3d/style.py`
  - Action: Add a failing docs/API audit test that scans public style source docstrings for full/complete Mapbox support overclaims.
  - Verification: `pytest tests/test_support_matrices_docs.py -q`
  - Done evidence: Test fails on existing overclaim wording and passes only when public API docs stay within local/provided feature scope.
  - Dependencies: T025

- [X] T029 [PRD: P0-R6-AC6] Reword public style API docstrings.
  - ID: T029
  - Title: Reword public style API docstrings
  - PRD tags: [P0-R6-AC6]
  - Files: `python/forge3d/style.py`
  - Action: Replace public style docstrings that imply complete/full Mapbox support with subset/local-feature wording consistent with the support matrix.
  - Verification: `pytest tests/test_support_matrices_docs.py -q`
  - Done evidence: Docs/API audit passes and no public source docstring claims complete/full Mapbox Style support.
  - Dependencies: T028

- [X] T030 [PRD: P0-R6-AC4, P0-R6-AC6] Run follow-up remediation verification and update matrix.
  - ID: T030
  - Title: Run follow-up remediation verification and update matrix
  - PRD tags: [P0-R6-AC4] [P0-R6-AC6]
  - Files: `docs/superpowers/state/requirements-verification-matrix.md`
  - Action: Run focused and full feature verification, then update P0-R6-AC4 and P0-R6-AC6 matrix evidence only when command output supports it.
  - Verification: `pytest tests/test_bundle_roundtrip.py tests/test_diagnostics_bundle_serialization.py tests/test_diagnostics_support_paths.py tests/test_diagnostics_style_support.py tests/test_diagnostics_contract.py tests/test_diagnostics_no_op_policy.py tests/test_support_matrices_docs.py tests/test_diagnostics_quickstart.py -q`
  - Done evidence: Matrix rows cite parsed-style unsupported-field diagnostics, public API overclaim audit, and passing command output.
  - Dependencies: T027, T029

- [X] T031 [PRD: P0-R6-AC4, P0-R6-AC6] Update blockers, ledger, and context pack after follow-up remediation.
  - ID: T031
  - Title: Update continuity state after follow-up remediation
  - PRD tags: [P0-R6-AC4] [P0-R6-AC6]
  - Files: `docs/superpowers/state/open-blockers.md`, `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/current-context-pack.md`, `specs/001-diagnostics-support-matrices/tasks.md`
  - Action: Close or update R-018 and R-019, mark T026-T031 complete only when evidence supports them, and record commands/results without claiming feature completion before analysis.
  - Verification: `rg -n "T026|T027|T028|T029|T030|T031|R-018|R-019|P0-R6-AC4|P0-R6-AC6" specs/001-diagnostics-support-matrices/tasks.md docs/superpowers/state/open-blockers.md docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md docs/superpowers/state/requirements-verification-matrix.md`
  - Done evidence: Continuity artifacts record remediated evidence, any `/speckit.analyze` outcome, and any remaining downstream non-feature gaps.
  - Dependencies: T030

## Phase 7: Completion Review Fixes

- [X] T032 [PRD: P0-R5-AC3, P0-R5-AC6, P0-R6-AC4, P0-R6-AC5, P0-R6-AC6] Add completion-review regression tests.
  - ID: T032
  - Title: Add completion-review regression tests
  - PRD tags: [P0-R5-AC3] [P0-R5-AC6] [P0-R6-AC4] [P0-R6-AC5] [P0-R6-AC6]
  - Files: `tests/test_diagnostics_contract.py`, `tests/test_diagnostics_support_paths.py`, `tests/test_diagnostics_style_support.py`, `tests/test_support_matrices_docs.py`
  - Action: Add failing tests for support-summary vocabulary rejection, per-label missing-glyph object IDs, symbol style underdeveloped truth, markdown support-matrix row validation, and Git visibility for required evidence files.
  - Verification: `pytest tests/test_diagnostics_contract.py::test_validation_report_rejects_unknown_support_summary_levels tests/test_diagnostics_support_paths.py::test_label_validator_preserves_missing_glyph_object_ids_per_label tests/test_diagnostics_style_support.py::test_symbol_style_support_matches_underdeveloped_matrix_truth tests/test_support_matrices_docs.py::test_required_evidence_files_are_not_git_ignored tests/test_support_matrices_docs.py::test_support_matrix_rows_use_prd_terms_and_remediation_columns -q`
  - Done evidence: Red run failed on the support-summary, missing-glyph object ID, symbol support, and Git visibility gaps before implementation.
  - Dependencies: T031

- [X] T033 [PRD: P0-R5-AC3, P0-R5-AC6, P0-R6-AC4, P0-R6-AC5] Implement completion-review behavior fixes.
  - ID: T033
  - Title: Implement completion-review behavior fixes
  - PRD tags: [P0-R5-AC3] [P0-R5-AC6] [P0-R6-AC4] [P0-R6-AC5]
  - Files: `python/forge3d/diagnostics.py`, `python/forge3d/style.py`, `tests/test_bundle_roundtrip.py`, `tests/test_diagnostics_bundle_serialization.py`
  - Action: Validate support-summary values against PRD support levels, emit one missing-glyph diagnostic per affected label object, classify `symbol` style layers as `underdeveloped` with `experimental_feature`, and update fixtures to use support-level values.
  - Verification: Same focused command as T032.
  - Done evidence: Focused command reported `5 passed`.
  - Dependencies: T032

- [X] T034 [PRD: P0-R6-AC6, Constitution VII, Constitution X] Make feature evidence Git-visible and restore unrelated historical docs.
  - ID: T034
  - Title: Make evidence visible and restore unrelated docs
  - PRD tags: [P0-R6-AC6] [Constitution VII] [Constitution X]
  - Files: `.gitignore`, `docs/superpowers/plans/2026-04-25-khumbu-sentinel-timelapse-implementation.md`, `docs/superpowers/plans/2026-05-05-khumbu-smooth-orbit-light-render-implementation.md`, `docs/superpowers/plans/3d-map-rendering-gaps-assessment.md`
  - Action: Remove broad ignores for `.specify/`, `.agents/`, `specs/`, `docs/guides`, and `docs/superpowers`; restore the unrelated historical plan files that had appeared as deleted.
  - Verification: `git check-ignore -v specs/001-diagnostics-support-matrices/tasks.md docs/guides/diagnostics_reference.md docs/guides/style_support_matrix.md docs/guides/label_support_matrix.md docs/guides/building_support_matrix.md docs/guides/tiles3d_support_matrix.md docs/guides/virtual_texturing_support_matrix.md docs/superpowers/state/requirements-verification-matrix.md docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md .specify/memory/constitution.md`
  - Done evidence: `git check-ignore` returned exit code 1 with no ignored evidence output; targeted historical plan paths show no deletion diff.
  - Dependencies: T033

- [X] T035 [PRD: P0-R5-AC1 through P0-R5-AC6, P0-R6-AC1 through P0-R6-AC6] Run completion-review verification and update continuity state.
  - ID: T035
  - Title: Run completion-review verification
  - PRD tags: [P0-R5-AC1] [P0-R5-AC2] [P0-R5-AC3] [P0-R5-AC4] [P0-R5-AC5] [P0-R5-AC6] [P0-R6-AC1] [P0-R6-AC2] [P0-R6-AC3] [P0-R6-AC4] [P0-R6-AC5] [P0-R6-AC6]
  - Files: `docs/superpowers/state/requirements-verification-matrix.md`, `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/current-context-pack.md`, `docs/superpowers/state/open-blockers.md`, `specs/001-diagnostics-support-matrices/tasks.md`
  - Action: Run the required full feature suite and update continuity artifacts with the new evidence while keeping R-011 open for unrelated remaining workspace changes.
  - Verification: `pytest tests/test_bundle_roundtrip.py tests/test_diagnostics_bundle_serialization.py tests/test_diagnostics_support_paths.py tests/test_diagnostics_style_support.py tests/test_diagnostics_contract.py tests/test_diagnostics_no_op_policy.py tests/test_support_matrices_docs.py tests/test_diagnostics_quickstart.py -q`
  - Done evidence: Full feature command reported `63 passed`; targeted `git status` showed `001` docs/spec/state files visible to Git and no deletion diff for restored historical plan files.
  - Dependencies: T034

## Dependencies and Execution Order

- Phase 1 gates source path ownership and diagnostics contract tests.
- T002, T004, T006, T008, T010, T012, and T014 can be authored in parallel once T001 completes because they target separate test files.
- Source implementation tasks depend on their corresponding tests: T003 after T002, T005 after T004, T007 after T006, T009 after T008, T011 after T010, and T013 after T012.
- Continuity updates T016 and T017 happen after verification evidence exists.
- Follow-up remediation must run after T025: T026 before T027, T028 before T029, T030 after T027 and T029, then T031.
- Completion-review fixes run after T031: T032 before T033, T034 after behavior fixes, then T035 after verification and Git visibility evidence.

## PRD Coverage Check

- P0-R5-AC1: T001, T002, T003, T004, T005, T013, T014, T015, T016
- P0-R5-AC2: T002, T003, T006, T007, T015, T016
- P0-R5-AC3: T002, T003, T006, T007, T015, T016, T020, T021, T032, T033, T035
- P0-R5-AC4: T001, T004, T005, T014, T015, T016, T017, T018, T019, T024, T025
- P0-R5-AC5: T002, T003, T008, T009, T015, T016
- P0-R5-AC6: T001, T006, T007, T008, T009, T011, T014, T015, T016, T017, T020, T021, T024, T025, T032, T033, T035
- P0-R6-AC1: T010, T012, T013, T015, T016
- P0-R6-AC2: T010, T012, T013, T015, T016
- P0-R6-AC3: T010, T011, T012, T013, T015, T016
- P0-R6-AC4: T010, T011, T013, T015, T016, T026, T027, T030, T031, T032, T033, T035
- P0-R6-AC5: T010, T011, T013, T015, T016, T022, T023, T024, T025, T032, T033, T035
- P0-R6-AC6: T012, T013, T015, T016, T017, T028, T029, T030, T031, T032, T034, T035
- Section 13 validation report and diagnostic inventory: T002, T003, T006, T007, T014, T015, T016, T020, T021
- Section 16 documentation: T012, T013, T014, T015, T016, T017, T024, T025
- Appendix B support classification discipline: T012, T013, T016, T017, T024, T025, T032, T033, T034, T035
- Milestone 0 support matrix cleanup: T012, T013, T015, T016, T017, T024, T025, T032, T034, T035
- Constitution IV deterministic evidence: T004, T005, T014, T015, T018, T019

## Detailed Pre-Implementation Coverage Audit Table

| PRD requirement | Acceptance criterion | Task IDs covering it | Test task IDs | Docs task IDs | Verification command | Risk if omitted |
|---|---|---|---|---|---|---|
| P0-R5 Production Diagnostics | P0-R5-AC1 structured diagnostic objects | T001, T002, T003, T004, T005, T013, T014, T015, T016 | T002, T004, T014 | T013, T016, T017 | `pytest tests/test_diagnostics_contract.py tests/test_diagnostics_bundle_serialization.py tests/test_diagnostics_quickstart.py -q` | Diagnostics degrade to logs, warnings, or ad hoc dictionaries that cannot support validation or bundles. |
| P0-R5 Production Diagnostics | P0-R5-AC2 severity values | T002, T003, T006, T007, T015, T016 | T002, T006 | T016 | `pytest tests/test_diagnostics_contract.py tests/test_diagnostics_support_paths.py -q` | Render policy cannot distinguish info, warning, error, and fatal states. |
| P0-R5 Production Diagnostics | P0-R5-AC3 affected layer/object IDs | T002, T003, T006, T007, T015, T016, T020, T021, T032, T033, T035 | T002, T006, T020, T032 | T016, T035 | `pytest tests/test_diagnostics_contract.py tests/test_diagnostics_support_paths.py -q` | Users cannot identify which layer or object needs remediation. |
| P0-R5 Production Diagnostics | P0-R5-AC4 bundle-serializable diagnostics | T001, T004, T005, T014, T015, T016, T017, T018, T019, T024, T025 | T004, T014, T018 | T013, T016, T017, T024, T025 | `pytest tests/test_bundle_roundtrip.py tests/test_diagnostics_bundle_serialization.py -q` | Review bundles lose diagnostic evidence and cannot reproduce validation state. |
| P0-R5 Production Diagnostics | P0-R5-AC5 fail-on-warning policy | T002, T003, T008, T009, T015, T016 | T002, T008 | T016 | `pytest tests/test_diagnostics_contract.py tests/test_diagnostics_no_op_policy.py -q` | Render workflows cannot enforce conservative validation behavior. |
| P0-R5 Production Diagnostics | P0-R5-AC6 unsupported never silently ignored | T001, T006, T007, T008, T009, T011, T014, T015, T016, T017, T020, T021, T024, T025, T032, T033, T035 | T006, T008, T014, T020, T032 | T013, T016, T017, T024, T025, T035 | `pytest tests/test_diagnostics_support_paths.py tests/test_diagnostics_no_op_policy.py tests/test_diagnostics_quickstart.py -q` | Unsupported, Pro-gated, placeholder, or experimental paths can pass as successful work. |
| P0-R6 Style Honesty | P0-R6-AC1 local/provided feature scope | T010, T012, T013, T015, T016 | T010, T012 | T012, T013, T016 | `pytest tests/test_diagnostics_style_support.py tests/test_support_matrices_docs.py -q` | Users can mistake local feature styling for streamed MVT support. |
| P0-R6 Style Honesty | P0-R6-AC2 supported layer list | T010, T012, T013, T015, T016 | T010, T012 | T012, T013, T016 | `pytest tests/test_diagnostics_style_support.py tests/test_support_matrices_docs.py -q` | Public docs cannot prove which style layers are supported. |
| P0-R6 Style Honesty | P0-R6-AC3 fill, line, and circle support where implemented | T010, T011, T012, T013, T015, T016 | T010, T012 | T012, T013, T016 | `pytest tests/test_diagnostics_style_support.py tests/test_support_matrices_docs.py -q` | Existing style capability can be overclaimed, underdocumented, or disconnected from overlays. |
| P0-R6 Style Honesty | P0-R6-AC4 unsupported style diagnostics | T010, T011, T013, T015, T016, T026, T027, T030, T031, T032, T033, T035 | T010, T026, T032 | T013, T016, T030, T031, T035 | `pytest tests/test_diagnostics_style_support.py -q` | Unsupported style fields or layer types can disappear silently, including after parsing. |
| P0-R6 Style Honesty | P0-R6-AC5 style output feeds vector and label workflows | T010, T011, T013, T015, T016, T022, T023, T024, T025, T032, T033, T035 | T010, T022, T032 | T013, T016, T024, T025, T035 | `pytest tests/test_diagnostics_style_support.py -q` | Style support remains isolated from typed overlay and future label workflows. |
| P0-R6 Style Honesty | P0-R6-AC6 no full Mapbox support claim | T012, T013, T015, T016, T017, T028, T029, T030, T031, T032, T034, T035 | T012, T028, T032 | T012, T013, T016, T017, T030, T031, T034, T035 | `pytest tests/test_support_matrices_docs.py -q` | Docs or public API docstrings overclaim full Mapbox Style or streamed vector-tile parity. |

## Audit Rerun Result

- PASS: Every P0-R5 and P0-R6 acceptance criterion has at least one task, one verification path, and test or diagnostic proof.
- PASS: Public diagnostics and style-support tasks are paired with docs/support-matrix wording tasks T012, T013, T016, and T017.
- PASS: Diagnostic requirements require structured objects, bundle serialization, severity, affected IDs, and no-op/unsupported-path negative tests rather than printed text only.
- PASS: Completion-review fixes T032-T035 added regression coverage for support-summary taxonomy, per-object missing glyph diagnostics, `symbol` style truth, support-matrix row audits, and Git-visible evidence; full feature verification reported `63 passed`.

## Extension Hooks

**Optional Hook**: git
Command: `/speckit.git.commit`
Description: Auto-commit after task generation

Prompt: Commit task changes?
To execute: `/speckit.git.commit`

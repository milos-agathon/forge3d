# Tasks: Deterministic LabelPlan

**Input**: `specs/003-deterministic-label-plan/plan.md`, `spec.md`, `research.md`, `data-model.md`, `quickstart.md`, and `contracts/label-plan-contract.md`

**Scope**: P0-R3, PRD Section 14, Milestone 2, deterministic label planning, rejection diagnostics, render/export payloads, and required continuity state updates.

**Required task format**: Every task below includes ID, title, PRD tags, files, action, verification, done evidence, and dependencies. The first checklist line preserves the Spec Kit task ID and `[P]` marker where work can be safely parallelized.

**Execution rule**: Tests must be written before or alongside implementation. Successful compile/render/export operations must return real accepted/rejected plan data or typed diagnostics, never empty placeholder success.

## Phase 1: Compiler Contract and Path Gates

- [X] T001 [PRD: P0-R3-AC1, P0-R3-AC2, P0-R3-AC8] Inspect LabelPlan ownership paths and dependencies.
  - ID: T001
  - Title: Inspect LabelPlan ownership paths
  - PRD tags: [P0-R3-AC1] [P0-R3-AC2] [P0-R3-AC8]
  - Files: expected `python/forge3d/label_plan.py`, `python/forge3d/__init__.py`, `python/forge3d/__init__.pyi`, `src/labels/layer.rs`, `src/labels/declutter.rs`, `src/labels/collision.rs`, `src/labels/projection.rs`, `src/labels/atlas.rs`, `tests/`, `docs/superpowers/state/implementation-ledger.md`
  - Action: Confirm the public module path, whether compiler logic is Python-first or Rust-backed, and where feature `001` diagnostics and feature `002` label inputs are available.
  - Verification: `rg -n "LabelPlan|AcceptedLabel|RejectedLabel|Keepout|priority|declutter|collision|atlas|typography" python src tests docs/superpowers/state`
  - Done evidence: Ledger records selected source/test/docs paths and dependency status before implementation.
  - Dependencies: None

- [X] T002 [P] [PRD: P0-R3-AC1, P0-R3-AC8] Add LabelPlan API contract tests.
  - ID: T002
  - Title: Add LabelPlan API contract tests
  - PRD tags: [P0-R3-AC1] [P0-R3-AC8]
  - Files: `tests/test_label_plan_contract.py`, expected `python/forge3d/label_plan.py`, `python/forge3d/__init__.pyi`
  - Action: Add tests for `LabelPlan.compile(...)` inputs, `accepted`, `rejected`, `diagnostics`, `bounds`, `seed`, `to_dict()`, `from_dict()`, `to_render_payload()`, and `to_export_payload()`.
  - Verification: `pytest tests/test_label_plan_contract.py -q`
  - Done evidence: Tests fail until public contract and typed outputs exist.
  - Dependencies: T001

- [X] T003 [P] [PRD: P0-R3-AC1] Add determinism and order-normalization tests.
  - ID: T003
  - Title: Add LabelPlan determinism tests
  - PRD tags: [P0-R3-AC1]
  - Files: `tests/test_label_plan_determinism.py`, expected `python/forge3d/label_plan.py`
  - Action: Add tests compiling identical inputs twice and compiling equivalent inputs with different dict/list/set/source order; compare accepted labels, rejected labels, diagnostics, bounds, seed, and serialized payload exactly.
  - Verification: `pytest tests/test_label_plan_determinism.py -q`
  - Done evidence: Tests fail on nondeterministic iteration, unstable sorting, or unseeded random behavior.
  - Dependencies: T001

- [X] T004 [PRD: P0-R3-AC1, P0-R3-AC8] Implement LabelPlan public models and deterministic serialization.
  - ID: T004
  - Title: Implement LabelPlan models and serialization
  - PRD tags: [P0-R3-AC1] [P0-R3-AC8]
  - Files: `python/forge3d/label_plan.py`, `python/forge3d/__init__.py`, `python/forge3d/__init__.pyi`, expected `python/forge3d/diagnostics.py`
  - Action: Implement `LabelPlan`, `AcceptedLabel`, `RejectedLabel`, `LabelCandidate`, `KeepoutRegion`, `PriorityClass`, deterministic sort keys, and JSON-friendly serialization.
  - Verification: `pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py -q`
  - Done evidence: Contract and determinism tests pass for the base compiler object model.
  - Dependencies: T002, T003

## Phase 2: Rejection Reasons and Candidates

- [X] T005 [P] [PRD: P0-R3-AC2] Add complete rejection reason tests.
  - ID: T005
  - Title: Add rejection reason tests
  - PRD tags: [P0-R3-AC2]
  - Files: `tests/test_label_plan_rejection_reasons.py`, expected `python/forge3d/label_plan.py`
  - Action: Add fixtures for `collision`, `outside_view`, `missing_glyph`, `priority_lost`, `keepout_region`, `terrain_occluded`, `invalid_geometry`, `unsupported_geometry_type`, and `empty_text`, plus `label_rejection_summary` and `missing_glyphs` diagnostics.
  - Verification: `pytest tests/test_label_plan_rejection_reasons.py -q`
  - Done evidence: Every rejected label has a required reason code and diagnostics are structured.
  - Dependencies: T004

- [X] T006 [PRD: P0-R3-AC2] Implement rejection retention and diagnostics.
  - ID: T006
  - Title: Implement rejection reason retention
  - PRD tags: [P0-R3-AC2]
  - Files: `python/forge3d/label_plan.py`, expected `python/forge3d/diagnostics.py`
  - Action: Retain rejected candidates with reason codes, diagnostic references, affected IDs, and deterministic ordering instead of dropping failed labels.
  - Verification: `pytest tests/test_label_plan_rejection_reasons.py -q`
  - Done evidence: Rejection reason tests pass with structured summaries.
  - Dependencies: T005

- [X] T007 [P] [PRD: P0-R3-AC3] Add point candidate tests.
  - ID: T007
  - Title: Add point candidate tests
  - PRD tags: [P0-R3-AC3]
  - Files: `tests/test_label_plan_point_candidates.py`, expected `python/forge3d/label_plan.py`
  - Action: Add tests for center, above, below, left, right, and radial candidates, including deterministic radial angle/order/jitter behavior for fixed seed.
  - Verification: `pytest tests/test_label_plan_point_candidates.py -q`
  - Done evidence: Tests fail until every required point candidate type is generated in stable order.
  - Dependencies: T004

- [X] T008 [PRD: P0-R3-AC3] Implement deterministic point candidates.
  - ID: T008
  - Title: Implement point candidate generation
  - PRD tags: [P0-R3-AC3]
  - Files: `python/forge3d/label_plan.py`
  - Action: Implement center/above/below/left/right/radial candidate generation with stable candidate IDs, bounds, scores, and seeded radial ordering.
  - Verification: `pytest tests/test_label_plan_point_candidates.py tests/test_label_plan_determinism.py -q`
  - Done evidence: Point candidate and determinism tests pass.
  - Dependencies: T007

- [X] T009 [P] [PRD: P0-R3-AC4] Add polygon candidate and invalid geometry tests.
  - ID: T009
  - Title: Add polygon candidate tests
  - PRD tags: [P0-R3-AC4]
  - Files: `tests/test_label_plan_polygon_candidates.py`, expected `python/forge3d/label_plan.py`
  - Action: Add tests for polygon centroid placement, visual-center/polylabel fallback, unsuitable centroid fallback behavior, and invalid polygon rejection with `invalid_geometry`.
  - Verification: `pytest tests/test_label_plan_polygon_candidates.py -q`
  - Done evidence: Tests fail until polygons produce centroid and fallback candidates or reason-coded rejection.
  - Dependencies: T004

- [X] T010 [PRD: P0-R3-AC4] Implement polygon candidate generation.
  - ID: T010
  - Title: Implement polygon candidate generation
  - PRD tags: [P0-R3-AC4]
  - Files: `python/forge3d/label_plan.py`
  - Action: Implement centroid candidate generation, visual-center/polylabel fallback, stable polygon ordering, and invalid geometry diagnostics.
  - Verification: `pytest tests/test_label_plan_polygon_candidates.py tests/test_label_plan_rejection_reasons.py -q`
  - Done evidence: Polygon tests pass and invalid geometry is not silently skipped.
  - Dependencies: T009

## Phase 3: Terrain, Keepouts, and Priority Solve

- [X] T011 [P] [PRD: P0-R3-AC5] Add terrain sampler and occlusion tests.
  - ID: T011
  - Title: Add terrain label tests
  - PRD tags: [P0-R3-AC5]
  - Files: `tests/test_label_plan_terrain.py`, expected `python/forge3d/label_plan.py`, inspect `python/forge3d/terrain_params.py`, inspect `src/labels/projection.rs`
  - Action: Add tests for known elevation samples from a deterministic sampler, unavailable sampler diagnostics, and `terrain_occluded` rejection when terrain visibility prevents truthful placement.
  - Verification: `pytest tests/test_label_plan_terrain.py -q`
  - Done evidence: Tests cover successful sampling and typed terrain diagnostics.
  - Dependencies: T004

- [X] T012 [PRD: P0-R3-AC5] Implement terrain sampling integration.
  - ID: T012
  - Title: Implement terrain sampling integration
  - PRD tags: [P0-R3-AC5]
  - Files: `python/forge3d/label_plan.py`, inspect `python/forge3d/terrain_params.py`
  - Action: Wire candidate generation to an active terrain sampler or transform when available, and emit structured diagnostics/rejections when sampling or visibility is unavailable.
  - Verification: `pytest tests/test_label_plan_terrain.py tests/test_label_plan_rejection_reasons.py -q`
  - Done evidence: Terrain tests pass without pretending unavailable sampler support exists.
  - Dependencies: T011

- [X] T013 [P] [PRD: P0-R3-AC6] Add keepout region tests.
  - ID: T013
  - Title: Add keepout region tests
  - PRD tags: [P0-R3-AC6]
  - Files: `tests/test_label_plan_keepouts.py`, expected `python/forge3d/label_plan.py`, inspect `python/forge3d/map_plate.py`
  - Action: Add title, legend, scale bar, north arrow, and manual rectangle keepout fixtures; assert intersecting candidates reject with `keepout_region`.
  - Verification: `pytest tests/test_label_plan_keepouts.py -q`
  - Done evidence: Tests fail until every required keepout type is active in placement.
  - Dependencies: T004

- [X] T014 [PRD: P0-R3-AC6] Implement keepout region handling.
  - ID: T014
  - Title: Implement keepout handling
  - PRD tags: [P0-R3-AC6]
  - Files: `python/forge3d/label_plan.py`, inspect `python/forge3d/map_plate.py`
  - Action: Implement keepout models, deterministic ordering, bounds intersection, and rejection with `keepout_region`.
  - Verification: `pytest tests/test_label_plan_keepouts.py tests/test_label_plan_rejection_reasons.py -q`
  - Done evidence: Keepout tests pass for all furniture and manual rectangle types.
  - Dependencies: T013

- [X] T015 [P] [PRD: P0-R3-AC7] Add priority class and tie-break tests.
  - ID: T015
  - Title: Add priority solve tests
  - PRD tags: [P0-R3-AC7]
  - Files: `tests/test_label_plan_priority.py`, expected `python/forge3d/label_plan.py`
  - Action: Add collision fixtures proving priority classes determine deterministic winners and losers, and equal-priority ties use stable sort keys.
  - Verification: `pytest tests/test_label_plan_priority.py -q`
  - Done evidence: Tests fail on nondeterministic or input-order-only collision winners.
  - Dependencies: T004

- [X] T016 [PRD: P0-R3-AC7] Implement deterministic priority solve.
  - ID: T016
  - Title: Implement priority collision solve
  - PRD tags: [P0-R3-AC7]
  - Files: `python/forge3d/label_plan.py`
  - Action: Implement priority classes, collision winner selection, `priority_lost` rejections, and deterministic tie-break policies.
  - Verification: `pytest tests/test_label_plan_priority.py tests/test_label_plan_determinism.py -q`
  - Done evidence: Priority and determinism tests pass.
  - Dependencies: T015

## Phase 4: Render/Export Payloads, Docs, and Continuity

- [X] T017 [P] [PRD: P0-R3-AC8, P0-R3-AC2] Add render/export payload and no-op tests.
  - ID: T017
  - Title: Add LabelPlan payload tests
  - PRD tags: [P0-R3-AC8] [P0-R3-AC2]
  - Files: `tests/test_label_plan_payloads.py`, expected `python/forge3d/label_plan.py`, inspect `python/forge3d/export.py`
  - Action: Add tests proving render/export payloads include accepted labels, bounds, typography, glyph references, diagnostics, seed, and deterministic ordering; unsupported backends must return typed diagnostics instead of placeholder success.
  - Verification: `pytest tests/test_label_plan_payloads.py -q`
  - Done evidence: Tests fail on empty payloads, lost rejected labels, or unsupported backend success.
  - Dependencies: T006, T008, T010, T012, T014, T016

- [X] T018 [PRD: P0-R3-AC8] Implement render/export payload methods.
  - ID: T018
  - Title: Implement LabelPlan payload methods
  - PRD tags: [P0-R3-AC8]
  - Files: `python/forge3d/label_plan.py`, inspect `python/forge3d/export.py`
  - Action: Implement `to_render_payload()` and `to_export_payload()` to return real payload data for accepted labels or typed diagnostics for unsupported output paths.
  - Verification: `pytest tests/test_label_plan_payloads.py tests/test_label_plan_contract.py -q`
  - Done evidence: Payload tests pass without placeholder render/export success.
  - Dependencies: T017

- [X] T019 [P] [PRD: P0-R3-AC1, P0-R3-AC2, P0-R3-AC6, P0-R3-AC8] Add LabelPlan docs audit tests.
  - ID: T019
  - Title: Add LabelPlan docs audit tests
  - PRD tags: [P0-R3-AC1] [P0-R3-AC2] [P0-R3-AC6] [P0-R3-AC8]
  - Files: `tests/test_label_plan_docs.py`, `docs/guides/label_plan_guide.md`, `docs/guides/label_support_matrix.md`, `docs/api/api_reference.rst`
  - Action: Add docs audit tests for deterministic placement, rejection reason vocabulary, point/polygon candidates, keepouts, priorities, terrain behavior, payload support, and unsupported/experimental wording.
  - Verification: `pytest tests/test_label_plan_docs.py -q`
  - Done evidence: Docs audit fails until LabelPlan guide and support text match implemented or diagnosed behavior.
  - Dependencies: T001

- [X] T020 [PRD: P0-R3-AC1, P0-R3-AC2, P0-R3-AC3, P0-R3-AC4, P0-R3-AC5, P0-R3-AC6, P0-R3-AC7, P0-R3-AC8] Add LabelPlan guide and API docs.
  - ID: T020
  - Title: Add LabelPlan documentation
  - PRD tags: [P0-R3-AC1] [P0-R3-AC2] [P0-R3-AC3] [P0-R3-AC4] [P0-R3-AC5] [P0-R3-AC6] [P0-R3-AC7] [P0-R3-AC8]
  - Files: `docs/guides/label_plan_guide.md`, `docs/guides/label_support_matrix.md`, `docs/api/api_reference.rst`, `docs/index.rst`
  - Action: Document compile inputs/outputs, deterministic seed/order behavior, rejection reasons, candidates, terrain, keepouts, priorities, render/export payloads, and unsupported backend diagnostics.
  - Verification: `pytest tests/test_label_plan_docs.py -q`
  - Done evidence: Docs audit passes and docs do not overclaim advanced curved/repeated/non-Latin behavior.
  - Dependencies: T019

- [X] T021 [P] [PRD: P0-R3-AC1, P0-R3-AC2, P0-R3-AC8] Add quickstart validation tests.
  - ID: T021
  - Title: Add LabelPlan quickstart tests
  - PRD tags: [P0-R3-AC1] [P0-R3-AC2] [P0-R3-AC8]
  - Files: `tests/test_label_plan_quickstart.py`, `specs/003-deterministic-label-plan/quickstart.md`, expected `python/forge3d/label_plan.py`
  - Action: Add automated quickstart checks for reproducible compile, rejection reason coverage, keepout/priority behavior, and render/export payload behavior.
  - Verification: `pytest tests/test_label_plan_quickstart.py -q`
  - Done evidence: Quickstart scenarios pass or emit typed diagnostics for unsupported payload paths.
  - Dependencies: T018, T020

- [X] T022 [PRD: P0-R3-AC1, P0-R3-AC2, P0-R3-AC3, P0-R3-AC4, P0-R3-AC5, P0-R3-AC6, P0-R3-AC7, P0-R3-AC8] Run full LabelPlan verification.
  - ID: T022
  - Title: Run full LabelPlan verification
  - PRD tags: [P0-R3-AC1] [P0-R3-AC2] [P0-R3-AC3] [P0-R3-AC4] [P0-R3-AC5] [P0-R3-AC6] [P0-R3-AC7] [P0-R3-AC8]
  - Files: `tests/test_label_plan_contract.py`, `tests/test_label_plan_determinism.py`, `tests/test_label_plan_rejection_reasons.py`, `tests/test_label_plan_point_candidates.py`, `tests/test_label_plan_polygon_candidates.py`, `tests/test_label_plan_terrain.py`, `tests/test_label_plan_keepouts.py`, `tests/test_label_plan_priority.py`, `tests/test_label_plan_payloads.py`, `tests/test_label_plan_docs.py`, `tests/test_label_plan_quickstart.py`
  - Action: Run the full feature test set and record the result summary.
  - Verification: `pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py tests/test_label_plan_polygon_candidates.py tests/test_label_plan_terrain.py tests/test_label_plan_keepouts.py tests/test_label_plan_priority.py tests/test_label_plan_payloads.py tests/test_label_plan_docs.py tests/test_label_plan_quickstart.py -q`
  - Done evidence: All feature tests pass or blockers are recorded with exact failing commands.
  - Dependencies: T021

- [X] T023 [PRD: P0-R3-AC1, P0-R3-AC2, P0-R3-AC3, P0-R3-AC4, P0-R3-AC5, P0-R3-AC6, P0-R3-AC7, P0-R3-AC8] Update requirements verification matrix.
  - ID: T023
  - Title: Update requirements verification matrix
  - PRD tags: [P0-R3-AC1] [P0-R3-AC2] [P0-R3-AC3] [P0-R3-AC4] [P0-R3-AC5] [P0-R3-AC6] [P0-R3-AC7] [P0-R3-AC8]
  - Files: `docs/superpowers/state/requirements-verification-matrix.md`
  - Action: Update P0-R3 rows with exact source, tests, docs, diagnostics, payload, and command evidence; do not mark `Verified` without successful verification output.
  - Verification: `rg -n "P0-R3-AC[1-8]|003-deterministic-label-plan|tests/test_label_plan" docs/superpowers/state/requirements-verification-matrix.md`
  - Done evidence: Matrix rows reference concrete artifacts and statuses match evidence.
  - Dependencies: T022

- [X] T024 [PRD: P0-R3-AC1, P0-R3-AC2, P0-R3-AC8] Update implementation ledger and current context pack.
  - ID: T024
  - Title: Update continuity state
  - PRD tags: [P0-R3-AC1] [P0-R3-AC2] [P0-R3-AC8]
  - Files: `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/current-context-pack.md`
  - Action: Record completed tasks, verification commands, deterministic comparison evidence, rejection diagnostics, payload support decisions, docs evidence, and residual blockers.
  - Verification: `rg -n "003-deterministic-label-plan|P0-R3|LabelPlan|label_rejection_summary|missing_glyphs" docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md`
  - Done evidence: Ledger and context pack contain current outcomes and no unverified success claims.
  - Dependencies: T023

## Dependencies and Execution Order

- T001 blocks source implementation.
- T002, T003, T005, T007, T009, T011, T013, T015, T019, and T021 can be authored in parallel after T001 where their fixture files do not overlap.
- Implementation follows tests: T004 after T002/T003, T006 after T005, T008 after T007, T010 after T009, T012 after T011, T014 after T013, T016 after T015, T018 after T017, and T020 after T019.
- T023 and T024 happen after verification evidence exists.

## PRD Coverage Check

- P0-R3-AC1: T001, T002, T003, T004, T019, T020, T021, T022, T023, T024
- P0-R3-AC2: T001, T005, T006, T017, T019, T020, T021, T022, T023, T024
- P0-R3-AC3: T007, T008, T020, T022, T023
- P0-R3-AC4: T009, T010, T020, T022, T023
- P0-R3-AC5: T011, T012, T020, T022, T023
- P0-R3-AC6: T013, T014, T019, T020, T022, T023
- P0-R3-AC7: T015, T016, T020, T022, T023
- P0-R3-AC8: T001, T002, T004, T017, T018, T019, T020, T021, T022, T023, T024

## Detailed Pre-Implementation Coverage Audit Table

| PRD requirement | Acceptance criterion | Task IDs covering it | Test task IDs | Docs task IDs | Verification command | Risk if omitted |
|---|---|---|---|---|---|---|
| P0-R3 Deterministic LabelPlan | P0-R3-AC1 stable accepted/rejected set for fixed inputs | T001, T002, T003, T004, T019, T020, T021, T022, T023, T024 | T002, T003, T019, T021 | T019, T020, T023, T024 | `pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_docs.py tests/test_label_plan_quickstart.py -q` | Offline label placement can change across runs or source ordering. |
| P0-R3 Deterministic LabelPlan | P0-R3-AC2 rejected labels include reason codes | T001, T005, T006, T017, T019, T020, T021, T022, T023, T024 | T005, T017, T019, T021 | T019, T020, T023, T024 | `pytest tests/test_label_plan_rejection_reasons.py tests/test_label_plan_payloads.py tests/test_label_plan_docs.py tests/test_label_plan_quickstart.py -q` | Rejected labels disappear without actionable reason codes or summaries. |
| P0-R3 Deterministic LabelPlan | P0-R3-AC3 point candidate set | T007, T008, T020, T022, T023 | T007 | T020, T023 | `pytest tests/test_label_plan_point_candidates.py tests/test_label_plan_determinism.py -q` | Basic point label placement lacks required candidate options. |
| P0-R3 Deterministic LabelPlan | P0-R3-AC4 polygon centroid and visual-center/polylabel fallback | T009, T010, T020, T022, T023 | T009 | T020, T023 | `pytest tests/test_label_plan_polygon_candidates.py tests/test_label_plan_rejection_reasons.py -q` | Polygon labels use poor or invalid anchors without diagnostics. |
| P0-R3 Deterministic LabelPlan | P0-R3-AC5 terrain elevation sampling | T011, T012, T020, T022, T023 | T011 | T020, T023 | `pytest tests/test_label_plan_terrain.py tests/test_label_plan_rejection_reasons.py -q` | Terrain label positions can be wrong or unavailable without diagnostic proof. |
| P0-R3 Deterministic LabelPlan | P0-R3-AC6 keepout regions | T013, T014, T019, T020, T022, T023 | T013, T019 | T019, T020, T023 | `pytest tests/test_label_plan_keepouts.py tests/test_label_plan_docs.py -q` | Titles, legends, scale bars, north arrows, or manual rectangles can be ignored. |
| P0-R3 Deterministic LabelPlan | P0-R3-AC7 priority classes | T015, T016, T020, T022, T023 | T015 | T020, T023 | `pytest tests/test_label_plan_priority.py tests/test_label_plan_determinism.py -q` | Collision outcomes depend on input order instead of documented priority. |
| P0-R3 Deterministic LabelPlan | P0-R3-AC8 render/export payloads | T001, T002, T004, T017, T018, T019, T020, T021, T022, T023, T024 | T002, T017, T019, T021 | T019, T020, T023, T024 | `pytest tests/test_label_plan_payloads.py tests/test_label_plan_contract.py tests/test_label_plan_docs.py tests/test_label_plan_quickstart.py -q` | Compiled plans cannot drive rendering/export or can return empty placeholder payloads. |

## Audit Rerun Result

- PASS: Every P0-R3 acceptance criterion has task coverage, test tasks, docs/support wording, and a verification command.
- PASS: Implementation tasks require deterministic serialization, exact comparisons, retained rejected labels, and typed diagnostics for unsupported payload paths.
- PASS: No task can be completed by a no-op because done evidence requires failing tests first, structured outputs, or recorded blockers.

## Extension Hooks

**Optional Hook**: git
Command: `/speckit.git.commit`
Description: Auto-commit after task generation

Prompt: Commit task changes?
To execute: `/speckit.git.commit`

# Tasks: Label API Truth

**Input**: `specs/002-label-api-truth/plan.md`, `spec.md`, `research.md`, `data-model.md`, `quickstart.md`, and `contracts/label-api-contract.md`

**Scope**: P0-R1, P0-R2, Milestone 1, label-related diagnostics, public high-level label workflows, and required continuity state updates.

**Required task format**: Every task below includes ID, title, PRD tags, files, action, verification, done evidence, and dependencies. The first checklist line preserves the Spec Kit task ID and `[P]` marker where work can be safely parallelized.

**Execution rule**: Tests must be written before or alongside source changes. No public label create/configuration command may return success for no state change, unused metadata, placeholder output, or raw-IPC-only behavior.

## Phase 1: Public Label Contract and Path Gates

- [X] T001 [PRD: P0-R1-AC1, P0-R1-AC4, P0-R1-AC6, P0-R2-AC1] Inspect Python and native label API ownership paths.
  - ID: T001
  - Title: Inspect label API ownership paths
  - PRD tags: [P0-R1-AC1] [P0-R1-AC4] [P0-R1-AC6] [P0-R2-AC1]
  - Files: `python/forge3d/viewer.py`, `python/forge3d/viewer.pyi`, `python/forge3d/viewer_ipc.py`, `python/forge3d/__init__.py`, `python/forge3d/__init__.pyi`, `src/viewer/cmd/labels_command.rs`, `src/viewer/ipc/protocol/response.rs`, `src/labels/`, `docs/superpowers/state/implementation-ledger.md`
  - Action: Verify whether high-level methods belong on `ViewerHandle` or another public wrapper, confirm native response and label manager signatures, and record exact files before implementation.
  - Verification: `rg -n "add_label|add_labels|add_line_label|curved|callout|typography|declutter|LabelManager|Response" python src tests docs/superpowers/state`
  - Done evidence: Ledger records the selected public wrapper path, native files to touch if needed, and any dependency on feature `001` diagnostics.
  - Dependencies: None

- [X] T002 [P] [PRD: P0-R1-AC4, P0-R1-AC1, P0-R1-AC6] Add public no-raw-IPC label workflow tests.
  - ID: T002
  - Title: Add high-level label workflow tests
  - PRD tags: [P0-R1-AC4] [P0-R1-AC1] [P0-R1-AC6]
  - Files: `tests/test_label_api_public_workflow.py`, `python/forge3d/viewer.py`, `python/forge3d/viewer.pyi`
  - Action: Add tests for `load_label_atlas`, `add_label`, `add_labels`, `set_labels_enabled`, `clear_labels`, and no direct `viewer_ipc` use in the basic workflow.
  - Verification: `pytest tests/test_label_api_public_workflow.py -q`
  - Done evidence: Tests fail until the public workflow can run without raw IPC and without no-op success.
  - Dependencies: T001

- [X] T003 [P] [PRD: P0-R1-AC1, P0-R1-AC6] Add stable ID and batch ordering tests.
  - ID: T003
  - Title: Add label stable ID tests
  - PRD tags: [P0-R1-AC1] [P0-R1-AC6]
  - Files: `tests/test_label_api_stable_ids.py`, `python/forge3d/viewer.py`, `python/forge3d/viewer_ipc.py`, `src/viewer/ipc/protocol/response.rs`
  - Action: Add tests asserting `add_label`, `add_labels`, line-label creation, callout creation, and label-related overlay creation return stable IDs where later reference is needed, and that batch outputs preserve input order with per-input diagnostics.
  - Verification: `pytest tests/test_label_api_stable_ids.py -q`
  - Done evidence: Tests fail on generic success responses, omitted IDs, nondeterministic order, or dropped per-label diagnostics.
  - Dependencies: T001

- [X] T004 [PRD: P0-R1-AC1, P0-R1-AC4, P0-R1-AC6] Implement stable public create responses.
  - ID: T004
  - Title: Implement stable label create responses
  - PRD tags: [P0-R1-AC1] [P0-R1-AC4] [P0-R1-AC6]
  - Files: `python/forge3d/viewer.py`, `python/forge3d/viewer.pyi`, `python/forge3d/viewer_ipc.py`, `python/forge3d/__init__.pyi`, inspect `src/viewer/cmd/labels_command.rs`, inspect `src/viewer/ipc/protocol/response.rs`
  - Action: Add high-level create wrappers and native/IPC response plumbing as needed so successful create operations return stable IDs and rejected inputs return typed diagnostics.
  - Verification: `pytest tests/test_label_api_public_workflow.py tests/test_label_api_stable_ids.py -q`
  - Done evidence: Public workflow and stable ID tests pass without raw IPC in the tested user workflow.
  - Dependencies: T002, T003

## Phase 2: Truthful Configuration State

- [X] T005 [P] [PRD: P0-R1-AC2, P0-R1-AC3, P0-R1-AC6] Add typography and declutter truthfulness tests.
  - ID: T005
  - Title: Add typography and declutter truthfulness tests
  - PRD tags: [P0-R1-AC2] [P0-R1-AC3] [P0-R1-AC6]
  - Files: `tests/test_label_api_configuration_truth.py`, `python/forge3d/viewer.py`, `src/labels/typography.rs`, `src/labels/declutter.rs`, `src/viewer/cmd/labels_command.rs`
  - Action: Add tests proving `set_label_typography()` and `set_declutter_algorithm()` mutate real state, affect layout/render metrics or serializable state, or return typed unsupported/experimental diagnostics.
  - Verification: `pytest tests/test_label_api_configuration_truth.py -q`
  - Done evidence: Tests fail if setters acknowledge success while leaving state unchanged. 2026-05-18 remediation updated these tests to require typography layout metrics and declutter placement-policy state instead of diagnostic-only deferral.
  - Dependencies: T001

- [X] T006 [P] [PRD: P0-R1-AC4, P0-R1-AC6] Add atlas, enabled-state, clear, and remove no-op tests.
  - ID: T006
  - Title: Add label state no-op tests
  - PRD tags: [P0-R1-AC4] [P0-R1-AC6]
  - Files: `tests/test_label_api_state_noops.py`, `python/forge3d/viewer.py`, `python/forge3d/viewer_ipc.py`, `src/viewer/cmd/labels_command.rs`
  - Action: Add tests proving atlas load, enable/disable, clear, and remove operations change real state or return typed diagnostics instead of placeholder success.
  - Verification: `pytest tests/test_label_api_state_noops.py -q`
  - Done evidence: Tests detect no-op command acknowledgements for label state changes.
  - Dependencies: T001

- [X] T007 [PRD: P0-R1-AC2, P0-R1-AC3, P0-R1-AC4, P0-R1-AC6] Implement truthful label configuration methods.
  - ID: T007
  - Title: Implement truthful label configuration methods
  - PRD tags: [P0-R1-AC2] [P0-R1-AC3] [P0-R1-AC4] [P0-R1-AC6]
  - Files: `python/forge3d/viewer.py`, `python/forge3d/viewer.pyi`, `python/forge3d/viewer_ipc.py`, `src/viewer/cmd/labels_command.rs`, `src/labels/typography.rs`, `src/labels/declutter.rs`, expected `python/forge3d/diagnostics.py`
  - Action: Wire public setters to real label state where available; otherwise return feature `001` diagnostics with `experimental_feature`, `unsupported`, or `placeholder_fallback` support classification.
  - Verification: `pytest tests/test_label_api_configuration_truth.py tests/test_label_api_state_noops.py tests/test_label_api_public_workflow.py -q`
  - Done evidence: Configuration tests pass and every unsupported setter fails honestly. 2026-05-18 remediation wires typography and declutter setters to public/native state, with unsupported declutter algorithms returning typed `placeholder_fallback`.
  - Dependencies: T004, T005, T006

## Phase 3: Line, Curved, and Terrain Label Truth

- [X] T008 [P] [PRD: P0-R2-AC1, P0-R2-AC2, P0-R2-AC4, P0-R2-AC5] Add line and curved label path tests.
  - ID: T008
  - Title: Add line and curved label tests
  - PRD tags: [P0-R2-AC1] [P0-R2-AC2] [P0-R2-AC4] [P0-R2-AC5]
  - Files: `tests/test_label_api_line_curved_paths.py`, `src/labels/line_label.rs`, `src/labels/curved.rs`, `src/labels/mod.rs`, `python/forge3d/viewer.py`
  - Action: Add horizontal, vertical, diagonal, and curved path fixtures that assert supported line labels emit glyph instances, glyph rotations follow tangents, and unsupported curved labels return `experimental_feature` or unsupported diagnostics.
  - Verification: `pytest tests/test_label_api_line_curved_paths.py -q`
  - Done evidence: Tests fail when line labels only store unused metadata or curved labels return unqualified success.
  - Dependencies: T001

- [X] T009 [P] [PRD: P0-R2-AC3, P0-R2-AC6, P0-R1-AC6] Add upside-down and terrain-elevated label tests.
  - ID: T009
  - Title: Add upside-down and terrain label tests
  - PRD tags: [P0-R2-AC3] [P0-R2-AC6] [P0-R1-AC6]
  - Files: `tests/test_label_api_line_edge_cases.py`, `src/labels/line_label.rs`, `src/labels/projection.rs`, `python/forge3d/terrain_params.py`, `python/forge3d/viewer.py`
  - Action: Add tests for upside-down avoidance or explicit unsupported diagnostics, terrain-elevated line labels when sampling is available, and typed diagnostics when terrain sampling is unavailable.
  - Verification: `pytest tests/test_label_api_line_edge_cases.py -q`
  - Done evidence: Tests distinguish implemented handling from unsupported/experimental terrain and orientation paths.
  - Dependencies: T001

- [X] T010 [PRD: P0-R2-AC1, P0-R2-AC2, P0-R2-AC3, P0-R2-AC4, P0-R2-AC5, P0-R2-AC6] Implement line, curved, and terrain label behavior or diagnostics.
  - ID: T010
  - Title: Implement label path behavior or diagnostics
  - PRD tags: [P0-R2-AC1] [P0-R2-AC2] [P0-R2-AC3] [P0-R2-AC4] [P0-R2-AC5] [P0-R2-AC6]
  - Files: `python/forge3d/viewer.py`, `python/forge3d/viewer.pyi`, `python/forge3d/viewer_ipc.py`, `src/viewer/cmd/labels_command.rs`, `src/labels/line_label.rs`, `src/labels/curved.rs`, `src/labels/projection.rs`, expected `python/forge3d/diagnostics.py`
  - Action: Emit renderable glyph instances and tangent rotations for supported line paths, implement or diagnose upside-down handling, and return typed experimental/unsupported diagnostics for unsupported curved or terrain-elevated paths.
  - Verification: `pytest tests/test_label_api_line_curved_paths.py tests/test_label_api_line_edge_cases.py -q`
  - Done evidence: Path tests pass without silent non-renderable line/curved label success.
  - Dependencies: T008, T009

- [X] T011 [P] [PRD: P0-R1-AC6, P0-R2-AC4, P0-R2-AC6] Add label diagnostics regression tests.
  - ID: T011
  - Title: Add label diagnostics regression tests
  - PRD tags: [P0-R1-AC6] [P0-R2-AC4] [P0-R2-AC6]
  - Files: `tests/test_label_api_diagnostics.py`, expected `python/forge3d/diagnostics.py`, `python/forge3d/viewer.py`
  - Action: Add negative tests for empty text, invalid path geometry, missing glyphs, experimental curved labels, unavailable terrain sampling, and placeholder/fallback label paths.
  - Verification: `pytest tests/test_label_api_diagnostics.py -q`
  - Done evidence: Tests prove unsupported, experimental, placeholder/fallback, and no-op label paths produce structured diagnostics.
  - Dependencies: T003, T005, T008

- [X] T012 [PRD: P0-R1-AC6, P0-R2-AC4, P0-R2-AC6] Wire structured label diagnostics.
  - ID: T012
  - Title: Wire label diagnostics
  - PRD tags: [P0-R1-AC6] [P0-R2-AC4] [P0-R2-AC6]
  - Files: `python/forge3d/viewer.py`, `python/forge3d/viewer_ipc.py`, expected `python/forge3d/diagnostics.py`, `src/viewer/cmd/labels_command.rs`
  - Action: Return structured diagnostics for invalid, unsupported, experimental, terrain-unavailable, and missing-glyph label paths with affected label/path identifiers where possible.
  - Verification: `pytest tests/test_label_api_diagnostics.py -q`
  - Done evidence: Diagnostic regression tests pass with no printed-string-only failures.
  - Dependencies: T010, T011

## Phase 4: Documentation, Examples, and Continuity

- [X] T013 [P] [PRD: P0-R1-AC5, P0-R2-AC3, P0-R2-AC4] Add label docs/support audit tests.
  - ID: T013
  - Title: Add label support docs audit tests
  - PRD tags: [P0-R1-AC5] [P0-R2-AC3] [P0-R2-AC4]
  - Files: `tests/test_label_api_docs_support.py`, `docs/guides/label_support_matrix.md`, `docs/api/api_reference.rst`, `docs/guides/feature_map.md`
  - Action: Add docs audit tests requiring support levels for point labels, line labels, curved labels, callouts, typography, decluttering, atlas loading, upside-down handling, terrain labels, and no production-ready claim for experimental paths.
  - Verification: `pytest tests/test_label_api_docs_support.py -q`
  - Done evidence: Docs audit fails until support levels are explicit and overclaims are removed.
  - Dependencies: T001

- [X] T014 [PRD: P0-R1-AC5, P0-R1-AC4, P0-R2-AC3, P0-R2-AC4] Update label API docs and support matrix.
  - ID: T014
  - Title: Update label API docs
  - PRD tags: [P0-R1-AC5] [P0-R1-AC4] [P0-R2-AC3] [P0-R2-AC4]
  - Files: `docs/guides/label_support_matrix.md`, `docs/api/api_reference.rst`, `docs/guides/feature_map.md`, `docs/index.rst`
  - Action: Document high-level label workflow, stable IDs, failure behavior, point/line/curved/callout support levels, typography/declutter/atlas support, and experimental or unsupported classifications.
  - Verification: `pytest tests/test_label_api_docs_support.py -q`
  - Done evidence: Docs audit passes and public docs do not imply production-ready curved/line behavior unless tests prove it.
  - Dependencies: T013

- [X] T015 [P] [PRD: P0-R1-AC4, P0-R1-AC1, P0-R1-AC5] Add quickstart/example smoke tests.
  - ID: T015
  - Title: Add label quickstart smoke tests
  - PRD tags: [P0-R1-AC4] [P0-R1-AC1] [P0-R1-AC5]
  - Files: `tests/test_label_api_quickstart.py`, `specs/002-label-api-truth/quickstart.md`, expected `examples/label_api_truth_basic.py`
  - Action: Add a smoke test for the quickstart workflow and an example that uses high-level methods only, including a check that the example does not import or call `viewer_ipc`.
  - Verification: `pytest tests/test_label_api_quickstart.py -q`
  - Done evidence: Quickstart smoke test passes and `rg -n "viewer_ipc" examples/label_api_truth_basic.py` finds no matches.
  - Dependencies: T004, T007, T014

- [X] T016 [PRD: P0-R1-AC1, P0-R1-AC2, P0-R1-AC3, P0-R1-AC4, P0-R1-AC5, P0-R1-AC6, P0-R2-AC1, P0-R2-AC2, P0-R2-AC3, P0-R2-AC4, P0-R2-AC5, P0-R2-AC6] Run full label API verification.
  - ID: T016
  - Title: Run full label API verification
  - PRD tags: [P0-R1-AC1] [P0-R1-AC2] [P0-R1-AC3] [P0-R1-AC4] [P0-R1-AC5] [P0-R1-AC6] [P0-R2-AC1] [P0-R2-AC2] [P0-R2-AC3] [P0-R2-AC4] [P0-R2-AC5] [P0-R2-AC6]
  - Files: `tests/test_label_api_public_workflow.py`, `tests/test_label_api_stable_ids.py`, `tests/test_label_api_configuration_truth.py`, `tests/test_label_api_state_noops.py`, `tests/test_label_api_line_curved_paths.py`, `tests/test_label_api_line_edge_cases.py`, `tests/test_label_api_diagnostics.py`, `tests/test_label_api_docs_support.py`, `tests/test_label_api_quickstart.py`
  - Action: Run the full label API feature test set and record the result summary.
  - Verification: `pytest tests/test_label_api_public_workflow.py tests/test_label_api_stable_ids.py tests/test_label_api_configuration_truth.py tests/test_label_api_state_noops.py tests/test_label_api_line_curved_paths.py tests/test_label_api_line_edge_cases.py tests/test_label_api_diagnostics.py tests/test_label_api_docs_support.py tests/test_label_api_quickstart.py -q`
  - Done evidence: All feature tests pass or blockers are recorded with exact failing commands. 2026-05-18 full label command reported `28 passed` after P0-R1-AC2/P0-R1-AC3 remediation.
  - Dependencies: T015

- [X] T017 [PRD: P0-R1-AC1, P0-R1-AC2, P0-R1-AC3, P0-R1-AC4, P0-R1-AC5, P0-R1-AC6, P0-R2-AC1, P0-R2-AC2, P0-R2-AC3, P0-R2-AC4, P0-R2-AC5, P0-R2-AC6] Update requirements verification matrix.
  - ID: T017
  - Title: Update requirements verification matrix
  - PRD tags: [P0-R1-AC1] [P0-R1-AC2] [P0-R1-AC3] [P0-R1-AC4] [P0-R1-AC5] [P0-R1-AC6] [P0-R2-AC1] [P0-R2-AC2] [P0-R2-AC3] [P0-R2-AC4] [P0-R2-AC5] [P0-R2-AC6]
  - Files: `docs/superpowers/state/requirements-verification-matrix.md`
  - Action: Update P0-R1 and P0-R2 rows with exact API, test, docs, diagnostics, and command evidence; do not mark `Verified` without successful verification output.
  - Verification: `rg -n "P0-R1-AC[1-6]|P0-R2-AC[1-6]|002-label-api-truth|tests/test_label_api" docs/superpowers/state/requirements-verification-matrix.md`
  - Done evidence: Matrix rows reference concrete artifacts and statuses match evidence. 2026-05-18 matrix marks P0-R1-AC2 and P0-R1-AC3 `Verified` with public/native typography and declutter evidence.
  - Dependencies: T016

- [X] T018 [PRD: P0-R1-AC4, P0-R1-AC6, P0-R2-AC4] Update implementation ledger and current context pack.
  - ID: T018
  - Title: Update continuity state
  - PRD tags: [P0-R1-AC4] [P0-R1-AC6] [P0-R2-AC4]
  - Files: `docs/superpowers/state/implementation-ledger.md`, `docs/superpowers/state/current-context-pack.md`
  - Action: Record completed tasks, verification commands, stable ID behavior, no-op/unsupported/experimental decisions, docs evidence, and any remaining label blockers.
  - Verification: `rg -n "002-label-api-truth|P0-R1|P0-R2|experimental_feature|placeholder_fallback|viewer_ipc" docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md`
  - Done evidence: Ledger and context pack contain current outcomes and no unverified success claims. 2026-05-18 continuity state records R-028/R-037 mitigated and keeps curved/terrain label deferrals diagnostic-bearing.
  - Dependencies: T017

## Dependencies and Execution Order

- T001 blocks all source implementation.
- T002, T003, T005, T006, T008, T009, T011, T013, and T015 can be authored in parallel after T001 because they target separate tests/docs.
- Implementation follows tests: T004 after T002/T003, T007 after T005/T006, T010 after T008/T009, T012 after T011, and T014 after T013.
- T017 and T018 must occur only after verification evidence exists.

## PRD Coverage Check

- P0-R1-AC1: T001, T003, T004, T015, T016, T017
- P0-R1-AC2: T005, T007, T016, T017
- P0-R1-AC3: T005, T007, T016, T017
- P0-R1-AC4: T001, T002, T004, T007, T014, T015, T016, T017, T018
- P0-R1-AC5: T013, T014, T015, T016, T017
- P0-R1-AC6: T001, T002, T003, T005, T006, T007, T009, T011, T012, T016, T017, T018
- P0-R2-AC1: T001, T008, T010, T016, T017
- P0-R2-AC2: T008, T010, T016, T017
- P0-R2-AC3: T009, T010, T013, T014, T016, T017
- P0-R2-AC4: T008, T010, T011, T012, T013, T014, T016, T017, T018
- P0-R2-AC5: T008, T010, T016, T017
- P0-R2-AC6: T009, T010, T011, T012, T016, T017

## Detailed Pre-Implementation Coverage Audit Table

| PRD requirement | Acceptance criterion | Task IDs covering it | Test task IDs | Docs task IDs | Verification command | Risk if omitted |
|---|---|---|---|---|---|---|
| P0-R1 Label API Truth | P0-R1-AC1 stable IDs for create paths | T001, T003, T004, T015, T016, T017 | T003, T015 | T014, T017 | `pytest tests/test_label_api_stable_ids.py tests/test_label_api_quickstart.py -q` | Created labels, callouts, line labels, or overlays cannot be inspected, updated, or removed reliably. |
| P0-R1 Label API Truth | P0-R1-AC2 typography mutates real state | T005, T007, T016, T017 | T005 | T014, T017 | `pytest tests/test_label_api_configuration_truth.py -q` | `set_label_typography()` can acknowledge success while changing nothing. |
| P0-R1 Label API Truth | P0-R1-AC3 declutter mutates behavior or typed unsupported | T005, T007, T016, T017 | T005 | T014, T017 | `pytest tests/test_label_api_configuration_truth.py -q` | `set_declutter_algorithm()` can become a no-op public control. |
| P0-R1 Label API Truth | P0-R1-AC4 basic workflow without raw IPC | T001, T002, T004, T007, T014, T015, T016, T017, T018 | T002, T015 | T014, T017, T018 | `pytest tests/test_label_api_public_workflow.py tests/test_label_api_quickstart.py -q` | MVP label workflows keep requiring raw `viewer_ipc` calls. |
| P0-R1 Label API Truth | P0-R1-AC5 support-level docs for label APIs | T013, T014, T015, T016, T017 | T013, T015 | T013, T014, T017 | `pytest tests/test_label_api_docs_support.py tests/test_label_api_quickstart.py -q` | Public docs can overclaim line, curved, typography, declutter, callout, or atlas support. |
| P0-R1 Label API Truth | P0-R1-AC6 no command returns no-op success | T001, T002, T003, T005, T006, T007, T009, T011, T012, T016, T017, T018 | T002, T003, T005, T006, T009, T011 | T014, T017, T018 | `pytest tests/test_label_api_public_workflow.py tests/test_label_api_stable_ids.py tests/test_label_api_configuration_truth.py tests/test_label_api_state_noops.py tests/test_label_api_diagnostics.py -q` | Label commands can pass tests while storing unused metadata or doing nothing. |
| P0-R2 Line and Curved Label Truth | P0-R2-AC1 line labels emit glyph instances | T001, T008, T010, T016, T017 | T008 | T014, T017 | `pytest tests/test_label_api_line_curved_paths.py -q` | `add_line_label()` can accept input while rendering no visible glyphs. |
| P0-R2 Line and Curved Label Truth | P0-R2-AC2 glyph rotation follows tangent | T008, T010, T016, T017 | T008 | T014, T017 | `pytest tests/test_label_api_line_curved_paths.py -q` | Supported line labels can render with incorrect path orientation. |
| P0-R2 Line and Curved Label Truth | P0-R2-AC3 upside-down handling or unsupported docs | T009, T010, T013, T014, T016, T017 | T009, T013 | T013, T014, T017 | `pytest tests/test_label_api_line_edge_cases.py tests/test_label_api_docs_support.py -q` | Orientation limitations remain hidden from users. |
| P0-R2 Line and Curved Label Truth | P0-R2-AC4 curved labels render or typed experimental/unsupported | T008, T010, T011, T012, T013, T014, T016, T017, T018 | T008, T011, T013 | T013, T014, T017, T018 | `pytest tests/test_label_api_line_curved_paths.py tests/test_label_api_diagnostics.py tests/test_label_api_docs_support.py -q` | Curved-label APIs can report success while only recording non-renderable metadata. |
| P0-R2 Line and Curved Label Truth | P0-R2-AC5 horizontal, vertical, diagonal, curved path tests | T008, T010, T016, T017 | T008 | T014, T017 | `pytest tests/test_label_api_line_curved_paths.py -q` | Path coverage misses orientation regressions. |
| P0-R2 Line and Curved Label Truth | P0-R2-AC6 terrain-elevated line labels or diagnostics | T009, T010, T011, T012, T016, T017 | T009, T011 | T014, T017 | `pytest tests/test_label_api_line_edge_cases.py tests/test_label_api_diagnostics.py -q` | Terrain label support can be overclaimed or silently unavailable. |

## Audit Rerun Result

- PASS: Every P0-R1 and P0-R2 acceptance criterion has tasks, tests, docs/support wording where public API behavior is exposed, and verification commands.
- PASS: Implementation tasks for public label APIs depend on no-op, diagnostics, and support-documentation tasks.
- PASS: Unsupported, experimental, terrain-unavailable, and placeholder label paths require typed diagnostics rather than printed text.

## Extension Hooks

**Optional Hook**: git  
Command: `/speckit.git.commit`  
Description: Auto-commit after task generation  

Prompt: Commit task changes?  
To execute: `/speckit.git.commit`

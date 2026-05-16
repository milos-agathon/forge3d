# Current Context Pack

Last updated: 2026-05-16  
Source PRD: `docs/superpowers/plans/prd.md`  
Constitution: `.specify/memory/constitution.md`  
Branch: `002-label-api-truth`  
Last commit: `e26d1b9`

## Project Goal

Build a truthful, typed, deterministic offline 3D map-production layer for
forge3d. MVP product shape: `MapScene + LabelPlan + ValidationReport + Bundle`.
Do not claim MVP-wide completion while P0-R4 `MapScene` remains incomplete.

## Current SpecKit Feature

Active feature: `specs/004-mapscene-mvp`.  
`.specify/feature.json` now points to `specs/004-mapscene-mvp` locally for
the next P0 feature after feature `003` completion.

## Task State

Feature `001-diagnostics-support-matrices`: T001-T035 done; matrix marks
P0-R5/P0-R6 `Verified`.

Feature `002-label-api-truth`: T001-T018 are now marked `[X]` in
`specs/002-label-api-truth/tasks.md`. A 2026-05-16 closure audit found no
remaining feature `002` implementation task IDs and refreshed the full label
verification command against the current multi-feature repo state.

Feature `003-deterministic-label-plan`: T001-T024 are now marked `[X]` in
`specs/003-deterministic-label-plan/tasks.md`. No incomplete task remains in
feature `003`.

No incomplete task remains in feature `002` or feature `003`.
Feature `004-mapscene-mvp` for P0-R4 is active. T001-T003 are complete; T004
is the next incomplete task.

## Matrix Status

Feature `001`:
- P0-R5-AC1 through P0-R5-AC6 are `Verified`.
- P0-R6-AC1 through P0-R6-AC6 are `Verified`.

Feature `002`:
- `Verified`: P0-R1-AC1, P0-R1-AC4, P0-R1-AC5, P0-R1-AC6, P0-R2-AC1, P0-R2-AC2, P0-R2-AC3, P0-R2-AC5.
- `Deferred with diagnostic`: P0-R1-AC2, P0-R1-AC3, P0-R2-AC4, P0-R2-AC6.

Deferred items are intentional honesty outcomes for native typography mutation,
alternate declutter behavior, production curved-label rendering, and
terrain-elevated line labels. The public API returns typed diagnostics instead
of success for those paths.

P0-R4 and remaining P1/P2 rows remain planned in later feature directories.

Feature `004`:
- T001 ownership inspection is complete. Selected source path is
  `python/forge3d/map_scene.py`, exported through `python/forge3d/__init__.py`
  and `python/forge3d/__init__.pyi`.
- T002-T003 added `tests/test_mapscene_recipe_contract.py` and implemented the
  minimal typed recipe model surface in `python/forge3d/map_scene.py`.
- Reuse prerequisites: `python/forge3d/diagnostics.py`,
  `python/forge3d/label_plan.py`, `python/forge3d/helpers/offscreen.py`,
  `python/forge3d/bundle.py`, style/building/3D Tiles/VT validators, pointcloud
  metadata, and memory helpers.
- No P0-R4 row is verified yet; this task only records ownership and
  dependency status.

Feature `003`:
- P0-R3-AC1 through P0-R3-AC8 are `Verified` in
  `docs/superpowers/state/requirements-verification-matrix.md`.
- T001 is ownership evidence; T002-T022 provide tests, implementation, docs,
  quickstart, payload, diagnostic, and full verification evidence; T023 records
  the matrix status; T024 records continuity.
- Selected implementation path is Python-first:
  `python/forge3d/label_plan.py`, exported through
  `python/forge3d/__init__.py` and `python/forge3d/__init__.pyi`.
- Feature `001` diagnostics are available via `python/forge3d/diagnostics.py`.
- Feature `002` label inputs are available through high-level `ViewerHandle`
  methods and `LabelOperationResult` / `LabelBatchResult`.

## Tests Run And Results

Feature `002` closure audit / verification drift refresh:
- Selected task IDs: no new implementation task IDs; T001-T018 are already
  marked `[X]`.
- Initial current-repo full label command:
  `pytest tests/test_label_api_public_workflow.py tests/test_label_api_stable_ids.py tests/test_label_api_configuration_truth.py tests/test_label_api_state_noops.py tests/test_label_api_line_curved_paths.py tests/test_label_api_line_edge_cases.py tests/test_label_api_diagnostics.py tests/test_label_api_docs_support.py tests/test_label_api_quickstart.py -q`
  -> `2 failed, 26 passed`.
- Root cause: `tests/test_label_api_docs_support.py` still expected
  deterministic `LabelPlan` to be `missing` and still required
  `.specify/feature.json` to point at historical feature `002`; both
  assumptions are stale after feature `003` made `LabelPlan` supported and
  feature `004` became active.
- Narrow maintenance fix: update the docs-support regression to expect
  deterministic `LabelPlan` as `supported` and to verify feature `002`
  artifacts independently of the active SpecKit pointer.
- Fresh full label command:
  `pytest tests/test_label_api_public_workflow.py tests/test_label_api_stable_ids.py tests/test_label_api_configuration_truth.py tests/test_label_api_state_noops.py tests/test_label_api_line_curved_paths.py tests/test_label_api_line_edge_cases.py tests/test_label_api_diagnostics.py tests/test_label_api_docs_support.py tests/test_label_api_quickstart.py -q`
  -> `28 passed`.
- Matrix audit:
  `rg -n "P0-R1-AC[1-6]|P0-R2-AC[1-6]|002-label-api-truth|tests/test_label_api" docs/superpowers/state/requirements-verification-matrix.md`
  -> exit 0.
- Continuity audit:
  `rg -n "002-label-api-truth|P0-R1|P0-R2|experimental_feature|placeholder_fallback|viewer_ipc" docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md`
  -> exit 0.
- Requirement status impact: no change. The verified/deferred split for
  feature `002` remains as recorded above.

Feature `004` T001 ownership inspection:
- Checklist `specs/004-mapscene-mvp/checklists/requirements.md`: complete
  (`0` incomplete).
- Prerequisite command:
  `.specify/scripts/powershell/check-prerequisites.ps1 -Json -RequireTasks -IncludeTasks`
  resolved `FEATURE_DIR` to `C:\Users\milos\forge3d\specs\004-mapscene-mvp`.
- Verification command:
  `rg -n "class (MapScene|SceneRecipe|ValidationReport|LabelPlan)|def render_png|render_rgba|save_bundle|viewer_ipc" python src tests docs/superpowers/state`
- Observed result: no implemented public `MapScene` or `SceneRecipe` product
  API was found. Existing prerequisites are present in
  `python/forge3d/diagnostics.py`, `python/forge3d/label_plan.py`,
  `python/forge3d/helpers/offscreen.py`, and `python/forge3d/bundle.py`.
- Additional focused inspection confirmed `Scene.render_rgba()` and
  `Scene.render_png()` are instance methods, offscreen rendering already calls
  `scene.render_rgba()`, and bundle state can preserve `ValidationReport`.
- No product tests were run for T001 because it is an ownership inspection
  task.

Feature `004` T002-T003 typed recipe models:
- Red `pytest tests/test_mapscene_recipe_contract.py -q` -> `2 failed` because
  `forge3d.TerrainSource` and the typed MapScene recipe API were missing.
- Green `pytest tests/test_mapscene_recipe_contract.py -q` -> `2 passed`.
- `python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_recipe_contract.py`
  -> exit 0.
- P0-R4 remains `Planned` in the matrix because validation/reporting, render,
  bundle, examples, docs, and full verification are not complete.

Feature `002` Phase 4/T016 verification:
- Red `pytest tests/test_label_api_docs_support.py -q` failed on missing docs/support rows, missing `placeholder_fallback` docs, raw IPC-first wording, stale contract signatures, and stale `.specify/feature.json`.
- Green `pytest tests/test_label_api_docs_support.py -q` -> `5 passed`.
- Red `pytest tests/test_label_api_quickstart.py -q` failed because `examples/label_api_truth_basic.py` did not exist and quickstart did not point to it.
- Green `pytest tests/test_label_api_quickstart.py -q` -> `4 passed`.
- `rg -n "viewer_ipc|send_ipc" examples/label_api_truth_basic.py` -> no matches.
- Full label command: `pytest tests/test_label_api_public_workflow.py tests/test_label_api_stable_ids.py tests/test_label_api_configuration_truth.py tests/test_label_api_state_noops.py tests/test_label_api_line_curved_paths.py tests/test_label_api_line_edge_cases.py tests/test_label_api_diagnostics.py tests/test_label_api_docs_support.py tests/test_label_api_quickstart.py -q` -> `28 passed`.

Earlier feature `002` evidence remains valid:
- Phase 1-3 Python subset -> `19 passed`.
- `cargo test test_glyph_rotation -q` -> `2 passed`.
- `cargo fmt -- --check` -> exit 0.
- `cargo check -q` -> exit 0.

Feature `003` T001 ownership inspection:
- Checklist `specs/003-deterministic-label-plan/checklists/requirements.md`:
  `16/16` complete, `0` incomplete.
- Verification command:
  `rg -n "LabelPlan|AcceptedLabel|RejectedLabel|Keepout|priority|declutter|collision|atlas|typography" python src tests docs/superpowers/state`
- Observed result: no implemented public `LabelPlan`, `AcceptedLabel`,
  `RejectedLabel`, or `KeepoutRegion` product API exists outside specs/state.
  Existing substrate is in `python/forge3d/diagnostics.py`,
  `python/forge3d/viewer.py`, `python/forge3d/viewer.pyi`, and
  `src/labels/*`.

Feature `003` T002-T004 base LabelPlan:
- Red `pytest tests/test_label_plan_contract.py -q` failed because
  `AcceptedLabel`/`LabelPlan` were not importable from `forge3d`.
- Red `pytest tests/test_label_plan_determinism.py -q` failed because
  `LabelPlan` was not importable from `forge3d`.
- A later combined green attempt initially failed on public API list semantics
  because normalized plan collections were tuples instead of lists.
- Green `pytest tests/test_label_plan_contract.py -q` -> `2 passed`.
- Green `pytest tests/test_label_plan_determinism.py -q` -> `3 passed`.
- Green `pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py -q` -> `5 passed`.

Feature `003` T005-T006 rejection reasons:
- Red `pytest tests/test_label_plan_rejection_reasons.py -q` -> `2 failed`
  because collision, priority loss, keepout, and terrain-occluded rejections
  were not retained yet.
- Green `pytest tests/test_label_plan_rejection_reasons.py -q` -> `2 passed`.
- Green `pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py -q` -> `7 passed`.
- `python -m py_compile python\forge3d\label_plan.py tests\test_label_plan_contract.py tests\test_label_plan_determinism.py tests\test_label_plan_rejection_reasons.py` -> exit 0.

Feature `003` T007-T008 point candidates:
- Red `pytest tests/test_label_plan_point_candidates.py -q` -> `2 failed`
  because accepted labels did not retain generated candidate sets.
- Green `pytest tests/test_label_plan_point_candidates.py -q` -> `2 passed`.
- Green `pytest tests/test_label_plan_point_candidates.py tests/test_label_plan_determinism.py -q` -> `5 passed`.
- Green `pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py -q` -> `9 passed`.
- `python -m py_compile python\forge3d\label_plan.py tests\test_label_plan_point_candidates.py` -> exit 0.

Feature `003` T009-T010 polygon candidates:
- Red `pytest tests/test_label_plan_polygon_candidates.py -q` -> `3 failed`
  because polygon labels were still treated as unsupported geometry.
- Green `pytest tests/test_label_plan_polygon_candidates.py -q` -> `3 passed`.
- Green `pytest tests/test_label_plan_polygon_candidates.py tests/test_label_plan_rejection_reasons.py -q` -> `5 passed`.
- Green `pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py tests/test_label_plan_polygon_candidates.py -q` -> `12 passed`.
- `python -m py_compile python\forge3d\label_plan.py tests\test_label_plan_polygon_candidates.py` -> exit 0.

Feature `003` T011-T012 terrain sampling:
- Red `pytest tests/test_label_plan_terrain.py -q` -> `3 failed` because
  sampler elevation, sampler visibility, and unavailable required terrain were
  not wired.
- Green `pytest tests/test_label_plan_terrain.py -q` -> `3 passed`.
- Green `pytest tests/test_label_plan_terrain.py tests/test_label_plan_rejection_reasons.py -q` -> `5 passed`.
- Green `pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py tests/test_label_plan_polygon_candidates.py tests/test_label_plan_terrain.py -q` -> `15 passed`.
- `python -m py_compile python\forge3d\label_plan.py tests\test_label_plan_terrain.py` -> exit 0.

Feature `003` T013-T014 keepout regions:
- `pytest tests/test_label_plan_keepouts.py -q` -> `6 passed`; no
  production-code change was needed because T006's generic keepout
  intersection already covered the required keepout kinds.
- Green `pytest tests/test_label_plan_keepouts.py tests/test_label_plan_rejection_reasons.py -q` -> `8 passed`.
- Green `pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py tests/test_label_plan_polygon_candidates.py tests/test_label_plan_terrain.py tests/test_label_plan_keepouts.py -q` -> `21 passed`.
- `python -m py_compile tests\test_label_plan_keepouts.py` -> exit 0.

Feature `003` T015-T016 priority solve:
- Red `pytest tests/test_label_plan_priority.py -q` -> `1 failed, 1 passed`
  because priority class rank was not used in collision solving yet.
- Green `pytest tests/test_label_plan_priority.py -q` -> `2 passed`.
- Green `pytest tests/test_label_plan_priority.py tests/test_label_plan_determinism.py -q` -> `5 passed`.
- Green `pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py tests/test_label_plan_polygon_candidates.py tests/test_label_plan_terrain.py tests/test_label_plan_keepouts.py tests/test_label_plan_priority.py -q` -> `23 passed`.
- `python -m py_compile python\forge3d\label_plan.py tests\test_label_plan_priority.py` -> exit 0.

Feature `003` T017-T018 payload methods:
- Red `pytest tests/test_label_plan_payloads.py -q` -> `1 failed, 1 passed`
  because unsupported backend handling raised `TypeError`.
- Green `pytest tests/test_label_plan_payloads.py -q` -> `2 passed`.
- Green `pytest tests/test_label_plan_payloads.py tests/test_label_plan_contract.py -q` -> `4 passed`.
- Green `pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py tests/test_label_plan_polygon_candidates.py tests/test_label_plan_terrain.py tests/test_label_plan_keepouts.py tests/test_label_plan_priority.py tests/test_label_plan_payloads.py -q` -> `25 passed`.
- `python -m py_compile python\forge3d\label_plan.py tests\test_label_plan_payloads.py` -> exit 0.

Feature `003` T019-T020 docs:
- Red `pytest tests/test_label_plan_docs.py -q` -> `2 failed` because the
  LabelPlan guide, support-matrix update, API reference, and docs index entry
  were missing.
- Green `pytest tests/test_label_plan_docs.py -q` -> `2 passed`.
- Green `pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py tests/test_label_plan_polygon_candidates.py tests/test_label_plan_terrain.py tests/test_label_plan_keepouts.py tests/test_label_plan_priority.py tests/test_label_plan_payloads.py tests/test_label_plan_docs.py -q` -> `27 passed`.
- `python -m py_compile tests\test_label_plan_docs.py` -> exit 0.

Feature `003` T021-T022 quickstart/full verification:
- Red `pytest tests/test_label_plan_quickstart.py -q` -> `2 failed` because
  the quickstart lacked concrete public API names and the first executable
  fixture had a glyph gap masking keepout rejection.
- Green `pytest tests/test_label_plan_quickstart.py -q` -> `2 passed`.
- `python -m py_compile tests\test_label_plan_quickstart.py` -> exit 0.
- Full T022 command `pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py tests/test_label_plan_polygon_candidates.py tests/test_label_plan_terrain.py tests/test_label_plan_keepouts.py tests/test_label_plan_priority.py tests/test_label_plan_payloads.py tests/test_label_plan_docs.py tests/test_label_plan_quickstart.py -q` -> `29 passed`.

Feature `003` T023 matrix:
- `rg -n "P0-R3-AC[1-8]|003-deterministic-label-plan|tests/test_label_plan" docs\superpowers\state\requirements-verification-matrix.md` shows P0-R3-AC1 through P0-R3-AC8 as `Verified` with feature `003` source/test/docs/diagnostic/payload evidence.

Feature `003` T024 continuity:
- `specs/003-deterministic-label-plan/tasks.md` has T001-T024 marked `[X]`.
- The next feature is `specs/004-mapscene-mvp` for P0-R4.

## Analyze Findings

The prior feature `002` red findings R1-R5 have explicit fixes:
- R1: During original feature `002` closure, `.specify/feature.json` targeted
  `002`. It now correctly targets active feature `004`; the 2026-05-16 closure
  audit changed the 002 docs regression to verify feature `002` artifacts
  independently of the mutable active pointer.
- R2: docs and feature map prefer high-level `ViewerHandle` label methods and keep raw IPC as advanced compatibility only.
- R3: docs/support tests, support matrix, API reference wording, quickstart, and example now exist.
- R4: `contracts/label-api-contract.md` now matches current `ViewerHandle` signatures and return types.
- R5: `tests/test_label_api_quickstart.py` repeats fixed inputs and verifies stable IDs, diagnostics, and glyph ordering.

Final read-only SpecKit-style consistency checks after T013-T018:
- Historical T013-T018 prerequisites resolved `FEATURE_DIR` to
  `specs/002-label-api-truth`. Current active prerequisites resolve to
  `specs/004-mapscene-mvp`, and feature `002` tests no longer require the
  active pointer to move backward.
- No unresolved placeholders/TBD markers were found in feature `002` spec, plan, tasks, or label API contract.
- No open `[ ]` task remains in feature `002`.
- P0-R1/P0-R2 tags are present across spec, plan, tasks, and matrix.
- No raw-label-IPC or production-ready curved/typography/declutter overclaim pattern was found in the updated label docs/contract/quickstart.

## Modified Files / Worktree Risk

R-011 remains open: the worktree still contains mixed feature `001`, feature
`002`, generated/log/PDB, deleted example, and unrelated/user-owned changes.
Do not revert unrelated changes.

New/updated feature `002` paths from this session include:
- `.specify/feature.json`
- `docs/guides/label_support_matrix.md`
- `docs/api/api_reference.rst`
- `docs/guides/feature_map.md`
- `specs/002-label-api-truth/contracts/label-api-contract.md`
- `specs/002-label-api-truth/quickstart.md`
- `specs/002-label-api-truth/tasks.md`
- `examples/label_api_truth_basic.py`
- `tests/test_label_api_docs_support.py`
- `tests/test_label_api_quickstart.py`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/current-context-pack.md`

Feature `002` closure-audit refresh additionally updated:
- `tests/test_label_api_docs_support.py`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/current-context-pack.md`

New/updated feature `003` paths from this session include:
- `.specify/feature.json`
- `python/forge3d/label_plan.py`
- `python/forge3d/__init__.py`
- `python/forge3d/__init__.pyi`
- `specs/003-deterministic-label-plan/tasks.md`
- `tests/test_label_plan_contract.py`
- `tests/test_label_plan_determinism.py`
- `tests/test_label_plan_rejection_reasons.py`
- `tests/test_label_plan_point_candidates.py`
- `tests/test_label_plan_polygon_candidates.py`
- `tests/test_label_plan_terrain.py`
- `tests/test_label_plan_keepouts.py`
- `tests/test_label_plan_priority.py`
- `tests/test_label_plan_payloads.py`
- `tests/test_label_plan_docs.py`
- `tests/test_label_plan_quickstart.py`
- `docs/guides/label_plan_guide.md`
- `docs/guides/label_support_matrix.md`
- `docs/api/api_reference.rst`
- `docs/index.rst`
- `specs/003-deterministic-label-plan/quickstart.md`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/current-context-pack.md`

## Open Blockers

- R-011: dirty worktree/change separation remains open before commit or PR.
- R-020: real `MapScene` quickstart remains downstream work for feature `004`.

## Next Exact Prompt

```text
Continue feature `004-mapscene-mvp` from T004 MapScene validation report tests. Do not reopen feature `002-label-api-truth` unless new SpecKit tasks are created for deferred diagnostic outcomes. Do not claim MVP-wide completion while P0-R4 remains incomplete, and keep R-011 dirty worktree/change separation open before commit or PR.
```

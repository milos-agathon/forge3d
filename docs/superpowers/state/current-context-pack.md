# Current Context Pack

Last updated: 2026-05-18
Source PRD: `docs/superpowers/plans/prd.md`  
Constitution: `.specify/memory/constitution.md`  
Branch: `005-map-assets-bundles-p1`
Last commit: `a864985`

## Project Goal

Build a truthful, typed, deterministic offline 3D map-production layer for
forge3d. MVP product shape: `MapScene + LabelPlan + ValidationReport + Bundle`.
Feature `004` now has native/offscreen PNG output for fixture-backed MVP
terrain/raster/vector/label recipes, source-derived compatibility output for
symbolic fixture recipes, and diagnostic-bearing unsupported layer paths.

## Current SpecKit Feature

Active implementation target for this checkpoint:
`specs/006-material-vt-large-scene-p2` Milestone 5, while the git branch and
SpecKit pointer still remain on feature `005`.

SpecKit pointer status: `.specify/feature.json` now targets
`specs/005-map-assets-bundles-p1`, and the prerequisite script now resolves
feature `005` for implementation commands.

Feature `006-material-vt-large-scene-p2` Milestone 5 is now implemented or
explicitly diagnosed through T029. The active SpecKit pointer has not been moved
from feature `005`, so future `/speckit.implement` prerequisite scripts may
still resolve to `005` unless the pointer is changed deliberately.

Independent feature `006` PRD compliance audit on 2026-05-18 reran the focused
P2 command and observed `35 passed in 0.97s`; `py_compile` for the changed P2
modules/tests exited `0`. Per human direction on 2026-05-18, the deleted
canonical MapScene examples / P0-R4-AC8 issue is not tracked as a blocker for
this checkpoint.

Feature `004` task execution remains complete through T029, and R-029 remains
remediated with native/offscreen fixture-backed render evidence.

## Task State

Feature `001-diagnostics-support-matrices`: T001-T035 done; matrix marks
P0-R5/P0-R6 `Verified`.

Feature `002-label-api-truth`: T001-T018 are now marked `[X]` in
`specs/002-label-api-truth/tasks.md`. A 2026-05-16 closure audit found no
remaining feature `002` implementation task IDs and refreshed the full label
verification command against the current multi-feature repo state. A
2026-05-18 remediation then resolved the prior P0-R1-AC2/P0-R1-AC3 blockers by
wiring typography state/layout metrics and declutter placement-policy state
through public and native paths.

Feature `003-deterministic-label-plan`: T001-T024 are now marked `[X]` in
`specs/003-deterministic-label-plan/tasks.md`. No incomplete task remains in
feature `003`.

No incomplete task remains in feature `002`, feature `003`, or feature `004`.
Feature `004-mapscene-mvp` task execution is complete through T029, including scoped
remediation subtasks T006A, T006B, T006C, T006D, T006E, T011A, T011B, T013A,
T013B, T016A, T016B, T017A, T017B, T020R, T021R, T020S, T021S, T020T, and
T021T. T019 full MapScene MVP verification passed, T020 updated the matrix, and
T021 updated continuity state. T022-T025 then corrected the render truth gate,
blocked synthetic PNG success, updated docs/state, and moved P0-R4-AC2 back to
`Blocked`; T026-T029 then implemented and verified source-derived PNG output;
the post-audit R-029 fix added native/offscreen fixture-backed render evidence,
updated docs/examples/state, and keeps P0-R4-AC2 `Verified`.

Feature `005-map-assets-bundles-p1`: T001 through T069 are now marked `[X]` in
`specs/005-map-assets-bundles-p1/tasks.md`. The 2026-05-18 RED/YELLOW
remediation pass fixed R-030 through R-035 without adding new product scope.
R-036/R-011 are mitigated for PR hygiene by staging only intended
feature/blocker-scope changes and leaving unrelated/generated/user-owned paths
unstaged. R-037 is mitigated because the global P0 label rows are now verified.

Feature `006-material-vt-large-scene-p2`: T001-T031 are now marked `[X]`.
T004/T005 keep VT normal/mask as deterministic `vt_unsupported_family`
diagnostic deferrals with `vt.normal`/`vt.mask` affected IDs. T006/T007 update
VT docs. T008-T012 add textured building tests/docs and explicit unsupported
diagnostics for texture+UV intent, missing texture path, missing UVs,
unsupported texture format, scalar fallback, and Pro-gated textured paths.
T013-T018 add deterministic repeated line-label placement, cartographic priority
presets, leader-line candidate metadata, and diagnostic deferrals for curved
labels and complex-script shaping. T019-T024 add opt-in `large_scene.resources`
summaries plus large-scene docs. T025-T028 add determinism/no-op and quickstart
coverage. T029 full P2 verification reported `35 passed in 0.93s` on the
latest rerun. T030 updated the requirements verification matrix and diagnostic
inventory; T031 updated the implementation ledger, current context pack, and
open-blockers state.

Current P2 status: P2-R1 and P2-R4 rows are `Verified`; P2-R2 texture runtime
rows are `Deferred with diagnostic` where end-to-end textured PBR rendering is
still unsupported; P2-R3 repeated/rules rows are `Verified`, while curved text
and complex-script shaping are `Deferred with diagnostic`.

Feature `006` T002 and T003 are now marked `[X]`. T002 added
`tests/test_p2_diagnostics_contract.py`; RED verification reported `4 failed`
before implementation because P2 diagnostic factories were missing and
`MapScene.validate()` did not emit building texture, cache/LOD, or instancing
P2 diagnostics. T003 added `P2_FEATURE_DIAGNOSTIC_CODES` plus factory helpers
for `missing_texture_path`, `missing_uvs`, `unsupported_texture_format`,
`unavailable_cache_lod_stats`, and `unsupported_instancing_path` in
`python/forge3d/diagnostics.py`, exported them through `python/forge3d/__init__.py`
and `python/forge3d/__init__.pyi`, and wired metadata adapters in
`python/forge3d/map_scene.py`. GREEN verification reported P2 diagnostics
`4 passed`, focused P1 regression `15 passed`, and product/test `py_compile`
exit `0`. Later T004-T031 completed the Milestone 5 scope, so the P2
acceptance rows now use `Verified` or `Deferred with diagnostic` according to
the current matrix.

Feature `005` now has verified evidence for:
- Label ingestion: point/line/polygon candidates, CRS transform through
  `forge3d.crs.transform_coords`, terrain sampling or
  `unavailable_terrain_sampler`, PRD-listed expressions, deterministic
  `LabelPlan` accepted/rejected output, and pre-render missing field/glyph
  diagnostics.
- Typography: `FontAtlas.default_latin`, `FontAtlas.from_font` typed asset
  diagnostics/range metadata, `FontFallbackRange`, `TypographySettings`,
  `unicode_coverage_gap`, multiline/callout layout metadata, and documented
  non-blocking complex-script shaping deferral.
- Buildings: `MapSceneBuildingLayer` support-status classification,
  CityJSON/GeoJSON render-prep diagnostics where native MapScene support is not
  available, geometry count/bounds summaries, scalar PBR review metadata docs,
  textured PBR `unsupported_feature`, and zero-geometry `placeholder_fallback`.
- 3D Tiles: public `Tiles3DLayer.from_tileset_json` / `from_b3dm`, supported vs
  unsupported format diagnostics, cache stats, LOD/SSE metadata, unsupported
  B3DM/GLB feature diagnostics, and docs that state no full Cesium runtime
  parity.
- Bundles: required P1 review fields including `supported_export_settings`,
  label sources/plans, diagnostics/state, deterministic guardrails,
  load/render when assets are available, and structured missing-asset
  diagnostics after load.

Feature `005` verification commands:
- `$p1 = rg --files tests | Where-Object { $_ -match 'test_p1_.*\.py$' }; python -m pytest @p1 -q` reported `51 passed`; latest blocker-cleanup rerun on 2026-05-18 reported `51 passed in 2.17s`.
- Combined MapScene guardrail plus P1 command reported `101 passed`; latest blocker-cleanup rerun on 2026-05-18 reported `101 passed in 4.34s`.
- `python -m py_compile python\forge3d\map_scene.py python\forge3d\diagnostics.py python\forge3d\__init__.py tests\test_p1_label_layer_crs_terrain.py tests\test_p1_label_expressions_plan_diagnostics.py tests\test_p1_typography_font_support.py tests\test_p1_building_workflow_support.py tests\test_p1_tiles3d_support.py tests\test_p1_bundle_roundtrip.py` exited `0`.
- `/speckit.analyze` read-only consistency gate on 2026-05-18: prerequisite
  script resolved feature `005`; optional before/after analyze commit hooks
  were not executed; artifact scan found 38 FRs, 16 SCs, 69 task IDs, zero
  unchecked task boxes, no unresolved placeholders, no missing P1
  acceptance-criterion task coverage, and no P1 matrix rows with non-verified
  statuses. No Critical/High spec/plan/tasks issue remains.
- Blocker-cleanup analyze rerun: prerequisite script resolved feature `005`,
  unchecked tasks `0`, P1 rows `30`, P1 rows not verified `0`, feature `005`
  open blockers `0`, placeholder hits `0`; status `PASS`.

## Matrix Status

Feature `001`:
- P0-R5-AC1 through P0-R5-AC6 are `Verified`.
- P0-R6-AC1 through P0-R6-AC6 are `Verified`.

Feature `002`:
- `Verified`: P0-R1-AC1 through P0-R1-AC6, P0-R2-AC1, P0-R2-AC2, P0-R2-AC3, P0-R2-AC5.
- `Verified` through typed diagnostic/unavailable paths: P0-R2-AC4, P0-R2-AC6.

The public API returns typed diagnostics instead of success for unsupported or
experimental label paths. Curved-label and terrain-elevated criteria are
satisfied by the explicit typed diagnostic/unavailable paths in their
acceptance criteria. Typography mutation and declutter behavior are verified by
public state/layout or placement-policy tests plus native `LabelManager` state
mutation evidence from the 2026-05-18 remediation.

Feature `004`:
- `Verified`: P0-R4-AC1 through P0-R4-AC8.
- P0-R4-AC2 is verified by fixture-backed native/offscreen render evidence, not
  by source-derived compatibility output alone.

Latest full MapScene verification reported `48 passed`.

Feature `005`:
- P1-R1-AC1 through P1-R5-AC5 are `Verified` in
  `docs/superpowers/state/requirements-verification-matrix.md`.
- R-030 through R-035 are mitigated/closed by feature `005` T009-T069 evidence.
- R-036 is mitigated for workspace hygiene: intended feature/blocker-scope
  files are staged, and unrelated/generated/user-owned paths remain unstaged.
- R-037 is mitigated after the 2026-05-18 label remediation verified global
  `P0-R1-AC2` and `P0-R1-AC3`.
- P1 verification:
  `$p1 = rg --files tests | Where-Object { $_ -match 'test_p1_.*\.py$' }; python -m pytest @p1 -q`
  -> latest blocker-cleanup rerun `51 passed in 2.17s`.
- Combined MapScene guardrail plus P1 verification:
  `python -m pytest tests\test_mapscene_validation.py tests\test_mapscene_save_bundle.py tests\test_mapscene_render_png.py tests\test_mapscene_docs.py tests\test_mapscene_examples.py tests\test_mapscene_quickstart.py tests\test_mapscene_recipe_contract.py tests\test_mapscene_render_policy.py tests\test_mapscene_support_status.py tests\test_mapscene_label_plan_integration.py @p1 -q`
  -> latest blocker-cleanup rerun `101 passed in 4.34s`.
- Compile check exited `0` for `python\forge3d\map_scene.py`,
  `python\forge3d\diagnostics.py`, `python\forge3d\__init__.py`, and new P1
  tests.
- SpecKit prerequisite script now resolves `FEATURE_DIR` to
  `C:\Users\milos\forge3d\specs\005-map-assets-bundles-p1`.
- T007-T008 verification:
  `$env:PYTHONPATH='python'; python -m pytest tests\test_p1_label_layer_geometry.py -q`
  -> RED `4 failed`, then GREEN `4 passed in 0.18s` after implementation.

Historical note: Feature `006` T001-T003 completed the dependency/diagnostic
foundation first. The later Milestone 5 pass completed T004-T031, so P2-R1
through P2-R4 are no longer planned-only; the current matrix records verified
and diagnostic-deferred outcomes.

Independent feature `004` PRD compliance audit refresh and R-029 remediation on 2026-05-17:
- Fresh command:
  `python -m pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_render_policy.py tests/test_mapscene_render_png.py tests/test_mapscene_save_bundle.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_docs.py -q`
  -> `48 passed in 1.36s`.
- R-029 remediation result: `tests/test_mapscene_render_png.py::test_render_uses_native_offscreen_for_real_terrain_and_raster_assets`
  proves real `.npy` terrain, PNG raster, inline vector, and label layers render
  through `scene.last_render_backend == "native/offscreen"` while the
  source-derived compositor is monkeypatched to fail. P0-R4-AC1 through
  P0-R4-AC8 remain verified without accepting task checkboxes as proof.
- Remaining scoped boundaries: point-cloud and supported-building render
  adapters, public 3D Tiles rendering, VT normal/mask runtime, textured PBR
  buildings, full Mapbox style parity, and feature `005` bundle load/render
  round trip remain incomplete/later-owned and must stay diagnostic-bearing or
  honestly documented.

Independent feature `004` PRD compliance audit on 2026-05-18:
- Scope: audited only `specs/004-mapscene-mvp` against the PRD, constitution,
  feature spec/plan/tasks, requirements matrix, changed-file summary, current
  code, support docs, and fresh local verification.
- Fresh command:
  `python -m pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_render_policy.py tests/test_mapscene_render_png.py tests/test_mapscene_save_bundle.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_docs.py -q`
  -> `48 passed in 1.64s`.
- Compile check:
  `python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_render_png.py tests\test_mapscene_examples.py tests\test_mapscene_quickstart.py tests\test_mapscene_docs.py examples\mapscene_terrain_raster.py examples\mapscene_vector_labels.py examples\mapscene_buildings_labels.py`
  -> exit `0`.
- Result: no new feature `004` P0-R4 blocker was found. P0-R4-AC1 through
  P0-R4-AC8 remain verified by concrete evidence. Remaining boundaries are
  later-owned or diagnostic-bearing: point-cloud and supported-building render
  adapters, public 3D Tiles rendering, VT normal/mask runtime, textured PBR
  buildings, full Mapbox style parity, and feature `005` bundle load/render
  round trip.

Feature `004`:
- T001 ownership inspection is complete. Selected source path is
  `python/forge3d/map_scene.py`, exported through `python/forge3d/__init__.py`
  and `python/forge3d/__init__.pyi`.
- T002-T003 added `tests/test_mapscene_recipe_contract.py` and implemented the
  minimal typed recipe model surface in `python/forge3d/map_scene.py`.
- T004-T006 added `tests/test_mapscene_validation.py` and
  `tests/test_mapscene_support_status.py`, then implemented structured
  `MapScene.validate()` adapters in `python/forge3d/map_scene.py` for CRS,
  style, missing glyphs, memory estimates, VT, public 3D Tiles intent,
  building support status, unsupported layer/output diagnostics, and render/save
  guard validation.
- T007-T008 added `tests/test_mapscene_label_plan_integration.py`, extended
  `tests/test_mapscene_validation.py`, and wired MapScene validation to
  deterministic `LabelPlan.compile()` with map-furniture keepouts, rejection
  summaries, missing-CRS diagnostics, building geometry-count memory estimates,
  and repeated validation determinism checks.
- T006A added source-identity/renderable-data validation for terrain, raster,
  vector, labels, and point-cloud recipe inputs.
- T006B/T006C added missing external asset and unsupported asset format
  diagnostics for terrain, raster, and point-cloud paths, plus blocked bundle
  persistence coverage for missing-asset scenes.
- T006D/T006E added terrain missing-CRS validation and supported-building
  source asset diagnostics for missing local sources and unsupported suffixes.
- T009-T011 added `tests/test_mapscene_render_png.py` and
  `tests/test_mapscene_render_policy.py`, then wired `MapScene.render()` to the
  deterministic offscreen PNG helper with validation-before-render and warning /
  error / fatal blocking policy.
- T012-T013 added `tests/test_mapscene_save_bundle.py`, then wired
  `MapScene.save_bundle()` to write deterministic review metadata, recipe
  intent, validation report, compiled label plans, deterministic label source
  references, and non-renderable status for blocked scenes.
- T011A/T011B added recipe-sensitive render regression coverage and replaced
  hash-only placeholder output with deterministic RGBA generation that changes
  when supported recipe data changes.
- T013A/T013B added bundle layer-source metadata coverage and writes
  `scene/layer_sources/` JSON payloads for terrain, raster, vector, label, and
  point-cloud inputs.
- T014/T015 added smoke-tested canonical examples:
  `examples/mapscene_terrain_raster.py`,
  `examples/mapscene_vector_labels.py`, and
  `examples/mapscene_buildings_labels.py`.
- T018 added quickstart validation for the canonical examples and required
  diagnostic-bearing negative paths.
- T020R/T021R updated release-gate honesty, blocker state, ledger, and this
  continuity pack for the red-remediation checkpoint.
- T020S/T021S updated matrix, ledger, and this continuity pack for the scoped
  external asset diagnostics checkpoint.
- T016B/T017B added a scoped diagnostics-reference audit for MapScene
  validation codes.
- T016/T017 added the full MVP docs/support audit and updated the Offline 3D
  Map Rendering guide, support matrices, and competitive positioning note.
- T020T/T021T updated matrix, ledger, and this continuity pack for the terrain
  CRS / building source diagnostics checkpoint.
- T016A-T017A added scoped docs drift tests and updated
  `docs/guides/offline_3d_map_rendering.md`, `docs/api/api_reference.rst`,
  `specs/004-mapscene-mvp/plan.md`, and
  `specs/004-mapscene-mvp/contracts/mapscene-contract.md` with the selected
  public `forge3d.map_scene` path. T016/T017 later completed the broader full
  docs/support audit.
- T019 ran the full MapScene MVP verification command and reported
  `44 passed in 0.84s`.
- T020 updated `docs/superpowers/state/requirements-verification-matrix.md` and
  moved P0-R4-AC1 through P0-R4-AC8 to `Verified`.
- T021 updated `docs/superpowers/state/implementation-ledger.md` and this
  context pack with final closure evidence and the next exact prompt.
- T022-T025 added render truth gate tests, implemented `placeholder_fallback`
  diagnostics for `mapscene.render_backend`, vector path loader, point-cloud
  MapScene render path, and supported building render adapter, blocked synthetic
  PNG writes, updated examples/docs/API/quickstart/contract wording, and updated
  matrix/ledger/context/blocker state. Focused RED reported `11 failed, 22
  passed`; focused GREEN reported `33 passed`; docs/quickstart reported
  `11 passed`; full MapScene reported `47 passed`; `py_compile` exited 0.
- T026-T029 added source-derived render tests, replaced the global blocked
  render backend with deterministic source-derived PNG output for supported
  non-blocked MVP scenes, updated examples/docs/API/quickstart/contract wording,
  and updated matrix/ledger/context/blocker state. Focused RED reported
  `8 failed, 14 passed`; focused GREEN reported `22 passed`; full MapScene
  reported `47 passed`; `py_compile` exited 0.
- Post-audit R-029 remediation added native/offscreen render evidence for real
  `.npy` terrain, PNG raster, inline vector, and label recipes; examples now
  generate local fixture assets and assert `native/offscreen`; docs and state
  distinguish native/offscreen evidence from source-derived compatibility
  output. Focused RED reported `1 failed`; focused render/docs/examples command
  reported `23 passed`; full MapScene reported `48 passed`; `py_compile`
  exited 0.
- Reuse prerequisites: `python/forge3d/diagnostics.py`,
  `python/forge3d/label_plan.py`, `python/forge3d/helpers/offscreen.py`,
  `python/forge3d/bundle.py`, style/building/3D Tiles/VT validators, pointcloud
  metadata, and memory helpers.
- P0-R4-AC1 through P0-R4-AC8 are verified for feature `004`. No incomplete
  task remains in `specs/004-mapscene-mvp/tasks.md`.
- Independent PRD compliance audit on 2026-05-17 originally confirmed feature
  `004` was not complete. Focused command
  `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py -q`
  reported `10 passed in 0.21s`, but that only verified typed construction and
  validation-scope diagnostics. R-021 through R-027 are mitigated after
  T026-T029; R-029 is mitigated by the native/offscreen remediation; R-028
  tracks blocked label acceptance rows in `docs/superpowers/state/open-blockers.md`.

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

Feature `005` Phase 1 prerequisite gate:
- RED prerequisite contracts:
  `$env:PYTHONPATH='python'; python -m pytest tests\test_p1_prerequisite_contracts.py -q`
  -> `2 failed, 1 passed` before implementation due to missing
  `Tiles3DLayer` and P1 API methods.
- RED diagnostics:
  `$env:PYTHONPATH='python'; python -m pytest tests\test_p1_diagnostics_contract.py -q`
  -> `2 failed` before implementation due to missing feature-local diagnostic
  factories.
- Fixture inventory:
  `$env:PYTHONPATH='python'; python -m pytest tests\test_p1_fixture_inventory.py -q`
  -> `3 passed`.
- GREEN combined P1 Phase 1:
  `$env:PYTHONPATH='python'; python -m pytest tests\test_p1_prerequisite_contracts.py tests\test_p1_diagnostics_contract.py tests\test_p1_fixture_inventory.py -q`
  -> `8 passed`.
- Shared MapScene regression subset:
  `$env:PYTHONPATH='python'; python -m pytest tests\test_mapscene_recipe_contract.py tests\test_mapscene_validation.py tests\test_mapscene_support_status.py tests\test_mapscene_label_plan_integration.py tests\test_mapscene_render_png.py tests\test_mapscene_render_policy.py tests\test_mapscene_save_bundle.py -q`
  -> `34 passed`.
- Compile check:
  `$env:PYTHONPATH='python'; python -m py_compile python\forge3d\diagnostics.py python\forge3d\map_scene.py python\forge3d\__init__.py tests\test_p1_prerequisite_contracts.py tests\test_p1_diagnostics_contract.py tests\test_p1_fixture_inventory.py`
  -> exit `0`.

Feature `005` T007-T008 LabelLayer geometry ingestion:
- RED:
  `$env:PYTHONPATH='python'; python -m pytest tests\test_p1_label_layer_geometry.py -q`
  -> `4 failed` before implementation.
- GREEN:
  `$env:PYTHONPATH='python'; python -m pytest tests\test_p1_label_layer_geometry.py -q`
  -> `4 passed in 0.18s`.

Feature `006` T002-T003 P2 diagnostic contract:
- RED:
  `$env:PYTHONPATH='python'; python -m pytest tests\test_p2_diagnostics_contract.py -q`
  -> `4 failed` before implementation.
- GREEN:
  `$env:PYTHONPATH='python'; python -m pytest tests\test_p2_diagnostics_contract.py -q`
  -> `4 passed`.
- Focused regression:
  `$env:PYTHONPATH='python'; python -m pytest tests\test_p1_diagnostics_contract.py tests\test_p1_building_workflow_support.py tests\test_p1_tiles3d_support.py -q`
  -> `15 passed`.
- Compile check:
  `$env:PYTHONPATH='python'; python -m py_compile python\forge3d\diagnostics.py python\forge3d\map_scene.py python\forge3d\__init__.py tests\test_p2_diagnostics_contract.py`
  -> exit `0`.
- Top-level export smoke:
  PowerShell here-string importing `forge3d` and asserting the P2 diagnostic
  inventory/factory helper exports exist
  -> `p2 exports ok`.

Feature `004` R-029 native/offscreen render remediation:
- Focused RED command:
  `python -m pytest tests/test_mapscene_render_png.py::test_render_uses_native_offscreen_for_real_terrain_and_raster_assets -q`
  -> `1 failed`.
- Focused render/docs/examples command:
  `python -m pytest tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_docs.py -q`
  -> `23 passed in 0.98s`.
- Full MapScene command:
  `python -m pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q`
  -> `48 passed in 1.36s`.
- Compile check:
  `python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_render_png.py tests\test_mapscene_examples.py tests\test_mapscene_quickstart.py tests\test_mapscene_docs.py examples\mapscene_terrain_raster.py examples\mapscene_vector_labels.py examples\mapscene_buildings_labels.py`
  -> exit 0.
- Requirement status impact: R-029 is mitigated; P0-R4-AC1 through P0-R4-AC8
  are `Verified`.

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

Feature `004` T004-T006 validation adapters:
- Red `pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py -q`
  -> `8 failed` because placeholder `MapScene.validate()` returned `ok`
  without typed diagnostics or layer summaries.
- Implemented validation adapters in `python/forge3d/map_scene.py` using
  existing diagnostics/style/label/VT contracts.
- Green `pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py -q`
  -> `8 passed`.
- Green `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py -q`
  -> `10 passed`.
- `python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_validation.py tests\test_mapscene_support_status.py`
  -> exit 0.
- P0-R4-AC1, P0-R4-AC5, P0-R4-AC6, and P0-R4-AC7 are now `Tested` for
  validation-scope evidence only. P0-R4 render, bundle, examples, docs,
  LabelPlan integration, and full verification remain incomplete.

Feature `004` independent PRD compliance audit:
- Git status at audit time: branch `004-mapscene-mvp`, dirty worktree with
  modified `python/forge3d/map_scene.py`, state files, `specs/004-mapscene-mvp/tasks.md`,
  unrelated/user-owned files, and untracked MapScene tests.
- Changed-file summary includes `python/forge3d/map_scene.py`,
  `tests/test_mapscene_validation.py`, `tests/test_mapscene_support_status.py`,
  `tests/test_mapscene_recipe_contract.py`, state files, and unrelated paths.
- Fresh focused command:
  `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py -q`
  -> `10 passed in 0.21s`.
- `python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_recipe_contract.py tests\test_mapscene_validation.py tests\test_mapscene_support_status.py`
  -> exit 0.
- Audit status at that time: P0-R4 was not complete because AC2 PNG render, AC3
  bundle save, AC8 examples/docs, MapScene LabelPlan integration, deterministic
  validation proof, missing-CRS diagnostics, building geometry-count memory
  evidence, and AC4 matrix status were incomplete. T007-T008 later mitigated the
  LabelPlan, deterministic validation, missing-CRS, geometry-count, and AC4
  matrix evidence gaps; render, bundle, examples, docs, quickstart, and full
  verification remain incomplete.

Feature `004` T007-T008 LabelPlan integration:
- Red `pytest tests/test_mapscene_label_plan_integration.py tests/test_mapscene_validation.py -q`
  -> `4 failed, 3 passed` because MapScene did not compile label plans, missing
  CRS metadata was treated as compatible, label rejection summaries were absent,
  and building geometry count did not contribute to memory estimates.
- Implemented MapScene LabelPlan integration in `python/forge3d/map_scene.py`:
  each `LabelLayer` compiles or rehydrates a deterministic `LabelPlan`, uses
  map-furniture keepouts, priority rules, typography, glyph atlas, output
  viewport, terrain, camera, and reproducibility seed, stores the plan in
  `compiled_label_plans`, and remaps diagnostics to the MapScene label layer.
- Added typed `missing_crs` diagnostics for absent layer CRS metadata and
  building `geometry_count` memory estimates.
- Green `pytest tests/test_mapscene_label_plan_integration.py tests/test_mapscene_validation.py -q`
  -> `7 passed`.
- Green `pytest tests/test_mapscene_label_plan_integration.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_recipe_contract.py -q`
  -> `13 passed`.
- `python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_label_plan_integration.py tests\test_mapscene_validation.py tests\test_mapscene_support_status.py tests\test_mapscene_recipe_contract.py`
  -> exit 0.
- P0-R4-AC1, AC4, AC5, AC6, and AC7 are now `Tested` for
  validation/LabelPlan-integration scope only. P0-R4 render, bundle, examples,
  docs, quickstart, and full verification remain incomplete.

Feature `004` red-only remediation checkpoint:
- Selected task IDs: T011A, T011B, T013A, T013B, T014, T015, T018, T020R,
  and T021R.
- Red `pytest tests/test_mapscene_render_png.py -q`
  -> `1 failed, 2 passed` because different supported recipe data produced
  byte-identical PNG bytes.
- Red `pytest tests/test_mapscene_save_bundle.py -q`
  -> `1 failed, 1 passed` because non-label layer source metadata such as
  `scene/layer_sources/terrain.json` was missing.
- Red `pytest tests/test_mapscene_examples.py -q`
  -> `4 failed` because canonical MapScene examples were absent.
- Red `pytest tests/test_mapscene_quickstart.py -q`
  -> `3 failed, 1 passed` because quickstart/example execution coverage was
  absent.
- Implemented recipe-sensitive deterministic render output, layer-source
  bundle metadata, three typed canonical examples, quickstart commands, and
  red-remediation matrix/ledger/context updates.
- Green `pytest tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py -q`
  -> `7 passed`.
- Green `pytest tests/test_mapscene_save_bundle.py -q`
  -> `2 passed`.
- Green `pytest tests/test_mapscene_examples.py -q`
  -> `4 passed`.
- Green `pytest tests/test_mapscene_quickstart.py -q`
  -> `4 passed`.
- Green selected combined command
  `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q`
  -> `34 passed`.
- `python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_render_png.py tests\test_mapscene_save_bundle.py tests\test_mapscene_examples.py tests\test_mapscene_quickstart.py examples\mapscene_terrain_raster.py examples\mapscene_vector_labels.py examples\mapscene_buildings_labels.py`
  -> exit 0.
- `rg -n "viewer_ipc|send_ipc" examples/mapscene_terrain_raster.py examples/mapscene_vector_labels.py examples/mapscene_buildings_labels.py`
  -> no matches.
- `.gitignore` now has narrow exceptions for the three MapScene examples, and
  `git status --untracked-files=all` lists them as untracked evidence.
- Requirement status impact at that checkpoint: P0-R4-AC8 moved to `Tested`; P0-R4-AC2 and
  P0-R4-AC3 have stronger data-sensitive render and source-metadata bundle
  evidence. No full feature `Verified` status was claimed then because
  T016/T017/T019/T020/T021 were still open.

Feature `004` external asset diagnostics checkpoint:
- Selected task IDs: T006B, T006C, T020S, and T021S.
- Red `pytest tests/test_mapscene_validation.py tests/test_mapscene_save_bundle.py -q`
  -> `3 failed, 10 passed` because missing local terrain/raster/point-cloud
  assets and unsupported asset suffixes were accepted as `ok`, and
  `save_bundle()` could mark a missing-asset scene renderable.
- Implemented `missing_external_asset` and `unsupported_asset_format`
  diagnostics in `python/forge3d/map_scene.py`, with explicit
  `asset_status: fixture` metadata for symbolic fixture recipes in selected
  tests/examples.
- Green `pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_save_bundle.py -q`
  -> `17 passed`.
- Green selected combined command
  `pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q`
  -> `37 passed`.
- `python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_validation.py tests\test_mapscene_save_bundle.py`
  -> exit 0.
- State/evidence `rg` for `T006B|T006C|T020S|T021S|missing_external_asset|unsupported_asset_format|Next Exact Prompt|P0-R1-AC2|P0-R1-AC3`
  -> exit 0.
- Requirement status impact: P0-R4-AC1, P0-R4-AC3, and P0-R4-AC7 have stronger
  no-op-prevention and bundle-diagnostic evidence. At that feature `004`
  checkpoint, P1-R5-AC3 remained planned for feature `005`; feature `005` now
  verifies full missing-asset bundle round-trip coverage with structured
  diagnostics.

Feature `004` terrain CRS / building source diagnostics checkpoint:
- Selected task IDs: T006D, T006E, T016B, T017B, T020T, and T021T.
- Red `python -m pytest tests/test_mapscene_validation.py tests/test_mapscene_docs.py -q -p no:cacheprovider`
  -> `3 failed, 12 passed` because terrain without CRS, supported building
  source asset failures, and diagnostics-reference inventory gaps were accepted
  or undocumented.
- Implemented terrain `missing_crs` validation, supported-building
  `missing_external_asset` / `unsupported_asset_format` validation, and scoped
  `docs/guides/diagnostics_reference.md` coverage for MapScene diagnostic
  codes.
- Green `python -m pytest tests/test_mapscene_validation.py tests/test_mapscene_docs.py -q -p no:cacheprovider`
  -> `15 passed`.
- Green `python -m pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_docs.py -q -p no:cacheprovider`
  -> `19 passed`.
- Green selected combined command
  `python -m pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_save_bundle.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q -p no:cacheprovider`
  -> `26 passed`.
- `python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_validation.py tests\test_mapscene_docs.py`
  -> exit 0.
- State/evidence `rg` for `T006D|T006E|T016B|T017B|T020T|T021T|missing_crs|missing_external_asset|unsupported_asset_format|Next Exact Prompt`
  -> exit 0.
- Requirement status impact: P0-R4-AC1, P0-R4-AC5, P0-R4-AC7, and scoped
  P0-R4-AC8 have stronger no-op-prevention and docs evidence. T019 full
  verification, T020 final matrix closure, and T021 final continuity closure
  remain open.

Feature `004` full docs/support audit checkpoint:
- Selected task IDs: T016 and T017.
- Red `python -m pytest tests/test_mapscene_docs.py -q -p no:cacheprovider`
  -> `4 failed, 3 passed` after correcting the support-level parser because
  the offline guide lacked canonical examples/support links, support matrices
  had stale current-ownership wording, competitive positioning used umbrella
  support wording, and required non-goal boundaries were absent.
- Updated `docs/guides/offline_3d_map_rendering.md`,
  `docs/guides/style_support_matrix.md`,
  `docs/guides/building_support_matrix.md`,
  `docs/guides/tiles3d_support_matrix.md`,
  `docs/guides/virtual_texturing_support_matrix.md`, and
  `docs/guides/competitive_positioning.md`.
- Green `python -m pytest tests/test_mapscene_docs.py -q -p no:cacheprovider`
  -> `7 passed`.
- Green selected docs/examples/quickstart command
  `python -m pytest tests/test_mapscene_docs.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q -p no:cacheprovider`
  -> `15 passed`.
- Green validation/support/docs command
  `python -m pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_docs.py -q -p no:cacheprovider`
  -> `23 passed`.
- `python -m py_compile tests\test_mapscene_docs.py`
  -> exit 0.
- Requirement status impact: P0-R4-AC7 and P0-R4-AC8 have full docs/support
  audit evidence for feature `004`; scoped P0-R4-AC1 docs evidence is stronger.
  T019 full verification, T020 final matrix closure, and T021 final continuity
  closure remain open.

Feature `004` full MVP verification and closure:
- Selected task IDs: T019, T020, and T021.
- Full verification command:
  `python -m pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q -p no:cacheprovider`
  -> `44 passed in 0.84s`.
- T020 updated `docs/superpowers/state/requirements-verification-matrix.md`;
  P0-R4-AC1 through P0-R4-AC8 were marked `Verified` at that historical
  checkpoint, before T022-T025 superseded the synthetic PNG evidence.
- T021 updated `docs/superpowers/state/implementation-ledger.md` and this
  context pack with final outcomes, residual blockers, and the next exact
  prompt.
- Requirement status impact at that checkpoint: feature `004-mapscene-mvp`
  was complete through T021. T022-T025 later blocked the synthetic P0-R4-AC2
  evidence, T026-T029 then superseded that blocker with source-derived
  compatibility output, and the R-029 remediation added native/offscreen
  fixture-backed render evidence. Current P0-R4-AC1 through P0-R4-AC8 status is
  `Verified`; P1/P2 feature rows remain planned.

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

R-011 remains open: the worktree still contains feature work plus unrelated or
user-owned changes, including `AGENTS.md`, a deleted example, logs/PDB output,
`src/viewer/event_loop/runner.rs`, and an untracked timelapse test. Do not
revert unrelated changes.

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

New/updated feature `004` paths from the T004-T006 validation session include:
- `python/forge3d/map_scene.py`
- `tests/test_mapscene_validation.py`
- `tests/test_mapscene_support_status.py`
- `specs/004-mapscene-mvp/tasks.md`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/current-context-pack.md`

New/updated feature `004` paths from the T007-T008 LabelPlan integration
session include:
- `python/forge3d/map_scene.py`
- `tests/test_mapscene_label_plan_integration.py`
- `tests/test_mapscene_validation.py`
- `specs/004-mapscene-mvp/tasks.md`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/current-context-pack.md`
- `docs/superpowers/state/open-blockers.md`

New/updated feature `004` paths from the red/yellow remediation batch include:
- `python/forge3d/map_scene.py`
- `tests/test_mapscene_validation.py`
- `tests/test_mapscene_support_status.py`
- `tests/test_mapscene_render_png.py`
- `tests/test_mapscene_render_policy.py`
- `tests/test_mapscene_save_bundle.py`
- `tests/test_mapscene_docs.py`
- `docs/guides/offline_3d_map_rendering.md`
- `docs/api/api_reference.rst`
- `specs/004-mapscene-mvp/plan.md`
- `specs/004-mapscene-mvp/contracts/mapscene-contract.md`
- `specs/004-mapscene-mvp/tasks.md`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/current-context-pack.md`

New/updated feature `004` paths from the red-only remediation checkpoint
include:
- `.gitignore`
- `python/forge3d/map_scene.py`
- `tests/test_mapscene_render_png.py`
- `tests/test_mapscene_save_bundle.py`
- `tests/test_mapscene_examples.py`
- `tests/test_mapscene_quickstart.py`
- `examples/mapscene_terrain_raster.py`
- `examples/mapscene_vector_labels.py`
- `examples/mapscene_buildings_labels.py`
- `specs/004-mapscene-mvp/quickstart.md`
- `specs/004-mapscene-mvp/tasks.md`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/open-blockers.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/current-context-pack.md`

New/updated feature `004` paths from the latest red diagnostics checkpoints
include:
- `python/forge3d/map_scene.py`
- `tests/test_mapscene_validation.py`
- `tests/test_mapscene_docs.py`
- `tests/test_mapscene_support_status.py`
- `tests/test_mapscene_label_plan_integration.py`
- `tests/test_mapscene_render_png.py`
- `tests/test_mapscene_render_policy.py`
- `tests/test_mapscene_save_bundle.py`
- `tests/test_mapscene_quickstart.py`
- `docs/guides/diagnostics_reference.md`
- `examples/mapscene_terrain_raster.py`
- `examples/mapscene_vector_labels.py`
- `examples/mapscene_buildings_labels.py`
- `specs/004-mapscene-mvp/tasks.md`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/current-context-pack.md`

New/updated feature `004` paths from the T019-T021 closure checkpoint include:
- `specs/004-mapscene-mvp/tasks.md`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/current-context-pack.md`

## Open Blockers

- R-011: mitigated for commit/PR scope by staging only intended
  feature/blocker changes and leaving unrelated/generated/user-owned paths
  unstaged.
- R-030 through R-035: mitigated by feature `005-map-assets-bundles-p1`
  T009-T069 evidence. Matrix P1-R1-AC1 through P1-R5-AC5 rows are `Verified`.
- R-036: mitigated. `git diff --cached --name-status` contains only intended
  feature/blocker-scope changes; `git diff --name-status` leaves unrelated or
  generated/user-owned paths unstaged: `AGENTS.md`, deleted
  `examples/turkiye_river_basins_3d.py`,
  `logs/.3c2cf94182465f0d10df58878528dd39234a1134-audit.json`,
  `python/forge3d/forge3d.pdb`, and `src/viewer/event_loop/runner.rs`.
  `tests/test_mount_hood_white_cloud_timelapse.py` remains untracked and
  unstaged.
- R-037: mitigated. Strict PRD product completion is no longer blocked by
  P0-R1-AC2/P0-R1-AC3; the 2026-05-18 remediation verified typography
  state/layout metrics and declutter placement-policy state. Fresh audit reruns
  reported full label command `28 passed in 2.37s`, focused Rust command `1 passed`,
  `cargo check -q` exit `0`, `cargo fmt -- --check` exit `0`, and Python
  `py_compile` exit `0`.
- R-012/R-013/R-014: mitigated in feature `006` by focused P2 tests/docs and
  diagnostic-bearing implementation. VT normal/mask remain unsupported
  diagnostic deferrals, textured PBR buildings remain unsupported diagnostic
  deferrals, and large-scene summaries are opt-in via
  `diagnostics_policy={"large_scene_summary": True}`.
- Feature `004` 2026-05-18 audit found no new P0-R4 blocker; keep the
  diagnostic-bearing/later-owned scope boundaries listed above explicit in any
  follow-up docs or PR text.
- R-020: mitigated by feature `003` LabelPlan guide/quickstart and feature
  `004` MapScene quickstart evidence; keep covered in docs/quickstart tests.
- R-021 through R-025: mitigated by T007-T008; keep them closed unless the
  focused MapScene LabelPlan/validation command regresses.
- R-026: mitigated by T026-T029. `MapScene.render()` no longer uses the
  recipe-sensitive synthetic compositor as PNG evidence; supported MVP
  symbolic fixture recipes write deterministic source-derived compatibility PNG
  output and unsupported layer paths keep typed diagnostics.
- R-027: mitigated by T014/T015/T018 examples/quickstart, T016/T017 full MVP
  docs/support audit, T019 full verification, and T024 wording corrections for
  the blocked render backend; keep covered by the full MapScene verification
  command.
- R-029: mitigated by the native/offscreen remediation. Fixture-backed `.npy`
  terrain, PNG raster, inline vector, and label recipes now render through
  native `Scene.render_rgba()` with `scene.last_render_backend ==
  "native/offscreen"`, while source-derived output remains compatibility-only.
  Keep incomplete point-cloud/building render adapters, 3D Tiles public Python
  rendering, VT normal/mask runtime, textured PBR buildings, full style parity,
  and feature `005` bundle round-trip scope diagnostic-bearing or documented.
- R-028: mitigated. P0-R1-AC2 and P0-R1-AC3 are verified by public and native
  typography/declutter state evidence plus fresh Python/Rust verification;
  keep curved labels and terrain-elevated line labels diagnostic-bearing through
  `experimental_feature`.
- External asset no-op risk from the earlier RED batch is mitigated by
  T006B/T006C for terrain/raster/point-cloud pre-render diagnostics and blocked
  bundle persistence; feature `005` now verifies missing-asset bundle
  round-trip replay with structured diagnostics.
- Missing terrain CRS and supported-building source no-op risks are mitigated
  by T006D/T006E; scoped diagnostics-reference omissions are mitigated by
  T016B/T017B; full MVP docs/support audit coverage is mitigated by T016/T017;
  T026-T029 corrected the final P0-R4 source-derived compatibility state; the
  R-029 remediation now verifies AC2 with native/offscreen fixture-backed render
  evidence instead of source-derived-only evidence.

## Next Exact Prompt

```text
[$speckit-implement](C:\Users\milos\forge3d\.agents\skills\speckit-implement\SKILL.md)

Feature `006-material-vt-large-scene-p2` has focused P2 verification evidence.
Continue only with requested follow-up scope. Keep unrelated/generated paths
separated; P2 diagnostic deferrals for VT normal/mask, textured PBR buildings,
curved text, and complex-script shaping remain intentional documented outcomes.
```

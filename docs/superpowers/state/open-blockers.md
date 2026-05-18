# Open Blockers

Source PRD: `docs/superpowers/plans/prd.md`  
Last updated: 2026-05-18
Current phase: multi-feature technical plans created for features `001` through `006`

No blocker prevented technical planning for features `001` through `006`. The items below must be resolved or deliberately defaulted during tasks or implementation before implementation claims can be made.

## Product Questions From PRD Section 20

| ID | Blocker / decision | Affects | Suggested resolution point | Status |
|---|---|---|---|---|
| B-001 | Should `MapScene` be a new top-level API or a wrapper around existing `Scene` and `ViewerHandle`? | `004-mapscene-mvp` | Constitution/spec for `004`; architecture plan before tasks. | Defaulted in plan: top-level typed product API that wraps existing substrate internally; exact implementation files TBD during tasks. |
| B-002 | Should `LabelPlan` be serializable as part of bundles by default, or should bundles store source labels and recompile on load? | `003-deterministic-label-plan`, `005-map-assets-bundles-p1` | Feature `003` spec and bundle model in feature `005`. | Defaulted for feature `005`: bundles store both source labels and compiled `LabelPlan` payloads where available; unsupported persistence or replay gaps emit structured diagnostics. |
| B-003 | What is the minimum supported CRS transform set for P0? | `001-diagnostics-support-matrices`, `004-mapscene-mvp` | Feature `001` or `004` clarify step. | Defaulted for feature `004`: no implicit CRS transforms; require matching CRS metadata or explicit compatible transform/policy, otherwise emit `crs_mismatch`. |
| B-004 | Should `MapScene.render()` support both offscreen native rendering and viewer snapshot rendering in the first release? | `004-mapscene-mvp` | Feature `004` plan. | Defaulted in plan: prefer native/offscreen PNG path for MVP; viewer snapshot optional or diagnostic unless task inspection proves it required and reliable. |
| B-005 | Which style expression subset is mandatory for P1 labels: only `{name}` and `get`, or also `concat`, `coalesce`, and casing? | `005-map-assets-bundles-p1` | Feature `005` clarify step; PRD currently lists all five in P1-R1-AC4. | Defaulted for feature `005`: require the full PRD-listed subset: `{name}`, `get`, `concat`, `coalesce`, and casing transforms. |
| B-006 | Which building workflows are public/open versus Pro-gated in the intended release packaging? | `001-diagnostics-support-matrices`, `005-map-assets-bundles-p1` | Feature `001` support matrix and feature `005` plan. | Defaulted for feature `005`: packaging may vary; validation must distinguish available native, `Pro-gated`, `placeholder/fallback`, and `unsupported` paths before render. |
| B-007 | Is 3D Tiles support intended for local fixtures and review only, or for large production tile hierarchies? | `005-map-assets-bundles-p1`, `006-material-vt-large-scene-p2` | Feature `005` clarify step. | Defaulted for feature `005`: scope to supported local fixtures and offline review or render preparation, not large production tile hierarchies or full Cesium runtime parity. |
| B-008 | What tolerance defines deterministic pixel stability for rendered regression tests? | `003-deterministic-label-plan`, `004-mapscene-mvp`, `005-map-assets-bundles-p1`, `006-material-vt-large-scene-p2` | Constitution and feature `004`/`005`/`006` test strategy. | Defaulted for features `004`, `005`, and `006`: exact pixel or fixture comparison for deterministic fixtures unless planning records a numeric tolerance before verification. |

## Repo-Specific Implementation Risks

| ID | Risk | Why it is specific to this repo | Mitigation required by PRD | Status |
|---|---|---|---|---|
| R-001 | Existing label IPC paths may acknowledge commands without mutating real `LabelManager` state. | Feature `002` T005-T018 blocks high-level no-op success for typography, declutter, atlas, enabled state, clear/remove, empty text, invalid geometry, missing glyphs, curved labels, terrain sampling, docs/support claims, and quickstart/example workflows. Raw IPC remains available as advanced compatibility surface, but docs now direct MVP label workflows to high-level `ViewerHandle` methods. | Keep high-level label verification in future feature checks; do not use raw IPC examples as MVP workflow proof. | Mitigated in `002`; raw IPC remains advanced compatibility only. |
| R-002 | Create calls may omit stable IDs across label, line-label, callout, or overlay paths. | Existing response protocol had generic success responses in the evidence anchors. | Feature `002` must enforce stable IDs where update/inspect/remove workflows need them. | Mitigated for `002` Phase 1 by `ViewerHandle` stable create wrappers, native client-supplied ID preservation, `IpcResponse::with_id(...)`, `tests/test_label_api_stable_ids.py`, and Rust IPC tests. Later update/inspect/remove workflows still need feature-specific coverage before full feature completion. |
| R-003 | Line/curved labels may store metadata but not emit renderable glyph instances. | Feature `002` T010 added native rotated text instances for line labels, corrected native tangent/upside-down rotation handling, and added high-level deterministic glyph placement/rotation state. T013-T018 added docs/support and quickstart evidence. Curved labels and terrain-elevated line labels return typed `experimental_feature` diagnostics and are documented as experimental instead of unqualified success. | Keep line/curved/terrain regressions in the full label verification set and do not claim production curved or terrain-elevated label support until later work implements it. | Mitigated in `002`; curved and terrain-elevated paths remain diagnostic deferrals. |
| R-004 | Style parser may silently drop unsupported Mapbox fields/layer types. | `python/forge3d/style.py` exists but is underdeveloped for full style support and not an MVT renderer. | Feature `001` must add `unsupported_style_field` and `unsupported_style_layer_type` diagnostics plus docs. | Mitigated in `001` by `validate_style_support(...)`, tests, and style support matrix. |
| R-005 | Building ingestion may fall back to zero geometry when Pro/native paths are unavailable. | PRD evidence identifies fallback branches in `python/forge3d/buildings.py`. Feature `005` now verifies `pro_gated_path`, `placeholder_fallback`, zero-geometry fallback, geometry-count/bounds summaries, and textured PBR `unsupported_feature` diagnostics through `tests/test_p1_building_workflow_support.py`. | Keep public MapScene building render adapters diagnostic-bearing where unavailable; do not claim full textured building render support until feature `006` implements it. | Mitigated in `001`/`005`; textured material runtime remains feature `006`. |
| R-006 | 3D Tiles Rust substrate may be overclaimed as public Python product readiness. | Rust B3DM/tile renderer infrastructure exists, but Python scene integration is incomplete. Feature `005` now verifies public `Tiles3DLayer` scene intent, supported/unsupported format diagnostics, cache/LOD metadata, unsupported tile feature diagnostics, and `python_public_3dtiles_incomplete` for incomplete public render paths through `tests/test_p1_tiles3d_support.py`. | Keep public 3D Tiles render parity diagnostic-bearing through `python_public_3dtiles_incomplete`; do not claim full Cesium runtime parity. | Mitigated in `001`/`005`; large production tile hierarchy work remains out of P1 scope. |
| R-007 | VT normal/mask families may remain placeholders while runtime pages only albedo. | PRD classifies VT normal/mask runtime support as missing in current runtime. | Feature `001`/`006` must diagnose `vt_unsupported_family` until implemented and tested. | Diagnostic contract and VT support matrix verified in `001`; runtime behavior remains open in `006`. |
| R-008 | `MapScene` could become a thin wrapper that still requires raw IPC for canonical workflows. | Current users stitch together viewer, IPC, style, bundle, CRS, terrain, label, building, and point-cloud APIs. | Feature `004` must prove examples use typed recipe APIs without raw IPC for MVP workflows. | Mitigated in `004` by canonical examples/tests and native/offscreen PNG render evidence; raw IPC is not required. |
| R-009 | Bundle save may omit enough scene state to reproduce review context. | Existing bundle substrate exists, but typed map-scene round-trip is a new product contract. Feature `005` now verifies P1 bundle fields, label sources/plans, diagnostics/state, deterministic manifests, available-asset load/render, and missing-asset replay diagnostics through `tests/test_p1_bundle_roundtrip.py` and `tests/test_p1_bundle_guardrails.py`. | Keep missing external assets structured and render-blocking; do not drop support status or diagnostics from review bundles. | Mitigated in `004`/`005`. |
| R-010 | Docs may overclaim support by saying missing/underdeveloped paths are supported. | The PRD explicitly protects classifications such as underdeveloped, Pro-gated, placeholder/fallback, experimental, unsupported, and non-goal. | Feature `001` must establish support matrices and docs wording discipline before implementation claims. | Mitigated in `001` by docs audit tests and support matrices. |
| R-011 | Existing dirty worktree contains feature work mixed with unrelated or not-yet-attributed paths. | Mitigated for commit/PR scope by staged/unstaged separation. `git diff --cached --name-status` now contains intended feature/blocker scope only: feature `005` specs/tests, feature `002` typography/declutter remediation, product API/docs/state updates, and `.specify`/`.gitignore` visibility changes. Unrelated/generated/user-owned paths remain unstaged: `AGENTS.md`, deleted `examples/turkiye_river_basins_3d.py`, `logs/.3c2cf94182465f0d10df58878528dd39234a1134-audit.json`, `python/forge3d/forge3d.pdb`, `src/viewer/event_loop/runner.rs`, and untracked `tests/test_mount_hood_white_cloud_timelapse.py`. | Preserve the staged/unstaged separation until commit/PR; do not include unstaged unrelated/generated paths unless explicitly requested. | Mitigated |
| R-012 | P2 VT normal/mask requests may be accepted but ignored by the native runtime. | Feature `006` T004/T005 added deterministic tests and implementation so non-albedo VT requests emit `vt_unsupported_family` with affected IDs `vt.normal`/`vt.mask`, classify `vt.normal`/`vt.mask` as `unsupported`, and block render before silent skip. VT docs state native runtime pages only `albedo`. Full P2 command reported `35 passed`. | Keep normal/mask as diagnostic deferrals until a future native VT implementation is planned, tested, and documented. | Mitigated in `006`; deferred with diagnostic, not native runtime support |
| R-013 | Textured building material support may fall back to scalar PBR without diagnostics. | Feature `006` T008-T012 added textured building MapScene fixture tests, texture diagnostics tests, docs audits, and MapScene implementation. Texture+UV intent now emits explicit `unsupported_feature` for the textured PBR render path; missing texture path, missing UVs, unsupported format, scalar fallback, and Pro-gated paths are diagnostic-bearing before render. Full P2 command reported `35 passed`. | Keep end-to-end textured PBR buildings classified as `unsupported`/`Deferred with diagnostic` until a real render path is implemented and tested. | Mitigated in `006`; deferred with diagnostic, not textured PBR support |
| R-014 | Large-scene stats may remain fragmented across lower-level modules. | Feature `006` T019-T024 added large-scene memory/cache/LOD/instancing/bottleneck tests, opt-in `large_scene.resources` validation summaries, docs audits, and offline-scope docs. The summary normalizes known metadata and reports unavailable stats instead of inventing precision. Full P2 command reported `35 passed`. | Keep large-scene summaries opt-in via `diagnostics_policy={"large_scene_summary": True}` unless a later feature changes the public default with regression updates. | Mitigated in `006` |
| R-015 | Feature `001` P0-R5-AC4 is only bundle-ready, not proven in a real scene bundle path. | Remediated by `tests/test_bundle_roundtrip.py::test_scene_bundle_roundtrip_preserves_validation_report` and `python/forge3d/bundle.py` preserving `ValidationReport` through `SceneState`, `LoadedBundle`, `save_bundle(...)`, and `load_bundle(...)`. Focused command `pytest tests/test_bundle_roundtrip.py tests/test_diagnostics_bundle_serialization.py -q` reported `20 passed`. | Keep downstream `MapScene.save_bundle()` and richer review-bundle payload requirements in features `004`/`005`, but do not reopen P0-R5-AC4 for feature `001` unless the bundle test regresses. | Mitigated in `001`; downstream MapScene/review-bundle integration remains planned. |
| R-016 | Feature `001` P0-R5-AC6 is proven for diagnostic factories and style validation, but not for every existing non-style workflow. | Remediated for feature `001` by public validators `validate_building_layer_support(...)`, `validate_tiles3d_support(...)`, `validate_terrain_vt_support(...)`, and `validate_label_support(...)`; `tests/test_diagnostics_support_paths.py` verifies placeholder/fallback, public Python 3D Tiles incompleteness, VT non-albedo families, line/curved experimental labels, missing glyphs, and affected layer/object IDs. Focused command reported `6 passed`. | Later features must call these validators from typed `MapScene`/render-prep workflows rather than duplicating diagnostic contracts. | Mitigated in `001`; automatic MapScene/render-prep wiring remains planned in `004`/`005`/`006`. |
| R-017 | Feature `001` P0-R6-AC5 style output compatibility is not proven through an actual `VectorOverlay` or `LabelLayer` workflow. | Remediated by `vector_overlay_configs_from_style(...)` producing `VectorOverlayConfig` payloads and `label_layer_contracts_from_style(...)` producing an `underdeveloped` future label contract; `tests/test_diagnostics_style_support.py` verifies both helpers. Focused command reported `5 passed`. | Later features must connect the label contract to the real `LabelLayer` once that product API exists, without claiming streamed MVT support. | Mitigated in `001`; real LabelLayer integration remains planned in `005`. |
| R-018 | Feature `001` P0-R6-AC4 was only verified for raw style dictionaries; unsupported paint/layout fields could disappear after `parse_style(...)`. | Remediated by `tests/test_diagnostics_style_support.py::test_parsed_style_support_reports_preserved_unsupported_fields`, `StyleLayer.unsupported_paint_fields`, `StyleLayer.unsupported_layout_fields`, `_parse_layer(...)` preservation, and `validate_style_support(...)` parsed-style diagnostics. Red command failed with status `ok` instead of `warning`; focused command `pytest tests/test_diagnostics_style_support.py -q` reported `6 passed`; full follow-up command reported `58 passed`. | Keep this regression in the feature test set; later style/render features must preserve or revalidate unsupported-field metadata instead of silently dropping it. | Mitigated in `001`. |
| R-019 | Feature `001` P0-R6-AC6 docs audit did not cover public API docstrings that could overclaim full Mapbox support. | Remediated by `tests/test_support_matrices_docs.py::test_public_style_api_docstrings_do_not_overclaim_mapbox_parity` and rewording `python/forge3d/style.py` `StyleSpec` docstring to local/provided feature styling scope. Red command failed on `Complete Mapbox GL Style specification`; focused command `pytest tests/test_support_matrices_docs.py -q` reported `4 passed`; full follow-up command reported `58 passed`. | Keep source-docstring overclaim checks in the docs audit; later public style docs must keep full Mapbox and streamed MVT claims out unless a future PRD decision changes scope. | Mitigated in `001`. |
| R-020 | PRD Section 16 still requires a real `MapScene` quickstart and `LabelPlan` guide once those APIs exist. | Independent audit rechecked this on 2026-05-16: feature `001` docs honestly marked full `MapScene` rendering and deterministic `LabelPlan` as `missing`/later-owned before those APIs existed. Feature `003` later added the deterministic LabelPlan guide/quickstart, and feature `004` T018 now adds quickstart commands for the typed MapScene examples. | Keep the quickstart and LabelPlan guide in the docs audit/full verification sets; reopen only if `tests/test_label_plan_docs.py`, `tests/test_label_plan_quickstart.py`, or `tests/test_mapscene_quickstart.py` regress. | Mitigated in `003` and `004` T018; broader feature `004` docs/support audit remains tracked by R-027. |

## Latest Independent Audit Note

On 2026-05-16, an independent PRD compliance audit rechecked
`specs/001-diagnostics-support-matrices` against the PRD, constitution, spec,
plan, tasks, requirements matrix, changed files, and fresh test output.

Fresh verification command:

```powershell
pytest tests/test_bundle_roundtrip.py tests/test_diagnostics_bundle_serialization.py tests/test_diagnostics_support_paths.py tests/test_diagnostics_style_support.py tests/test_diagnostics_contract.py tests/test_diagnostics_no_op_policy.py tests/test_support_matrices_docs.py tests/test_diagnostics_quickstart.py -q
```

Observed result: `63 passed in 0.63s`.

Audit result: no new P0-R5/P0-R6 acceptance-criterion gap was found for feature
`001`. Incomplete line/curved labels, buildings, 3D Tiles, VT normal/mask, and
style parity are represented by typed diagnostics and support-matrix wording
rather than overclaimed as supported.

Remaining gaps have narrowed: R-011 broader worktree/change separation is now
mitigated by staged/unstaged separation, while R-020 is mitigated by the later
feature `003` LabelPlan guide and feature `004` MapScene quickstart evidence.

## Feature 004 Independent Audit Note

On 2026-05-17, an independent PRD compliance audit rechecked
`specs/004-mapscene-mvp` against the PRD, constitution, spec, plan, tasks,
requirements matrix, changed-file summary, current code, and fresh focused test
output.

Fresh verification command:

```powershell
pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py -q
```

Observed result: `10 passed in 0.21s`.

Audit result: feature `004` was not complete. T001-T006 had evidence for typed
recipe construction and validation-scope diagnostics, but P0-R4 still lacked
verified PNG render output, reproducible review bundle save, MapScene-integrated
deterministic LabelPlan compilation, canonical examples, docs/support audits,
full determinism proof, and full matrix continuity. T007-T008 later mitigated
the LabelPlan, missing-CRS, building geometry-count memory, validation
determinism, and AC4 matrix-evidence gaps. Render, bundle, examples, docs,
quickstart, and full verification remain open.

| ID | Gap | Evidence | Required fix | Status |
|---|---|---|---|---|
| R-021 | P0-R4-AC4 typed recipe is tested in code but remains `Planned` in the requirements matrix. | `docs/superpowers/state/requirements-verification-matrix.md` P0-R4-AC4 row; `tests/test_mapscene_recipe_contract.py`; focused command above. | Updated by T007-T008 evidence row and P0-R4-AC4 matrix status to `Tested`; keep full `Verified` status blocked until render/bundle/examples/docs/full verification. | Mitigated in `004` T007-T008 |
| R-022 | Missing CRS metadata is not diagnosed; `_same_crs(None, ...)` treats missing CRS as compatible. | `python/forge3d/map_scene.py` `_same_crs`; CRS validation branch only fires when both CRS values exist; spec edge case requires explicit missing/unknown CRS status. | Added `tests/test_mapscene_validation.py::test_validate_reports_missing_layer_crs_without_assuming_compatibility` and typed `missing_crs` diagnostic. | Mitigated in `004` T007-T008 |
| R-023 | GPU estimate does not prove geometry-count contribution for building layers. | `MapSceneBuildingLayer.geometry_count` is summarized, but memory estimate tests cover output, terrain/raster dimensions, and point counts only. | Added `tests/test_mapscene_validation.py::test_validate_includes_building_geometry_count_in_gpu_memory_estimate` and conservative geometry-count estimate. | Mitigated in `004` T007-T008 |
| R-024 | Deterministic validation ordering is not independently proven. | Current focused tests assert some ordering, but no repeated `validate().to_dict()` equality test exists for mixed diagnostics/layers. | Added `tests/test_mapscene_label_plan_integration.py::test_label_layer_compiles_deterministic_label_plan_and_rejection_summary`, which compares repeated report and plan payloads exactly. | Mitigated in `004` T007-T008 |
| R-025 | MapScene does not yet compile deterministic LabelPlans or surface label rejection summaries from `LabelPlan.compile()`. | `MapScene.compiled_label_plans` remains empty; T007/T008 are still open; spec requires label rejection summaries where applicable. | Added MapScene LabelPlan integration in `python/forge3d/map_scene.py` and `tests/test_mapscene_label_plan_integration.py`. | Mitigated in `004` T007-T008 |
| R-026 | `MapScene.render()` must not use the recipe-sensitive synthetic compositor as real PNG evidence. | The RED/YELLOW audit found `MapScene.render()` wrote deterministic recipe-hash PNGs without reading source terrain/raster/vector data, so P0-R4-AC2 was overclaimed. | T022-T025 blocked the synthetic backend with typed diagnostics; T026-T029 then replaced the global blocked backend with a source-derived compatibility adapter; R-029 remediation added native/offscreen fixture-backed render evidence for the PRD completion claim. | Mitigated in `004`; focused source-derived command reported `22 passed`, focused native/render docs/examples command reported `23 passed`, full MapScene command reported `48 passed`, and P0-R4-AC2 is `Verified`. |
| R-027 | Canonical MapScene examples and full MVP docs/support audit are absent or incomplete. | Initially no `examples/mapscene_*.py` files were found. T014/T015/T018 added typed examples and quickstart commands; T016/T017 added full docs/support audit; T022-T024 corrected render-backend overclaim wording; T026-T029 updated examples/docs for source-derived compatibility output; R-029 remediation updated terrain/raster and vector/label examples to generate real assets and assert native/offscreen rendering. | Keep examples/docs/quickstart in the full MapScene verification set and preserve typed diagnostics for unsupported paths. | Mitigated for examples/docs/support wording and PNG output evidence. |
| R-028 | Prior P0 label rows used diagnostic deferral for criteria that still require implementation or a human release decision. | Mitigated by 2026-05-18 remediation: `tests/test_label_api_configuration_truth.py::test_set_label_typography_mutates_state_and_layout_metrics` verifies public typography state and deterministic layout metrics; `tests/test_label_api_configuration_truth.py::test_set_declutter_algorithm_mutates_placement_policy_state` verifies public declutter placement-policy state; `src/labels/mod.rs::tests::test_label_manager_typography_and_declutter_state_mutate` verifies native `LabelManager` typography and declutter state mutation. Full label command reported `28 passed`; focused Rust command reported `1 passed`; docs support command reported `5 passed`; `cargo check -q`, `cargo fmt -- --check`, and Python `py_compile` exited `0`. | Keep typography/declutter tests in the full label verification set. Curved labels and terrain-elevated line labels remain diagnostic-bearing through `experimental_feature` and must not be overclaimed. | Mitigated; matrix marks P0-R1-AC2 and P0-R1-AC3 as `Verified` |
| R-029 | P0-R4-AC2 PNG output was only partially verified because prior evidence proved deterministic source-derived PNG files, not a native/offscreen terrain/raster/vector render adapter. | `specs/004-mapscene-mvp/plan.md` requires wrapping native `Scene.render_png` / `render_rgba` where available and says the render path should prefer native/offscreen PNG. The prior `python/forge3d/map_scene.py` path always called `_render_source_derived_rgba(...)`. | Post-audit remediation added `_render_native_offscreen_rgba(...)`, fixture-backed `.npy` terrain and PNG raster loading, native `Scene.set_height_from_r32f`, `set_raster_overlay`, `set_camera_look_at`, instance `render_rgba()`, vector/label compositing, canonical examples with generated real assets, and `tests/test_mapscene_render_png.py::test_render_uses_native_offscreen_for_real_terrain_and_raster_assets`, which monkeypatches the source-derived fallback to fail and asserts `scene.last_render_backend == "native/offscreen"`. | Mitigated in `004`; focused native regression initially failed, focused render/docs/examples command reported `23 passed`, full MapScene command reported `48 passed`, and P0-R4-AC2 remains `Verified`. |

Incomplete line/curved labels, buildings, 3D Tiles, VT normal/mask, and style
support are not accepted as complete. Current evidence shows several of these
paths have typed diagnostics and support-matrix wording from features `001`,
`002`, and feature `004` validation, render, bundle, example, and quickstart
tests. Feature `004` now has full docs/support audit coverage and a current
full MapScene command reporting `48 passed`. The strict audit's R-029
native/offscreen render-evidence gap is mitigated. The prior P0 label blocker
scope R-028 is also mitigated by native/public typography and declutter state
evidence from the 2026-05-18 remediation.

### Feature 004 Audit Refresh - 2026-05-17

This independent PRD compliance refresh audited only `specs/004-mapscene-mvp`
against the PRD, constitution, feature spec/plan/tasks, requirements matrix,
changed-file summary, current implementation, support docs, and fresh test
output.

Fresh verification command:

```powershell
python -m pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_render_policy.py tests/test_mapscene_render_png.py tests/test_mapscene_save_bundle.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_docs.py -q
```

Observed result after R-029 remediation: `48 passed in 1.36s`.

Audit result after remediation: R-029 is mitigated. The suite now includes a
native/offscreen fixture-backed terrain/raster/vector/label render regression
that fails if `_render_source_derived_rgba(...)` is used, and verifies
`scene.last_render_backend == "native/offscreen"`. P0-R4-AC1 through
P0-R4-AC8 remain verified by file/test/doc evidence, not by task checkbox
status, with the following scope boundaries still preserved:

- Point-cloud, supported-building render adapters, 3D Tiles public Python
  rendering, VT normal/mask runtime, textured PBR buildings, and full Mapbox
  style parity remain incomplete or later-owned and must stay diagnostic-bearing
  or honestly documented.
- The current 004 evidence proves deterministic review bundle contents and
  diagnostic persistence, but not the later feature `005` full bundle load/render
  round trip.
- Repo-wide MVP release readiness no longer depends on R-028; P0-R1-AC2 and
  P0-R1-AC3 are verified by the 2026-05-18 typography/declutter remediation.
  R-011 worktree/change separation is mitigated by staged/unstaged separation.

### Feature 004 Independent PRD Compliance Audit - 2026-05-18

This audit reviewed only `specs/004-mapscene-mvp` against the PRD,
constitution, feature spec/plan/tasks, requirements matrix, changed-file
summary, implementation evidence, support docs, and fresh local verification.

Fresh verification:

```powershell
$env:PYTHONPATH='python'; python -m pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_render_policy.py tests/test_mapscene_render_png.py tests/test_mapscene_save_bundle.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_docs.py -q
```

Observed result: `48 passed in 1.64s`.

Compile verification:

```powershell
$env:PYTHONPATH='python'; python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_render_png.py tests\test_mapscene_examples.py tests\test_mapscene_quickstart.py tests\test_mapscene_docs.py examples\mapscene_terrain_raster.py examples\mapscene_vector_labels.py examples\mapscene_buildings_labels.py
```

Observed result: exit `0`.

Audit result: no new feature `004` P0-R4 blocker was found. P0-R4-AC1 through
P0-R4-AC8 remain verified by concrete file/test/doc evidence. The remaining
scope boundaries are not feature `004` blockers: point-cloud and supported
building render adapters, public 3D Tiles rendering, VT normal/mask runtime,
textured PBR buildings, full Mapbox style parity, and feature `005` bundle
load/render round-trip remain incomplete or later-owned and must continue to
be typed-diagnostic-bearing or honestly documented.

### Feature 005 Independent PRD Compliance Audit - 2026-05-18

This audit reviewed `specs/005-map-assets-bundles-p1` against the PRD,
constitution, feature spec/plan/tasks, requirements matrix, changed-file
summary, current implementation, support docs, and fresh local verification.

Fresh verification:

```powershell
python -m pytest tests\test_p1_prerequisite_contracts.py tests\test_p1_diagnostics_contract.py tests\test_p1_fixture_inventory.py tests\test_p1_docs_support_matrix.py tests\test_p1_bundle_guardrails.py tests\test_p1_red_yellow_remediation.py -q
```

Observed result: `18 passed in 1.18s`.

Selected regression verification:

```powershell
$env:PYTHONPATH='python'; python -m pytest tests\test_mapscene_recipe_contract.py tests\test_mapscene_validation.py tests\test_mapscene_support_status.py tests\test_mapscene_label_plan_integration.py tests\test_mapscene_render_png.py tests\test_mapscene_render_policy.py tests\test_mapscene_save_bundle.py -q
```

Observed result: `34 passed in 0.62s`.

Compile verification:

```powershell
python -m py_compile python\forge3d\diagnostics.py python\forge3d\map_scene.py python\forge3d\__init__.py tests\test_p1_prerequisite_contracts.py tests\test_p1_diagnostics_contract.py tests\test_p1_fixture_inventory.py tests\test_p1_docs_support_matrix.py tests\test_p1_bundle_guardrails.py tests\test_p1_red_yellow_remediation.py
```

Observed result: exit `0`.

Independent audit refresh on 2026-05-18 reran the full P1 and MapScene
guardrail evidence after T009-T069 were completed:

```powershell
$env:PYTHONPATH='python'; $p1 = rg --files tests | Where-Object { $_ -match 'test_p1_.*\.py$' }; python -m pytest @p1 -q
```

Observed result: `51 passed in 2.05s`.

```powershell
$env:PYTHONPATH='python'; $p1 = rg --files tests | Where-Object { $_ -match 'test_p1_.*\.py$' }; python -m pytest tests\test_mapscene_validation.py tests\test_mapscene_save_bundle.py tests\test_mapscene_render_png.py tests\test_mapscene_docs.py tests\test_mapscene_examples.py tests\test_mapscene_quickstart.py tests\test_mapscene_recipe_contract.py tests\test_mapscene_render_policy.py tests\test_mapscene_support_status.py tests\test_mapscene_label_plan_integration.py @p1 -q
```

Observed result: `101 passed in 4.38s` on the latest audit refresh.

Compile verification:

```powershell
$env:PYTHONPATH='python'; python -m py_compile python\forge3d\map_scene.py python\forge3d\diagnostics.py python\forge3d\__init__.py tests\test_p1_label_layer_crs_terrain.py tests\test_p1_label_expressions_plan_diagnostics.py tests\test_p1_typography_font_support.py tests\test_p1_building_workflow_support.py tests\test_p1_tiles3d_support.py tests\test_p1_bundle_roundtrip.py
```

Observed result: exit `0`.

Audit result: feature `005` P1 acceptance criteria are verified by concrete
test, source, docs, diagnostic, bundle, and matrix evidence. The previous
global P0 release-gate blocker for `P0-R1-AC2` and `P0-R1-AC3` is mitigated by
the 2026-05-18 label remediation and fresh audit reruns. The remaining blocker
is worktree hygiene: unrelated or unattributed changes must be separated before
commit/PR.

| ID | Gap | Evidence | Required fix | Status |
|---|---|---|---|---|
| R-030 | P1-R1 LabelLayer data ingestion is only partially tested. Geometry ingestion for P1-R1-AC1 is now tested, but CRS transform, terrain sampling, expression evaluation, deterministic LabelPlan output, and missing field/glyph validation are not verified. | Mitigated by `tests/test_p1_label_layer_crs_terrain.py`, `tests/test_p1_label_expressions_plan_diagnostics.py`, and `tests/test_p1_label_layer_geometry.py`; P1 command reported `51 passed`, guardrail command reported `101 passed`; matrix marks P1-R1-AC1 through P1-R1-AC6 `Verified`. | None for this blocker; keep line-label production limitations diagnostic-bearing through `experimental_feature`. | Mitigated |
| R-031 | P1-R2 typography/font handling is not verified beyond fixture existence and diagnostic factory scaffolding. | Mitigated by `tests/test_p1_typography_font_support.py`, `FontAtlas`, `FontFallbackRange`, `TypographySettings`, docs wording for complex-script shaping, P1 command `51 passed`, and matrix P1-R2-AC1 through P1-R2-AC7 `Verified`. | None for this blocker; do not claim complex-script shaping beyond documented non-blocking deferral. | Mitigated |
| R-032 | P1-R3 integrated building workflow is not verified for support-status classification, fixture render prep, geometry counts/bounds, scalar PBR docs, textured PBR unsupported diagnostics, or zero-geometry fallback diagnostics. | Mitigated by `tests/test_p1_building_workflow_support.py`, typed textured PBR `unsupported_feature`, scalar/textured PBR docs, P1 command `51 passed`, and matrix P1-R3-AC1 through P1-R3-AC6 `Verified`. | None for this blocker; public MapScene building render adapters remain diagnostic-bearing where unavailable. | Mitigated |
| R-033 | P1-R4 public 3D Tiles scene integration is not verified beyond constructor/API shape and docs wording. | Mitigated by `tests/test_p1_tiles3d_support.py`, cache/LOD metadata, unsupported format/feature diagnostics, no-Cesium docs audit, P1 command `51 passed`, and matrix P1-R4-AC1 through P1-R4-AC6 `Verified`. | None for this blocker; public 3D Tiles render parity remains diagnostic-bearing through `python_public_3dtiles_incomplete`. | Mitigated |
| R-034 | P1-R5 full bundle round-trip is not verified. | Mitigated by `tests/test_p1_bundle_roundtrip.py`, `tests/test_p1_bundle_guardrails.py`, P1 bundle `supported_export_settings`, available-asset load/render, missing-asset replay diagnostics, P1 command `51 passed`, and matrix P1-R5-AC1 through P1-R5-AC5 `Verified`. | None for this blocker; keep missing external assets structured and render-blocking. | Mitigated |
| R-035 | The requirements matrix is correct to keep P1 rows `Planned`; any feature-complete claim would contradict current state evidence. | Mitigated by updated matrix rows P1-R1-AC1 through P1-R5-AC5 all `Verified` with exact test evidence; tasks T009-T069 marked `[X]`; guardrail command reported `101 passed`. | None for this blocker. | Mitigated |
| R-036 | Diff contains unrelated or unattributed workspace changes that must be separated before commit/PR. | Mitigated by staging only the intended feature/blocker scope. `git diff --cached --name-status` now contains feature `005` specs/tests, feature `002` typography/declutter remediation, product API/docs/state updates, and `.specify`/`.gitignore` visibility changes. `git diff --name-status` remains limited to unstaged unrelated/generated/user-owned paths: `AGENTS.md`, deleted `examples/turkiye_river_basins_3d.py`, `logs/.3c2cf94182465f0d10df58878528dd39234a1134-audit.json`, `python/forge3d/forge3d.pdb`, and `src/viewer/event_loop/runner.rs`; `git status --short --untracked-files=all` also leaves `tests/test_mount_hood_white_cloud_timelapse.py` unstaged. | Preserve this staged/unstaged separation until commit/PR; do not commit the unstaged unrelated/generated paths unless separately requested. | Mitigated |
| R-037 | Strict PRD product completion was blocked by global P0 rows outside feature `005` P1 scope. | Mitigated by the 2026-05-18 label remediation. `docs/superpowers/state/requirements-verification-matrix.md` now marks `P0-R1-AC2` and `P0-R1-AC3` as `Verified`; fresh audit reruns reported full label command `28 passed`, focused Rust command `1 passed`, `cargo check -q` exit `0`, `cargo fmt -- --check` exit `0`, and Python `py_compile` exit `0`. | Preserve the R-036/R-011 staged/unstaged worktree separation before commit or PR; do not reopen R-037 unless a P0 row regresses to non-verified. | Mitigated |

## Feature 006 Independent Audit Note

On 2026-05-18, an independent PRD compliance audit rechecked
`specs/006-material-vt-large-scene-p2` against the PRD, constitution, feature
spec, plan, tasks, requirements matrix, changed-file summary, current
implementation, docs, and fresh local verification.

Fresh P2 verification:

```powershell
$env:PYTHONPATH='python'; python -m pytest tests\test_p2_diagnostics_contract.py tests\test_p2_vt_family_validation.py tests\test_p2_vt_docs.py tests\test_p2_textured_building_mapscene.py tests\test_p2_building_texture_diagnostics.py tests\test_p2_building_texture_docs.py tests\test_p2_advanced_labels_repeated_curved.py tests\test_p2_advanced_label_rules.py tests\test_p2_complex_shaping_decision.py tests\test_p2_large_scene_memory.py tests\test_p2_large_scene_cache_lod_instancing.py tests\test_p2_large_scene_bottlenecks.py tests\test_p2_large_scene_docs.py tests\test_p2_determinism_noop.py tests\test_p2_support_docs.py tests\test_p2_quickstart.py -q
```

Observed result: `35 passed in 0.97s`.

Compile verification:

```powershell
$env:PYTHONPATH='python'; python -m py_compile python\forge3d\label_plan.py python\forge3d\map_scene.py python\forge3d\terrain_params.py tests\test_p2_diagnostics_contract.py tests\test_p2_vt_family_validation.py tests\test_p2_textured_building_mapscene.py tests\test_p2_building_texture_diagnostics.py tests\test_p2_advanced_labels_repeated_curved.py tests\test_p2_advanced_label_rules.py tests\test_p2_complex_shaping_decision.py tests\test_p2_large_scene_memory.py tests\test_p2_large_scene_cache_lod_instancing.py tests\test_p2_large_scene_bottlenecks.py tests\test_p2_determinism_noop.py tests\test_p2_quickstart.py
```

Observed result: exit `0`.

Audit result: the P2 slice has focused test evidence for implemented or
diagnostic-deferred VT, textured building, advanced label, and large-scene
requirements. No feature `006` blocker remains in this file.

## Non-Goals To Keep Out Of Scope

- Web-first globe engine or hosted tile-provider ecosystem.
- Reimplementing Mapbox GL / Cesium-style live tile delivery as the main product goal.
- Blender-style general DCC software.
- Unreal-style open-world game/editor tooling.
- Full 3D Tiles / Cesium-grade global runtime parity in the first release.
- Full Mapbox Style Specification support in the first release.
- Full international text shaping in the first release.
- Non-map rendering features such as general animation, character rendering, non-geospatial modeling, game logic, simulation, or cinematic production features.
- Treating Pro-gated native features as missing when the assessment shows that they exist but are not public/open or not integrated.

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
| R-005 | Building ingestion may fall back to zero geometry when Pro/native paths are unavailable. | PRD evidence identifies fallback branches in `python/forge3d/buildings.py`. | Feature `001`/`005` must emit `pro_gated_path` and `placeholder_fallback` diagnostics. | Diagnostic contract verified in `001`; asset-path wiring remains open in `005`. |
| R-006 | 3D Tiles Rust substrate may be overclaimed as public Python product readiness. | Rust B3DM/tile renderer infrastructure exists, but Python scene integration is incomplete. | Feature `001`/`005` must distinguish Rust infrastructure, public Python API, supported formats, and unsupported formats. | Support matrix and diagnostic contract verified in `001`; public layer wiring remains open in `005`. |
| R-007 | VT normal/mask families may remain placeholders while runtime pages only albedo. | PRD classifies VT normal/mask runtime support as missing in current runtime. | Feature `001`/`006` must diagnose `vt_unsupported_family` until implemented and tested. | Diagnostic contract and VT support matrix verified in `001`; runtime behavior remains open in `006`. |
| R-008 | `MapScene` could become a thin wrapper that still requires raw IPC for canonical workflows. | Current users stitch together viewer, IPC, style, bundle, CRS, terrain, label, building, and point-cloud APIs. | Feature `004` must prove examples use typed recipe APIs without raw IPC for MVP workflows. | Mitigated in `004` by canonical examples/tests and native/offscreen PNG render evidence; raw IPC is not required. |
| R-009 | Bundle save may omit enough scene state to reproduce review context. | Existing bundle substrate exists, but typed map-scene round-trip is a new product contract. | Feature `004`/`005` must test bundle contents and round-trip behavior. | Partially mitigated in `004` for deterministic diagnostic review metadata and layer source payloads; full load/render round-trip remains planned in `005`. |
| R-010 | Docs may overclaim support by saying missing/underdeveloped paths are supported. | The PRD explicitly protects classifications such as underdeveloped, Pro-gated, placeholder/fallback, experimental, unsupported, and non-goal. | Feature `001` must establish support matrices and docs wording discipline before implementation claims. | Mitigated in `001` by docs audit tests and support matrices. |
| R-011 | Existing dirty worktree contains feature `001` work mixed with feature `002` label work and unrelated or not-yet-attributed paths on branch `002-label-api-truth`. | Completion-review fixes on 2026-05-16 previously addressed ignored `001` evidence. This Phase 3 session found broad `.gitignore` rules still hid the required `002` state/task artifacts, then added narrow exceptions for `docs/superpowers/state/*.md` and `specs/002-label-api-truth/tasks.md`; targeted status now shows those artifacts as Git-visible untracked files. Broader `git status` still shows feature `001` diagnostics/support files, feature `002` label changes, modified docs/logs/PDB, and other paths. | Before any PR or commit, separate intentional feature `001` diagnostics/support-matrix changes from feature `002` Phase 1-3 changes and unrelated/user-owned workspace changes; do not revert user changes blindly. | Open; required `002` state/task evidence visibility mitigated, broader change-separation still open. |
| R-012 | P2 VT normal/mask requests may be accepted but ignored by the native runtime. | Feature `006` inventory found Python `VTLayerFamily` accepts `normal` and `mask`, while `src/terrain/renderer/virtual_texture.rs` pages only `albedo`. | Feature `006` must implement native normal/mask runtime support or emit `vt_unsupported_family` before render. | Open |
| R-013 | Textured building material support may fall back to scalar PBR without diagnostics. | Feature `006` inventory found scalar `BuildingMaterial` and general texture helpers, but no building texture path/UV diagnostics workflow. | Feature `006` must add real textured building support or diagnostics for missing texture path, missing UVs, unsupported format, and fallback material behavior. | Open |
| R-014 | Large-scene stats may remain fragmented across lower-level modules. | Feature `006` inventory found point-cloud, 3D Tiles, terrain scatter, material VT, and global memory stats, but no unified product `ValidationReport`. | Feature `006` must normalize available stats or emit `unavailable_cache_lod_stats`/`unsupported_instancing_path` diagnostics. | Open |
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

Remaining gaps have narrowed: R-011 broader worktree/change separation remains
open before commit or PR, while R-020 is mitigated by the later feature `003`
LabelPlan guide and feature `004` MapScene quickstart evidence.

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
| R-028 | Prior P0 label rows used diagnostic deferral for criteria that still require implementation or a human release decision. | `P0-R1-AC2` requires typography mutation plus visible/layout effect; `P0-R1-AC3` requires real declutter behavior or explicit unsupported typed error. Current feature `002` evidence prevents no-op success with `experimental_feature` diagnostics but does not satisfy those exact acceptance criteria. | Implement native behavior, change the diagnostic to an explicit unsupported typed error where the AC permits it, or record a human product decision that diagnostic-only behavior is acceptable for MVP. | Open; matrix marks P0-R1-AC2 and P0-R1-AC3 as `Blocked` |
| R-029 | P0-R4-AC2 PNG output was only partially verified because prior evidence proved deterministic source-derived PNG files, not a native/offscreen terrain/raster/vector render adapter. | `specs/004-mapscene-mvp/plan.md` requires wrapping native `Scene.render_png` / `render_rgba` where available and says the render path should prefer native/offscreen PNG. The prior `python/forge3d/map_scene.py` path always called `_render_source_derived_rgba(...)`. | Post-audit remediation added `_render_native_offscreen_rgba(...)`, fixture-backed `.npy` terrain and PNG raster loading, native `Scene.set_height_from_r32f`, `set_raster_overlay`, `set_camera_look_at`, instance `render_rgba()`, vector/label compositing, canonical examples with generated real assets, and `tests/test_mapscene_render_png.py::test_render_uses_native_offscreen_for_real_terrain_and_raster_assets`, which monkeypatches the source-derived fallback to fail and asserts `scene.last_render_backend == "native/offscreen"`. | Mitigated in `004`; focused native regression initially failed, focused render/docs/examples command reported `23 passed`, full MapScene command reported `48 passed`, and P0-R4-AC2 remains `Verified`. |

Incomplete line/curved labels, buildings, 3D Tiles, VT normal/mask, and style
support are not accepted as complete. Current evidence shows several of these
paths have typed diagnostics and support-matrix wording from features `001`,
`002`, and feature `004` validation, render, bundle, example, and quickstart
tests. Feature `004` now has full docs/support audit coverage and a current
full MapScene command reporting `48 passed`. The strict audit's R-029
native/offscreen render-evidence gap is mitigated. Remaining P0 blocker scope
is R-028 for P0-R1-AC2/P0-R1-AC3 label behavior or a documented human decision.

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
- Repo-wide MVP release readiness also depends on R-028 (`P0-R1-AC2` and
  `P0-R1-AC3`) or an explicit human product decision, plus R-011 worktree/change
  separation before commit or PR.

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

# Implementation Ledger

Source PRD: `docs/superpowers/plans/prd.md`  
Constitution: `.specify/memory/constitution.md` v1.0.3  
SpecKit playbook: `docs/superpowers/plans/speckit-playbook.md`  
Current feature: `004-mapscene-mvp` R-029 remediation complete; P0-R4-AC2 native/offscreen render output verified
Current SpecKit feature directory: `specs/004-mapscene-mvp` until the next feature handoff updates the active pointer
Current git branch: `004-mapscene-mvp`
Last known commit: `e26d1b9`  
Last updated: 2026-05-18

## 2026-05-18 Feature 004 Independent PRD Compliance Audit

Target feature: `specs/004-mapscene-mvp`.

This was an audit-only session. No product code was changed. The audit reviewed
only feature `004` against the PRD, constitution, feature spec/plan/tasks,
requirements matrix, changed-file summary, current implementation, support
docs, and fresh local verification.

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
P0-R4-AC8 remain verified by concrete file/test/doc evidence. Remaining scope
boundaries are later-owned or diagnostic-bearing rather than feature `004`
blockers: point-cloud and supported-building render adapters, public 3D Tiles
rendering, VT normal/mask runtime, textured PBR buildings, full Mapbox style
parity, and feature `005` bundle load/render round trip.

## 2026-05-17 Feature 004 Native/Offscreen Render Evidence Remediation

Target feature: `specs/004-mapscene-mvp`.

Selected finding: R-029 from the independent `004-mapscene-mvp` PRD compliance
audit.

This was a narrow post-audit fix. The prior T026-T029 source-derived adapter
remains as deterministic compatibility output for symbolic fixture recipes, but
it is no longer the evidence used to satisfy P0-R4-AC2. The fix wires
fixture-backed supported terrain/raster/vector/label recipes through the native
`Scene.render_rgba()` path and updates examples/docs/state accordingly.

PRD acceptance criteria covered: `P0-R4-AC2`, `P0-R4-AC7`, `P0-R4-AC8`, and
Constitution III/IV/V/VIII/X.

Red verification before implementation:

```powershell
python -m pytest tests/test_mapscene_render_png.py::test_render_uses_native_offscreen_for_real_terrain_and_raster_assets -q
```

Observed red result: `1 failed`. The failure showed `MapScene.render()` still
called `_render_source_derived_rgba(...)`; the test monkeypatched that function
to raise, so no native/offscreen evidence existed for real fixture-backed
terrain/raster/vector/label inputs.

Implemented:

- `python/forge3d/map_scene.py`: added `_render_native_offscreen_rgba(...)` and
  related loaders. The native path creates a native `Scene`, uploads `.npy`
  height data with `set_height_from_r32f`, uploads PNG raster overlays with
  `set_raster_overlay`, applies orbit camera settings with
  `set_camera_look_at`, calls the instance `render_rgba()` method, composites
  inline vector/label layers, records `last_render_backend`, and stores that
  backend in review-bundle metadata.
- `tests/test_mapscene_render_png.py`: added the R-029 regression test using
  real generated `.npy` terrain, PNG raster, inline vector data, a label layer,
  a fixed reproducibility profile, and a monkeypatched source-derived fallback.
- `examples/mapscene_terrain_raster.py` and
  `examples/mapscene_vector_labels.py`: generate local fixture assets and
  return `render_backend`; example/quickstart tests assert `native/offscreen`.
- `docs/guides/offline_3d_map_rendering.md`,
  `docs/api/api_reference.rst`,
  `specs/004-mapscene-mvp/quickstart.md`, and
  `specs/004-mapscene-mvp/contracts/mapscene-contract.md`: distinguish
  native/offscreen fixture-backed output from source-derived compatibility
  output.
- `docs/superpowers/state/requirements-verification-matrix.md`,
  `docs/superpowers/state/implementation-ledger.md`,
  `docs/superpowers/state/current-context-pack.md`, and
  `docs/superpowers/state/open-blockers.md`: update R-029 and P0-R4-AC2 state.

Selected verification commands after implementation:

```powershell
python -m pytest tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_docs.py -q
python -m pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q
python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_render_png.py tests\test_mapscene_examples.py tests\test_mapscene_quickstart.py tests\test_mapscene_docs.py examples\mapscene_terrain_raster.py examples\mapscene_vector_labels.py examples\mapscene_buildings_labels.py
```

Observed green results: focused render/docs/examples command `23 passed in
0.98s`; full MapScene command `48 passed in 1.36s`; `py_compile` exited 0.

Requirement status impact: R-029 moves from `Open` to `Mitigated in 004`.
`P0-R4-AC2` remains `Verified`, now with native/offscreen fixture-backed render
evidence instead of source-derived-only evidence. `P0-R1-AC2` and
`P0-R1-AC3` remain blocked pending native label behavior or a human product
decision.

## 2026-05-17 Feature 004 Source-Derived PNG Render Completion

Target feature: `specs/004-mapscene-mvp`.

Selected task IDs: T026, T027, T028, and T029.

This was one coherent RED/YELLOW remediation batch: T026 converted the blocked
P0-R4-AC2 finding into source-derived PNG tests, T027 implemented the
source-derived render adapter, T028 updated public docs/examples/contract
wording, and T029 updated durable state. The batch is deliberately limited to
the active feature `004` render truth gap and does not mix in the remaining
feature `002` label blockers.

PRD acceptance criteria covered: `P0-R4-AC2`, `P0-R4-AC7`, `P0-R4-AC8`, and
Constitution III/IV/V/VII/VIII/X.

Red verification before implementation:

```powershell
$env:PYTHONPATH='python'; python -m pytest tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_docs.py -q -p no:cacheprovider
```

Observed red result: `8 failed, 14 passed`. Failures showed the T022-T025
blocked backend still failed the source-derived PNG acceptance path, supported
examples did not produce PNG artifacts, and docs/API/quickstart wording was
still stale.

Implemented:

- `python/forge3d/map_scene.py`: replaced the global render-backend
  `placeholder_fallback` block with `_render_source_derived_rgba(...)` for
  supported non-blocked MVP scenes, writes deterministic PNG output, records
  `mapscene.render_backend` and `mapscene.render_png` as supported after a
  successful write, and keeps layer-specific typed diagnostics for unsupported
  vector path-loader, point-cloud, building, and other fallback paths.
- `tests/test_mapscene_render_png.py`,
  `tests/test_mapscene_render_policy.py`,
  `tests/test_mapscene_examples.py`,
  `tests/test_mapscene_quickstart.py`, and
  `tests/test_mapscene_docs.py`: updated render, policy, example, quickstart,
  and docs assertions for source-derived PNG output.
- `docs/guides/offline_3d_map_rendering.md`,
  `docs/api/api_reference.rst`,
  `specs/004-mapscene-mvp/quickstart.md`, and
  `specs/004-mapscene-mvp/contracts/mapscene-contract.md`: reworded the public
  contract from blocked placeholder output to supported deterministic
  source-derived PNG output for non-blocked MVP scenes.
- `specs/004-mapscene-mvp/tasks.md`,
  `docs/superpowers/state/requirements-verification-matrix.md`,
  `docs/superpowers/state/implementation-ledger.md`,
  `docs/superpowers/state/current-context-pack.md`, and
  `docs/superpowers/state/open-blockers.md`: recorded T026-T029 evidence,
  P0-R4-AC2 verified status, R-026 mitigation, remaining P0 label blockers, and
  the next exact prompt.

Selected verification commands after implementation:

```powershell
$env:PYTHONPATH='python'; python -m pytest tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_docs.py -q -p no:cacheprovider
$env:PYTHONPATH='python'; python -m pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q -p no:cacheprovider
$env:PYTHONPATH='python'; python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_render_png.py tests\test_mapscene_examples.py tests\test_mapscene_quickstart.py tests\test_mapscene_docs.py examples\mapscene_terrain_raster.py examples\mapscene_vector_labels.py
```

Observed green results: focused command `22 passed in 0.56s`; full MapScene
command `47 passed in 0.86s`; `py_compile` exited 0.

Requirement status impact: `P0-R4-AC2` moves from `Blocked` to `Verified`.
R-026 is mitigated for supported MVP terrain/raster/vector/label recipes.
`P0-R4-AC1` through `P0-R4-AC8` are now all `Verified` for feature `004`.
`P0-R1-AC2` and `P0-R1-AC3` remain blocked pending native label behavior or a
human product decision.

## 2026-05-17 Feature 004 RED/YELLOW Remediation Checkpoint

Target feature: `specs/004-mapscene-mvp`.

Selected task IDs: T022, T023, T024, and T025.

This was one coherent remediation batch: T022 added render-truth and
renderability RED tests, T023 implemented typed diagnostics and blocked the
synthetic render path, T024 updated docs/API/quickstart/contract wording, and
T025 updated durable state. The batch deliberately did not treat the
recipe-sensitive compositor as real PNG render evidence.

PRD acceptance criteria covered: `P0-R4-AC2`, `P0-R4-AC3`, `P0-R4-AC7`,
`P0-R4-AC8`, and Constitution III/IV/V/VII/VIII/X.

Red verification before implementation:

```powershell
$env:PYTHONPATH='python'; python -m pytest tests/test_mapscene_render_png.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_save_bundle.py -q -p no:cacheprovider
```

Observed red result: `11 failed, 22 passed`. Failures showed
`MapScene.render()` still wrote synthetic PNG output, vector path-only layers,
point-cloud layers, and supported building layers lacked renderability
diagnostics, examples claimed render success, and bundles marked synthetic
scenes renderable.

Docs red verification:

```powershell
$env:PYTHONPATH='python'; python -m pytest tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q -p no:cacheprovider
```

Observed red result after adding the docs assertion: `1 failed, 10 passed`.

Implemented:

- `python/forge3d/map_scene.py`: added `placeholder_fallback` diagnostics for
  `mapscene.render_backend`, vector path loader, point-cloud MapScene render
  path, and supported building render adapter. `MapScene.render()` now blocks
  before writing synthetic PNG output. `MapScene.save_bundle()` stays
  deterministic but records non-renderable review status when diagnostics block.
- `tests/test_mapscene_render_png.py`,
  `tests/test_mapscene_validation.py`,
  `tests/test_mapscene_support_status.py`,
  `tests/test_mapscene_render_policy.py`,
  `tests/test_mapscene_save_bundle.py`,
  `tests/test_mapscene_examples.py`,
  `tests/test_mapscene_quickstart.py`, and
  `tests/test_mapscene_docs.py`: updated tests for honest blocked render,
  layer diagnostics, examples, quickstart, bundle renderability, and docs.
- `examples/mapscene_terrain_raster.py` and
  `examples/mapscene_vector_labels.py`: now return `blocked_by_diagnostics`
  when render is blocked instead of claiming PNG success.
- `docs/guides/offline_3d_map_rendering.md`,
  `docs/api/api_reference.rst`,
  `specs/004-mapscene-mvp/quickstart.md`, and
  `specs/004-mapscene-mvp/contracts/mapscene-contract.md`: reworded render
  support as current `placeholder/fallback` backend and bundle scope as review
  metadata plus diagnostics.
- `docs/superpowers/plans/2026-05-17-mapscene-red-yellow-remediation.md`:
  saved the remediation plan used for this batch.
- `specs/004-mapscene-mvp/tasks.md`,
  `docs/superpowers/state/requirements-verification-matrix.md`,
  `docs/superpowers/state/implementation-ledger.md`,
  `docs/superpowers/state/current-context-pack.md`, and
  `docs/superpowers/state/open-blockers.md`: recorded revised evidence and
  current blocker state.

Selected verification commands after implementation:

```powershell
$env:PYTHONPATH='python'; python -m pytest tests/test_mapscene_render_png.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_save_bundle.py -q -p no:cacheprovider
$env:PYTHONPATH='python'; python -m pytest tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q -p no:cacheprovider
$env:PYTHONPATH='python'; python -m pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q -p no:cacheprovider
$env:PYTHONPATH='python'; python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_validation.py tests\test_mapscene_support_status.py tests\test_mapscene_render_png.py tests\test_mapscene_render_policy.py tests\test_mapscene_save_bundle.py tests\test_mapscene_examples.py tests\test_mapscene_quickstart.py examples\mapscene_terrain_raster.py examples\mapscene_vector_labels.py examples\mapscene_buildings_labels.py
```

Observed green results: focused command `33 passed`; docs/quickstart command
`11 passed`; full MapScene command `47 passed in 0.84s`; `py_compile` exited 0.

Requirement status impact: `P0-R4-AC2` moves from `Verified` to `Blocked`
because no source-derived PNG render adapter is proven. `P0-R4-AC3` remains
`Verified` for deterministic diagnostic review bundles, not full load/render
round-trip. `P0-R4-AC7` and `P0-R4-AC8` have stronger no-op and support-honesty
evidence. `P0-R1-AC2` and `P0-R1-AC3` remain blocked pending native label
behavior or a human product decision.

## 2026-05-17 Feature 004 Full MVP Verification And Closure Checkpoint

Target feature: `specs/004-mapscene-mvp`.

Selected task IDs: T019, T020, and T021.

This was one coherent RED-only closure batch: T019 provides the full feature
verification gate, T020 updates the PRD acceptance matrix from that exact
evidence, and T021 updates continuity state plus the next prompt. No additional
implementation tasks were started after the full verification passed.

PRD acceptance criteria covered: `P0-R4-AC1` through `P0-R4-AC8`, Section 21,
and Constitution VII/X.

Selected verification command:

```powershell
$env:PYTHONPATH='python'; python -m pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q -p no:cacheprovider
```

Observed result: `44 passed in 0.84s`.

Implemented:

- `specs/004-mapscene-mvp/tasks.md`: marked T019, T020, and T021 complete with
  the exact full-feature verification evidence.
- `docs/superpowers/state/requirements-verification-matrix.md`: added the
  T019-T021 evidence row and moved P0-R4-AC1 through P0-R4-AC8 from `Tested`
  to `Verified`.
- `docs/superpowers/state/implementation-ledger.md` and
  `docs/superpowers/state/current-context-pack.md`: recorded the closure
  evidence, residual blockers, worktree risk, and next exact prompt.

Requirement status impact at that historical checkpoint: feature
`004-mapscene-mvp` P0-R4 rows were marked `Verified`. T022-T025 later blocked
the synthetic P0-R4-AC2 evidence, and T026-T029 then superseded that blocker
with source-derived PNG render evidence. Current P0-R4-AC1 through P0-R4-AC8
status is `Verified`; P1/P2 rows remain planned for later feature directories.
`P0-R1-AC2` and `P0-R1-AC3` remain blocked pending native behavior or a human
product decision; this closure does not alter those release-gate blockers.

Residual risk: R-011 dirty worktree/change separation remains open before any
commit or PR. `.specify/feature.json` still points to `specs/004-mapscene-mvp`
at closure time, so the next feature handoff must update or confirm the active
feature context before executing feature `005`.

## 2026-05-17 Feature 004 Full Docs/Support Audit Checkpoint

Target feature: `specs/004-mapscene-mvp`.

Selected task IDs: T016 and T017.

This was one coherent RED-only batch: T016 adds the full MVP docs/support audit
coverage and T017 updates the offline MapScene guide plus support matrices to
satisfy that audit. It closes the R-027 docs/support gap without starting T019
full feature verification or final T020/T021 closure.

PRD acceptance criteria covered: `P0-R4-AC8`, `P0-R4-AC7`, scoped
`P0-R4-AC1`, Section 16, FR-021, and FR-022.

Red verification before implementation:

```powershell
python -m pytest tests/test_mapscene_docs.py -q -p no:cacheprovider
```

Observed red result after correcting the audit parser: `4 failed, 3 passed`.
Failures showed the offline MapScene guide lacked canonical example and support
guide anchors, style/building/3D Tiles/VT support matrices had stale or missing
current MapScene diagnostic/ownership wording, competitive positioning used
umbrella support wording, and required non-goal boundary rows were absent.

Implemented:

- `tests/test_mapscene_docs.py`: added broad docs/support audit coverage for
  canonical examples, guide links, support taxonomy, current MapScene
  diagnostics/ownership, and competitive-positioning boundaries.
- `docs/guides/offline_3d_map_rendering.md`: added canonical examples,
  support references, and the selected example/quickstart verification command.
- `docs/guides/style_support_matrix.md`,
  `docs/guides/building_support_matrix.md`,
  `docs/guides/tiles3d_support_matrix.md`,
  `docs/guides/virtual_texturing_support_matrix.md`, and
  `docs/guides/competitive_positioning.md`: updated current support rows and
  overclaim boundaries without presenting P1/P2 gaps as supported.
- `specs/004-mapscene-mvp/tasks.md`,
  `docs/superpowers/state/requirements-verification-matrix.md`,
  `docs/superpowers/state/implementation-ledger.md`, and
  `docs/superpowers/state/current-context-pack.md`: recorded this docs/support
  checkpoint.

Selected verification commands after implementation:

```powershell
python -m pytest tests/test_mapscene_docs.py -q -p no:cacheprovider
python -m pytest tests/test_mapscene_docs.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q -p no:cacheprovider
python -m pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_docs.py -q -p no:cacheprovider
python -m py_compile tests\test_mapscene_docs.py
rg -n "T016|T017|MVP docs/support audit|Next Exact Prompt|P0-R4-AC7|P0-R4-AC8" docs/superpowers/state/requirements-verification-matrix.md docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md specs/004-mapscene-mvp/tasks.md
```

Observed green results: focused docs command `7 passed`;
docs/examples/quickstart command `15 passed`; validation/support/docs command
`23 passed`; `py_compile` exited 0; final state/evidence `rg` exited 0.

Requirement status impact at that checkpoint: `P0-R4-AC7` and `P0-R4-AC8` had full
docs/support audit evidence for feature `004`, and scoped `P0-R4-AC1`
documentation evidence was stronger. No feature `004` `Verified` status was
claimed then because T019/T020/T021 were still open.

## 2026-05-17 Feature 004 Terrain CRS / Building Source Diagnostics Checkpoint

Target feature: `specs/004-mapscene-mvp`.

Selected task IDs: T006D, T006E, T016B, T017B, T020T, and T021T.

This was one coherent RED-only batch: all selected tasks close validation and
documentation truth gaps where missing terrain CRS and supported building source
asset failures could produce a successful validation report or remain absent
from the diagnostics reference. The batch does not start T019 full verification
or the remaining broad T016/T017 docs/support audit.

PRD acceptance criteria covered: `P0-R4-AC1`, `P0-R4-AC5`, `P0-R4-AC7`,
scoped `P0-R4-AC8`, Section 21, and Constitution VII/X.

Red verification before implementation:

```powershell
python -m pytest tests/test_mapscene_validation.py tests/test_mapscene_docs.py -q -p no:cacheprovider
```

Observed red result: `3 failed, 12 passed`. Failures showed terrain with
missing CRS metadata and supported building layers with missing/unsupported
source assets still returned `ok`, and `docs/guides/diagnostics_reference.md`
did not document the scoped MapScene diagnostic inventory.

Implemented:

- `python/forge3d/map_scene.py`: added terrain `missing_crs` validation and
  supported-building source path checks for `missing_external_asset` and
  `unsupported_asset_format`.
- `tests/test_mapscene_validation.py`: added terrain missing-CRS and supported
  building-source failure regressions.
- `tests/test_mapscene_docs.py` and `docs/guides/diagnostics_reference.md`:
  added and satisfied a scoped diagnostic inventory audit.
- `specs/004-mapscene-mvp/tasks.md`,
  `docs/superpowers/state/requirements-verification-matrix.md`,
  `docs/superpowers/state/implementation-ledger.md`, and
  `docs/superpowers/state/current-context-pack.md`: recorded this scoped
  checkpoint.

Selected verification commands after implementation:

```powershell
python -m pytest tests/test_mapscene_validation.py tests/test_mapscene_docs.py -q -p no:cacheprovider
python -m pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_docs.py -q -p no:cacheprovider
python -m pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_save_bundle.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q -p no:cacheprovider
python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_validation.py tests\test_mapscene_docs.py
rg -n "T006D|T006E|T016B|T017B|T020T|T021T|missing_crs|missing_external_asset|unsupported_asset_format|Next Exact Prompt" specs/004-mapscene-mvp/tasks.md docs/superpowers/state/requirements-verification-matrix.md docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md
```

Observed green results: validation/docs command `15 passed`;
validation/support/docs command `19 passed`; combined selected regression
`26 passed`; `py_compile` exited 0; final state/evidence `rg` exited 0.

Requirement status impact at that checkpoint: `P0-R4-AC1`, `P0-R4-AC5`,
`P0-R4-AC7`, and scoped `P0-R4-AC8` received stronger no-op-prevention and docs
evidence. No feature `004` `Verified` status was claimed because final
verification and closure tasks remained open.

## 2026-05-17 Feature 004 External Asset Diagnostics Checkpoint

Target feature: `specs/004-mapscene-mvp`.

Selected task IDs: T006B, T006C, T020S, and T021S.

This was one coherent RED-only batch: all selected tasks close the no-op success
risk where non-empty terrain, raster, and point-cloud paths could be treated as
renderable recipe inputs even when the local asset was missing or the suffix was
outside the MVP-supported set. The batch also updates continuity evidence for
that same narrow asset-diagnostics fix. It does not start T016/T017 full
docs/support audit, T019 full feature verification, T020 final matrix closure,
or T021 final continuity closure.

PRD acceptance criteria covered: `P0-R4-AC1`, `P0-R4-AC3`, `P0-R4-AC7`, Section
21 bundle/diagnostic behavior, and Constitution VII/X. `P0-R1-AC2` and
`P0-R1-AC3` remain `Blocked`; this checkpoint does not alter their release-gate
state.

Red verification before implementation:

```powershell
pytest tests/test_mapscene_validation.py tests/test_mapscene_save_bundle.py -q
```

Observed red result: `3 failed, 10 passed`. Failures showed
`MapScene.validate()` returned `ok` for missing external assets and unsupported
asset formats, and `MapScene.save_bundle()` returned `ok`/renderable for a
missing-asset scene.

Implemented:

- `python/forge3d/map_scene.py`: added typed `missing_external_asset` and
  `unsupported_asset_format` diagnostics for terrain, raster, and point-cloud
  paths; blocked affected layer summaries as unsupported; preserved blocked
  bundle renderability behavior through existing validation-report persistence.
- `tests/test_mapscene_validation.py`: added missing external asset and
  unsupported asset format red tests; made symbolic fixture paths explicit with
  `asset_status: fixture`.
- `tests/test_mapscene_save_bundle.py`: added missing-asset bundle persistence
  regression; made symbolic fixture paths explicit with `asset_status: fixture`.
- `tests/test_mapscene_support_status.py`,
  `tests/test_mapscene_label_plan_integration.py`,
  `tests/test_mapscene_render_png.py`, `tests/test_mapscene_render_policy.py`,
  `tests/test_mapscene_quickstart.py`, and canonical MapScene examples: marked
  symbolic fixture paths as intentional fixtures so strict local-asset checks do
  not mask the behavior under test.
- `specs/004-mapscene-mvp/tasks.md`,
  `docs/superpowers/state/requirements-verification-matrix.md`,
  `docs/superpowers/state/implementation-ledger.md`, and
  `docs/superpowers/state/current-context-pack.md`: recorded this scoped
  red-remediation checkpoint.

Selected verification commands after implementation:

```powershell
pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_save_bundle.py -q
pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q
python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_validation.py tests\test_mapscene_save_bundle.py
rg -n "T006B|T006C|T020S|T021S|missing_external_asset|unsupported_asset_format|Next Exact Prompt|P0-R1-AC2|P0-R1-AC3" specs/004-mapscene-mvp/tasks.md docs/superpowers/state/requirements-verification-matrix.md docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md docs/superpowers/state/open-blockers.md
```

Observed green results: focused validation/support/bundle command `17 passed`;
combined selected MapScene command `37 passed`; `py_compile` exited 0; final
state/evidence `rg` exited 0.

Requirement status impact at that checkpoint: `P0-R4-AC1`, `P0-R4-AC3`, and `P0-R4-AC7` received
stronger no-op-prevention and bundle-diagnostic evidence. `P1-R5-AC3` remains
`Planned` for feature `005` because full missing-asset bundle round-trip
coverage is outside that selected RED batch. No feature `004` `Verified` status
was claimed then because T016/T017/T019/T020/T021 were still open.

## 2026-05-17 Feature 004 Red-Only Remediation Checkpoint

Target feature: `specs/004-mapscene-mvp`.

Selected task IDs: T011A, T011B, T013A, T013B, T014, T015, T018,
T020R, and T021R.

This was one coherent RED-only batch: all selected tasks close evidence gaps
from the prior analysis that prevented honest P0-R4 release claims without
starting the remaining broad docs/support or final verification tasks. The
batch covers recipe-sensitive PNG output, review-bundle source metadata,
canonical typed examples, quickstart execution, and release-gate continuity
honesty.

PRD acceptance criteria covered: `P0-R4-AC2`, `P0-R4-AC3`, `P0-R4-AC8`,
Section 16, Section 21, and Constitution III/IV/VII/X. `P0-R1-AC2` and
`P0-R1-AC3` were reclassified as `Blocked` in the matrix because their current
diagnostic-only behavior prevents no-op success but does not satisfy the exact
acceptance criteria without implementation or a human product decision.

No additional task IDs were started. Original T016/T017, T019, T020, and T021
remain open for the full MVP docs/support audit, full feature verification,
final matrix update, and final continuity closure.

Red verification before implementation:

```powershell
pytest tests/test_mapscene_render_png.py -q
pytest tests/test_mapscene_save_bundle.py -q
pytest tests/test_mapscene_examples.py -q
pytest tests/test_mapscene_quickstart.py -q
```

Observed red results: render PNG `1 failed, 2 passed` because different
supported recipe data produced byte-identical PNG bytes; bundle `1 failed,
1 passed` because `scene/layer_sources/terrain.json` was missing; examples
`4 failed` because canonical examples were absent; quickstart `3 failed,
1 passed` because quickstart/example execution coverage was absent.

Implemented:

- `python/forge3d/map_scene.py`: recipe-sensitive deterministic RGBA rendering
  and non-label layer source metadata payloads for MapScene bundles.
- `tests/test_mapscene_render_png.py`: recipe-sensitive render regression.
- `tests/test_mapscene_save_bundle.py`: terrain, raster, vector, label, and
  point-cloud source metadata assertions.
- `tests/test_mapscene_examples.py` and `tests/test_mapscene_quickstart.py`:
  canonical example and quickstart execution coverage.
- `examples/mapscene_terrain_raster.py`,
  `examples/mapscene_vector_labels.py`, and
  `examples/mapscene_buildings_labels.py`: typed MapScene workflows with no raw
  viewer IPC.
- `.gitignore`: narrow exceptions so the selected MapScene examples are
  Git-visible despite the repo-wide `examples/*` ignore rule.
- `specs/004-mapscene-mvp/quickstart.md` and
  `specs/004-mapscene-mvp/tasks.md`: targeted quickstart commands and scoped
  red-remediation task evidence.
- `docs/superpowers/state/requirements-verification-matrix.md`,
  `docs/superpowers/state/open-blockers.md`,
  `docs/superpowers/state/implementation-ledger.md`, and
  `docs/superpowers/state/current-context-pack.md`: scoped matrix, blocker,
  ledger, and continuity updates.

Selected verification commands after implementation:

```powershell
pytest tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py -q
pytest tests/test_mapscene_save_bundle.py -q
pytest tests/test_mapscene_examples.py -q
pytest tests/test_mapscene_quickstart.py -q
pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_save_bundle.py tests/test_mapscene_examples.py tests/test_mapscene_docs.py tests/test_mapscene_quickstart.py -q
python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_render_png.py tests\test_mapscene_save_bundle.py tests\test_mapscene_examples.py tests\test_mapscene_quickstart.py examples\mapscene_terrain_raster.py examples\mapscene_vector_labels.py examples\mapscene_buildings_labels.py
rg -n "viewer_ipc|send_ipc" examples/mapscene_terrain_raster.py examples/mapscene_vector_labels.py examples/mapscene_buildings_labels.py
git check-ignore -v examples/mapscene_terrain_raster.py examples/mapscene_vector_labels.py examples/mapscene_buildings_labels.py
```

Observed green results: render/render-policy `7 passed`; bundle `2 passed`;
examples `4 passed`; quickstart `4 passed`; combined selected MapScene command
`34 passed`; `py_compile` exited 0; raw IPC `rg` returned no matches; `git
status --untracked-files=all` lists the three MapScene examples as untracked
after the `.gitignore` exceptions.

Requirement status impact at that checkpoint: `P0-R4-AC8` moved to `Tested`;
`P0-R4-AC2` and `P0-R4-AC3` received stronger recipe-sensitive/renderable-data
and source metadata evidence. No feature `004` `Verified` status was claimed
then because T016/T017/T019/T020/T021 were still open.

## 2026-05-17 Feature 004 Red/Yellow Remediation Batch

Target feature: `specs/004-mapscene-mvp`.

Selected task IDs: T006A, T009, T010, T011, T012, T013, T016A, and T017A.
This is one coherent red/yellow remediation batch: it closes the analyzed
source-data validation gap, wires the currently public render and bundle methods
so they produce real artifacts or diagnostics, and removes stale public API/docs
wording without starting canonical examples or quickstart tasks.

The broader original T016/T017 docs/support tasks remained open at this
checkpoint because canonical examples and quickstart evidence were still
planned. Later T014/T015/T018 completed the examples and quickstart; T016/T017
still own the full docs/support audit. Recording scoped T016A/T017A avoids
claiming full AC8 docs/example closure for this earlier batch.

Optional before-implement hook `/speckit.git.commit` was identified and not
executed because it is optional and the worktree already contains unrelated or
pre-existing dirty files.

PRD acceptance criteria covered in this session: `P0-R4-AC1`, `P0-R4-AC2`,
`P0-R4-AC3`, `P0-R4-AC4`, and `P0-R4-AC7`. At this checkpoint `P0-R4-AC8`,
canonical examples, quickstart, full docs/support audit, final
matrix/continuity closure, and MVP `Verified` status were still planned; later
T014/T015/T018 moved AC8 to `Tested`.

Red command:

```powershell
pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_save_bundle.py tests/test_mapscene_docs.py -q
```

Observed red result: `12 failed, 10 passed`. Failures covered missing
source-identity/renderable-data diagnostics, unwired `MapScene.render()`,
unwired `MapScene.save_bundle()`, missing `forge3d.map_scene` API docs, and
stale feature-004 `TBD` wording.

Implemented:

- `python/forge3d/map_scene.py`: added `missing_source_identity` and
  `missing_renderable_data` diagnostics, deterministic PNG rendering through
  `python/forge3d/helpers/offscreen.py`, render warning/error/fatal policy
  enforcement, deterministic MapScene review-bundle writing, compiled label plan
  and label source reference persistence, and blocked-bundle renderability
  metadata.
- `tests/test_mapscene_validation.py`: added source identity and renderable-data
  diagnostics tests.
- `tests/test_mapscene_render_png.py`: added deterministic PNG render tests.
- `tests/test_mapscene_render_policy.py`: added render policy/no-op tests.
- `tests/test_mapscene_save_bundle.py`: added deterministic review-bundle tests.
- `tests/test_mapscene_docs.py`: added scoped MapScene API/docs drift tests.
- `tests/test_mapscene_support_status.py`: updated the blocking render/save guard
  expectations to the implemented behavior.
- `docs/guides/offline_3d_map_rendering.md`,
  `docs/api/api_reference.rst`, `specs/004-mapscene-mvp/plan.md`, and
  `specs/004-mapscene-mvp/contracts/mapscene-contract.md`: documented the
  selected public `forge3d.map_scene` path and removed stale later-owned/TBD
  wording for implemented MapScene API pieces.
- `specs/004-mapscene-mvp/tasks.md`: marked T009-T013 complete and recorded
  scoped remediation subtasks T006A/T016A/T017A; T014/T015/T016/T017/T018-T021
  remain open.

Green commands:

```powershell
pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_save_bundle.py tests/test_mapscene_docs.py -q
```

Result: `22 passed`.

```powershell
pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_label_plan_integration.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_render_png.py tests/test_mapscene_render_policy.py tests/test_mapscene_save_bundle.py tests/test_mapscene_docs.py -q
```

Result: `25 passed`.

```powershell
python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_validation.py tests\test_mapscene_support_status.py tests\test_mapscene_render_png.py tests\test_mapscene_render_policy.py tests\test_mapscene_save_bundle.py tests\test_mapscene_docs.py
```

Result: exited 0.

Requirement status impact: `P0-R4-AC2` and `P0-R4-AC3` move from `Planned` to
`Tested`; `P0-R4-AC1`, `P0-R4-AC4`, and `P0-R4-AC7` receive stronger tested
evidence. No `Verified` status is claimed because examples, quickstart, full
docs/support audit, and full feature verification remain incomplete.

## 2026-05-17 Feature 004 T007-T008 LabelPlan Integration Session

Target feature: `specs/004-mapscene-mvp`.

Selected task IDs: T007 and T008. This is one coherent dependency batch:
MapScene LabelPlan integration and the validation-scope audit blockers it
unblocks. T009-T021 were not started because render, bundle, examples, docs,
quickstart, full verification, and final continuity depend on this checkpoint.

Optional before-implement hook `/speckit.git.commit` was identified and not
executed because it is optional and the worktree contains unrelated/user-owned
dirty files.

PRD acceptance criteria covered in this session: `P0-R4-AC1`, `P0-R4-AC4`,
`P0-R4-AC5`, `P0-R4-AC6`, and `P0-R4-AC7`. This is validation and LabelPlan
integration evidence only; it does not claim PNG render, bundle save, examples,
docs, quickstart, or MVP-wide completion.

Pre-edit state:

```text
## 004-mapscene-mvp...origin/004-mapscene-mvp
 M AGENTS.md
 M docs/superpowers/state/current-context-pack.md
 M docs/superpowers/state/implementation-ledger.md
 M docs/superpowers/state/open-blockers.md
 M docs/superpowers/state/requirements-verification-matrix.md
 D examples/turkiye_river_basins_3d.py
 M logs/.3c2cf94182465f0d10df58878528dd39234a1134-audit.json
 M python/forge3d/forge3d.pdb
 M python/forge3d/map_scene.py
 M specs/004-mapscene-mvp/tasks.md
 M src/viewer/event_loop/runner.rs
?? tests/test_mapscene_support_status.py
?? tests/test_mapscene_validation.py
?? tests/test_mount_hood_white_cloud_timelapse.py
```

T007 added `tests/test_mapscene_label_plan_integration.py` and added validation
edge assertions to `tests/test_mapscene_validation.py`. The new tests cover:

- deterministic `LabelPlan.compile()` integration during `MapScene.validate()`
- `MapScene.compiled_label_plans` storage keyed by label layer ID
- map-furniture keepouts feeding LabelPlan rejection behavior
- `label_rejection_summary` diagnostics surfacing in the validation report
- repeated fixed-input `validate().to_dict()` and plan payload equality
- missing layer CRS metadata producing typed diagnostics instead of compatibility
- building `geometry_count` contributing to GPU memory estimates

Red verification:

```powershell
pytest tests/test_mapscene_label_plan_integration.py tests/test_mapscene_validation.py -q
```

Observed red result: `4 failed, 3 passed`. Failures were expected and came from
missing `compiled_label_plans["labels"]`, missing CRS metadata reporting `ok`,
missing `label_rejection_summary`, and absent building geometry memory estimate.

T008 implemented the MapScene LabelPlan integration in
`python/forge3d/map_scene.py`. The implementation compiles or rehydrates a
`LabelPlan` for each `LabelLayer`, passes map-furniture keepouts, priority
rules, typography, glyph atlas, output viewport, camera, terrain, and
reproducibility seed, stores the plan in `compiled_label_plans`, and remaps plan
diagnostics to the MapScene label layer. It also adds typed `missing_crs`
diagnostics and building geometry-count memory estimates.

Green verification:

```powershell
pytest tests/test_mapscene_label_plan_integration.py tests/test_mapscene_validation.py -q
pytest tests/test_mapscene_label_plan_integration.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py tests/test_mapscene_recipe_contract.py -q
python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_label_plan_integration.py tests\test_mapscene_validation.py tests\test_mapscene_support_status.py tests\test_mapscene_recipe_contract.py
```

Observed results: focused LabelPlan/validation command `7 passed`; selected
regression command `13 passed`; `py_compile` exited 0.

Updated artifacts:

- `tests/test_mapscene_label_plan_integration.py`
- `tests/test_mapscene_validation.py`
- `python/forge3d/map_scene.py`
- `specs/004-mapscene-mvp/tasks.md`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/current-context-pack.md`
- `docs/superpowers/state/open-blockers.md`

Requirement status impact: P0-R4-AC1, AC4, AC5, AC6, and AC7 are `Tested` for
validation/LabelPlan integration scope. P0-R4-AC2, AC3, AC8, and full feature
`Verified` status remain planned. Feature `004` tasks T007 and T008 are now
marked `[X]`; the next incomplete task is T009 for deterministic PNG render
tests.

## 2026-05-17 Feature 004 T004-T006 Validation Adapters Session

Target feature: `specs/004-mapscene-mvp`.

Selected task IDs: T004, T005, and T006. No spec, plan, or task regeneration
was performed. The optional before-implement git commit hook was identified in
`.specify/extensions.yml` and not executed because it is optional and the
worktree contains unrelated/user-owned dirty files.

PRD acceptance criteria covered in this session: `P0-R4-AC1`, `P0-R4-AC5`,
`P0-R4-AC6`, and `P0-R4-AC7`. This is validation-scope evidence only; it does
not claim PNG render, bundle save, examples, docs, LabelPlan integration, or
MVP-wide completion.

Pre-edit state:

```text
## 004-mapscene-mvp...origin/004-mapscene-mvp
 M AGENTS.md
 D examples/turkiye_river_basins_3d.py
 M logs/.3c2cf94182465f0d10df58878528dd39234a1134-audit.json
 M python/forge3d/forge3d.pdb
 M src/viewer/event_loop/runner.rs
?? tests/test_mount_hood_white_cloud_timelapse.py
```

Checklist and SpecKit prerequisite state:

```powershell
.specify\scripts\powershell\check-prerequisites.ps1 -Json -RequireTasks -IncludeTasks
```

Observed result: active `FEATURE_DIR` resolved to
`C:\Users\milos\forge3d\specs\004-mapscene-mvp` with available docs
`research.md`, `data-model.md`, `contracts/`, `quickstart.md`, and `tasks.md`.
`specs/004-mapscene-mvp/checklists/requirements.md` had 16 total items, 16
complete, and 0 incomplete.

T004 added `tests/test_mapscene_validation.py`. The tests cover structured
`ValidationReport` output, `last_validation_report`, layer summaries,
deterministic support summaries, CRS mismatch without implicit transforms,
style support diagnostics, missing-glyph diagnostics, GPU memory estimates,
warning/error/fatal blocking status, and unsupported output format diagnostics.

T005 added `tests/test_mapscene_support_status.py`. The tests cover
Pro-gated, placeholder/fallback, experimental, unsupported building intent,
terrain VT non-albedo families, incomplete public 3D Tiles intent, unknown
recipe layer diagnostics, and render/save no-op guards preserving blocking
validation diagnostics without writing misleading artifacts.

Red verification:

```powershell
pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py -q
```

Observed red result: `8 failed`; all failures came from the placeholder
`MapScene.validate()` returning status `ok` without the required diagnostics or
layer summaries.

T006 implemented MapScene validation adapters in `python/forge3d/map_scene.py`.
The implementation reuses existing diagnostics/style/label/VT contracts and
adds MapScene-specific validation glue for:

- structured `ValidationReport` and `LayerSummary` output before render
- no implicit CRS transforms
- missing glyphs through `validate_label_support(...)`
- style diagnostics through `validate_style_support(...)`
- GPU memory estimates from output size, terrain/raster metadata, and point counts
- Pro-gated, placeholder/fallback, experimental, and unsupported building intent
- public 3D Tiles incomplete intent via `python_public_3dtiles_incomplete`
- non-albedo VT families via `validate_terrain_vt_support(...)`
- unsupported output format and unknown recipe layer diagnostics
- render/save guard behavior that validates before raising until later render
  and bundle tasks wire real output

Green verification:

```powershell
pytest tests/test_mapscene_validation.py tests/test_mapscene_support_status.py -q
pytest tests/test_mapscene_recipe_contract.py tests/test_mapscene_validation.py tests/test_mapscene_support_status.py -q
python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_validation.py tests\test_mapscene_support_status.py
```

Observed results: focused validation/support command `8 passed`; recipe
regression plus validation/support command `10 passed`; `py_compile` exited 0.

Updated artifacts:

- `tests/test_mapscene_validation.py`
- `tests/test_mapscene_support_status.py`
- `python/forge3d/map_scene.py`
- `specs/004-mapscene-mvp/tasks.md`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/current-context-pack.md`

Requirement status impact: P0-R4-AC1, P0-R4-AC5, P0-R4-AC6, and P0-R4-AC7
are `Tested` in the requirements matrix for validation-scope evidence only.
P0-R4-AC2, P0-R4-AC3, P0-R4-AC8, and full feature `Verified` status remain
planned. Feature `004` tasks T004, T005, and T006 are now marked `[X]`; the
next incomplete task is T007 for MapScene LabelPlan integration tests.

## 2026-05-16 Feature 002 Closure Audit / Verification Drift Session

Target feature: `specs/002-label-api-truth`.

Selected task IDs: no new implementation task IDs. `specs/002-label-api-truth/tasks.md`
already has T001-T018 marked `[X]`, and the ledger/matrix agree that feature
`002` is task-complete. This session therefore stayed within a T013/T016/T017/T018
evidence refresh and did not reopen deferred native behavior work.

PRD acceptance criteria covered by the audit: `P0-R1-AC1` through
`P0-R1-AC6` and `P0-R2-AC1` through `P0-R2-AC6`.

Initial verification:

```powershell
pytest tests/test_label_api_public_workflow.py tests/test_label_api_stable_ids.py tests/test_label_api_configuration_truth.py tests/test_label_api_state_noops.py tests/test_label_api_line_curved_paths.py tests/test_label_api_line_edge_cases.py tests/test_label_api_diagnostics.py tests/test_label_api_docs_support.py tests/test_label_api_quickstart.py -q
```

Observed result: `2 failed, 26 passed`. Root cause investigation found stale
assertions in `tests/test_label_api_docs_support.py`, not missing feature `002`
implementation: deterministic `LabelPlan` is now `supported` after feature
`003`, and `.specify/feature.json` correctly points at active feature
`specs/004-mapscene-mvp` after feature `003` continuity. A historical feature
`002` regression test should verify feature `002` artifacts independently of
the mutable active SpecKit pointer.

Narrow maintenance change:

- Updated `tests/test_label_api_docs_support.py` so the support matrix expects
  deterministic `LabelPlan` as `supported`.
- Replaced the active-pointer assertion with an artifact existence check for
  `specs/002-label-api-truth` while still validating that the active feature
  pointer is a real `specs/...` directory.

Green verification:

```powershell
pytest tests/test_label_api_public_workflow.py tests/test_label_api_stable_ids.py tests/test_label_api_configuration_truth.py tests/test_label_api_state_noops.py tests/test_label_api_line_curved_paths.py tests/test_label_api_line_edge_cases.py tests/test_label_api_diagnostics.py tests/test_label_api_docs_support.py tests/test_label_api_quickstart.py -q
rg -n "P0-R1-AC[1-6]|P0-R2-AC[1-6]|002-label-api-truth|tests/test_label_api" docs/superpowers/state/requirements-verification-matrix.md
rg -n "002-label-api-truth|P0-R1|P0-R2|experimental_feature|placeholder_fallback|viewer_ipc" docs/superpowers/state/implementation-ledger.md docs/superpowers/state/current-context-pack.md
```

Observed results: full label command `28 passed`; both `rg` audit commands
exited 0.

Requirement status impact: no status changes. Feature `002` remains closed at
the task level. `P0-R1-AC1`, `P0-R1-AC4`, `P0-R1-AC5`, `P0-R1-AC6`,
`P0-R2-AC1`, `P0-R2-AC2`, `P0-R2-AC3`, and `P0-R2-AC5` remain `Verified`.
`P0-R1-AC2`, `P0-R1-AC3`, `P0-R2-AC4`, and `P0-R2-AC6` remain
`Deferred with diagnostic` because native typography mutation, alternate
declutter behavior, production curved-label rendering, and terrain-elevated
line-label sampling are still intentionally not claimed by feature `002`.

No additional tasks were started. The next incomplete task remains feature
`004-mapscene-mvp` T004 for MapScene validation report tests.

## 2026-05-16 Feature 004 T002-T003 Typed Recipe Models Session

Target feature: `specs/004-mapscene-mvp`.

This session continued after T001 and executed only T002 and T003.

T002 `[P0-R4-AC4]` added `tests/test_mapscene_recipe_contract.py` for the
typed recipe construction contract. The test constructs `MapScene`,
`SceneRecipe`, `TerrainSource`, `RasterOverlay`, `VectorOverlay`, `LabelLayer`,
`PointCloudLayer`, product-level building intent via `MapSceneBuildingLayer`,
`MapFurnitureLayer`, `OrbitCamera`, `LightingPreset`, and `OutputSpec`. It also
asserts deterministic serialization shape, keyword construction, public
exports, and that the product building-intent layer does not overwrite the
existing legacy `forge3d.BuildingLayer`.

Red verification:

```powershell
pytest tests/test_mapscene_recipe_contract.py -q
```

Observed red result: `2 failed`; `forge3d.TerrainSource` was not present,
confirming the public typed recipe API did not exist.

T003 `[P0-R4-AC4]` implemented the minimal typed recipe model surface in
`python/forge3d/map_scene.py` and exported it through `python/forge3d/__init__.py`
and `python/forge3d/__init__.pyi`. The new module provides deterministic
`to_dict()` methods for the recipe components and a `MapScene` constructor that
accepts either a `SceneRecipe` or keyword recipe components. `MapScene.render()`
and `MapScene.save_bundle()` still fail honestly until their later feature
tasks wire real PNG and bundle behavior.

Green verification:

```powershell
pytest tests/test_mapscene_recipe_contract.py -q
python -m py_compile python\forge3d\map_scene.py tests\test_mapscene_recipe_contract.py
```

Observed results: recipe contract tests `2 passed`; `py_compile` exited 0.

Feature `004` tasks T002 and T003 are now marked `[X]`. No P0-R4 matrix row is
advanced yet because validation, diagnostics, render, bundle, docs, examples,
and full feature verification remain incomplete. The next incomplete task is
T004 for MapScene validation report tests.

## 2026-05-16 Feature 004 T001 Ownership Inspection Session

Target feature: `specs/004-mapscene-mvp`.

T001 `[P0-R4-AC1] [P0-R4-AC4] [P0-R4-AC7]` inspected the current MapScene
ownership paths and P0 dependencies before any product API implementation.
There is no ledger/tasks conflict: `specs/002-label-api-truth/tasks.md` has
T001-T018 marked `[X]`, the ledger says features `002` and `003` are complete,
and `.specify/feature.json` plus
`.specify/scripts/powershell/check-prerequisites.ps1 -Json -RequireTasks -IncludeTasks`
resolve the active feature to `specs/004-mapscene-mvp`.

Verification:

```powershell
rg -n "class (MapScene|SceneRecipe|ValidationReport|LabelPlan)|def render_png|render_rgba|save_bundle|viewer_ipc" python src tests docs/superpowers/state
```

Observed result: no implemented public `MapScene` or `SceneRecipe` product
class was found. The exact command returned only state/doc references for
MapScene-related ownership plus existing render/bundle/raw-IPC references.
Follow-up focused inspection confirmed the available dependencies:

- Chosen product API path: implement the new typed contract in
  `python/forge3d/map_scene.py`, with public exports and type stubs in
  `python/forge3d/__init__.py` and `python/forge3d/__init__.pyi`.
- Diagnostics prerequisite: `python/forge3d/diagnostics.py` exists with
  `Diagnostic`, `LayerSummary`, `ValidationReport`, render failure policies,
  required diagnostic factories, and `validate_label_support(...)`.
- LabelPlan prerequisite: `python/forge3d/label_plan.py` exists with
  `LabelPlan`, accepted/rejected/candidate models, keepouts, priorities, and
  render/export payload methods.
- Render path: prefer `python/forge3d/helpers/offscreen.py` for MapScene MVP
  PNG output, since it already handles native `Scene.render_rgba()` as an
  instance method and deterministic PNG writing; native `Scene.render_png` and
  `Scene.render_rgba` are instance methods in the Rust/PyO3 scene API and
  stubs.
- Bundle path: use `python/forge3d/bundle.py` as the review-bundle substrate
  because it has `SceneState`, `LoadedBundle`, deterministic manifest support,
  and `ValidationReport` preservation. `save_bundle()`/`load_bundle()` remain
  Pro-gated, so later MapScene bundle implementation must either compose a
  truthful limited metadata bundle or emit typed diagnostics instead of
  claiming unsupported public bundle success.
- Validation adapters to reuse: `python/forge3d/style.py`
  `validate_style_support(...)`, `vector_overlay_configs_from_style(...)`, and
  `label_layer_contracts_from_style(...)`; `python/forge3d/buildings.py`
  `validate_building_layer_support(...)`; `python/forge3d/tiles3d.py`
  `validate_tiles3d_support(...)`; `python/forge3d/terrain_params.py`
  `validate_terrain_vt_support(...)`; point-cloud metadata and memory inputs
  from `python/forge3d/pointcloud.py` and `python/forge3d/mem.py`.
- Existing raw IPC remains only a lower-level compatibility path. Canonical
  feature `004` examples must use the new typed MapScene API and must not call
  `viewer_ipc` directly.

No P0-R4 matrix row changes in this T001 inspection-only task. P0-R4 remains
`Planned` until tests, implementation, docs, examples, render, bundle, and full
verification evidence exist. Feature `004` task T001 is now marked `[X]`; the
next incomplete task is T002 for typed recipe construction tests.

## 2026-05-16 Feature 003 T024 Continuity Finalization Session

Target feature: `specs/003-deterministic-label-plan`.

T024 `[P0-R3-AC1] [P0-R3-AC2] [P0-R3-AC8]` updated this implementation
ledger, `docs/superpowers/state/current-context-pack.md`, and
`specs/003-deterministic-label-plan/tasks.md` after the matrix update.
Feature `003` now has T001-T024 marked `[X]`, P0-R3-AC1 through P0-R3-AC8 are
`Verified` in the requirements matrix, and the next SpecKit feature is
`specs/004-mapscene-mvp` for P0-R4. `.specify/feature.json` now points to
`specs/004-mapscene-mvp` for the next continuation.

This does not claim MVP-wide completion. P0-R4 `MapScene` remains incomplete,
and R-011 dirty worktree/change separation remains open before commit or PR.

## 2026-05-16 Feature 003 T023 Matrix Update Session

Target feature: `specs/003-deterministic-label-plan`.

T023 `[P0-R3-AC1] [P0-R3-AC2] [P0-R3-AC3] [P0-R3-AC4] [P0-R3-AC5]
[P0-R3-AC6] [P0-R3-AC7] [P0-R3-AC8]` updated
`docs/superpowers/state/requirements-verification-matrix.md`. P0-R3-AC1
through P0-R3-AC8 are now `Verified` with references to
`python/forge3d/label_plan.py`, the focused `tests/test_label_plan_*.py`
files, docs/quickstart evidence, diagnostics (`missing_glyphs`,
`label_rejection_summary`, `placeholder_fallback`), payload behavior, and the
full T022 command result.

Verification:

```powershell
rg -n "P0-R3-AC[1-8]|003-deterministic-label-plan|tests/test_label_plan" docs\superpowers\state\requirements-verification-matrix.md
```

Observed result: all P0-R3 rows reference concrete feature `003` artifacts and
show `Verified`; diagnostic rows for `missing_glyphs` and
`label_rejection_summary` include feature `003` evidence.

Feature `003` task T023 is now marked `[X]`. The next incomplete task is T024
for final continuity-state verification.

## 2026-05-16 Feature 003 T021-T022 Quickstart And Full Verification Session

Target feature: `specs/003-deterministic-label-plan`.

This session continued after T020 and executed T021 and T022.

T021 `[P0-R3-AC1] [P0-R3-AC2] [P0-R3-AC8]` added
`tests/test_label_plan_quickstart.py` for the documented quickstart scenario.
Red verification:

```powershell
pytest tests/test_label_plan_quickstart.py -q
```

Observed red result: `2 failed`; the quickstart did not name the public
`LabelPlan.compile`, `KeepoutRegion`, `PriorityClass`, payload, or diagnostic
APIs, and the first executable fixture revision accidentally allowed a glyph
gap to mask the intended keepout rejection. The fixture was corrected and
`specs/003-deterministic-label-plan/quickstart.md` was updated with a concrete
public API scenario.

Green verification:

```powershell
pytest tests/test_label_plan_quickstart.py -q
python -m py_compile tests\test_label_plan_quickstart.py
```

Observed results: quickstart tests `2 passed`; `py_compile` exited 0.

T022 ran the full LabelPlan feature verification:

```powershell
pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py tests/test_label_plan_polygon_candidates.py tests/test_label_plan_terrain.py tests/test_label_plan_keepouts.py tests/test_label_plan_priority.py tests/test_label_plan_payloads.py tests/test_label_plan_docs.py tests/test_label_plan_quickstart.py -q
```

Observed result: `29 passed`.

Feature `003` tasks T021 and T022 are now marked `[X]`. The next incomplete
task is T023 for requirements verification matrix updates.

## 2026-05-16 Feature 003 T019-T020 LabelPlan Docs Session

Target feature: `specs/003-deterministic-label-plan`.

This session continued after T018 and executed only T019 and T020.

T019 `[P0-R3-AC1] [P0-R3-AC2] [P0-R3-AC6] [P0-R3-AC8]` added
`tests/test_label_plan_docs.py` for the LabelPlan guide, support matrix, API
reference, and docs index. Red verification:

```powershell
pytest tests/test_label_plan_docs.py -q
```

Observed red result: `2 failed`; `docs/guides/label_plan_guide.md` was
missing, the label support matrix still described deterministic LabelPlan as
`missing`, and the API/index entries were absent.

T020 documented the implemented LabelPlan compiler in
`docs/guides/label_plan_guide.md`, updated
`docs/guides/label_support_matrix.md`, added `forge3d.label_plan` to
`docs/api/api_reference.rst`, and linked the guide from `docs/index.rst`. The
docs cover compile inputs/outputs, deterministic seed/order behavior,
accepted/rejected/candidate models, required rejection reasons, point and
polygon candidates, terrain sampling behavior, keepouts, priorities,
render/export payloads, and unsupported backend diagnostics without claiming
advanced curved/repeated or complex-script label rendering.

Final verification:

```powershell
pytest tests/test_label_plan_docs.py -q
pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py tests/test_label_plan_polygon_candidates.py tests/test_label_plan_terrain.py tests/test_label_plan_keepouts.py tests/test_label_plan_priority.py tests/test_label_plan_payloads.py tests/test_label_plan_docs.py -q
python -m py_compile tests\test_label_plan_docs.py
```

Observed results: docs tests `2 passed`; current LabelPlan subset `27 passed`;
`py_compile` exited 0.

Feature `003` tasks T019 and T020 are now marked `[X]`. The next incomplete
task is T021 for quickstart validation tests. P0-R3 matrix rows remain
`Planned` until T023 records feature-level evidence.

## 2026-05-16 Feature 003 T017-T018 Payload Methods Session

Target feature: `specs/003-deterministic-label-plan`.

This session continued after T016 and executed only T017 and T018.

T017 `[P0-R3-AC8] [P0-R3-AC2]` added `tests/test_label_plan_payloads.py` for
render/export payload preservation and unsupported backend diagnostics. The
tests assert accepted labels, rejected labels, diagnostics, bounds, seed,
typography, glyphs, and candidates are preserved, and unsupported render/export
backend requests return typed diagnostics instead of empty placeholder success.
Red verification:

```powershell
pytest tests/test_label_plan_payloads.py -q
```

Observed red result: `1 failed, 1 passed`; default payloads already preserved
real data, but unsupported backend handling was not implemented and raised a
`TypeError`.

T018 `[P0-R3-AC8]` implemented optional backend-aware payload methods in
`python/forge3d/label_plan.py`. `to_render_payload()` and
`to_export_payload()` preserve the full plan by default, add deterministic
`backend` and `supported` fields, and append a `placeholder_fallback`
diagnostic when an unsupported backend is requested.

Final verification:

```powershell
pytest tests/test_label_plan_payloads.py -q
pytest tests/test_label_plan_payloads.py tests/test_label_plan_contract.py -q
pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py tests/test_label_plan_polygon_candidates.py tests/test_label_plan_terrain.py tests/test_label_plan_keepouts.py tests/test_label_plan_priority.py tests/test_label_plan_payloads.py -q
python -m py_compile python\forge3d\label_plan.py tests\test_label_plan_payloads.py
```

Observed results: payload tests `2 passed`; payload plus contract tests
`4 passed`; current LabelPlan subset `25 passed`; `py_compile` exited 0.

Feature `003` tasks T017 and T018 are now marked `[X]`. The next incomplete
task is T019 for LabelPlan docs audit tests. P0-R3 matrix rows remain
`Planned` until T023 records feature-level evidence.

## 2026-05-16 Feature 003 T015-T016 Priority Solve Session

Target feature: `specs/003-deterministic-label-plan`.

This session continued after T014 and executed only T015 and T016.

T015 `[P0-R3-AC7]` added `tests/test_label_plan_priority.py` for priority
class collision winners and stable equal-priority tie breaks. Red
verification:

```powershell
pytest tests/test_label_plan_priority.py -q
```

Observed red result: `1 failed, 1 passed`; equal-priority stable tie behavior
already worked, but `priority_rules` were only serialized and did not influence
collision winners.

T016 `[P0-R3-AC7]` implemented deterministic priority-class scoring in
`python/forge3d/label_plan.py`. Candidate scores now combine priority class
rank from `PriorityClass` rules with per-label priority before collision
solving. Collision rejection details retain candidate and winner priority
classes in addition to bounds and numeric scores.

Final verification:

```powershell
pytest tests/test_label_plan_priority.py -q
pytest tests/test_label_plan_priority.py tests/test_label_plan_determinism.py -q
pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py tests/test_label_plan_polygon_candidates.py tests/test_label_plan_terrain.py tests/test_label_plan_keepouts.py tests/test_label_plan_priority.py -q
python -m py_compile python\forge3d\label_plan.py tests\test_label_plan_priority.py
```

Observed results: priority tests `2 passed`; priority plus determinism
`5 passed`; current LabelPlan subset `23 passed`; `py_compile` exited 0.

Feature `003` tasks T015 and T016 are now marked `[X]`. The next incomplete
task is T017 for render/export payload and no-op tests. P0-R3 matrix rows
remain `Planned` until T023 records feature-level evidence.

## 2026-05-16 Feature 003 T013-T014 Keepout Regions Session

Target feature: `specs/003-deterministic-label-plan`.

This session continued after T012 and executed only T013 and T014.

T013 `[P0-R3-AC6]` added `tests/test_label_plan_keepouts.py` for required
keepout kinds: `title`, `legend`, `scale_bar`, `north_arrow`, and
`manual_rectangle`. The tests verify intersecting candidates reject with
`keepout_region`, retain candidate identity and keepout details, and
non-intersecting keepout mappings are preserved deterministically in plan
bounds.

Verification:

```powershell
pytest tests/test_label_plan_keepouts.py -q
```

Observed result: `6 passed`. The tests did not require new production-code
edits because T006's generic keepout intersection already covered every
required keepout kind.

T014 `[P0-R3-AC6]` is satisfied by the already implemented keepout models,
deterministic payload ordering, bounds intersection, and `keepout_region`
rejection behavior in `python/forge3d/label_plan.py`.

Final verification:

```powershell
pytest tests/test_label_plan_keepouts.py tests/test_label_plan_rejection_reasons.py -q
pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py tests/test_label_plan_polygon_candidates.py tests/test_label_plan_terrain.py tests/test_label_plan_keepouts.py -q
python -m py_compile tests\test_label_plan_keepouts.py
```

Observed results: keepout plus rejection tests `8 passed`; current base
LabelPlan subset `21 passed`; `py_compile` exited 0.

Feature `003` tasks T013 and T014 are now marked `[X]`. The next incomplete
task is T015 for priority-class and tie-break coverage. P0-R3 matrix rows
remain `Planned` until T023 records feature-level evidence.

## 2026-05-16 Feature 003 T011-T012 Terrain Sampling Session

Target feature: `specs/003-deterministic-label-plan`.

This session continued after T010 and executed only T011 and T012.

T011 `[P0-R3-AC5]` added `tests/test_label_plan_terrain.py` for deterministic
terrain sampling, required-terrain unavailable diagnostics, and visibility
rejection with retained terrain sample details. Red verification:

```powershell
pytest tests/test_label_plan_terrain.py -q
```

Observed red result: `3 failed`; the compiler ignored sampler elevation,
accepted labels that required terrain without a sampler, and did not sample
visibility false from a terrain object.

T012 `[P0-R3-AC5]` implemented terrain sampling integration in
`python/forge3d/label_plan.py`. Labels can request terrain with
`requires_terrain` or `terrain_mode`, sampler objects with `sample(...)` or
callable terrain providers are invoked deterministically, accepted point
candidates use sampled elevation, visibility false rejects the candidate with
`terrain_occluded`, and unavailable required terrain emits a
`placeholder_fallback` diagnostic plus the rejection summary instead of
accepting placeholder success.

Final verification:

```powershell
pytest tests/test_label_plan_terrain.py -q
pytest tests/test_label_plan_terrain.py tests/test_label_plan_rejection_reasons.py -q
pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py tests/test_label_plan_polygon_candidates.py tests/test_label_plan_terrain.py -q
python -m py_compile python\forge3d\label_plan.py tests\test_label_plan_terrain.py
```

Observed results: terrain tests `3 passed`; terrain plus rejection tests
`5 passed`; current base LabelPlan subset `15 passed`; `py_compile` exited 0.

Feature `003` tasks T011 and T012 are now marked `[X]`. The next incomplete
task is T013 for keepout-region coverage. P0-R3 matrix rows remain `Planned`
until T023 records feature-level evidence.

## 2026-05-16 Feature 003 T009-T010 Polygon Candidates Session

Target feature: `specs/003-deterministic-label-plan`.

This session continued after T008 and executed only T009 and T010.

T009 `[P0-R3-AC4]` added `tests/test_label_plan_polygon_candidates.py` for
polygon centroid candidates, visual-center fallback when the centroid is
unsuitable, and invalid polygon geometry rejection with `invalid_geometry`.
Red verification:

```powershell
pytest tests/test_label_plan_polygon_candidates.py -q
```

Observed red result: `3 failed`; polygon labels were rejected as
`unsupported_geometry_type`, including invalid polygons that should be
classified as `invalid_geometry` once polygon support exists.

T010 `[P0-R3-AC4]` implemented polygon candidate generation in
`python/forge3d/label_plan.py`. Polygon labels now parse a GeoJSON-style outer
ring, reject malformed or zero-area polygons with `invalid_geometry`, generate
`centroid` and `visual_center` candidates, select the centroid when it is
inside the polygon, and select a deterministic visual-center fallback when the
centroid is unsuitable. The visual-center fallback is intentionally simple and
deterministic; broader cartographic quality remains bounded by later
verification and documentation tasks.

Final verification:

```powershell
pytest tests/test_label_plan_polygon_candidates.py -q
pytest tests/test_label_plan_polygon_candidates.py tests/test_label_plan_rejection_reasons.py -q
pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py tests/test_label_plan_polygon_candidates.py -q
python -m py_compile python\forge3d\label_plan.py tests\test_label_plan_polygon_candidates.py
```

Observed results: polygon tests `3 passed`; polygon plus rejection tests
`5 passed`; current base LabelPlan subset `12 passed`; `py_compile` exited 0.

Feature `003` tasks T009 and T010 are now marked `[X]`. The next incomplete
task is T011 for terrain sampler and occlusion coverage. P0-R3 matrix rows
remain `Planned` until T023 records feature-level evidence.

## 2026-05-16 Feature 003 T007-T008 Point Candidates Session

Target feature: `specs/003-deterministic-label-plan`.

This session continued after T006 and executed only T007 and T008.

T007 `[P0-R3-AC3]` added `tests/test_label_plan_point_candidates.py` for the
required point candidate types `center`, `above`, `below`, `left`, `right`,
and radial alternatives. The tests assert stable candidate IDs and ordering,
deterministic directional anchors, seeded radial angle output, and exact
round-trip serialization. Red verification:

```powershell
pytest tests/test_label_plan_point_candidates.py -q
```

Observed red result: `2 failed`; `AcceptedLabel` exposed only the selected
`candidate`, and serialized accepted labels did not include the generated
candidate set.

T008 `[P0-R3-AC3]` implemented deterministic point candidate generation in
`python/forge3d/label_plan.py`. `AcceptedLabel` now retains a deterministic
`candidates` list while preserving `candidate` as the selected placement.
Point candidates include center and cardinal offsets plus seeded radial
alternatives with stable candidate IDs, bounds, scores, ordering keys, and
radial detail metadata. Seeded radial jitter uses a stable SHA-256-derived
unit interval instead of Python's randomized hash.

Final verification:

```powershell
pytest tests/test_label_plan_point_candidates.py -q
pytest tests/test_label_plan_point_candidates.py tests/test_label_plan_determinism.py -q
pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py tests/test_label_plan_point_candidates.py -q
python -m py_compile python\forge3d\label_plan.py tests\test_label_plan_point_candidates.py
```

Observed results: point-candidate tests `2 passed`; point plus determinism
`5 passed`; current base LabelPlan subset `9 passed`; `py_compile` exited 0.

Feature `003` tasks T007 and T008 are now marked `[X]`. The next incomplete
task is T009 for polygon candidate and invalid-geometry coverage. P0-R3 matrix
rows remain `Planned` until T023 records feature-level evidence.

## 2026-05-16 Feature 003 T005-T006 Rejection Reasons Session

Target feature: `specs/003-deterministic-label-plan`.

This session continued after T004 and executed only T005 and T006.

T005 `[P0-R3-AC2]` added `tests/test_label_plan_rejection_reasons.py` with
fixtures covering the required reason vocabulary: `collision`, `outside_view`,
`missing_glyph`, `priority_lost`, `keepout_region`, `terrain_occluded`,
`invalid_geometry`, `unsupported_geometry_type`, and `empty_text`. The tests
also assert `missing_glyphs` and `label_rejection_summary` diagnostics,
candidate IDs for candidate-level rejections, deterministic payload
round-trip, and structured rejection details. Red verification:

```powershell
pytest tests/test_label_plan_rejection_reasons.py -q
```

Observed red result: `2 failed`; the existing base compiler retained only
`empty_text`, `missing_glyph`, `outside_view`, `invalid_geometry`, and
`unsupported_geometry_type`. It did not yet retain `collision`,
`priority_lost`, `keepout_region`, or `terrain_occluded`.

T006 `[P0-R3-AC2]` implemented basic rejection retention and diagnostics in
`python/forge3d/label_plan.py`. Point candidates now carry terrain sample
details where supplied, explicit invisible terrain samples reject with
`terrain_occluded`, active keepout rectangles reject intersecting candidates
with `keepout_region`, and a deterministic collision pass rejects colliding
losers with `priority_lost` when the winner has higher priority or
`collision` for equal-priority ties. The compiler still leaves detailed point
candidate generation, polygon candidates, terrain sampler integration,
furniture keepout breadth, and full priority-class solving to their later
task groups.

The earlier T002 contract fixture was also adjusted so its sample legend
keepout no longer overlaps labels while expecting both labels to be accepted.
This keeps the contract test aligned with now-active keepout behavior.

Final verification:

```powershell
pytest tests/test_label_plan_rejection_reasons.py -q
pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py tests/test_label_plan_rejection_reasons.py -q
python -m py_compile python\forge3d\label_plan.py tests\test_label_plan_contract.py tests\test_label_plan_determinism.py tests\test_label_plan_rejection_reasons.py
```

Observed results: rejection tests `2 passed`; combined base LabelPlan subset
`7 passed`; `py_compile` exited 0.

Feature `003` tasks T005 and T006 are now marked `[X]`. The next incomplete
task is T007 for deterministic point-candidate coverage. P0-R3 matrix rows
remain `Planned` until T023 records feature-level evidence.

## 2026-05-16 Feature 003 T002-T004 Base LabelPlan Session

Target feature: `specs/003-deterministic-label-plan`.

This session continued from T001 and executed only the next task group:
T002, T003, and T004. No spec, plan, or task regeneration was performed.

T002 `[P0-R3-AC1] [P0-R3-AC8]` added
`tests/test_label_plan_contract.py` for the public `LabelPlan.compile(...)`
contract, accepted/rejected/diagnostic fields, deterministic bounds, seed,
serialization round-trip, and render/export payload shells. Initial red
verification failed because the public `LabelPlan` and related classes were
not importable from `forge3d`.

T003 `[P0-R3-AC1]` added `tests/test_label_plan_determinism.py` for repeated
compile determinism, equivalent source-order normalization, and mapping key
order independent serialization. Initial red verification failed because
`LabelPlan` was not importable.

T004 `[P0-R3-AC1] [P0-R3-AC8]` implemented the base public object model in
`python/forge3d/label_plan.py` and exported it from
`python/forge3d/__init__.py` and `python/forge3d/__init__.pyi`. The new module
defines `LabelPlan`, `AcceptedLabel`, `RejectedLabel`, `LabelCandidate`,
`KeepoutRegion`, `PriorityClass`, `PAYLOAD_VERSION`, and
`REJECTION_REASONS`. The base compiler now produces stable accepted/rejected
lists for point labels, retains typed rejection reasons for empty text,
missing glyphs, invalid geometry, unsupported geometry, and outside-viewport
labels, emits feature `001` diagnostics for missing glyphs and rejection
summaries, and serializes to JSON-friendly render/export payloads.

During green verification, the combined command first failed on public API
list semantics because normalized `accepted`/`rejected` values were tuples.
The implementation was adjusted to preserve deterministic ordering while
exposing lists. Final verification:

```powershell
pytest tests/test_label_plan_contract.py -q
pytest tests/test_label_plan_determinism.py -q
pytest tests/test_label_plan_contract.py tests/test_label_plan_determinism.py -q
```

Observed results: contract tests `2 passed`; determinism tests `3 passed`;
combined T004 command `5 passed`.

Feature `003` tasks T002, T003, and T004 are now marked `[X]`. The next
incomplete task is T005 for complete rejection-reason coverage. P0-R3 matrix
rows remain `Planned` until the feature-level matrix update task T023 records
full evidence across all P0-R3 acceptance criteria.

## 2026-05-16 Feature 003 T001 Ownership Inspection Session

Target feature: `specs/003-deterministic-label-plan`.

Before changing files, this continuation read the PRD, constitution, current
context pack, implementation ledger, requirements matrix, open blockers,
feature `002` spec/plan/tasks, git status, recent commits, current diff
summary, relevant logs, and feature `002` checklist. The ledger and
`specs/002-label-api-truth/tasks.md` agreed that feature `002` has no open
task IDs: T001-T018 are complete. A final read-only SpecKit-style consistency
pass for feature `002` found no blocking artifact issue, no unresolved
placeholders, no open task checkbox, and no raw-IPC/overclaim regression in the
updated label docs, contract, or quickstart. The only noted workflow risk is
that `.specify/feature.json` is ignored by `.gitignore`, although it has been
updated locally for the active feature.

Per the context pack, the next MVP feature after `002` is
`003-deterministic-label-plan` for P0-R3. This session then read the feature
`003` spec, plan, research, data model, contract, quickstart, tasks, and
checklist. The feature `003` checklist `requirements.md` is complete
(`16/16`, `0` incomplete). `.specify/feature.json` was updated to point at
`specs/003-deterministic-label-plan` for subsequent SpecKit commands.

T001 `[P0-R3-AC1] [P0-R3-AC2] [P0-R3-AC8]` completed ownership inspection.
Verification command:

```powershell
rg -n "LabelPlan|AcceptedLabel|RejectedLabel|Keepout|priority|declutter|collision|atlas|typography" python src tests docs/superpowers/state
```

Observed result: no implemented public `LabelPlan`, `AcceptedLabel`,
`RejectedLabel`, or `KeepoutRegion` product API was found outside specs/state.
Feature `001` diagnostics are available in `python/forge3d/diagnostics.py` and
exported from `python/forge3d/__init__.py`; the reusable diagnostic helpers
include `Diagnostic`, `ValidationReport`, `missing_glyphs_diagnostic`, and
`label_rejection_summary_diagnostic`. Feature `002` truthful label inputs are
available through high-level `ViewerHandle` methods and `LabelOperationResult`
/ `LabelBatchResult` in `python/forge3d/viewer.py` and stubs in
`python/forge3d/viewer.pyi`.

Selected ownership for implementation: start with a Python-first public module
`python/forge3d/label_plan.py`, exported through `python/forge3d/__init__.py`
and `python/forge3d/__init__.pyi`. Reuse feature `001` diagnostics directly
instead of inventing a second diagnostic model. Treat existing Rust label
substrate as inspected implementation reference and possible later
acceleration: `src/labels/layer.rs` has `LabelFeature`, `FeatureGeometry`,
`LabelLayer`, and `GeneratedLabel`; `src/labels/declutter.rs` has
`PlacementCandidate`, `DeclutterConfig`, seeded annealing, and greedy priority
sorting; `src/labels/collision.rs` and `src/labels/rtree.rs` provide collision
checks; `src/labels/projection.rs` has projection helpers but incomplete
occlusion; `src/labels/atlas.rs` exposes glyph lookup and layout; and
`src/labels/typography.rs` exposes typography settings and glyph advance
helpers. No Rust implementation was changed in T001.

Remaining feature `003` work starts with T002 and T003, which can run in
parallel after T001: API contract tests and determinism/order-normalization
tests. No P0-R3 requirement status changed yet; matrix rows remain `Planned`
until red/green test and implementation evidence exists.

## 2026-05-16 Feature 002 Phase 4 Docs/Quickstart Completion Session

Target feature: `specs/002-label-api-truth`.

Before changing files, this session read the PRD, constitution, context pack,
implementation ledger, requirements matrix, open blockers, feature `002`
spec/plan/tasks, checklist, git status, recent commits, current diff summary,
and relevant logs. The ledger and `tasks.md` agreed that the next open group
was T013-T015, followed by T016-T018. `.specify/feature.json` still pointed at
feature `001`; this session fixed it to `specs/002-label-api-truth` rather
than relying on an override.

T013 `[P0-R1-AC5] [P0-R2-AC3] [P0-R2-AC4]` added
`tests/test_label_api_docs_support.py` before docs changes. Red verification
`pytest tests/test_label_api_docs_support.py -q` failed on missing required
label support rows, missing `placeholder_fallback` docs, public docs still
preferring raw label IPC wording, stale contract signatures, and the stale
SpecKit feature pointer.

T014 `[P0-R1-AC5] [P0-R1-AC4] [P0-R2-AC3] [P0-R2-AC4]` updated
`docs/guides/label_support_matrix.md`, `docs/api/api_reference.rst`,
`docs/guides/feature_map.md`, and
`specs/002-label-api-truth/contracts/label-api-contract.md`. A later
read-only consistency check also found stale `TBD` wording in
`specs/002-label-api-truth/plan.md` and the contract header; both were amended
to name `ViewerHandle` in `python/forge3d/viewer.py` and stubs in
`python/forge3d/viewer.pyi` as the selected high-level API surface. The docs
now state support levels for point labels, batch labels, line labels, curved
labels, callouts, typography controls, decluttering controls, atlas loading,
missing glyph diagnostics, upside-down line handling, terrain-elevated line
labels, and deterministic `LabelPlan`. Curved, typography, declutter, and
terrain-elevated paths are documented as diagnostic/experimental where
implementation does not claim production support. Green verification
`pytest tests/test_label_api_docs_support.py -q` reported `5 passed`.

T015 `[P0-R1-AC4] [P0-R1-AC1] [P0-R1-AC5]` added
`tests/test_label_api_quickstart.py` before adding the example. Red
verification `pytest tests/test_label_api_quickstart.py -q` failed because
`examples/label_api_truth_basic.py` did not exist and the quickstart did not
point to it. The session added `examples/label_api_truth_basic.py`, a
GPU-free high-level `ViewerHandle` smoke workflow, and updated
`specs/002-label-api-truth/quickstart.md`. The example exercises stable IDs,
batch ordering, line glyph ordering/rotation, callout and vector overlay IDs,
state toggling/clearing, `missing_glyphs`, and `experimental_feature`
diagnostics for curved labels, terrain-elevated line labels, typography, and
decluttering. Green verification `pytest tests/test_label_api_quickstart.py -q`
reported `4 passed`; `rg -n "viewer_ipc|send_ipc" examples/label_api_truth_basic.py`
returned no matches.

T016 `[P0-R1-AC1] [P0-R1-AC2] [P0-R1-AC3] [P0-R1-AC4] [P0-R1-AC5] [P0-R1-AC6]
[P0-R2-AC1] [P0-R2-AC2] [P0-R2-AC3] [P0-R2-AC4] [P0-R2-AC5] [P0-R2-AC6]`
ran the full feature verification:

```powershell
pytest tests/test_label_api_public_workflow.py tests/test_label_api_stable_ids.py tests/test_label_api_configuration_truth.py tests/test_label_api_state_noops.py tests/test_label_api_line_curved_paths.py tests/test_label_api_line_edge_cases.py tests/test_label_api_diagnostics.py tests/test_label_api_docs_support.py tests/test_label_api_quickstart.py -q
```

Observed result: `28 passed in 1.57s`.

T017 updated `docs/superpowers/state/requirements-verification-matrix.md`.
P0-R1-AC1, AC4, AC5, AC6 and P0-R2-AC1, AC2, AC3, AC5 are now `Verified` for
feature `002`. P0-R1-AC2, P0-R1-AC3, P0-R2-AC4, and P0-R2-AC6 remain
`Deferred with diagnostic` because the feature intentionally returns typed
diagnostics instead of claiming native typography mutation, alternate
declutter behavior, production curved-label glyph rendering, or terrain
sampling.

T018 updated this ledger, `docs/superpowers/state/current-context-pack.md`,
and marked T013-T018 complete in `specs/002-label-api-truth/tasks.md`. This
session does not claim MVP-wide completion: P0-R3 `LabelPlan` and P0-R4
`MapScene` remain planned in later feature directories, and R-011 dirty
worktree/change separation remains open before commit or PR.

Final read-only SpecKit-style consistency checks resolved
`FEATURE_DIR` to `specs/002-label-api-truth`, found no unresolved
placeholder/TBD markers in feature `002` spec/plan/tasks/contract, found no
open feature `002` task checkboxes, confirmed P0-R1/P0-R2 tags across
spec/plan/tasks/matrix, and found no raw-label-IPC or production-ready
curved/typography/declutter overclaim pattern in updated label docs,
contract, or quickstart.

## 2026-05-16 Feature 002 Phase 3 Line/Curved Label Truth Session

Target feature: `specs/002-label-api-truth`.

Per dependency order, this session executed only the next incomplete task group:
Phase 3 T008 through T012.

T008 `[P0-R2-AC1] [P0-R2-AC2] [P0-R2-AC4] [P0-R2-AC5]` added
`tests/test_label_api_line_curved_paths.py` before implementation. Red
verification `pytest tests/test_label_api_line_curved_paths.py -q` failed with
missing `line_label_glyph_instances` state and missing
`ViewerHandle.add_curved_label(...)`. After implementation, the focused
command reported `2 passed`.

T009 `[P0-R2-AC3] [P0-R2-AC6] [P0-R1-AC6]` added
`tests/test_label_api_line_edge_cases.py` before implementation. Red
verification `pytest tests/test_label_api_line_edge_cases.py -q` failed with
missing line glyph state and unsupported `terrain_mode`. After implementation,
the focused command reported `2 passed`.

T010 `[P0-R2-AC1] [P0-R2-AC2] [P0-R2-AC3] [P0-R2-AC4] [P0-R2-AC5] [P0-R2-AC6]`
implemented line, curved, and terrain label behavior or diagnostics. Public
`ViewerHandle.add_line_label(...)` now validates text/path inputs, returns
typed diagnostics for unavailable terrain sampling, records deterministic
`line_label_glyph_instances` with glyph, position, tangent rotation, ordering
key, and upside-down adjustment state, and preserves stable IDs for accepted
line labels. `ViewerHandle.add_curved_label(...)` now returns a typed
`experimental_feature` result instead of unqualified success. Native text
instances now carry rotation, `text_overlay.wgsl` rotates glyph quads around
their centers, `LabelManager` emits rotated glyph quads for line labels, and
native line placement uses the tangent angle directly with normalized
upside-down flipping.
Verification `pytest tests/test_label_api_line_curved_paths.py tests/test_label_api_line_edge_cases.py -q`
reported `4 passed`; `cargo test test_glyph_rotation -q` reported `2 passed`;
`cargo fmt -- --check` and `cargo check -q` exited 0 with the existing
path-canonicalization warning.

T011 `[P0-R1-AC6] [P0-R2-AC4] [P0-R2-AC6]` added
`tests/test_label_api_diagnostics.py` before implementation. Red verification
`pytest tests/test_label_api_diagnostics.py -q` failed because empty text,
invalid line geometry, missing glyphs, curved labels, and terrain-elevated
line labels still returned success or missing methods. After implementation,
the focused command reported `5 passed`.

T012 `[P0-R1-AC6] [P0-R2-AC4] [P0-R2-AC6]` wired structured diagnostics
through high-level label create/path methods. Empty text and invalid line
geometry return `placeholder_fallback`; non-ASCII glyph gaps return
`missing_glyphs`; curved and terrain-elevated paths return
`experimental_feature`. Verification `pytest tests/test_label_api_diagnostics.py -q`
reported `5 passed`.

This session also found that broad `.gitignore` rules for `docs/superpowers/`
and `specs/` hid the required state/task artifacts updated for feature `002`.
Added narrow exceptions for `docs/superpowers/state/*.md` and
`specs/002-label-api-truth/tasks.md`; targeted status now shows those files as
Git-visible untracked evidence.

Additional regression verification:

```powershell
pytest tests/test_label_api_public_workflow.py tests/test_label_api_stable_ids.py tests/test_label_api_configuration_truth.py tests/test_label_api_state_noops.py tests/test_label_api_line_curved_paths.py tests/test_label_api_line_edge_cases.py tests/test_label_api_diagnostics.py -q
cargo test test_glyph_rotation -q
cargo fmt -- --check
cargo check -q
```

Observed results: `19 passed` for the Python label subset; `cargo test
test_glyph_rotation -q` reported `2 passed`; both Cargo check/format commands
exited 0. Pytest emitted cache warnings because `.pytest_cache` could not be
created under the current Windows permissions. Cargo emitted the existing
`could not canonicalize path: 'C:\Users\milos'` warning.

Requirement status after this phase: P0-R1-AC6 is `Tested` for Phase 1-3
no-op prevention. P0-R2-AC1, AC2, AC3, and AC5 are `Tested`. P0-R2-AC4 and
P0-R2-AC6 are `Deferred with diagnostic` because curved labels and
terrain-elevated line labels now fail honestly rather than claiming production
rendering/sampling. P0-R1-AC5 docs, T013-T015 docs/quickstart, T016 full label
verification, T017 matrix finalization, T018 continuity, and `/speckit.analyze`
remain for a later session.

## 2026-05-16 Feature 001 Diagnostics Example Session

Target feature: `specs/001-diagnostics-support-matrices`.

Added `examples/diagnostics_support_matrices_demo.py` as a runnable, GPU-free
example that showcases feature `001` diagnostics/support-matrix behavior. The
example builds one deterministic `ValidationReport` from:

- `validate_style_support(...)` for local/provided style support,
  unsupported fields, unsupported layer types, and `symbol` underdeveloped
  truth.
- `validate_building_layer_support(...)` for zero-geometry
  `placeholder_fallback`.
- `validate_tiles3d_support(...)` for
  `python_public_3dtiles_incomplete`.
- `validate_terrain_vt_support(...)` for non-albedo
  `vt_unsupported_family`.
- `validate_label_support(...)` for line/curved `experimental_feature` and
  per-label `missing_glyphs`.
- Manual diagnostics for `crs_mismatch`, `pro_gated_path`,
  `estimated_gpu_memory`, and `label_rejection_summary`.

The example writes a bundle-ready JSON report to
`examples/out/diagnostics_support_matrices_report.json` by default and prints a
compact status/diagnostic summary. Added a narrow `.gitignore` exception so the
new source example is Git-visible while generated `examples/out/` output stays
ignored.

Verification:

```powershell
python examples/diagnostics_support_matrices_demo.py --output examples/out/diagnostics_support_matrices_report.json
python -m py_compile examples/diagnostics_support_matrices_demo.py
git status --short --untracked-files=all | Select-String -Pattern "diagnostics_support|examples|.gitignore"
```

Observed results: the example printed `status: error`, listed structured
diagnostics for CRS, building fallback, Pro-gated building, 3D Tiles, style,
VT, memory, labels, and wrote the JSON report; `py_compile` exited 0; status
shows `?? examples/diagnostics_support_matrices_demo.py` and `M .gitignore`
with generated output remaining ignored.

## 2026-05-16 Feature 001 Independent PRD Compliance Audit Update

Target feature: `specs/001-diagnostics-support-matrices`.

This audit rechecked feature `001` as an independent PRD compliance review, not
as implementation work. Scope covered `docs/superpowers/plans/prd.md`,
`.specify/memory/constitution.md`, feature spec/plan/tasks, the requirements
verification matrix, changed files, and fresh test output.

Fresh verification command:

```powershell
pytest tests/test_bundle_roundtrip.py tests/test_diagnostics_bundle_serialization.py tests/test_diagnostics_support_paths.py tests/test_diagnostics_style_support.py tests/test_diagnostics_contract.py tests/test_diagnostics_no_op_policy.py tests/test_support_matrices_docs.py tests/test_diagnostics_quickstart.py -q
```

Observed result: `63 passed in 0.63s`.

Result: no new P0-R5/P0-R6 acceptance-criterion gap was found for feature
`001`. Incomplete line/curved labels, buildings, 3D Tiles, VT normal/mask, and
style parity are represented by typed diagnostics and support-matrix wording
rather than overclaimed as supported. No requirements-verification status was
changed. Remaining gaps were refreshed in
`docs/superpowers/state/open-blockers.md` and
`docs/superpowers/state/current-context-pack.md`: R-011 dirty worktree/change
separation remains open, and R-020 real `MapScene` quickstart plus
deterministic `LabelPlan` guide remains downstream work for features `004` and
`003`.

## 2026-05-16 Feature 001 Completion-Review Fix Session

Target feature: `specs/001-diagnostics-support-matrices`.

This session addressed completion-review findings against feature `001` without
claiming unrelated feature `002` or `005` work.

T032 `[P0-R5-AC3] [P0-R5-AC6] [P0-R6-AC4] [P0-R6-AC5] [P0-R6-AC6]` added
regression tests before implementation for:

- `ValidationReport.supported_features` and `unsupported_features` rejecting
  support levels outside PRD Appendix B.
- `validate_label_support(...)` preserving affected `object_id` on
  `missing_glyphs` diagnostics.
- `validate_style_support(...)` classifying `symbol` style layers consistently
  with the style support matrix as `underdeveloped`.
- support-matrix markdown rows using exact PRD support terms and diagnostics or
  documentation-boundary/remediation text for every non-`supported` row.
- required `001` spec/docs/state evidence not being hidden by `.gitignore`.

Red verification:

```powershell
pytest tests/test_diagnostics_contract.py::test_validation_report_rejects_unknown_support_summary_levels tests/test_diagnostics_support_paths.py::test_label_validator_preserves_missing_glyph_object_ids_per_label tests/test_diagnostics_style_support.py::test_symbol_style_support_matches_underdeveloped_matrix_truth tests/test_support_matrices_docs.py::test_required_evidence_files_are_not_git_ignored tests/test_support_matrices_docs.py::test_support_matrix_rows_use_prd_terms_and_remediation_columns -q
```

Observed result: `4 failed, 1 passed`, with failures matching support-summary
validation, per-object missing glyphs, symbol style truth, and Git visibility.

T033 implemented the behavior fixes in `python/forge3d/diagnostics.py` and
`python/forge3d/style.py`. `ValidationReport` now validates support-summary
values against PRD support levels, `validate_label_support(...)` emits one
`missing_glyphs` diagnostic per affected label object, and
`validate_style_support(...)` reports `symbol` text layers as
`underdeveloped` with `experimental_feature` instead of treating them as fully
unsupported. Existing test fixtures that used `diagnostics: structured` as a
support level were corrected to `supported`.

T034 removed broad `.gitignore` entries for `.specify/`, `.agents/`, `specs/`,
`docs/guides`, and `docs/superpowers`, then restored the unrelated historical
plan files that had appeared as deleted under `docs/superpowers/plans/`.
Targeted `git check-ignore` for required `001` specs/docs/state paths returned
exit code 1 with no ignored-evidence output. Targeted status/diff checks showed
the restored historical plan files no longer have deletion diffs.

Focused green verification reported `5 passed`.

T035 ran the full feature verification:

```powershell
pytest tests/test_bundle_roundtrip.py tests/test_diagnostics_bundle_serialization.py tests/test_diagnostics_support_paths.py tests/test_diagnostics_style_support.py tests/test_diagnostics_contract.py tests/test_diagnostics_no_op_policy.py tests/test_support_matrices_docs.py tests/test_diagnostics_quickstart.py -q
```

Observed result: `63 passed`.

R-011 remains open for broader worktree/change separation because the workspace
still contains unrelated feature `002` label/viewer changes and other
not-yet-attributed paths on branch `005-map-assets-bundles-p1`. The ignored
evidence and historical plan deletion parts of R-011 are mitigated.

## 2026-05-16 Feature 002 Phase 2 Configuration Truth Session

Target feature: `specs/002-label-api-truth`.

Per dependency order, this session executed only the next incomplete task group:
T005, T006, and T007.

T005 `[P0-R1-AC2] [P0-R1-AC3] [P0-R1-AC6]` added
`tests/test_label_api_configuration_truth.py` before implementation. Red
verification `pytest tests/test_label_api_configuration_truth.py -q` failed
with missing `ViewerHandle.set_label_typography(...)` and
`ViewerHandle.set_declutter_algorithm(...)`. After implementation, the focused
command reported `2 passed`.

T006 `[P0-R1-AC4] [P0-R1-AC6]` added
`tests/test_label_api_state_noops.py` before implementation. Red verification
`pytest tests/test_label_api_state_noops.py -q` failed because
`load_label_atlas(...)`/`clear_labels()` returned `None` and
`ViewerHandle.remove_label(...)` was missing. After implementation, the focused
command reported `2 passed`.

T007 `[P0-R1-AC2] [P0-R1-AC3] [P0-R1-AC4] [P0-R1-AC6]` implemented
`LabelOperationResult`, `ViewerHandle.label_configuration_state()`, serializable
state updates for `load_label_atlas`, `set_labels_enabled`, `clear_labels`, and
`remove_label`, plus typed `experimental_feature` diagnostics for
`set_label_typography` and `set_declutter_algorithm` instead of forwarding the
known native no-op commands. Public stubs were updated in
`python/forge3d/viewer.pyi`.

Verification:

```powershell
pytest tests/test_label_api_configuration_truth.py -q
pytest tests/test_label_api_state_noops.py -q
pytest tests/test_label_api_public_workflow.py tests/test_label_api_stable_ids.py -q
pytest tests/test_label_api_configuration_truth.py tests/test_label_api_state_noops.py tests/test_label_api_public_workflow.py -q
```

Observed results: `2 passed`, `2 passed`, `6 passed`, and `6 passed`.

Requirement status after this phase: P0-R1-AC2 and P0-R1-AC3 are recorded as
`Deferred with diagnostic` because the high-level API now fails honestly rather
than claiming unsupported native typography/declutter mutation. P0-R1-AC4
remains `Tested` with added state-operation evidence. P0-R1-AC6 remains
`Planned` overall because line/curved/glyph no-op coverage is still owned by
T008-T012.

## 2026-05-16 Feature 001 Independent PRD Compliance Audit

Target feature: `specs/001-diagnostics-support-matrices`.

Audit scope covered `docs/superpowers/plans/prd.md`,
`.specify/memory/constitution.md`, the feature spec/plan/tasks, the
requirements verification matrix, changed files, and fresh verification output.

Fresh verification command:

```powershell
pytest tests/test_bundle_roundtrip.py tests/test_diagnostics_bundle_serialization.py tests/test_diagnostics_support_paths.py tests/test_diagnostics_style_support.py tests/test_diagnostics_contract.py tests/test_diagnostics_no_op_policy.py tests/test_support_matrices_docs.py tests/test_diagnostics_quickstart.py -q
```

Observed result: `58 passed`.

Result: no new P0-R5/P0-R6 acceptance-criterion gap was found for feature
`001`. Remaining gaps were recorded in `docs/superpowers/state/open-blockers.md`
and `docs/superpowers/state/current-context-pack.md`: R-011 dirty-worktree and
change-separation risk remains open, and R-020 real `MapScene` quickstart plus
`LabelPlan` guide remains downstream work for features `004` and `003`.

## 2026-05-15 Feature 002 Phase 1 Implementation Session

Target feature: `specs/002-label-api-truth`.
The prerequisite script resolved the active feature directory to
`specs/001-diagnostics-support-matrices`; that feature already has T001-T031
complete. Per dependency order, this pass continued with the next incomplete
MVP feature, `specs/002-label-api-truth`, and prioritized only Phase 1
T001-T004.

T001 `[P0-R1-AC1] [P0-R1-AC4] [P0-R1-AC6] [P0-R2-AC1]` completed ownership
inspection before source implementation. Verification command:
`rg -n "add_label|add_labels|add_line_label|curved|callout|typography|declutter|LabelManager|Response" python src tests docs/superpowers/state`.
Observed result: raw helpers existed in `python/forge3d/viewer_ipc.py`, while
`ViewerHandle` lacked high-level label methods. Native ID plumbing touched
`src/viewer/ipc/protocol/request.rs`,
`src/viewer/ipc/protocol/response.rs`,
`src/viewer/ipc/protocol/translate/labels.rs`,
`src/viewer/ipc/protocol/translate/overlays.rs`,
`src/viewer/cmd/labels_command.rs`,
`src/viewer/cmd/vector_overlay_command.rs`, `src/labels/mod.rs`,
`src/viewer/state/labels.rs`,
`src/viewer/terrain/scene/overlays.rs`, and
`src/viewer/terrain/vector_overlay/core.rs`.

T002 `[P0-R1-AC4] [P0-R1-AC1] [P0-R1-AC6]` added
`tests/test_label_api_public_workflow.py` before implementation. Red
verification `pytest tests/test_label_api_public_workflow.py -q` failed with
`AttributeError: 'ViewerHandle' object has no attribute 'load_label_atlas'`.
After implementation, the same task command reported `2 passed`.

T003 `[P0-R1-AC1] [P0-R1-AC6]` added
`tests/test_label_api_stable_ids.py` before implementation. Red verification
first failed on missing `ViewerHandle.add_label(...)`/batch/line methods, then
failed on missing `ViewerHandle.add_vector_overlay(...)` after the overlay
stable-ID criterion was added. After implementation, the same task command
reported `4 passed`.

T004 `[P0-R1-AC1] [P0-R1-AC4] [P0-R1-AC6]` implemented high-level
`ViewerHandle` label methods, `LabelBatchResult`, public stubs, stable
client-supplied create IDs, native IPC response ID serialization, and native
manager ID preservation for point labels, line labels, callouts, curved-label
requests routed through the current line-label substrate, and vector overlays.
Verification commands:
`pytest tests/test_label_api_public_workflow.py tests/test_label_api_stable_ids.py -q`
reported `6 passed`; `cargo test test_parse_label_and_overlay_create_ids`
reported `1 passed`; `cargo test test_response_serialization_with_created_id`
reported `1 passed`; `cargo fmt` completed successfully.

Requirement status after this phase: P0-R1-AC1 and P0-R1-AC4 are `Tested` in
the matrix for Phase 1 evidence. P0-R1-AC6 remains `Planned` overall because
truthful typography, declutter, atlas/state no-op behavior, line/curved
diagnostics, and renderable glyph emission are owned by later tasks T005-T012.
P0-R2 rows remain `Planned`; this pass does not claim line-label glyph
emission, tangent rotation, curved-label production support, upside-down
handling, or terrain-elevated label behavior.

## 2026-05-15 Feature 001 Implementation Session

Target feature: `specs/001-diagnostics-support-matrices`.
Requested path `specs/specs/001-diagnostics-support-matrices/checklists/tasks.md`
does not exist; implementation is using the existing task file
`specs/001-diagnostics-support-matrices/tasks.md`. The SpecKit prerequisite
script still resolves `FEATURE_DIR` to
`specs/005-map-assets-bundles-p1`, so this mismatch remains recorded as
context rather than changing generated feature state.

T001 `[P0-R5-AC1] [P0-R5-AC4] [P0-R5-AC6]` completed path ownership
inspection before source implementation. Verification command:
`rg -n "Diagnostic|ValidationReport|RenderFailurePolicy|SeverityPolicy" python tests docs/superpowers/state`.
Observed result: no product-level `Diagnostic`, `ValidationReport`,
`LayerSummary`, `SupportMatrixEntry`, `RenderFailurePolicy`, or
`SeverityPolicy` exists under `python/forge3d` or `tests`; matches are limited
to docs/state planning text and unrelated existing validation patterns.
Chosen paths are `python/forge3d/diagnostics.py`,
`tests/test_diagnostics_contract.py`,
`tests/test_diagnostics_bundle_serialization.py`, package exports in
`python/forge3d/__init__.py`, and public stubs in
`python/forge3d/__init__.pyi`. No implementation code existed before T001
completed. `.gitignore` already contains Python, Rust, generated output,
environment, docs build, and temporary-file patterns, so no ignore-file change
was made.

T002 `[P0-R5-AC1] [P0-R5-AC2] [P0-R5-AC3] [P0-R5-AC5]` added
`tests/test_diagnostics_contract.py` before implementation. Red verification:
`pytest tests/test_diagnostics_contract.py -q` failed during collection with
`ModuleNotFoundError: No module named 'forge3d.diagnostics'`, proving the
contract did not yet exist. After implementation, the same command reported
`17 passed`.

T003 `[P0-R5-AC1] [P0-R5-AC2] [P0-R5-AC3] [P0-R5-AC5]` implemented the public
diagnostics data models and exports in `python/forge3d/diagnostics.py`,
`python/forge3d/__init__.py`, and `python/forge3d/__init__.pyi`. The contract
test command `pytest tests/test_diagnostics_contract.py -q` reported
`17 passed`.

T004 `[P0-R5-AC4] [P0-R5-AC1]` added
`tests/test_diagnostics_bundle_serialization.py` before serialization
implementation. Red verification:
`pytest tests/test_diagnostics_bundle_serialization.py -q` failed during
collection with `ModuleNotFoundError: No module named 'forge3d.diagnostics'`.
After implementation, the same command reported `5 passed`.

T005 `[P0-R5-AC4] [P0-R5-AC1]` implemented deterministic diagnostic and
validation-report serialization with stable diagnostic ordering. Combined
verification command:
`pytest tests/test_diagnostics_contract.py tests/test_diagnostics_bundle_serialization.py -q`
reported `22 passed`. Requirement rows P0-R5-AC1 through P0-R5-AC5 are now
`Tested` at this checkpoint. Later T006-T015 in this same feature completed
the remaining unsupported-path, style, docs, quickstart, and full-verification
work.

Follow-up red-finding fix session normalized `.specify/feature.json` to
`specs/001-diagnostics-support-matrices` while intentionally leaving the dirty
git branch `005-map-assets-bundles-p1` unchanged. Rerun command
`.specify\scripts\powershell\check-prerequisites.ps1 -Json -RequireTasks -IncludeTasks`
returned `FEATURE_DIR` as
`C:\Users\milos\forge3d\specs\001-diagnostics-support-matrices`.

T006-T007 `[P0-R5-AC2] [P0-R5-AC3] [P0-R5-AC6]` added
`tests/test_diagnostics_support_paths.py` and implemented required diagnostic
inventory helpers in `python/forge3d/diagnostics.py`. Covered codes:
`crs_mismatch`, `missing_glyphs`, `pro_gated_path`, `placeholder_fallback`,
`experimental_feature`, `vt_unsupported_family`,
`python_public_3dtiles_incomplete`, `estimated_gpu_memory`, and
`label_rejection_summary`.

T008-T009 `[P0-R5-AC5] [P0-R5-AC6]` added
`tests/test_diagnostics_no_op_policy.py` and verified warning-policy behavior
plus blocking behavior for `pro_gated_path` and `placeholder_fallback`.

T010-T011 `[P0-R6-AC1]` through `[P0-R6-AC5]` plus `[P0-R5-AC6]` added
`tests/test_diagnostics_style_support.py` and implemented
`forge3d.style.validate_style_support(...)`. The helper reports supported
local/provided `fill`, `line`, and `circle` layer styling and emits
`unsupported_style_layer_type` / `unsupported_style_field` diagnostics for
unsupported layer types and fields.

T012-T013 `[P0-R6-AC1]` through `[P0-R6-AC6]` plus `[P0-R5-AC1]` added docs
audit tests and source docs under `docs/guides/` for diagnostics, style,
labels, buildings, 3D Tiles, virtual texturing, offline map rendering, and
competitive positioning. `docs/index.rst` now includes those guides.

T014-T015 completed quickstart tests and full feature verification. Fresh full
command:
`pytest tests/test_diagnostics_contract.py tests/test_diagnostics_bundle_serialization.py tests/test_diagnostics_support_paths.py tests/test_diagnostics_no_op_policy.py tests/test_diagnostics_style_support.py tests/test_support_matrices_docs.py tests/test_diagnostics_quickstart.py -q`
reported `35 passed`.

T016-T017 updated `docs/superpowers/state/requirements-verification-matrix.md`,
this ledger, and `docs/superpowers/state/current-context-pack.md`. P0-R5 and
P0-R6 rows are now `Verified` for feature `001` diagnostics/support-matrix
scope. Full `MapScene`, `LabelPlan`, render-path wiring, and scene-bundle
write/load behavior remain owned by features `003`, `004`, `005`, and `006`;
this session does not claim those downstream integrations.

## 2026-05-15 Feature 001 Strict Audit Gap Remediation

Target feature: `specs/001-diagnostics-support-matrices`.
This pass addressed strict audit findings R-015 through R-017 without expanding
feature `001` into full `MapScene`, `LabelPlan`, or renderer ownership.

T018 `[P0-R5-AC4]` added
`tests/test_bundle_roundtrip.py::test_scene_bundle_roundtrip_preserves_validation_report`
before bundle implementation. Red verification:
`pytest tests/test_bundle_roundtrip.py tests/test_diagnostics_bundle_serialization.py -q`
failed with `TypeError: save_bundle() got an unexpected keyword argument
'validation_report'`.

T019 `[P0-R5-AC4]` implemented diagnostic persistence through
`python/forge3d/bundle.py`: `SceneState.validation_report`,
`LoadedBundle.validation_report`, `save_bundle(..., validation_report=...)`,
and `load_bundle(...)`. Focused verification command
`pytest tests/test_bundle_roundtrip.py tests/test_diagnostics_bundle_serialization.py -q`
reported `20 passed`.

T020 `[P0-R5-AC6] [P0-R5-AC3]` added public non-style validator tests before
implementation in `tests/test_diagnostics_support_paths.py`. Red verification
`pytest tests/test_diagnostics_support_paths.py -q` reported `4 failed, 2
passed`, with failures caused by missing validator imports.

T021 `[P0-R5-AC6] [P0-R5-AC3]` implemented public validators:
`validate_building_layer_support(...)` in `python/forge3d/buildings.py`,
`validate_tiles3d_support(...)` in `python/forge3d/tiles3d.py`,
`validate_terrain_vt_support(...)` in `python/forge3d/terrain_params.py`, and
`validate_label_support(...)` in `python/forge3d/diagnostics.py`. The focused
command `pytest tests/test_diagnostics_support_paths.py -q` reported
`6 passed`.

T022 `[P0-R6-AC5]` added style workflow handoff tests before helper
implementation in `tests/test_diagnostics_style_support.py`. Red verification
`pytest tests/test_diagnostics_style_support.py -q` reported `2 failed, 3
passed`, with failures caused by missing helper imports.

T023 `[P0-R6-AC5]` implemented `vector_overlay_configs_from_style(...)` and
`label_layer_contracts_from_style(...)` in `python/forge3d/style.py`, with
package exports and stubs. Focused verification command
`pytest tests/test_diagnostics_style_support.py -q` reported `5 passed`.

T024 `[P0-R5-AC4] [P0-R5-AC6] [P0-R6-AC5]` ran the strict audit remediation
suite:
`pytest tests/test_bundle_roundtrip.py tests/test_diagnostics_bundle_serialization.py tests/test_diagnostics_support_paths.py tests/test_diagnostics_style_support.py tests/test_diagnostics_contract.py tests/test_diagnostics_no_op_policy.py tests/test_support_matrices_docs.py tests/test_diagnostics_quickstart.py -q`
reported `56 passed`.

T025 `[P0-R5-AC4] [P0-R5-AC6] [P0-R6-AC5]` updated
`docs/superpowers/state/requirements-verification-matrix.md`,
`docs/superpowers/state/open-blockers.md`,
`docs/superpowers/state/current-context-pack.md`, this ledger, and
`specs/001-diagnostics-support-matrices/tasks.md`. R-015 through R-017 are
mitigated for feature `001`; downstream render/MapScene/LabelPlan integration
remains in later feature rows. `/speckit.analyze` passed after remediation:
20 functional requirements, 10 success criteria, 25 tasks, 0 incomplete tasks,
no unresolved placeholders, no constitution conflicts, and no findings.

## 2026-05-15 Feature 001 Follow-Up Audit Gap Remediation

Target feature: `specs/001-diagnostics-support-matrices`.
This pass addressed follow-up PRD compliance audit findings R-018 and R-019
without expanding feature `001` into full `MapScene`, `LabelPlan`, or renderer
ownership.

T026 `[P0-R6-AC4]` added
`tests/test_diagnostics_style_support.py::test_parsed_style_support_reports_preserved_unsupported_fields`
before implementation. Red verification:
`pytest tests/test_diagnostics_style_support.py -q` reported `1 failed, 5
passed`; the failure showed `validate_style_support(parse_style(raw_style))`
returned status `ok` instead of `warning` for unsupported `line-gradient` and
`line-sort-key` fields.

T027 `[P0-R6-AC4]` implemented parsed-style unsupported-field preservation in
`python/forge3d/style.py` by adding `StyleLayer.unsupported_paint_fields` and
`StyleLayer.unsupported_layout_fields`, populating them in `_parse_layer(...)`,
and using them in `validate_style_support(...)` when raw layer dictionaries are
not available. Focused verification command
`pytest tests/test_diagnostics_style_support.py -q` reported `6 passed`.

T028 `[P0-R6-AC6]` added
`tests/test_support_matrices_docs.py::test_public_style_api_docstrings_do_not_overclaim_mapbox_parity`
before docstring remediation. Red verification:
`pytest tests/test_support_matrices_docs.py -q` reported `1 failed, 3 passed`;
the failure found `Complete Mapbox GL Style specification` in
`python/forge3d/style.py`.

T029 `[P0-R6-AC6]` reworded the public `StyleSpec` docstring in
`python/forge3d/style.py` to local/provided feature styling scope. Focused
verification command `pytest tests/test_support_matrices_docs.py -q` reported
`4 passed`.

T030 `[P0-R6-AC4] [P0-R6-AC6]` ran the follow-up remediation suite:
`pytest tests/test_bundle_roundtrip.py tests/test_diagnostics_bundle_serialization.py tests/test_diagnostics_support_paths.py tests/test_diagnostics_style_support.py tests/test_diagnostics_contract.py tests/test_diagnostics_no_op_policy.py tests/test_support_matrices_docs.py tests/test_diagnostics_quickstart.py -q`
reported `58 passed`.

T031 `[P0-R6-AC4] [P0-R6-AC6]` updated
`docs/superpowers/state/requirements-verification-matrix.md`,
`docs/superpowers/state/open-blockers.md`,
`docs/superpowers/state/current-context-pack.md`, this ledger, and
`specs/001-diagnostics-support-matrices/tasks.md`. R-018 and R-019 are
mitigated for feature `001`. Speckit analysis pass after remediation found 20
functional requirements, 10 success criteria, 31 tasks, 0 incomplete tasks, no
unresolved placeholders, no constitution conflicts, and no findings. Optional
pre/after analyze git commit hooks were present and not executed.

## Session Scope

This planning session used `/speckit.plan` scope across six existing feature
specifications:

- `specs/001-diagnostics-support-matrices/spec.md`
- `specs/002-label-api-truth/spec.md`
- `specs/003-deterministic-label-plan/spec.md`
- `specs/004-mapscene-mvp/spec.md`
- `specs/005-map-assets-bundles-p1/spec.md`
- `specs/006-material-vt-large-scene-p2/spec.md`

It created or replaced `plan.md`, created `data-model.md`, `quickstart.md`,
and one `contracts/*.md` file for each feature, and appended Phase 0 planning
decisions to each feature's `research.md`. It also updated the SpecKit marker
in `AGENTS.md` to point to the six current plans. No product code
implementation was performed.

An earlier session ran `/speckit.clarify` for feature,
`005-map-assets-bundles-p1`, against the PRD and constitution. It reviewed
ambiguous behavior, acceptance criteria, support-level classification, MVP
blocking, testability, determinism, documentation wording, and new-chat
continuity evidence. No product code implementation was performed.

An earlier prerequisite result in this session resolved to feature `006`, so
`specs/006-material-vt-large-scene-p2/spec.md` was also clarified with P2
defaults before the latest `.specify/feature.json`, `git status`, and
prerequisite check confirmed that the active SpecKit feature at that time was
`005`.

Files changed by this session:

- `specs/002-label-api-truth/research.md`
- `specs/003-deterministic-label-plan/research.md`
- `specs/004-mapscene-mvp/research.md`
- `specs/001-diagnostics-support-matrices/research.md`
- `specs/005-map-assets-bundles-p1/spec.md`
- `specs/005-map-assets-bundles-p1/plan.md`
- `specs/005-map-assets-bundles-p1/research.md`
- `specs/005-map-assets-bundles-p1/tasks.md`
- `specs/006-material-vt-large-scene-p2/spec.md`
- `specs/006-material-vt-large-scene-p2/research.md`
- `docs/superpowers/state/current-context-pack.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/open-blockers.md`

The requirements verification matrix status did not change: all P0/P1/P2 rows
remain `Planned`, with no implementation, test, or verification evidence
claimed.

This session created `specs/004-mapscene-mvp/research.md` before technical
planning. It inspected the requested source artifacts, continuity state, and
existing repo paths for typed MapScene/SceneRecipe coverage, validation and
diagnostics, PNG render paths, bundle paths, canonical example substrate,
terrain/raster/vector/label/building/point-cloud/map-furniture APIs,
native/Rust paths, suspected placeholder/fallback/no-op paths, Pro-gated
paths, tests, fixtures, docs, examples, and unknowns. No product code
implementation was performed, and no product tests were run.

A later pre-plan inventory session for feature
`001-diagnostics-support-matrices` created
`specs/001-diagnostics-support-matrices/research.md`. It inspected the PRD,
constitution, feature spec, continuity state, and existing repo paths for
diagnostics, support matrices, style support, buildings, 3D Tiles, labels, VT,
bundles, CRS, memory/device utilities, tests, fixtures, docs, examples, public
Python APIs, native/Rust paths, placeholder/fallback/no-op paths, Pro-gated
paths, and unknowns. No product code implementation was performed, and no
product tests were run.

The follow-up implementation inventory session created
`specs/005-map-assets-bundles-p1/research.md` before technical planning. It
inspected source artifacts and existing repo paths for P1 label ingestion,
typography/font handling, buildings, 3D Tiles, and bundle round-trip behavior.
The inventory records actual modules, tests, fixtures, docs/examples,
native/Rust paths, Python public API paths, suspected placeholders/fallbacks,
Pro-gated paths, unknowns, and commands used. No code implementation was
performed.

The hostile plan review session on 2026-05-15 targeted
`specs/005-map-assets-bundles-p1/plan.md` against
`docs/superpowers/plans/prd.md`, `.specify/memory/constitution.md`, and
`specs/005-map-assets-bundles-p1/spec.md`. The original plan content was the
SpecKit template, so the review treated acceptance coverage, tests,
diagnostics/support honesty, determinism, no-op prevention, raw IPC avoidance,
bundle/docs requirements, continuity artifacts, scope boundaries, and file-path
accuracy as failing before amendment. The session replaced `plan.md` with a
concrete P1 plan containing no current red items, five documented accepted
yellow risks/open blockers, an acceptance mapping for P1-R1-AC1 through
P1-R5-AC5, and a required future task-generation checklist. No product code was
implemented and no requirement status changed.

A follow-up hostile review on 2026-05-15 found one remaining red planning gap:
the PRD Section 11 required API shape was implied but not explicitly locked for
`label_layer.compile_labels(...)`, `BuildingLayer.from_mesh(...)`,
`Tiles3DLayer.from_tileset_json(...)`, and `Tiles3DLayer.from_b3dm(...)`. The
plan was amended with an Exact PRD API Surface section, updated acceptance
mapping rows, and explicit task-generation requirements for those methods.
Re-review found no remaining red items. No product code was implemented and no
requirement status changed.

The pre-implementation task coverage audit on 2026-05-15 targeted
`specs/005-map-assets-bundles-p1/tasks.md`. The first pass found no missing
exact P1 acceptance-criterion task tags, but failed two governance checks:
P0 dependency deferral needed an explicit typed diagnostic plus blocker record,
and public P1 API implementation tasks needed stronger docs/support-level
coverage. The task list was amended so T002/T006 require a fatal
`p0_dependency_missing` diagnostic/blocker record when P0-owned contracts are
unavailable, T065 covers every public P1 API implementation task and all P1
acceptance criteria, and the file includes a 43-row detailed coverage audit
table with task IDs, test IDs, docs IDs, verification commands, and omission
risks. Rerun checks passed. No product code was implemented, no product tests
were run, and no requirements-verification status changed.

The multi-feature pre-implementation task coverage audit on 2026-05-15
targeted all requested task files:

- `specs/001-diagnostics-support-matrices/tasks.md`
- `specs/002-label-api-truth/tasks.md`
- `specs/003-deterministic-label-plan/tasks.md`
- `specs/004-mapscene-mvp/tasks.md`
- `specs/005-map-assets-bundles-p1/tasks.md`
- `specs/006-material-vt-large-scene-p2/tasks.md`

The first automated pass found 180 tasks, zero missing required task fields,
zero missing PRD acceptance-criterion coverage, zero implementation tasks
without test or diagnostic proof, and zero printed-text-only diagnostic
requirements. The audit still failed the requested artifact format because
features `001`, `002`, `003`, `004`, and `006` lacked persisted detailed
coverage audit tables. The task files were amended with detailed tables
covering PRD requirement, acceptance criterion, covering task IDs, test task
IDs, docs task IDs, verification command, and risk if omitted. Feature `004`
also received explicit PRD Section 21 MVP must-include coverage rows, and
feature `005` received an explicit audit PASS section.

Rerun checks passed with detailed tables and audit PASS sections present in all
six task files. No product code was implemented, no product tests were run, and
no requirements-verification status changed.

This session created `specs/002-label-api-truth/research.md` before technical
planning. It inspected the requested source artifacts, continuity state, and
existing repo paths for public label APIs, raw viewer IPC, native label
manager/rendering substrate, IPC response shape, tests, fixtures, docs,
examples, suspected no-op/fallback paths, Pro-gated paths, and unknowns. No
code implementation was performed, and no product tests were run.

This session created `specs/003-deterministic-label-plan/research.md` before
technical planning. It inspected the requested source artifacts, continuity
state, and existing repo paths for deterministic label planning, existing
native label layer/declutter/collision/projection/atlas/typography substrate,
Python public API surfaces, tests, fixtures, docs, examples, suspected
placeholder/fallback/no-op paths, Pro-gated paths, and unknowns. No code
implementation was performed, and no product tests were run.

The hostile plan review session on 2026-05-15 targeted
`specs/001-diagnostics-support-matrices/plan.md` against
`docs/superpowers/plans/prd.md`, `.specify/memory/constitution.md`, and
`specs/001-diagnostics-support-matrices/spec.md`. The review identified
planning risks in unresolved file-path ownership, explicit test coverage,
diagnostics/support honesty enforcement, deterministic serialization,
no-op success prevention, raw IPC avoidance, bundle-ready diagnostics,
documentation path accuracy, new-chat continuity, and P0/P1/P2 scope
boundaries. The plan was amended with a red/yellow/green risk table containing
no red items, accepted yellow risks for P0/P1/P2 support-matrix references and
path changes, fixed planned source/test/docs paths, required diagnostic
coverage by code, no-op/raw-IPC/determinism tests, and a future
task-generation checklist. No product code was implemented and no
requirements-verification status changed.

This session created `specs/006-material-vt-large-scene-p2/research.md` before
technical planning. It inspected the requested source artifacts, continuity
state, and existing repo paths for VT albedo/normal/mask support, textured
building materials and UV/texture prerequisites, advanced static label
placement, large-scene memory/cache/LOD/instancing diagnostics, tests,
fixtures, docs, examples, native/Rust paths, Python public API paths,
suspected placeholder/fallback/no-op paths, Pro-gated paths, and unknowns. No
code implementation was performed, and no product tests were run.

## Clarifications Applied To Feature 005

Five PRD-aligned default clarifications were recorded in the current spec:

- Bundles store both source labels and compiled `LabelPlan` payloads where
  available; persistence or replay gaps emit structured diagnostics.
- Feature-local structured diagnostics are required for `missing_label_field`,
  `unicode_coverage_gap`, `unsupported_tile_format`,
  `unsupported_tile_feature`, `missing_external_asset`, and
  `unavailable_terrain_sampler`.
- Diagnostic support does not make unavailable building or 3D Tiles rendering
  paths `supported`.
- Deterministic manifests, diagnostics, label plans, and round-trip outputs use
  stable ordering and exact comparisons unless planning records a numeric
  tolerance before verification.
- P1 deferrals do not block P0 MVP readiness unless a documented human decision
  expands release scope.

## Current Task State

| Item | State |
|---|---|
| PRD read | Done for feature `002` research inventory and earlier feature `005` clarification |
| Constitution read | Done |
| Continuity state read | Done |
| Extension hooks checked | Done; before/after clarify commit hooks are optional and were not executed |
| Feature `005` spec | Clarified and revalidated at `specs/005-map-assets-bundles-p1/spec.md` |
| Requirements matrix | Feature `001` P0-R5/P0-R6 rows are `Verified`; feature `002` P0-R1-AC1/P0-R1-AC4/P0-R1-AC5/P0-R1-AC6 and P0-R2-AC1/P0-R2-AC2/P0-R2-AC3/P0-R2-AC4/P0-R2-AC5/P0-R2-AC6 are `Verified`; feature `002` P0-R1-AC2/P0-R1-AC3 are `Blocked`; feature `003` P0-R3-AC1 through P0-R3-AC8 are `Verified`; feature `004` P0-R4-AC1 through P0-R4-AC8 are `Verified`; remaining P1/P2 rows remain planned in later feature directories. |
| Feature `005` research inventory | Created at `specs/005-map-assets-bundles-p1/research.md` |
| Feature `002` research inventory | Created at `specs/002-label-api-truth/research.md` |
| Feature `003` research inventory | Created at `specs/003-deterministic-label-plan/research.md` |
| Feature `004` research inventory | Created at `specs/004-mapscene-mvp/research.md` |
| Feature `006` research inventory | Created at `specs/006-material-vt-large-scene-p2/research.md` |
| Feature `001` technical plan | Created with plan, data model, contract, quickstart, and research decisions |
| Feature `001` hostile plan review | Completed; `plan.md` amended to remove red review findings and document accepted yellow risks |
| Feature `002` technical plan | Created with plan, data model, contract, quickstart, and research decisions |
| Feature `003` technical plan | Created with plan, data model, contract, quickstart, and research decisions |
| Feature `004` technical plan | Created with plan, data model, contract, quickstart, and research decisions |
| Feature `005` technical plan | Replaced template with actual P1 plan; created data model, contract, quickstart, and research decisions |
| Feature `005` hostile plan review | Completed; `plan.md` amended to remove red review findings, document accepted yellow risks, and lock the exact PRD Section 11 API surface |
| Feature `005` task coverage audit | Completed; `tasks.md` now records typed P0 dependency diagnostics, all-public-API docs/support coverage, and a detailed coverage audit table |
| Feature `006` technical plan | Created with plan, data model, contract, quickstart, and research decisions |
| Code implementation | Feature `001` diagnostics/support-matrix implementation completed through T035; feature `002` T001-T018 completed stable public create responses, high-level configuration diagnostics/state no-op handling, line glyph placement/rotation plus curved/terrain diagnostics, docs/support wording, quickstart/example smoke coverage, matrix update, and continuity update; feature `003` T001-T024 completed deterministic `LabelPlan` models, rejection reasons, point/polygon candidates, terrain sampling, keepouts, priority solving, payloads, docs, quickstart, matrix update, and continuity update; feature `004` T001-T029 plus R-029 remediation completed typed MapScene construction, validation, LabelPlan integration, native/offscreen PNG rendering for fixture-backed MVP scenes, source-derived compatibility output, review-bundle source metadata, canonical examples, quickstart, docs/support audit, matrix update, and continuity update. |
| Product verification tests | Feature `001` completion-review command reported `63 passed`; feature `002` full label command reported `28 passed`; feature `003` full T022 LabelPlan command reported `29 passed`; feature `004` focused native/render docs/examples command reported `23 passed` and full MapScene command reported `48 passed` with P0-R4-AC2 verified. |

Follow-up `/speckit.clarify` review on 2026-05-15 rechecked the current
feature spec against the PRD, constitution, and continuity artifacts. No
additional PRD-blocking ambiguity required a human product decision, and the
existing five safest-default clarifications remain the active feature contract.

Implementation inventory on 2026-05-15 found no implemented public `MapScene`,
`SceneRecipe`, `LabelPlan`, `ValidationReport`, `Diagnostic`, public Python
`LabelLayer`, or public Python `Tiles3DLayer` outside specs/docs/state. It
identified lower-level substrate in `python/forge3d/buildings.py`,
`python/forge3d/tiles3d.py`, `python/forge3d/bundle.py`,
`python/forge3d/style.py`, `python/forge3d/style_expressions.py`,
`python/forge3d/crs.py`, `src/labels`, `src/import`, `src/tiles3d`, and
`src/bundle`.

Feature `001` inventory on 2026-05-15 found no implemented product-level
`Diagnostic`, `ValidationReport`, `LayerSummary`, `SupportMatrixEntry`,
`RenderFailurePolicy`, or `SeverityPolicy` outside specs/docs/state. It
identified suspected placeholder/fallback/no-op paths in
`python/forge3d/buildings.py`, `python/forge3d/tiles3d.py`,
`python/forge3d/style.py`, `src/viewer/cmd/labels_command.rs`,
`src/labels/mod.rs`, `python/forge3d/terrain_params.py`,
`src/terrain/render_params/native_vt.rs`, `src/terrain/renderer/virtual_texture.rs`,
and `python/forge3d/bundle.py`.

Feature `002` inventory on 2026-05-15 found the current public Python label
workflow in `python/forge3d/viewer_ipc.py`, with no `add_labels` batch helper
or high-level `ViewerHandle` label methods. Native label substrate exists in
`src/labels`, `src/viewer/state/labels.rs`, and
`src/viewer/cmd/labels_command.rs`, but current IPC responses are generic and
do not expose created label/callout/vector overlay IDs. Suspected no-op or
truthfulness gaps include typography and declutter commands, curved-label
routing through line labels, callout style fields being ignored, line-label
glyph placement not becoming visible text instances, atlas ASCII fallback, and
load/remove commands acknowledging success without structured diagnostics.

Feature `003` inventory on 2026-05-15 found no implemented product
`LabelPlan` compile API, accepted/rejected plan objects, keepout model, plan
diagnostics, or deterministic plan serialization outside specs/docs/state.
Native substrate exists in `src/labels/layer.rs`, `src/labels/declutter.rs`,
`src/labels/rtree.rs`, `src/labels/collision.rs`,
`src/labels/projection.rs`, `src/labels/atlas.rs`, and
`src/labels/typography.rs`, but current paths do not retain rejected labels
with reason codes, expose center/above/below/left/right/radial candidates,
support keepout regions, implement polygon visual-center/polylabel fallback, or
prove terrain-occlusion behavior.

Feature `004` inventory on 2026-05-15 found no implemented public `MapScene`,
`SceneRecipe`, `TerrainSource`, `RasterOverlay`, `VectorOverlay`,
product-level `LabelLayer`, `PointCloudLayer`, `MapFurnitureLayer`,
`OutputSpec`, `ValidationReport`, product-level `Diagnostic`, `ReviewBundle`,
or public `LabelPlan` implementation outside specs/docs/state. Existing
substrate exists in `python/forge3d/viewer.py`, `python/forge3d/viewer_ipc.py`,
native `src/scene` render paths, `src/terrain/renderer`,
`python/forge3d/terrain_params.py`, `python/forge3d/style.py`,
`python/forge3d/buildings.py`, `python/forge3d/pointcloud.py`,
`python/forge3d/bundle.py`, `python/forge3d/crs.py`, `src/labels`,
`src/import`, `src/tiles3d`, `src/pointcloud`, `src/bundle`, and map-furniture
helpers. The typed validation/render/bundle/examples product contract remains
unimplemented.

Feature `006` inventory on 2026-05-15 found no implemented product-level
`MapScene`, `SceneRecipe`, `ValidationReport`, `Diagnostic`, or `LabelPlan`
surface for P2 material/VT/large-scene workflows outside specs/docs/state.
VT `normal` and `mask` are accepted by Python settings but native runtime pages
only `albedo`; building materials are scalar PBR with Pro-gated native ingest
and placeholder fallback paths; advanced labels have raw IPC/native substrate
but no deterministic P2 plan contract; large-scene cache, LOD, memory, and
instancing stats exist in lower-level modules but are not unified into
pre-render product diagnostics.

## Verification Commands

Latest product verification for feature `004` is recorded in the
source-derived PNG render completion section above. Older feature `006` static
inspection commands are recorded in
`specs/006-material-vt-large-scene-p2/research.md`. Earlier specification
validation commands:

```powershell
rg -n "\[NEEDS CLARIFICATION:|\[FEATURE NAME\]|\$ARGUMENTS|\[PRD: <AC-ID>\]" specs\005-map-assets-bundles-p1\spec.md
rg -n "P1-R1-AC[1-6]|P1-R2-AC[1-7]|P1-R3-AC[1-6]|P1-R4-AC[1-6]|P1-R5-AC[1-5]|FR-035|FR-036|FR-037|FR-038|SC-014|SC-015|SC-016" specs\005-map-assets-bundles-p1\spec.md
rg -n "missing_label_field|unicode_coverage_gap|unsupported_tile_format|unsupported_tile_feature|missing_external_asset|unavailable_terrain_sampler|MVP readiness|Clarifications" specs\005-map-assets-bundles-p1\spec.md
git status --short --branch --untracked-files=all
```

Observed validation state:

- `spec.md` contains no unresolved template placeholders.
- `spec.md` contains exactly five clarification bullets for the 2026-05-15
  session.
- P1 requirements remain non-MVP-blocking.
- Support wording no longer treats diagnostic support as support for
  unavailable rendering or ingestion paths.
- Feature-local diagnostics and deterministic bundle comparison defaults are
  captured.

## Current Git State Observed

```text
## 004-mapscene-mvp...origin/004-mapscene-mvp
 M .gitignore
 M AGENTS.md
 M docs/api/api_reference.rst
 M docs/guides/building_support_matrix.md
 M docs/guides/competitive_positioning.md
 M docs/guides/diagnostics_reference.md
 M docs/guides/offline_3d_map_rendering.md
 M docs/guides/style_support_matrix.md
 M docs/guides/tiles3d_support_matrix.md
 M docs/guides/virtual_texturing_support_matrix.md
 M docs/superpowers/state/current-context-pack.md
 M docs/superpowers/state/implementation-ledger.md
 M docs/superpowers/state/open-blockers.md
 M docs/superpowers/state/requirements-verification-matrix.md
 D examples/turkiye_river_basins_3d.py
 M logs/.3c2cf94182465f0d10df58878528dd39234a1134-audit.json
 M python/forge3d/forge3d.pdb
 M python/forge3d/map_scene.py
 M specs/004-mapscene-mvp/contracts/mapscene-contract.md
 M specs/004-mapscene-mvp/plan.md
 M specs/004-mapscene-mvp/quickstart.md
 M specs/004-mapscene-mvp/tasks.md
 M src/viewer/event_loop/runner.rs
?? examples/mapscene_buildings_labels.py
?? examples/mapscene_terrain_raster.py
?? examples/mapscene_vector_labels.py
?? tests/test_mapscene_docs.py
?? tests/test_mapscene_examples.py
?? tests/test_mapscene_label_plan_integration.py
?? tests/test_mapscene_quickstart.py
?? tests/test_mapscene_render_png.py
?? tests/test_mapscene_render_policy.py
?? tests/test_mapscene_save_bundle.py
?? tests/test_mapscene_support_status.py
?? tests/test_mapscene_validation.py
?? tests/test_mount_hood_white_cloud_timelapse.py
```

The dirty workspace contains unrelated or pre-existing feature changes. Do not
revert them. Use LFS-filter-disabled status if default `git status` hits the
`python/forge3d/forge3d.pdb` clean-filter access error.

## Next Exact Prompt

```text
[$speckit-implement](C:\Users\milos\forge3d\.agents\skills\speckit-implement\SKILL.md)

Implement only the remaining P0 label blockers P0-R1-AC2 and P0-R1-AC3:
make `set_label_typography` mutate measurable native/layout state and make
`set_declutter_algorithm` either change placement behavior or return an
explicit unsupported typed error accepted by the PRD. Before coding, identify
selected task IDs, why they form a coherent batch, PRD acceptance criteria
covered, and verification commands. Preserve feature `004` R-029 native/offscreen
render evidence plus source-derived compatibility output, keep P0-R4-AC2
`Verified`, and keep R-011 dirty worktree/change separation open before commit
or PR.
```

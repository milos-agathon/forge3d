# Feature Research: Label API Truth

**Feature directory**: `specs/002-label-api-truth`
**Date**: 2026-05-15
**Scope**: feature-specific implementation inventory before technical planning; no code implementation.

## Source Artifacts Inspected

- `docs/superpowers/plans/prd.md`
- `.specify/memory/constitution.md`
- `specs/002-label-api-truth/spec.md`
- `docs/superpowers/state/current-context-pack.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/open-blockers.md`

## Commands Used For Inspection

```powershell
git status --short
Get-Content -Raw docs/superpowers/plans/prd.md
Get-Content -Raw .specify/memory/constitution.md
Get-Content -Raw specs/002-label-api-truth/spec.md
if (Test-Path specs/002-label-api-truth/research.md) { Get-Content -Raw specs/002-label-api-truth/research.md } else { Write-Output '<missing>' }
Get-Content -Raw docs/superpowers/state/current-context-pack.md
Get-Content -Raw docs/superpowers/state/implementation-ledger.md
Get-Content -Raw docs/superpowers/state/requirements-verification-matrix.md
Get-Content -Raw docs/superpowers/state/open-blockers.md
rg --files | rg -i "(label|atlas|glyph|viewer_ipc|viewer|diagnostic|support|fixture|example|docs|tests|src/labels|viewer/cmd|ipc/protocol|\.pyi$)"
rg -n "label|Label|glyph|atlas|typograph|declutter|callout|curved|line_label|add_label|clear_labels|set_labels_enabled|load_label_atlas|set_label_typography|set_declutter" python/forge3d src/labels src/viewer/cmd src/viewer/ipc src/viewer/state tests docs examples assets -g "*.py" -g "*.pyi" -g "*.rs" -g "*.md" -g "*.json"
rg -n "MapScene|SceneRecipe|LabelPlan|ValidationReport|Diagnostic|LabelLayer|Label ID|label_id|label_ids" python src tests docs specs -g "*.py" -g "*.pyi" -g "*.rs" -g "*.md"
rg -n "Pro|pro_gated|feature.*pro|license|fallback|placeholder|no-op|noop|experimental|unsupported|TODO|stub|not implemented|return.*Ok|Success|Ack|Label" python/forge3d src/labels src/viewer/cmd src/viewer/ipc src/viewer/state tests docs examples -g "*.py" -g "*.pyi" -g "*.rs" -g "*.md"
rg --files fixtures tests assets examples docs python src | rg -i "(label|glyph|font|atlas|fuji|picking|overlay|mapbox|fixture|golden)"
rg -n "^def (add_label|add_labels|add_line_label|add_curved_label|add_callout|remove_label|clear_labels|set_labels_enabled|load_label_atlas|set_label_zoom|set_max_visible_labels|set_label_typography|set_declutter_algorithm|add_vector_overlay|send_ipc)|^class" python/forge3d/viewer_ipc.py python/forge3d/viewer.py python/forge3d/viewer.pyi python/forge3d/__init__.pyi
rg -n "AddLabel|AddLineLabel|AddCurvedLabel|AddCallout|RemoveLabel|ClearLabels|SetLabelsEnabled|LoadLabelAtlas|SetLabelTypography|SetDeclutterAlgorithm|add_label|add_line_label|add_callout|success|label_id|Response" src/viewer/cmd/labels_command.rs src/viewer/ipc/protocol/request.rs src/viewer/ipc/protocol/response.rs src/viewer/ipc/protocol/payloads.rs src/viewer/ipc/protocol/translate/labels.rs src/viewer/state/labels.rs src/viewer/scene_review.rs
rg -n "pub struct LabelManager|pub fn (new|add_label|add_line_label|remove_label|set_label_style|get_label|get_label_mut|clear|set_enabled|load_atlas|update|render_instances|line_label|glyph|pick_at)|Process line labels|TODO|not implemented|depth buffer|compute_line_label|compute_curved|pub struct|pub enum" src/labels/mod.rs src/labels/types.rs src/labels/line_label.rs src/labels/curved.rs src/labels/typography.rs src/labels/declutter.rs src/labels/atlas.rs src/labels/layer.rs src/labels/callout.rs src/labels/projection.rs
$lines = Get-Content python/forge3d/viewer_ipc.py; $lines[190..630]
$lines = Get-Content src/viewer/cmd/labels_command.rs; $lines[0..280]
$lines = Get-Content src/viewer/ipc/protocol/response.rs; $lines[0..90]
$lines = Get-Content src/labels/mod.rs; $lines[60..450]
$lines = Get-Content src/labels/mod.rs; $lines[450..530]
$lines = Get-Content src/labels/line_label.rs; $lines[0..260]
$lines = Get-Content src/labels/curved.rs; $lines[0..285]
$lines = Get-Content src/labels/atlas.rs; $lines[250..340]
$lines = Get-Content src/viewer/state/labels.rs; $lines[0..90]
rg -n "handle_cmd|IpcResponse::success|IpcResponse::error|println|cmd_success|labels_command" src/viewer/cmd src/viewer/ipc src/viewer/event_loop -g "*.rs"
$lines = Get-Content src/viewer/cmd/handler.rs; $lines[0..180]
$lines = Get-Content src/viewer/cmd/ipc_command.rs; $lines[0..220]
$lines = Get-Content tests/test_viewer_ipc.py; $lines[0..380]
rg -n "label|add_label|add_line_label|curved|callout|typography|declutter|atlas|id" tests/test_viewer_ipc.py tests/test_api_contracts.py tests/test_labels_pybindings.py tests/test_export_svg.py tests/test_bundle_roundtrip.py docs/guides/data_and_scene_workflows.md docs/start/architecture.md docs/gallery/02-mount-fuji-labels.md docs/examples/index.md examples/fuji_labels_demo.py -g "*.py" -g "*.md"
rg -n "LabelFlags|LabelStyle|EXPECTED_CLASSES|EXPECTED_FUNCTIONS|TestLabelBindings|Label bindings|Label" tests/test_api_contracts.py python/forge3d/__init__.py python/forge3d/__init__.pyi src/py_module/classes.rs src/py_module/functions.rs src/lib.rs
rg -n "remove_callout|add_callout|RemoveCallout|IpcRequest|ViewerCmd" src/viewer/ipc/protocol/request.rs src/viewer/viewer_enums.rs src/viewer/viewer_enums/commands.rs python/forge3d/viewer_ipc.py src/viewer/ipc/protocol/translate/labels.rs
rg -n "add_label|add_labels|set_labels_enabled|load_label_atlas|set_label_typography|set_declutter_algorithm|add_line_label|add_curved_label|add_callout|clear_labels" python/forge3d docs examples tests -g "*.py" -g "*.pyi" -g "*.md"
rg -n "def add_labels|add_labels|AddLabels|LabelBatch|Batch" python src tests docs examples specs/002-label-api-truth/spec.md
rg -n "MapScene|SceneRecipe|LabelPlan|ValidationReport|Diagnostic|class Diagnostic|@dataclass.*Diagnostic" python/forge3d src tests examples docs -g "*.py" -g "*.pyi" -g "*.rs" -g "*.md"
rg -n "add_vector_overlay|AddVectorOverlay|vector_overlay|id=|Loaded.*id|Response" python/forge3d/viewer_ipc.py src/viewer/cmd/vector_overlay_command.rs src/viewer/ipc/protocol/request.rs src/viewer/ipc/protocol/translate/overlays.rs src/viewer/ipc/protocol/response.rs
rg -n "label_manager|labels_enabled|label_count|load_label_atlas|update_labels|upload_to_renderer|visible_count|text_overlay|LabelManager" src/viewer src/core/text_overlay.rs src/shaders/text_overlay.wgsl
rg -n "viewer_ipc|add_label|LabelStyle|LabelFlags|__all__|Label" python/forge3d/__init__.py python/forge3d/__init__.pyi python/forge3d/viewer.pyi
```

## Phase 0 Planning Decisions

- Decision: Add high-level public label methods around existing viewer/label substrate, with final file paths TBD after inspecting `ViewerHandle` and stubs during tasks.
  Rationale: Current public workflow is mostly raw `viewer_ipc`; the MVP requires a basic label workflow without raw IPC.
  Alternatives considered: Leave raw IPC as the product workflow, rejected by P0-R1-AC4.

- Decision: Successful create operations must use real stable IDs from native/viewer state where possible.
  Rationale: Python-only synthetic IDs could drift from native state and would not satisfy update/inspect/remove workflows.
  Alternatives considered: Return generic success objects, rejected because current generic responses are part of the truthfulness gap.

- Decision: Typography, declutter, line, and curved label paths must either prove real state/render effects or return diagnostics.
  Rationale: The inventory found likely no-op or incomplete paths; no-op success is constitutionally blocked.
  Alternatives considered: Keep accepting commands for compatibility, rejected unless compatibility is paired with typed unsupported/experimental diagnostics.

## 1. Existing Modules, Classes, And Functions Relevant To This Feature

### Python modules

- `python/forge3d/viewer_ipc.py`
  - Raw IPC helpers: `send_ipc`, `add_label`, `remove_label`, `clear_labels`, `set_labels_enabled`, `load_label_atlas`, `add_line_label`, `set_label_zoom`, `set_max_visible_labels`, `add_curved_label`, `add_callout`, `remove_callout`, `set_label_typography`, and `set_declutter_algorithm`.
  - `add_label`, `add_line_label`, `add_curved_label`, and `add_callout` docstrings say responses may include created IDs, but the inspected native IPC response struct has no label ID field.
  - No `add_labels` batch helper was found outside the feature spec.

- `python/forge3d/viewer.py`
  - `ViewerHandle` has `send_ipc`, `load_bundle`, `load_overlay`, `load_point_cloud`, review-layer methods, and snapshot/camera helpers.
  - No high-level `ViewerHandle.add_label`, `add_labels`, `add_line_label`, `add_curved_label`, `add_callout`, `clear_labels`, `set_labels_enabled`, `load_label_atlas`, `set_label_typography`, or `set_declutter_algorithm` wrapper was found.

- `python/forge3d/viewer.pyi`
  - Stubs `ViewerHandle` but no typed label methods were found.

- `python/forge3d/__init__.py`
  - Imports and exports `viewer_ipc` as a module.
  - Does not appear to re-export raw label helpers directly as top-level functions.
  - Re-exports unrelated label-like dataclasses as `ExportLabel`, `ExportLabelStyle`, and `StyleLabelStyle`; these belong to export/style paths, not the viewer label API.

- `python/forge3d/__init__.pyi`
  - Contains native `LabelFlags` and `LabelStyle` type stubs.
  - No public `LabelBatchResult`, product-level label diagnostic class, or high-level viewer label methods were found.

- `python/forge3d/export.py`
  - Contains SVG/export-side `LabelStyle`, `Label`, `ExportScene.add_label`, and SVG label rendering support.
  - This is relevant as an existing cartographic label/export path, but it is Pro-gated SVG/PDF export substrate rather than the interactive/native viewer label API owned by this feature.

- `python/forge3d/style.py`
  - Contains style-side `LabelStyle`, `layout_to_label_style`, and `layer_to_label_style`.
  - Relevant to future label styling, but not a public high-level label creation API.

### Rust/native modules

- `src/labels/types.rs`
  - Defines `LabelId`, `LineLabelPlacement`, `LabelFlags`, `LabelStyle`, `LabelData`, `LineLabelData`, `GlyphPlacement`, and `LeaderLine`.

- `src/labels/mod.rs`
  - Defines `LabelManager`.
  - Relevant methods include `new`, `load_atlas`, `load_atlas_from_files`, `add_label`, `add_line_label`, `remove_label`, `set_label_style`, `get_label`, `get_label_mut`, `clear`, `set_enabled`, `is_enabled`, `label_count`, `set_zoom`, `get_zoom`, `set_max_visible`, `resize`, `leader_lines`, `pick_at`, `update`, `update_with_camera`, `upload_to_renderer`, `atlas_view`, and `visible_count`.
  - Point labels are converted into `TextInstance` values through `MsdfAtlas::layout_text`.
  - Line labels compute `glyph_positions` during update, but the code comments that line label rotation is tracked and not applied by glyph rendering yet.

- `src/labels/line_label.rs`
  - Defines `compute_line_label_placement` and `compute_glyph_advances`.
  - Includes tests for path length and sampling.
  - Computes per-glyph `GlyphPlacement` with position, rotation, and upside-down flipping logic.

- `src/labels/curved.rs`
  - Defines `PathPoint`, `SampledPath`, `CurvedGlyphInstance`, `CurvedTextLayout`, `layout_curved_text`, and `project_curved_layout`.
  - Native viewer command handling does not appear to use this curved layout path; `AddCurvedLabel` is currently translated into `LabelManager::add_line_label(..., LineLabelPlacement::Along, 0.0)`.

- `src/labels/atlas.rs`
  - Defines `GlyphMetrics` and `MsdfAtlas`.
  - Relevant methods include `load`, `load_from_files`, `get_glyph`, `measure_text`, and `layout_text`.
  - If glyph parsing produces no glyphs, the atlas creates fallback ASCII metrics.

- `src/labels/typography.rs`
  - Defines `TypographySettings`, `KerningTable`, `compute_advances_with_typography`, `TextCase`, and `apply_text_case`.
  - This is relevant substrate for `set_label_typography`, but no wiring from the viewer typography command to `LabelManager` state was found.

- `src/labels/declutter.rs`
  - Defines `PlacementCandidate`, `DeclutterConfig`, `DeclutterResult`, and `DeclutterAlgorithm`.
  - This is relevant substrate for declutter behavior, but no wiring from the viewer declutter command to `LabelManager` state was found.

- `src/labels/callout.rs` and `src/labels/leader.rs`
  - Define callout and leader-line substrate.
  - Native `AddCallout` command handling currently creates a normal label with offset/leader style and ignores several callout style fields; see suspected no-op paths.

- `src/labels/projection.rs`
  - Defines `LabelProjector`.
  - The depth-occlusion method notes that depth-buffer readback is not implemented.

- `src/labels/layer.rs`
  - Defines native `LabelFeature`, `FeatureGeometry`, `PlacementStrategy`, `LabelLayerConfig`, `GeneratedLabel`, `LabelPlacementType`, and `LabelLayer`.
  - More directly relevant to future data-driven `LabelLayer` work, but also relevant context for line/curved label placement status.
  - No PyO3 registration for this `LabelLayer` was identified in this feature scan.

- `src/labels/py_bindings.rs`
  - Defines PyO3 classes `PyLabelFlags` and `PyLabelStyle`.
  - Registered in `src/py_module/classes.rs`.

- `src/viewer/state/labels.rs`
  - Adds viewer methods `add_label`, `remove_label`, `clear_labels`, `set_labels_enabled`, `labels_enabled`, `label_count`, `load_label_atlas`, `update_labels`, and `resize_labels`.
  - These methods provide native state mutation but are not directly exposed as high-level Python `ViewerHandle` label methods.

- `src/viewer/cmd/labels_command.rs`
  - Handles `ViewerCmd` variants for labels.
  - Creates label IDs internally for point labels, line labels, curved labels, and callouts, then prints them to stdout.
  - Returns only `true` to command dispatch, not a structured response containing IDs or diagnostics.

- `src/viewer/ipc/protocol/request.rs`
  - Defines IPC request variants including `AddLabel`, `AddLineLabel`, `RemoveLabel`, `ClearLabels`, `SetLabelsEnabled`, `LoadLabelAtlas`, `SetLabelZoom`, `SetMaxVisibleLabels`, `AddCurvedLabel`, `AddCallout`, `RemoveCallout`, `SetLabelTypography`, and `SetDeclutterAlgorithm`.

- `src/viewer/ipc/protocol/translate/labels.rs`
  - Translates label `IpcRequest` variants into `ViewerCmd` variants.

- `src/viewer/ipc/protocol/response.rs`
  - Defines `IpcResponse` with `ok`, `error`, `stats`, `pick_events`, lasso/bundle/terrain/report/review fields.
  - No label ID, callout ID, overlay ID, diagnostic list, support-level field, or batch-result field was found.

- `src/viewer/ipc/server.rs`
  - Converts accepted queued commands to `IpcResponse::success()`.
  - This is a central reason that create commands can mutate native state while still returning no created ID to Python.

- `src/viewer/cmd/vector_overlay_command.rs`
  - `AddVectorOverlay` gets an internal ID from the terrain viewer and prints it.
  - Like labels, no IPC response field was found for returning the created overlay ID.

- `src/core/text_overlay.rs` and `src/shaders/text_overlay.wgsl`
  - Text overlay renderer and shader path used by label visible instances.

## 2. Existing Tests And Fixtures Relevant To This Feature

### Tests

- `tests/test_api_contracts.py::TestLabelBindings`
  - Verifies native `_forge3d.LabelStyle` and `_forge3d.LabelFlags` registration, defaults, field setters, color tuple behavior, repr, and kwargs construction.
  - Does not test viewer label creation, stable IDs, line-label glyph emission, typography command mutation, declutter command mutation, or diagnostics.

- `tests/test_labels_pybindings.py`
  - Proxy test importing `TestLabelBindings`.

- `tests/test_viewer_ipc.py`
  - Tests NDJSON formatting, response parsing, `ViewerHandle`, bundle install state, and scene-review query/mutation commands.
  - It includes scene-review label payload examples, but no tests were found for `viewer_ipc.add_label`, `add_line_label`, `add_curved_label`, `add_callout`, `clear_labels`, `set_labels_enabled`, `load_label_atlas`, `set_label_typography`, or `set_declutter_algorithm`.

- `tests/test_bundle_roundtrip.py`
  - Tests generic bundle/review-layer labels as dictionaries in `SceneBaseState` and `ReviewLayer`.
  - Relevant to label review state, not direct proof of viewer label API truthfulness.

- `tests/test_export_svg.py` and `tests/test_export_projection.py`
  - Test export-side `ExportScene.add_label`, SVG label rendering, labels disabled, bounds including labels, XML escaping, and multiple labels.
  - These tests cover Pro-gated export labels, not native viewer labels or IPC responses.

- Rust inline tests in `src/labels/line_label.rs`
  - Cover path length and path sampling.
  - They do not prove full viewer render emission for line/curved labels.

- Rust inline tests in `src/labels/curved.rs`, `src/labels/rtree.rs`, `src/labels/typography.rs`, and `src/labels/layer.rs`
  - Relevant native unit coverage exists in module files, but this inventory did not run tests and did not find public Python contract tests for P0 label API truth.

### Fixtures and assets

- `assets/fonts/default_atlas.json`
- `assets/fonts/default_atlas.png`
- `assets/gpkg/Mount_Fuji_places.gpkg`
- `assets/tif/Mount_Fuji_30m.tif`
- `tests/fixtures/mapbox_streets_v8.json`
- `docs/gallery/images/02-mount-fuji-labels.png`

No dedicated horizontal/vertical/diagonal/curved line-label test fixture files were found during this scan.

## 3. Existing Docs And Examples Relevant To This Feature

- `docs/guides/data_and_scene_workflows.md`
  - Documents viewer overlays and labels through raw IPC helpers such as `forge3d.viewer_ipc.add_label` and `forge3d.viewer_ipc.set_labels_enabled`.

- `docs/start/architecture.md`
  - States vector overlays and labels use raw IPC or `forge3d.viewer_ipc`.

- `docs/examples/index.md`
  - Lists `examples/fuji_labels_demo.py` as a label placement, typography, priorities, zoom ranges, and decluttering example using `viewer_ipc.add_label` and label settings.

- `docs/gallery/02-mount-fuji-labels.md`
  - Shows raw command payloads for labels and viewer snapshot use.

- `docs/viewer/index.md`
  - Mentions label placement and typography and lists `fuji_labels_demo.py`.

- `examples/fuji_labels_demo.py`
  - Uses raw IPC functions for atlas loading, `set_labels_enabled`, point labels, line labels, curved labels, callouts, typography, declutter algorithm, zoom, and max visible labels.
  - Filters place names to ASCII-renderable values because of atlas limitations.

- `examples/picking_demo.py`
  - Uses `set_labels_enabled`, `add_label`, and `load_label_atlas` through raw IPC for picking/label demo behavior.

No public label support matrix document was found for point labels, line labels, curved labels, callouts, typography, decluttering, atlas loading, missing glyph behavior, or experimental/unsupported paths.

## 4. Native/Rust Paths Relevant To This Feature

- `src/labels/mod.rs`
- `src/labels/types.rs`
- `src/labels/line_label.rs`
- `src/labels/curved.rs`
- `src/labels/atlas.rs`
- `src/labels/typography.rs`
- `src/labels/declutter.rs`
- `src/labels/collision.rs`
- `src/labels/rtree.rs`
- `src/labels/projection.rs`
- `src/labels/callout.rs`
- `src/labels/leader.rs`
- `src/labels/layer.rs`
- `src/labels/py_bindings.rs`
- `src/viewer/state/labels.rs`
- `src/viewer/cmd/labels_command.rs`
- `src/viewer/cmd/vector_overlay_command.rs`
- `src/viewer/cmd/handler.rs`
- `src/viewer/ipc/server.rs`
- `src/viewer/ipc/protocol/request.rs`
- `src/viewer/ipc/protocol/response.rs`
- `src/viewer/ipc/protocol/translate/labels.rs`
- `src/viewer/ipc/protocol/translate/overlays.rs`
- `src/viewer/viewer_enums/commands.rs`
- `src/viewer/viewer_struct.rs`
- `src/viewer/init/viewer_new.rs`
- `src/viewer/render/main_loop/frame_setup.rs`
- `src/viewer/render/main_loop/secondary.rs`
- `src/viewer/input/viewer_input.rs`
- `src/viewer/scene_review.rs`
- `src/core/text_overlay.rs`
- `src/shaders/text_overlay.wgsl`
- `src/export/svg_labels.rs`

## 5. Python Public API Paths Relevant To This Feature

- `python/forge3d/viewer_ipc.py`
  - Existing public module with raw label IPC helpers.
  - This is currently the main public Python label creation/configuration surface found by the scan.

- `python/forge3d/viewer.py`
  - Existing `ViewerHandle` public class, but no high-level label wrapper methods found.
  - `ViewerHandle.send_ipc` allows raw command usage, which the feature spec says should not be required for the basic MVP label workflow.

- `python/forge3d/viewer.pyi`
  - Existing type stub for `ViewerHandle`; no label methods found.

- `python/forge3d/__init__.py`
  - Publicly exports the `viewer_ipc` module but not individual label helper functions.

- `python/forge3d/__init__.pyi`
  - Public stubs for native `_forge3d.LabelFlags` and `_forge3d.LabelStyle`.
  - No public stubs found for `add_labels`, label batch results, structured label diagnostics, or high-level viewer label methods.

- `python/forge3d/export.py`
  - Public export-side labels and SVG generation; relevant but separate from native viewer label truth.

- `python/forge3d/style.py`
  - Public style-side label style conversion; relevant but separate from native viewer label creation and mutation.

## 6. Suspected Placeholder, Fallback, Or No-Op Paths

- `src/viewer/ipc/protocol/response.rs`
  - `IpcResponse` has no created label ID/callout ID/vector overlay ID field.
  - This makes the current raw IPC response contract unable to satisfy stable-ID acceptance criteria without a response change or different API path.

- `src/viewer/ipc/server.rs`
  - Accepted queued commands are acknowledged with `IpcResponse::success()`, which contains no label-specific payload.
  - This likely explains why Python helpers document optional IDs while native responses do not expose them.

- `src/viewer/cmd/labels_command.rs`
  - `AddLabel`, `AddLineLabel`, `AddCurvedLabel`, and `AddCallout` create internal IDs and print them, but the IDs are not returned in IPC responses.
  - `RemoveLabel` and `RemoveCallout` print whether removal succeeded, but still return `true` from command handling; the IPC caller likely receives success even if the target ID was absent.
  - `LoadLabelAtlas` logs errors but returns `true` from command handling even when atlas load fails.
  - `SetLabelTypography` ignores all payload fields and prints that settings were updated. No `LabelManager` typography state mutation was found.
  - `SetDeclutterAlgorithm` ignores seed/max-iterations and only prints the algorithm name. No `LabelManager` declutter state mutation was found.
  - `AddCurvedLabel` ignores `tracking` and `center_on_path`, then stores the request as a line label with `LineLabelPlacement::Along`.
  - `AddCallout` ignores background color, border color, border width, corner radius, and padding, then creates a normal label with optional offset/leader style.

- `src/labels/mod.rs`
  - Line labels are stored and have `glyph_positions` computed, but `visible_instances` are not extended for line-label glyphs in the inspected code path. The code comments that line labels track rotation but glyph rendering does not apply it yet.
  - Missing glyphs are not obviously collected into structured diagnostics before render; `MsdfAtlas::layout_text` appears to skip characters whose glyphs are absent.

- `src/labels/atlas.rs`
  - If no glyphs are parsed, fallback ASCII metrics are created. This may hide atlas parsing failure unless the feature adds diagnostics.
  - Space has a special fallback in measurement.

- `src/labels/projection.rs`
  - Depth occlusion notes that depth-buffer readback is not implemented. Terrain-elevated or terrain-occluded line-label behavior should be treated as uncertain until verified.

- `python/forge3d/viewer_ipc.py`
  - `add_label`, `add_line_label`, `add_curved_label`, and `add_callout` docstrings say responses may contain IDs, but they simply return `send_ipc(...)` and do not enforce/validate ID presence.
  - `set_label_typography` and `set_declutter_algorithm` expose public raw helpers, but native command handlers appear to acknowledge without real state mutation.

- `python/forge3d/viewer.py`
  - `ViewerHandle.send_ipc` keeps raw IPC available. No high-level public label workflow was found that hides raw IPC while enforcing truthful results.

- `add_labels` batch creation
  - No implementation found. This is a missing path for FR-002 and must not be assumed.

- `MapScene`, `SceneRecipe`, `LabelPlan`, `ValidationReport`, and `Diagnostic`
  - No implementation found under `python/forge3d`, `src`, `tests`, `examples`, or `docs` outside specs/state references. Feature `002` should not assume these product contracts already exist.

## 7. Pro-Gated Paths

- No label-specific `_check_pro_access(...)` gate was found in `python/forge3d/viewer_ipc.py`, `python/forge3d/viewer.py`, or `src/labels`.

- Adjacent cartographic/export paths are Pro-gated and may affect examples/docs but are not the core viewer label API:
  - `python/forge3d/export.py` gates SVG/PDF export.
  - `python/forge3d/map_plate.py` gates map plate composition.
  - `python/forge3d/style.py` gates Mapbox style loading/application.
  - `python/forge3d/bundle.py` gates scene bundle save/load.

- `tests/test_pro_gating.py` covers those adjacent Pro gates.

Uncertain: PRD/spec wording includes `Pro-gated` as a support-level classification for any label behavior that depends on native or Pro-only paths, but this scan did not identify a Pro-only label renderer path.

## 8. Unknowns That Need Verification

- `specs/002-label-api-truth/plan.md` is missing. Technical planning has not started.

- Whether the final public API should be implemented as `ViewerHandle` methods, top-level helper functions, `viewer_ipc` result validation, or a combination needs design. Existing docs point users to raw IPC, while the feature spec requires a basic workflow without raw IPC.

- How to return created IDs through the current asynchronous IPC model needs design. The native command dispatcher returns `true`, and the IPC server responds generically after queueing commands.

- Whether command execution can synchronously return real mutation results, or whether a query/inspection API is needed to validate state after queued commands, needs verification.

- `add_labels` batch API shape is not present. Per-label diagnostics and stable input-order correspondence need a new contract.

- Structured diagnostic classes are not implemented yet in inspected product paths. Feature `002` may depend on feature `001` for shared diagnostic types or may need feature-local compatible diagnostics.

- Missing glyph detection policy is unclear. The atlas can measure/layout known glyphs and skip missing glyphs, but no structured missing-glyph report was found.

- Typography substrate exists in `src/labels/typography.rs`, but how it should integrate with `LabelManager`, `MsdfAtlas::layout_text`, and visible layout metrics is unresolved.

- Declutter substrate exists in `src/labels/declutter.rs`, but how algorithm choice should affect `LabelManager::update_with_camera` is unresolved.

- Line-label support is underdeveloped: native placement exists, but renderable glyph instance emission for line labels was not found. Horizontal, vertical, diagonal, curved, and terrain-elevated path behavior require focused tests.

- Curved text substrate exists in `src/labels/curved.rs`, but the viewer command path appears not to use it. It is uncertain whether curved labels should be implemented now or classified as `experimental`/`unsupported` with diagnostics for this feature.

- Callout substrate exists, but current viewer command handling appears to create offset labels rather than full callout boxes. Need verify whether true callout rendering exists elsewhere before claiming support.

- Overlay creation stable IDs are part of P0-R1-AC1 where labels depend on overlays. `src/viewer/cmd/vector_overlay_command.rs` prints vector overlay IDs, but no IPC response field was found.

- Terrain-elevated line-label behavior depends on terrain sampling availability. No direct label terrain sampler integration was found.

- Existing examples may overclaim "Plan 3" typography/declutter/curved/callout behavior. Documentation updates need to preserve PRD support terms and avoid treating demo commands as production-supported until tests prove them.

- No product tests were run in this inventory. All findings are from static inspection.

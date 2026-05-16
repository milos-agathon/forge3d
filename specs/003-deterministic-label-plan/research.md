# Feature Research: Deterministic LabelPlan

**Feature directory**: `specs/003-deterministic-label-plan`  
**Date**: 2026-05-15  
**Scope**: feature-specific implementation inventory before technical planning; no code implementation.

## Source Artifacts Inspected

- `docs/superpowers/plans/prd.md`
- `.specify/memory/constitution.md`
- `specs/003-deterministic-label-plan/spec.md`
- `docs/superpowers/state/current-context-pack.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/open-blockers.md`
- `AGENTS.md`

## Commands Used For Inspection

```powershell
Get-Content -Raw docs/superpowers/plans/prd.md
Get-Content -Raw .specify/memory/constitution.md
Get-Content -Raw specs/003-deterministic-label-plan/spec.md
if (Test-Path specs/003-deterministic-label-plan/plan.md) { Get-Content -Raw specs/003-deterministic-label-plan/plan.md } else { 'NO_PLAN_MD' }
if (Test-Path specs/003-deterministic-label-plan/research.md) { Get-Content -Raw specs/003-deterministic-label-plan/research.md } else { 'NO_RESEARCH_MD' }
Get-Content -Raw docs/superpowers/state/current-context-pack.md
Get-Content -Raw docs/superpowers/state/implementation-ledger.md
Get-Content -Raw docs/superpowers/state/requirements-verification-matrix.md
Get-Content -Raw docs/superpowers/state/open-blockers.md
git status --short --branch
rg --files | rg "(^python/forge3d|^src|^tests|^examples|^docs|^specs/003|^assets|^fixtures|^data)"
rg -n "LabelPlan|LabelLayer|label plan|deterministic label|label_rejection|missing_glyph|keepout|polylabel|visual[-_ ]center|candidate|terrain_occluded|priority_lost|outside_view|empty_text|unsupported_geometry_type|invalid_geometry" -S
rg -n "class .*Label|def .*label|add_label|line_label|curved_label|LabelManager|LabelStyle|LabelFlags|LabelInstance|LabelLayout|Glyph|atlas|declutter|collision|rtree|RTree|rstar|priority" python src tests examples docs -S
rg -n "^(pub )?(struct|enum|fn|impl)|^    pub fn|^    fn|LabelPlan|LabelLayer|PlacementCandidate|Declutter|LabelFeature|FeatureGeometry|GeneratedLabel|LabelPlacementType|LabelManager|MsdfAtlas|LabelProjector|LabelData|LineLabelData|LabelStyle|LabelFlags" src/labels -S
rg -n "^(class|def|@dataclass)|LabelPlan|LabelLayer|Diagnostic|ValidationReport|Label|Style|add_label|viewer_ipc|terrain|elevation|sample" python/forge3d -S
rg -n "LabelPlan|LabelLayer|Diagnostic|missing_glyph|label_rejection|declutter|keepout|LabelStyle|LabelFlags|add_label|line_label|curved|candidate|priority|collision|terrain|elevation|polylabel|centroid" tests examples docs -S
rg -n "LabelPlan|LabelLayer|ValidationReport|Diagnostic|MapScene|SceneRecipe" python src tests examples docs --glob '!docs/_build/**' -S
rg -n "check_pro_access|requires_pro|Pro|pro|LicenseError|feature.*label|cfg\(feature|features =|extension-module|label" python/forge3d src Cargo.toml pyproject.toml -S
rg -n "pub mod labels|labels::|PyLabel|add_class::<.*Label|LabelStyle|LabelFlags|LabelLayer|LabelPlan|m.add" src/lib.rs src/py_module/classes.rs src/py_module/functions.rs src/labels/py_bindings.rs -S
rg --files assets tests examples docs python src | rg -i "(label|glyph|atlas|font|fuji|mapbox|style|gpkg|dem|terrain|fixture)"
$lines = Get-Content src/labels/layer.rs; $lines[0..230]
$lines = Get-Content src/labels/layer.rs; $lines[230..470]
$lines = Get-Content src/labels/declutter.rs; $lines[0..320]
$lines = Get-Content src/labels/mod.rs; $lines[240..445]
$lines = Get-Content python/forge3d/viewer_ipc.py; $lines[190..630]
Get-Content -Raw Cargo.toml
Get-Content -Raw pyproject.toml
```

## Phase 0 Planning Decisions

- Decision: Implement `LabelPlan` as a deterministic product compiler contract before relying on render output.
  Rationale: The repo has label/collision substrate but no accepted/rejected plan model, reason-code retention, or stable serialization.
  Alternatives considered: Reuse live viewer declutter output directly, rejected because it does not provide the required deterministic rejected-label diagnostics.

- Decision: Normalize all source labels, candidates, diagnostics, and payloads through stable ordering keys.
  Rationale: PRD Section 14.4 and the constitution require fixed inputs and seed to produce identical plans independent of map/set/filesystem order.
  Alternatives considered: Preserve caller order only, rejected because equivalent unordered inputs must still compare identically.

- Decision: Render/export fulfillment for this feature is a real deterministic payload or a typed diagnostic, not full `MapScene.render()`.
  Rationale: Full orchestration belongs to feature `004`, while P0-R3 still requires accepted plan data to be consumable downstream.
  Alternatives considered: Defer render/export payloads entirely, rejected by P0-R3-AC8.

## 1. Existing Modules, Classes, And Functions Relevant To This Feature

### Product-level `LabelPlan` surface

- No implemented `LabelPlan`, `AcceptedLabel`, `RejectedLabel`, `LabelCandidate`, `KeepoutRegion`, `PriorityClass`, product-level `Diagnostic`, `ValidationReport`, `MapScene`, or `SceneRecipe` class/function was found under `python/forge3d`, `src`, `tests`, `examples`, or non-built docs outside specs/state references.
- `specs/003-deterministic-label-plan/plan.md` does not exist yet. Technical planning has not started.

### Native label substrate

- `src/labels/mod.rs`
  - Defines `LabelManager`.
  - Relevant methods: `new`, `load_atlas`, `load_atlas_from_files`, `add_label`, `add_line_label`, `remove_label`, `set_label_style`, `get_label`, `get_label_mut`, `clear`, `set_enabled`, `is_enabled`, `label_count`, `set_zoom`, `get_zoom`, `set_max_visible`, `resize`, `leader_lines`, `pick_at`, `update`, `update_with_camera`, `upload_to_renderer`, `atlas_view`, and `visible_count`.
  - `update_with_camera` currently performs priority sorting, projection, R-tree collision insertion, visibility, leader-line generation, and text instance emission for point labels.
  - Line labels are processed after point labels, with glyph positions computed but not emitted into `visible_instances`; the code comments that line labels track rotation but glyph rendering does not apply it yet.

- `src/labels/types.rs`
  - Defines `LabelId`, `LineLabelPlacement`, `LabelFlags`, `LabelStyle`, `LabelData`, `LineLabelData`, `GlyphPlacement`, and `LeaderLine`.
  - `LabelStyle.priority`, zoom range, offset, rotation, leader flag, and horizon fade are directly relevant to scoring, placement, accepted payloads, and bounds.

- `src/labels/layer.rs`
  - Defines `FeatureType`, `LabelFeature`, `FeatureGeometry`, `PlacementStrategy`, `LabelLayerConfig`, `GeneratedLabel`, `LabelPlacementType`, `LabelLayer`, `MapboxSymbolLayer`, and helpers `features_from_points`, `features_from_lines`, `label_layer_from_mapbox_style`, and `apply_mapbox_text_style`.
  - `FeatureGeometry::centroid()` exists for point, line, and polygon inputs. For polygons it currently averages vertices; this is not a visual-center/polylabel implementation.
  - `LabelLayer::generate_labels()` creates `GeneratedLabel` entries and skips empty text silently. This is useful substrate but conflicts with this feature's need to retain rejected entries with `empty_text`.
  - Priority is derived from a `population` property when present. This is an existing heuristic, not a product-level priority-class contract.

- `src/labels/declutter.rs`
  - Defines `PlacementCandidate`, `DeclutterConfig`, `DeclutterResult`, `DeclutterAlgorithm`, `declutter_greedy`, `declutter_annealing`, and `declutter`.
  - `DeclutterConfig.seed` and local `SimpleRng` support reproducible annealing for fixed candidate order.
  - `declutter_greedy()` sorts only by descending priority. Tie behavior for equal priorities is uncertain because no stable secondary sort key was found.
  - `DeclutterResult` returns visible labels and positions, but no rejected labels or rejection reason codes.

- `src/labels/rtree.rs` and `src/labels/collision.rs`
  - Provide collision data structures: `LabelRTree`, `LabelBounds`, `CollisionGrid`, insertion, overlap checks, and queries.
  - These are relevant to `collision` / `priority_lost`, but they do not carry diagnostic reason objects.

- `src/labels/projection.rs`
  - Defines `LabelProjector` with `project`, `project_with_occlusion`, and `screen_size`.
  - `project_with_occlusion` notes depth-buffer readback is not implemented, so `terrain_occluded` support is uncertain.

- `src/labels/atlas.rs`
  - Defines `GlyphMetrics` and `MsdfAtlas`.
  - Relevant methods include `load`, `load_from_files`, `get_glyph`, `measure_text`, and `layout_text`.
  - If metric parsing yields no glyphs, fallback ASCII metrics are created; missing-glyph diagnostics are not exposed as structured plan diagnostics.

- `src/labels/typography.rs`
  - Defines `TypographySettings`, `KerningTable`, `compute_advances_with_typography`, `TextCase`, and `apply_text_case`.
  - Relevant for typography input to `LabelPlan`, but no product-level compile API was found using these settings.

- `src/labels/line_label.rs`
  - Defines `compute_line_label_placement` and `compute_glyph_advances`.
  - Relevant to future advanced or experimental line placement, but advanced repeated line labels and full curved text are non-goals for feature `003`.

- `src/labels/curved.rs`
  - Defines `SampledPath`, `CurvedGlyphInstance`, `CurvedTextLayout`, `layout_curved_text`, and `project_curved_layout`.
  - Relevant substrate, but curved labels are not part of this feature's minimum point/polygon compiler except as an unsupported/experimental path if encountered.

- `src/labels/callout.rs` and `src/labels/leader.rs`
  - Provide callout/leader-line geometry helpers. These may influence keepouts or render/export payloads, but no `LabelPlan` integration was found.

- `src/export/svg_labels.rs`
  - Can generate SVG text for visible `LabelData` with screen positions. Relevant to "compiled plan can be exported", but it consumes existing `LabelData`, not a `LabelPlan`.

### Python modules

- `python/forge3d/viewer_ipc.py`
  - Existing raw IPC label helpers: `add_label`, `remove_label`, `clear_labels`, `set_labels_enabled`, `load_label_atlas`, `add_line_label`, `set_label_zoom`, `set_max_visible_labels`, `add_curved_label`, `add_callout`, `remove_callout`, `set_label_typography`, `set_declutter_algorithm`.
  - Relevant to current label workflow, but not a deterministic offline compiler.

- `python/forge3d/viewer.py` and `python/forge3d/viewer.pyi`
  - `ViewerHandle` exposes `send_ipc`, `load_overlay`, `load_point_cloud`, bundle/review methods, and camera/snapshot helpers.
  - No high-level `LabelPlan` or `ViewerHandle.compile_label_plan` method was found.

- `python/forge3d/__init__.py` and `python/forge3d/__init__.pyi`
  - Native PyO3 `LabelFlags` and `LabelStyle` stubs exist.
  - No public `LabelPlan`, `LabelLayer`, `KeepoutRegion`, `Diagnostic`, or plan compile stub was found.

- `python/forge3d/style.py` and `python/forge3d/style_expressions.py`
  - Existing style-side label conversion and expression infrastructure may be relevant to future label sources, but no `LabelPlan` integration was found.

- `python/forge3d/bundle.py`
  - Contains `SceneBaseState` and label coercion for review/bundle labels via dictionaries.
  - No compiled `LabelPlan` serialization model was found.

- `python/forge3d/export.py`
  - Contains export-side `LabelStyle`, `Label`, `ExportScene.add_label`, and SVG export support. This is a separate public cartographic export path, not the requested deterministic compiler.

## 2. Existing Tests And Fixtures Relevant To This Feature

### Python tests

- `tests/test_api_contracts.py::TestLabelBindings`
  - Covers `_forge3d.LabelStyle` and `_forge3d.LabelFlags` construction, defaults, setters, repr, and keyword construction.
  - Does not cover deterministic plan compilation, accepted/rejected outputs, reason codes, candidate generation, keepouts, terrain sampling, or render/export payloads.

- `tests/test_labels_pybindings.py`
  - Proxy import for label binding tests.

- `tests/test_export_svg.py` and `tests/test_export_projection.py`
  - Cover export-side labels, SVG output, hidden labels, bounds, escaping, and projection.
  - Useful for export payload expectations, but not proof of `LabelPlan`.

- `tests/test_style_parser.py`
  - Includes `LabelStyle` conversion tests for style parsing. Relevant if label sources consume style settings.

- `tests/test_bundle_roundtrip.py`
  - Tests dictionary label state in `SceneBaseState` and `ReviewLayer`. Relevant to bundle persistence expectations, but no compiled plan round-trip was found.

- `tests/test_viewer_ipc.py`
  - Tests IPC formatting and `ViewerHandle` helper behavior. No deterministic label plan tests were found.

### Rust inline tests

- `src/labels/declutter.rs`
  - Tests greedy non-overlap, overlapping priority, and box overlap.
  - Does not assert deterministic equal-priority tiebreaking or rejected reason codes.

- `src/labels/rtree.rs` and `src/labels/collision.rs`
  - Test collision insertion/query basics.

- `src/labels/layer.rs`
  - Tests label layer creation, `features_from_points`, and `generate_labels`.
  - Does not test invalid geometry, empty text as retained rejection, polygon visual-center fallback, keepouts, or deterministic serialization.

- `src/labels/line_label.rs`, `src/labels/curved.rs`, `src/labels/typography.rs`, `src/labels/leader.rs`, `src/labels/callout.rs`, `src/labels/projection.rs`, and `src/export/svg_labels.rs`
  - Contain module-local unit tests relevant to placement, typography, callout, projection, and SVG helpers.

### Fixtures and assets

- `assets/fonts/default_atlas.json`
- `assets/fonts/default_atlas.png`
- `assets/gpkg/Mount_Fuji_places.gpkg`
- `assets/tif/Mount_Fuji_30m.tif`
- `python/forge3d/data/mini_dem.npy`
- `tests/fixtures/mapbox_streets_v8.json`
- `docs/gallery/images/02-mount-fuji-labels.png`

No dedicated `LabelPlan` fixture, rejected-reason fixture set, keepout fixture, invalid polygon fixture, or terrain-occlusion fixture was found during static inspection.

## 3. Existing Docs And Examples Relevant To This Feature

- `docs/superpowers/plans/prd.md`
  - Defines the `LabelPlan.compile(...)` conceptual API, required outputs, rejection reason vocabulary, determinism requirements, and diagnostics.

- `specs/003-deterministic-label-plan/spec.md`
  - Feature-specific contract for deterministic accepted/rejected plans, point/polygon candidates, keepouts, priorities, terrain sampling, missing glyphs, and render/export payloads.

- `docs/guides/data_and_scene_workflows.md`
  - Documents current label workflows via `forge3d.viewer_ipc.add_label`, not `LabelPlan`.

- `docs/guides/feature_map.md`
  - Lists vector overlays and labels as `viewer.send_ipc`, `viewer_ipc.add_vector_overlay`, and `viewer_ipc.add_label`.

- `docs/start/architecture.md`
  - Describes labels through raw IPC or `forge3d.viewer_ipc`.

- `docs/examples/index.md`, `docs/viewer/index.md`, and `docs/gallery/02-mount-fuji-labels.md`
  - Point to or demonstrate existing Fuji label workflows. These are useful examples of desired inputs, not deterministic plan docs.

- `examples/fuji_labels_demo.py`
  - Uses raw IPC label creation, priorities, typography, declutter, line labels, curved labels, callouts, and atlas loading.
  - Relevant source of sample label data and behavior expectations, but it is demo/IPC oriented and not a `LabelPlan` example.

- `examples/picking_demo.py`
  - Uses raw IPC labels and atlas loading for picking demo behavior.

No `LabelPlan Guide`, deterministic label compiler example, keepout/priority example, or rejection diagnostics doc was found.

## 4. Native/Rust Paths Relevant To This Feature

- `src/labels/mod.rs`
- `src/labels/types.rs`
- `src/labels/layer.rs`
- `src/labels/declutter.rs`
- `src/labels/rtree.rs`
- `src/labels/collision.rs`
- `src/labels/projection.rs`
- `src/labels/atlas.rs`
- `src/labels/typography.rs`
- `src/labels/line_label.rs`
- `src/labels/curved.rs`
- `src/labels/callout.rs`
- `src/labels/leader.rs`
- `src/labels/py_bindings.rs`
- `src/export/svg_labels.rs`
- `src/style/converters.rs`
- `src/style/expressions/mod.rs`
- `src/style/sprite.rs`
- `src/viewer/state/labels.rs`
- `src/viewer/cmd/labels_command.rs`
- `src/viewer/ipc/protocol/request.rs`
- `src/viewer/ipc/protocol/response.rs`
- `src/viewer/ipc/protocol/translate/labels.rs`
- `src/geo/reproject.rs` (uncertain relevance for CRS-transformed label sources; feature `003` assumes terrain/camera inputs but full CRS ingestion is later work)
- `src/terrain/camera.rs`
- `src/camera/mod.rs`
- `src/terrain/renderer/py_api.rs` (terrain rendering context; direct elevation sampler integration for labels not verified)
- `src/terrain/analysis.rs` (uncertain terrain sampling relevance)

## 5. Python Public API Paths Relevant To This Feature

- `python/forge3d/__init__.py`
  - Re-exports style/export label types and `viewer_ipc`, but no `LabelPlan`.

- `python/forge3d/__init__.pyi`
  - Stubs `_forge3d.LabelStyle` and `_forge3d.LabelFlags`; no `LabelPlan` stubs found.

- `python/forge3d/viewer_ipc.py`
  - Existing public raw label command helpers. This may be a compatibility input/output bridge, but it is not deterministic plan compilation.

- `python/forge3d/viewer.py` and `python/forge3d/viewer.pyi`
  - Existing viewer handle surface; no plan compiler.

- `python/forge3d/export.py`
  - Public export-side label objects and SVG export.

- `python/forge3d/style.py` and `python/forge3d/style_expressions.py`
  - Public style conversion/expression surface that may be consumed by later label-layer ingestion.

- `python/forge3d/bundle.py`
  - Public bundle save/load model that may eventually need compiled plan serialization.

- `python/forge3d/crs.py`
  - Public CRS utilities. Relevant if feature planning chooses to normalize source geometries before plan compilation, but full data-driven `LabelLayer` ingestion is outside feature `003`.

- `python/forge3d/terrain_scatter.py`, `python/forge3d/camera_rigs.py`, and `python/forge3d/terrain_params.py`
  - Existing terrain/camera helper surfaces. Terrain elevation sampling for labels needs focused design; no direct `LabelPlan` sampler was found.

## 6. Suspected Placeholder, Fallback, Or No-Op Paths

- `LabelPlan` itself is missing; any current API claiming deterministic accepted/rejected plans would be invented.

- `src/labels/layer.rs::LabelLayer::generate_labels`
  - Silently skips empty text. Feature `003` requires rejected labels or candidates with `empty_text`.
  - Uses simple centroid averaging for polygon geometry. Feature `003` requires centroid plus visual-center/polylabel fallback.
  - Does not validate invalid coordinates, zero-area polygons, unsupported geometry, or non-finite bounds with reason codes.

- `src/labels/declutter.rs`
  - Returns only visible IDs and positions. It does not preserve rejected labels or distinguish `collision` from `priority_lost`.
  - Equal-priority ordering is uncertain because greedy sort has no explicit deterministic secondary key.
  - Annealing is seeded, but stable results still depend on stable input candidate ordering.

- `src/labels/mod.rs::LabelManager::update_with_camera`
  - Performs live/current-view placement and collision, not an inspectable offline plan with accepted/rejected diagnostics.
  - Uses `HashMap` values for labels before sorting by priority. Equal-priority ordering is uncertain unless normalized elsewhere.
  - Labels outside projection are made invisible without retaining a structured `outside_view` rejection object.
  - Missing glyphs appear to be skipped by atlas layout rather than surfaced as `missing_glyphs` diagnostics.
  - Line-label positions are computed, but render instances are not emitted for line-label glyphs in the inspected path.

- `src/labels/projection.rs`
  - `project_with_occlusion` has an unimplemented depth-buffer readback note; `terrain_occluded` should be treated as uncertain until verified or diagnosed.

- `src/labels/atlas.rs`
  - Fallback ASCII metrics can hide atlas parsing failures. This must not be confused with full glyph coverage.

- `python/forge3d/viewer_ipc.py`
  - Existing label helper docstrings mention optional created IDs, but native IPC response shape does not include label IDs. This is mostly feature `002`, but it also affects whether compiled plans can be rendered via current viewer IPC.

- `python/forge3d/bundle.py`
  - Bundle labels are generic dictionaries. No compiled plan persistence, accepted/rejected ordering, bounds, seed, or diagnostics serialization was found.

- `python/forge3d/export.py`
  - Export labels can render SVG text, but no failure path for unsupported plan-export backends was found in relation to `LabelPlan`.

## 7. Pro-Gated Paths

- No label-specific Pro gate was found in `src/labels`, `python/forge3d/viewer_ipc.py`, or `python/forge3d/viewer.py`.

- Adjacent public paths that could affect render/export examples are Pro-gated:
  - `python/forge3d/export.py` gates SVG/PDF export through license checks.
  - `python/forge3d/map_plate.py` gates map plate composition.
  - `python/forge3d/style.py` gates Mapbox style loading/application.
  - `python/forge3d/bundle.py` gates scene bundle save/load.

- Build-feature gating relevant to this feature:
  - `Cargo.toml` defines `extension-module`, `proj`, `cog_streaming`, `copc_laz`, and other feature flags.
  - `pyproject.toml` maturin features are `["extension-module", "weighted-oit", "enable-tbn", "enable-gpu-instancing", "copc_laz"]`; `proj` and `cog_streaming` are not enabled in the Python wheel feature list.
  - `src/labels/py_bindings.rs` is reachable through `extension-module` registration, but only `PyLabelStyle` and `PyLabelFlags` are registered in `src/py_module/classes.rs`.

Uncertain: Terrain sampling for labels may require native terrain runtime state, raster/Python utilities, or future `MapScene` integration. No Pro-only label terrain sampler was found.

## 8. Unknowns That Need Verification

- Whether `LabelPlan` should be implemented primarily in Python, Rust, or a hybrid. Existing reusable placement/collision pieces are Rust-side, while the requested public API shape is Python-facing.

- How `LabelPlan.compile()` will accept label sources before feature `005` public `LabelLayer` ingestion exists. Existing Rust `LabelLayer` is not registered as a Python class.

- Whether feature `003` should introduce a feature-local diagnostic type or depend on feature `001` product diagnostics. No product-level `Diagnostic` implementation was found.

- Exact camera contract for compile-time projection: current Rust paths use `Mat4` view-projection, while the PRD example passes `camera=...` and `viewport=OutputSpec(...)`.

- Exact terrain sampler contract. Existing terrain code can render and sample data in several contexts, but no public label elevation sampler was found.

- Whether `terrain_occluded` is implementable in feature `003` or should emit a typed diagnostic when depth/terrain visibility data is unavailable.

- How to generate point candidates beyond the current single-position/offset model. Existing code supports label offsets but does not expose center/above/below/left/right/radial candidate enumeration as plan data.

- Whether a polylabel/visual-center dependency should be added or implemented locally. Existing polygon centroid is simple vertex averaging and likely insufficient.

- How keepout regions should be represented: title box, legend, scale bar, north arrow, and manual rectangles are required, but no existing keepout model was found.

- Deterministic ordering keys need design for labels, candidates, accepted labels, rejected labels, diagnostics, bounds, and serialization. Existing `HashMap` usage and priority-only sorting are not enough.

- How missing glyph coverage should be checked without relying on render-time `layout_text` skipping. `MsdfAtlas::get_glyph` exists, but no public coverage report exists.

- How render/export payloads should be shaped before full `MapScene.render()` exists. `src/export/svg_labels.rs` and `python/forge3d/export.py` are possible export substrates, but neither consumes `LabelPlan`.

- Whether compiled plans should serialize through `python/forge3d/bundle.py` in feature `003` or be bundle-ready only, leaving actual bundle round-trip to features `004`/`005`.

- Current branch and continuity files point at feature `005` or `002` in places, while this request targets `003-deterministic-label-plan`. This inventory treats `003` as the target by explicit user request.

- No product tests were run in this inventory. Findings are from static inspection only.

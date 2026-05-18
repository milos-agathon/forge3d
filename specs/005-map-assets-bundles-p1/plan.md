# Implementation Plan: Map Assets and Bundle Round-Trip P1

**Branch**: `005-map-assets-bundles-p1` | **Date**: 2026-05-15 | **Spec**: `specs/005-map-assets-bundles-p1/spec.md`
**Input**: Feature specification from `specs/005-map-assets-bundles-p1/spec.md`

## Summary

Implement P1 integrated map assets on top of the P0 product contract:
data-driven `LabelLayer` ingestion, production typography basics, integrated
`BuildingLayer`, public `Tiles3DLayer`, and deterministic map-scene bundle
round-trip behavior. This plan is intentionally hostile to overclaiming:
unavailable building, 3D Tiles, typography, font, terrain-sampling, and bundle
replay paths must emit structured diagnostics and preserve exact PRD support
classifications instead of being treated as render support.

The plan depends on the P0 contracts from features `001` through `004`:
`Diagnostic`, `ValidationReport`, `LabelPlan`, `MapScene`, `SceneRecipe`, and
`Bundle`. P1 implementation must extend those contracts, not create a parallel
scene model.

## Technical Context

**Language/Version**: Python 3.x public API with Rust/PyO3 substrate where
existing native paths are available.
**Primary Dependencies**: Feature `001` diagnostics/support matrices, feature
`002` label API truth, feature `003` deterministic `LabelPlan`, feature `004`
typed `MapScene`, plus existing `python/forge3d/buildings.py`,
`python/forge3d/tiles3d.py`, `python/forge3d/bundle.py`,
`python/forge3d/crs.py`, `python/forge3d/style.py`,
`python/forge3d/style_expressions.py`, `src/labels`, `src/import`,
`src/tiles3d`, and `src/bundle`.
**Storage**: Deterministic bundle manifests and payloads, external asset
references, source labels, compiled `LabelPlan` payloads where available,
diagnostics, layer summaries, and supported export settings.
**Testing**: `pytest` for Python API, validation, diagnostics, deterministic
serialization, fixtures, examples, and docs audits; Rust/native tests only when
bindings or native implementation paths change.
**Target Platform**: Offline Python map rendering and review workflows.
**Project Type**: Product-layer library API.
**Performance Goals**: Validation must run before render and expose cache/LOD
or memory information where available; deterministic fixtures use exact
comparison unless this plan is amended with a numeric tolerance before
verification.
**Constraints**: No raw IPC in canonical P1 workflows; no no-op success; no
implicit support upgrade for diagnostic-only paths; no full Cesium runtime
parity, full Mapbox GL parity, complex-script shaping, textured PBR building
implementation, VT normal/mask runtime, browser delivery, hosted tiles, DCC,
game/editor, or non-map rendering scope.
**Scale/Scope**: P1-R1 through P1-R5, Milestone 4, and P1 dependencies on
P0-R5/P0-R6 diagnostics and support-level honesty.

## Constitution Check

- [x] PRD traceability: plan maps P1-R1-AC1 through P1-R5-AC5, Milestone 4,
  and P1 diagnostic dependencies to implementation and verification evidence.
- [x] Offline map scope: plan is limited to offline Python map assets,
  validation, rendering preparation, and review bundles.
- [x] API truthfulness: ingestion, typography, building, tiles, bundle save,
  and bundle load must produce real state/artifacts or typed diagnostics.
- [x] Determinism: feature ordering, expression evaluation, terrain-sampling
  decisions, diagnostics ordering, fallback selection, manifests, label-plan
  payloads, and round-trip comparisons use stable ordering and exact fixture
  comparison unless a future amendment records a numeric tolerance.
- [x] Diagnostics first: unavailable, unsupported, experimental, Pro-gated,
  placeholder/fallback, missing-asset, missing-field, Unicode, terrain-sampler,
  and tile-feature paths are validated before render.
- [x] Typed product contract: P1 extends `MapScene`, `SceneRecipe`,
  `LabelPlan`, `ValidationReport`, `Diagnostic`, and `Bundle`; raw IPC remains
  internal or advanced tooling only.
- [x] Evidence plan: each in-scope acceptance criterion has required code,
  test, docs, diagnostics, and verification-matrix evidence below.
- [x] Support wording: plan uses PRD Appendix B terms exactly and does not
  overclaim 3D Tiles, buildings, fonts, style, VT, or non-Latin shaping.
- [x] Compatibility: additive wrappers and adapters preserve existing APIs and
  surface Pro/fallback state rather than hiding it.
- [x] Continuity: completion requires updates to
  `docs/superpowers/state/current-context-pack.md`,
  `docs/superpowers/state/implementation-ledger.md`, and, when status changes,
  `docs/superpowers/state/requirements-verification-matrix.md`.

## Project Structure

```text
specs/005-map-assets-bundles-p1/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── p1-assets-bundle-contract.md
└── tasks.md
```

Existing paths to inspect during task generation and implementation:

```text
python/forge3d/__init__.py
python/forge3d/__init__.pyi
python/forge3d/buildings.py
python/forge3d/tiles3d.py
python/forge3d/bundle.py
python/forge3d/crs.py
python/forge3d/style.py
python/forge3d/style_expressions.py
python/forge3d/viewer.py
python/forge3d/viewer_ipc.py
src/labels/
src/import/
src/tiles3d/
src/bundle/
src/py_module/functions/io_import.rs
src/py_module/classes.rs
assets/fonts/default_atlas.json
assets/fonts/default_atlas.png
assets/geojson/
tests/test_buildings_extrude.py
tests/test_buildings_cityjson.py
tests/test_buildings_materials.py
tests/test_3dtiles_parse.py
tests/test_3dtiles_sse.py
tests/test_bundle_roundtrip.py
tests/test_bundle_render.py
tests/test_bundle_cli.py
tests/test_style_parser.py
tests/test_crs_reproject.py
tests/test_crs_auto.py
tests/test_labels_pybindings.py
docs/guides/
docs/gallery/
docs/tutorials/
docs/api/api_reference.rst
examples/
```

**Structure Decision**: P1 APIs should live in the P0 product-layer modules
established by feature `004`, with exports added in `python/forge3d/__init__.py`
and stubs in `python/forge3d/__init__.pyi` as needed. If feature `004` has not
created those modules when P1 starts, P1 tasks must block or depend on the P0
tasks rather than creating a second `MapScene` or `LabelPlan` implementation.

## Exact PRD API Surface

Task generation must preserve the PRD Section 11 required API shape exactly
unless implementation proves a method impossible and records a typed diagnostic,
docs update, and accepted blocker:

```python
LabelLayer.from_geodataframe(gdf, text_field="name", crs="EPSG:4326", ...)
LabelLayer.from_features(features, text_field="name", crs="EPSG:4326", ...)
LabelLayer.from_style_layer(style_layer, features, text_field="name", ...)
label_layer.compile_labels(camera, viewport, terrain)

BuildingLayer.from_geojson(...)
BuildingLayer.from_cityjson(...)
BuildingLayer.from_mesh(...)

Tiles3DLayer.from_tileset_json(...)
Tiles3DLayer.from_b3dm(...)
```

`compile_labels()` must feed feature `003`'s `LabelPlan.compile()` contract
instead of creating a second planner. `from_mesh()` must be treated as part of
the public building API shape even if the implementation can only validate or
diagnose unsupported mesh-derived building content in this feature.

## Hostile Review Risk Table

| Area reviewed | Initial gap found | Current risk | Required amendment now present |
|---|---|---|---|
| PRD acceptance-criterion coverage | Template had no P1 AC mapping. | Green | Acceptance Mapping covers P1-R1-AC1 through P1-R5-AC5 plus Milestone 4. |
| Test coverage | Template had no tests. | Green | Test Strategy First lists positive, negative, determinism, docs, and bundle tests. |
| Diagnostics and support-level honesty | No diagnostic inventory or classification guardrails. | Green | Diagnostics section requires PRD codes plus feature-local codes and exact support terms. |
| Determinism/reproducibility | No ordering, seed, or comparison policy. | Green | Determinism policy requires stable ordering and exact fixture comparisons by default. |
| No-op success prevention | No guard against placeholder success. | Green | No-op prevention is a gating rule for ingestion, font atlas, buildings, tiles, and bundles. |
| Raw IPC avoidance | Template did not ban raw IPC for workflows. | Green | API and test strategy require P1 workflows through typed product APIs only. |
| Bundle/documentation requirements | No bundle schema or docs audit. | Green | Bundle strategy and tasks require manifests, label sources/plans, diagnostics, and support matrices. |
| New-chat continuity artifacts | No state update requirement. | Green | Constitution check and verification section require context pack, ledger, and matrix updates. |
| P0/P1/P2 boundaries | No boundary statement. | Green | Scope boundary section keeps P1 non-MVP-blocking and excludes P2/non-goals. |
| File-path accuracy | Template paths and option blocks were generic. | Green | Project Structure lists actual feature, source, docs, test, fixture, and native paths. |
| Exact PRD API shape | Plan implied but did not lock `compile_labels`, `from_mesh`, `from_tileset_json`, and `from_b3dm`. | Green | Exact PRD API Surface now names required methods and blocks silent omission during task generation. |

No red items remain in this plan. Yellow risks below are accepted planning
risks or open blockers and must stay visible during task generation.

## Accepted Yellow Risks and Open Blockers

| ID | Yellow risk / blocker | Why accepted at planning time | Required handling |
|---|---|---|---|
| Y-001 | P1 depends on P0 `Diagnostic`, `ValidationReport`, `LabelPlan`, and `MapScene` contracts that may not exist when implementation starts. | Feature `005` is explicitly P1 follow-on work and not MVP-blocking. | `tasks.md` must create dependencies on features `001` through `004` and must not implement parallel product contracts. |
| Y-002 | Building native availability varies by package/license and may be Pro-gated. | PRD requires honest distinction, not guaranteed open availability. | Validation must emit `pro_gated_path`, `placeholder_fallback`, or `unsupported` before render where native support is unavailable. |
| Y-003 | 3D Tiles public Python render integration may be incomplete even though Rust substrate exists. | PRD classifies this as underdeveloped public integration. | Use supported local fixtures where possible; otherwise emit `python_public_3dtiles_incomplete`, `unsupported_tile_format`, or `unsupported_tile_feature`. |
| Y-004 | TTF/OTF atlas generation support is uncertain in the current label pipeline. | Research found text/font substrate but no clear label atlas generator. | Implement generation only if real; otherwise return typed diagnostics and document unavailable support. |
| Y-005 | Exact pixel comparison may not be viable for every render path. | Constitution permits tolerance only when documented before verification. | Keep exact fixture comparisons for plan/manifest/diagnostic outputs; add a numeric tolerance amendment before any non-exact pixel verification. |

## Test Strategy First

1. `LabelLayer` ingestion tests for point, line, and polygon features,
   unsupported/invalid geometry, stable feature ordering, missing text fields,
   missing glyphs, CRS transforms, unavailable CRS transforms, terrain-sampling
   success, and unavailable terrain sampler diagnostics.
2. Label expression tests for `{name}`, `get`, `concat`, `coalesce`,
   `upcase`, and `downcase`; missing properties must emit
   `missing_label_field` instead of silently producing empty labels.
3. `LabelLayer` to `LabelPlan` integration tests proving accepted/rejected
   plan output and deterministic repeated compilation for fixed inputs.
4. Typography tests for default Latin atlas coverage, Unicode coverage gaps,
   fallback ranges, kerning, tracking, line-height, multiline labels, callouts,
   and typed unsupported diagnostics where controls are exposed but unavailable.
5. Font atlas generation tests for TTF/OTF where real generation exists, or
   diagnostics tests proving generation failures do not create usable-looking
   placeholder atlases.
6. Building validation tests for available native path, `Pro-gated`,
   `placeholder/fallback`, `unsupported`, zero geometry, geometry counts,
   bounding boxes, scalar PBR documentation, and textured PBR unsupported
   diagnostics unless implemented end to end.
7. Building render tests for CityJSON and GeoJSON fixtures only where the
   native path is actually available.
8. `Tiles3DLayer` tests for supported local fixture loading, unsupported tile
   formats, unsupported B3DM/GLB features, cache stats where available,
   screen-space-error or LOD config where available, and diagnostics where not.
9. Bundle tests proving deterministic save/load of terrain metadata, layer
   metadata, camera, lighting, output spec, source labels, compiled
   `LabelPlan` payloads where available, diagnostics, export settings, external
   asset references, support status, and review metadata.
10. Missing external asset tests for bundle load and validation.
11. No-op success tests for label ingestion, atlas generation, building ingest,
   3D Tiles load, bundle save, and bundle load.
12. Docs audits for support matrices, 3D Tiles no-Cesium-parity wording,
   building scalar/textured material status, typography coverage, bundle
   round-trip behavior, diagnostics reference, and no raw IPC in canonical P1
   examples.

## Implementation Strategy

1. Gate P1 implementation on the P0 product API modules from feature `004`.
   If missing, generate dependent tasks instead of creating duplicate classes.
2. Extend `LabelLayer` constructors to ingest feature dictionaries,
   GeoDataFrames where available, and style-derived label rules, using
   `python/forge3d/crs.py` and `python/forge3d/style_expressions.py` for
   existing CRS and expression substrate.
3. Normalize label feature order by stable layer ID, feature ID, geometry type,
   source order, and expression key before candidate generation or diagnostics.
4. Connect ingested labels to `LabelPlan.compile()` from feature `003`; missing
   `LabelPlan` functionality is a blocker, not a P1-local reimplementation.
5. Add typography/font objects to the public product contract only where they
   can affect layout, validation, serialization, or typed diagnostics.
6. Wrap existing `python/forge3d/buildings.py` and native import substrate in a
   `BuildingLayer` product adapter that records support status, geometry
   counts, bounding boxes, material status, and renderability.
7. Wrap existing `python/forge3d/tiles3d.py` and any native `src/tiles3d`
   bindings that exist into a `Tiles3DLayer` product adapter scoped to local
   fixtures and offline review/render preparation.
8. Extend the `MapScene` bundle path from feature `004` to store deterministic
   P1 asset state, source labels, compiled label plans where available,
   diagnostics, support status, missing-asset diagnostics, and review metadata.
9. Update docs and examples after behavior is implemented or diagnosed; docs
   must not claim support before code and tests prove it.

## Diagnostics And Support-Level Honesty

P1 must preserve the PRD diagnostic inventory where applicable:

```text
crs_mismatch
missing_glyphs
pro_gated_path
placeholder_fallback
experimental_feature
python_public_3dtiles_incomplete
estimated_gpu_memory
label_rejection_summary
```

P1 must add and test feature-local structured diagnostics:

```text
missing_label_field
unicode_coverage_gap
unsupported_tile_format
unsupported_tile_feature
missing_external_asset
unavailable_terrain_sampler
```

Every diagnostic must include code, severity, message, remediation, support
level where applicable, affected layer ID, affected object or feature ID where
known, and deterministic serialization fields suitable for bundles.

Diagnostic support does not make unavailable building ingestion, 3D Tiles
rendering, font generation, terrain sampling, textured PBR, complex shaping, or
bundle replay `supported`. Those paths keep their exact support level:
`supported`, `underdeveloped`, `missing`, `Pro-gated`, `placeholder/fallback`,
`experimental`, `unsupported`, or `non-goal`.

## Determinism And Reproducibility Policy

- Feature IDs, source order, diagnostics, layer summaries, bundle manifest
  entries, external asset references, label source payloads, compiled
  `LabelPlan` payloads, fallback decisions, and support summaries must use
  deterministic ordering.
- Label expression evaluation must not depend on Python dict iteration order
  unless keys are normalized first.
- CRS transform reporting and terrain-sampling decisions must produce stable
  diagnostics for fixed inputs.
- Bundle save/load/validate fixtures use exact serialized comparison by
  default.
- Render or pixel-output tests use exact comparison only unless this plan is
  amended before verification with a numeric tolerance and rationale.

## No-Op Success Prevention

The following paths must fail honestly or return typed diagnostics instead of
success:

- `LabelLayer` constructors that produce no labels due to missing fields,
  invalid geometry, unsupported geometry, missing glyphs, unavailable CRS
  transform, or unavailable terrain sampler.
- `FontAtlas.from_font()` or equivalent atlas generation that cannot load the
  font, cannot generate coverage, or falls back to placeholder glyph metrics.
- Building ingestion that returns zero geometry, placeholder arrays, unavailable
  native support, unsupported format, unsupported textured material data, or
  Pro-gated paths.
- `Tiles3DLayer` loading that parses metadata but cannot produce supported
  render/review content.
- Bundle save that omits required scene intent, diagnostics, label sources,
  label plans where available, support status, or external asset references.
- Bundle load that cannot reconstruct required state or locate referenced
  external assets.

## Raw IPC Avoidance

P1 canonical examples and tests must use typed product APIs, not
`python/forge3d/viewer_ipc.py`. Existing IPC helpers may remain implementation
substrate behind adapters, but a user must be able to exercise data-driven
labels, buildings, 3D Tiles validation, and bundle round-trip through
`MapScene`, `LabelLayer`, `BuildingLayer`, `Tiles3DLayer`, `ValidationReport`,
`Diagnostic`, and `Bundle` contracts.

## Acceptance Mapping

| PRD AC | Components | Required tests | Required docs/diagnostics |
|---|---|---|---|
| P1-R1-AC1 | `LabelLayer.from_features`, `from_geodataframe`, `from_style_layer` | Point, line, polygon, unsupported geometry tests | Geometry support docs and affected feature diagnostics |
| P1-R1-AC2 | CRS adapter | Transform and mismatch tests | `crs_mismatch` remediation |
| P1-R1-AC3 | Terrain sampling policy | Available sampler and unavailable sampler tests | `unavailable_terrain_sampler` |
| P1-R1-AC4 | Label expression evaluator | `{name}`, `get`, `concat`, `coalesce`, casing tests | `missing_label_field` |
| P1-R1-AC5 | `label_layer.compile_labels()` feeding `LabelPlan.compile()` | Accepted/rejected and deterministic compile tests | `label_rejection_summary` |
| P1-R1-AC6 | Pre-render label validation | Missing field and glyph tests | `missing_label_field`, `missing_glyphs` |
| P1-R2-AC1 | `FontAtlas.default_latin` | Coverage fixture tests | Default Latin atlas coverage docs |
| P1-R2-AC2 | `FontAtlas.from_font` | TTF/OTF success or typed failure tests | Font generation support status |
| P1-R2-AC3 | `FontFallbackRange` | Deterministic fallback selection tests | Fallback range docs |
| P1-R2-AC4 | Unicode coverage validation | Unicode gap tests | `unicode_coverage_gap` |
| P1-R2-AC5 | `TypographySettings` | Kerning/tracking/line-height metric tests or diagnostics | Exposed-control support levels |
| P1-R2-AC6 | Multiline/callout support | Layout/render tests or unsupported diagnostics | Multiline/callout support matrix |
| P1-R2-AC7 | Optional shaping note | Non-blocking design/doc test | Complex shaping marked non-goal/deferred unless reprioritized |
| P1-R3-AC1 | `BuildingLayer.from_geojson`, `from_cityjson`, `from_mesh` validation | Native, Pro-gated, fallback, unsupported tests | Support status diagnostics |
| P1-R3-AC2 | `MapScene` building render prep | CityJSON/GeoJSON render tests where native path exists | No support claim when unavailable |
| P1-R3-AC3 | Building summaries | Geometry count and bounds tests | Layer diagnostics include counts/bounds |
| P1-R3-AC4 | Scalar PBR status | Docs audit | Building support matrix |
| P1-R3-AC5 | Textured PBR status | End-to-end tests or unsupported diagnostic tests | Textured PBR support wording |
| P1-R3-AC6 | Zero-geometry fallback handling | Placeholder/fallback negative tests | `placeholder_fallback` |
| P1-R4-AC1 | `Tiles3DLayer.from_tileset_json`, `from_b3dm` public API | Supported local fixture load tests | Public API docs |
| P1-R4-AC2 | Tile format validation | Supported/unsupported format tests | `unsupported_tile_format` |
| P1-R4-AC3 | Cache stats adapter | Available cache stats tests or unavailable diagnostics | Layer summary docs |
| P1-R4-AC4 | LOD/SSE config | Config tests or unsupported diagnostics | LOD support wording |
| P1-R4-AC5 | B3DM/GLB validation | Unsupported feature tests | `unsupported_tile_feature` |
| P1-R4-AC6 | 3D Tiles docs | Docs wording audit | Explicit no full Cesium runtime parity |
| P1-R5-AC1 | Bundle manifest/schema | Save tests for required fields | Deterministic manifest docs |
| P1-R5-AC2 | Bundle load/reconstruct | Available-asset round-trip tests | Renderability support status |
| P1-R5-AC3 | Missing asset validation | Missing external asset tests | `missing_external_asset` |
| P1-R5-AC4 | Review state | Review metadata tests | Bundle review docs |
| P1-R5-AC5 | Round-trip automation | Save/load/validate deterministic tests | Verification matrix evidence |
| Milestone 4 | Integrated assets | End-to-end label/building/tile/bundle workflow tests | Support matrix and exit review |

## Bundle And Documentation Requirements

Bundle round-trip must preserve or diagnose:

- terrain source metadata
- layer metadata and support status
- camera, lighting, and output spec
- source labels
- compiled `LabelPlan` payloads where available
- diagnostics and layer summaries
- supported export settings
- external asset references and missing-asset diagnostics
- review metadata needed to reconstruct scene intent

Required docs/support updates:

- data-driven label ingestion guide or API docs
- typography/font coverage guide
- building support matrix with scalar PBR and textured PBR status
- 3D Tiles support matrix stating no full Cesium runtime parity
- bundle round-trip guide with source-label and compiled-plan policy
- diagnostics reference entries for feature-local codes
- canonical P1 examples using typed APIs only

## Scope Boundaries

- P1 work is not MVP-blocking unless a documented human decision expands MVP
  scope.
- P1 may depend on P0 behavior but must not weaken P0 gates.
- Full complex-script shaping, VT normal/mask runtime support, textured PBR
  building implementation, large production 3D Tiles hierarchies, full Cesium
  runtime parity, streamed MVT rendering, full Mapbox GL parity, browser
  delivery, hosted tile services, DCC tooling, game/editor tooling, and
  non-map rendering are out of scope unless separately reprioritized.
- Textured PBR buildings satisfy P1 only by being implemented end to end or
  explicitly marked `unsupported` through diagnostics and docs.

## Tasks That Must Be Generated Later

`/speckit.tasks` must generate tasks for:

1. P0 dependency checks for `Diagnostic`, `ValidationReport`, `LabelPlan`,
   `MapScene`, and `Bundle` contracts.
2. Public P1 API/stub/export updates for `LabelLayer`, `FontAtlas`,
   `FontFallbackRange`, `TypographySettings`, `BuildingLayer`, `Tiles3DLayer`,
   and bundle helpers.
3. `LabelLayer` ingestion implementation and tests for P1-R1-AC1 through
   P1-R1-AC6, including `from_geodataframe`, `from_features`,
   `from_style_layer`, and `label_layer.compile_labels()`.
4. Typography/font coverage implementation or typed diagnostics and tests for
   P1-R2-AC1 through P1-R2-AC7.
5. Building adapter implementation, support-status diagnostics, fixture tests,
   and docs for P1-R3-AC1 through P1-R3-AC6, including `from_geojson`,
   `from_cityjson`, and `from_mesh`.
6. 3D Tiles adapter implementation, fixture/negative tests, cache/LOD behavior,
   and docs for P1-R4-AC1 through P1-R4-AC6, including `from_tileset_json`
   and `from_b3dm`.
7. Bundle manifest/schema/load/save changes and deterministic round-trip tests
   for P1-R5-AC1 through P1-R5-AC5.
8. Feature-local diagnostic code definitions, serialization tests, remediation
   text, and bundle inclusion tests.
9. No-op success negative tests across ingestion, atlas generation, buildings,
   tiles, bundle save, and bundle load.
10. Deterministic ordering tests and exact fixture comparisons for manifests,
    diagnostics, label plans, layer summaries, and round-trip outputs.
11. Docs/support-matrix updates and docs wording audits.
12. Example updates proving typed API usage without raw IPC.
13. Updates to `docs/superpowers/state/requirements-verification-matrix.md`,
    `docs/superpowers/state/implementation-ledger.md`, and
    `docs/superpowers/state/current-context-pack.md`.

## Verification Matrix And Ledger

No P1 row may move beyond `Planned` until implementation, tests, diagnostics,
docs, and exact verification commands are recorded. `Verified` is allowed only
after code evidence, test evidence, docs evidence, diagnostic evidence where
relevant, and state artifact updates exist.

This plan review itself does not change requirement status in
`docs/superpowers/state/requirements-verification-matrix.md`; rows remain
`Planned` because no product code or product tests were implemented.

## Rollback And Safe Failure

If P1 implementation encounters missing P0 contracts, unavailable native
building paths, incomplete 3D Tiles render paths, unsupported font generation,
missing external assets, or unsupported bundle replay, it must stop at typed
diagnostics and docs rather than reporting successful renderable output.
Misleading artifacts are worse than blocked artifacts for this feature.

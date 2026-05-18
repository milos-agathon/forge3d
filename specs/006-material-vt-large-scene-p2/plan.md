# Implementation Plan: Material, VT, and Large-Scene P2

**Branch**: `006-material-vt-large-scene-p2` | **Date**: 2026-05-15 | **Spec**: `specs/006-material-vt-large-scene-p2/spec.md`
**Input**: Feature specification from `specs/006-material-vt-large-scene-p2/spec.md`

## Summary

Address P2 polish gaps by either implementing or diagnosing VT normal/mask runtime support, textured building materials, advanced deterministic static labels, and large-scene memory/cache/LOD/instancing diagnostics. Existing substrate is in `python/forge3d/terrain_params.py`, native terrain VT runtime, `python/forge3d/buildings.py`, `textures.py`, `materials.py`, native import/IO modules, `viewer_ipc.py`, `src/labels`, `python/forge3d/pointcloud.py`, `tiles3d.py`, `geometry.py`, and terrain scatter/instancing APIs. New product paths extend P0/P1 APIs and are TBD until those foundations exist.

## Technical Context

**Language/Version**: Python 3.x typed API with Rust/PyO3 native rendering/diagnostics substrate.  
**Primary Dependencies**: P0/P1 `MapScene`, diagnostics, LabelPlan, building/tile/bundle APIs, VT runtime, texture/material helpers, point-cloud/tile/cache/LOD/instancing stats.  
**Storage**: Scene validation reports, texture asset references, advanced label plans, resource summaries, support matrices.  
**Testing**: pytest for validation, diagnostics, fixture comparisons; Rust tests if VT runtime/material/label/instancing code changes.  
**Target Platform**: Offline map production and large-scene review, not live globe streaming.  
**Project Type**: Product-layer library extension plus possible native runtime work.  
**Performance Goals**: Large-scene validation summarizes known metadata without expensive render where possible.  
**Constraints**: P2 does not block MVP; exact deterministic fixture comparison unless tolerance is recorded before verification.  
**Scale/Scope**: P2-R1 through P2-R4, Milestone 5.

## Constitution Check

- [x] PRD traceability: P2-R1-AC1 through P2-R4-AC5.
- [x] Offline map scope: material/VT/label/large-scene polish for offline maps.
- [x] API truthfulness: real support or explicit diagnostics; no skipped family/material/stat success.
- [x] Determinism: stable validation, resource summaries, advanced label plans, support status reports.
- [x] Diagnostics first: VT, texture, UV, fallback, cache/LOD, instancing, bottleneck diagnostics before render where possible.
- [x] Typed product contract: extends P0/P1 product objects without redefining MVP.
- [x] Evidence plan: mapped below.
- [x] Support wording: unavailable P2 paths are not `supported`.
- [x] Compatibility: additive; no MVP behavior changes unless explicitly chosen later.
- [x] Continuity: update state artifacts during implementation.

## Project Structure

```text
specs/006-material-vt-large-scene-p2/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── p2-material-vt-large-scene-contract.md
└── tasks.md
```

Existing paths to inspect during tasks:

```text
python/forge3d/terrain_params.py
src/terrain/render_params/native_vt.rs
src/terrain/renderer/virtual_texture.rs
python/forge3d/buildings.py
python/forge3d/textures.py
python/forge3d/materials.py
src/import/
src/io/gltf_read.rs
src/io/obj_read.rs
python/forge3d/viewer_ipc.py
src/labels/
python/forge3d/pointcloud.py
src/pointcloud/
python/forge3d/tiles3d.py
src/tiles3d/
python/forge3d/geometry.py
src/scene/py_api/instanced_mesh.rs
src/terrain/renderer/py_api.rs
```

**Structure Decision**: Extend P0/P1 product APIs. If P0/P1 foundations are missing, tasks must block or depend on them. Native VT normal/mask implementation is optional only if validation emits `vt_unsupported_family` and docs state exact status.

## Test Strategy First

1. VT tests for albedo-only and albedo+normal/mask requests; non-albedo requests render real support or emit `vt_unsupported_family`.
2. Negative no-silent-skip test proving requested normal/mask families are not ignored while render reports success.
3. Textured building tests for valid albedo texture/UVs, missing UVs, missing texture paths, unsupported texture format, scalar fallback diagnostics.
4. Advanced label tests for repeated line labels, curved text, road/river presets, leader lines, priority presets, and complex-shaping deferral.
5. Large-scene tests for memory budget estimates, cache/LOD stats where available, unavailable stats diagnostics, instancing support, and bottleneck layer summaries.
6. Exact deterministic comparison tests for validation reports, resource summaries, advanced label outputs, and support matrices.
7. Docs audit for VT support, building texture support, advanced label support, and large-scene offline/non-live wording.

## Implementation Strategy

1. Reuse P0/P1 diagnostics and add feature-local codes: `missing_texture_path`, `missing_uvs`, `unsupported_texture_format`, `unavailable_cache_lod_stats`, `unsupported_instancing_path`.
2. For VT, choose native normal/mask implementation only after inspecting runtime cost and shader/resource paths; otherwise validate as unsupported before render.
3. For building textures, extend material intent only where existing building geometry exposes UVs and texture references; otherwise diagnose missing UVs/paths/fallbacks.
4. Extend `LabelPlan` advanced modes with deterministic repeated line labels, curved status, road/river presets, leader lines, and priority presets only where real placement can be proven.
5. Normalize point-cloud, 3D Tiles, terrain scatter, material VT, and global memory stats into large-scene summaries where available.
6. Keep unavailable stats as explicit diagnostics rather than invented estimates.

## Acceptance Mapping

| PRD AC | Components | Tests | Docs | Diagnostics |
|---|---|---|---|---|
| P2-R1-AC1..AC4 | VT family support | albedo/normal/mask tests | VT matrix | `vt_unsupported_family` |
| P2-R2-AC1..AC5 | textured building materials | texture/UV/path/fallback tests | building texture matrix | missing texture/UV/fallback |
| P2-R3-AC1..AC6 | advanced labels | repeated/curved/rules/leader/priority tests | advanced label docs | experimental/unsupported |
| P2-R4-AC1 | memory estimates | budget tests | large-scene docs | `estimated_gpu_memory` |
| P2-R4-AC2 | cache/LOD stats | per-layer stats tests | large-scene docs | unavailable stats |
| P2-R4-AC3 | instancing | instancing support tests | instancing note | unsupported instancing |
| P2-R4-AC4 | bottleneck summary | diagnostics tests | large-scene docs | bottleneck layer details |
| P2-R4-AC5 | scope docs | docs audit | large-scene docs | N/A |

## Migration And Compatibility

P2 is additive and non-MVP-blocking. Existing VT settings that accept normal/mask must not silently skip in product workflows; validation may become stricter through `MapScene`. Scalar building material fallback remains possible only when diagnostics/docs make fallback explicit. Existing advanced raw label APIs remain, but production support requires deterministic plan evidence.

## Verification Matrix And Ledger

P2 rows remain `Planned` until real implementation or `Deferred with diagnostic` evidence exists. Any deferred P2 gap needs structured diagnostics, docs, and negative tests before status changes. Record verification commands and exact artifact paths in matrix and ledger.

## Rollback And Safe Failure

If native VT normal/mask support is unstable, disable it through validation diagnostics. If texture paths/UVs are missing, render must not claim textured PBR. If cache/LOD/instancing stats are unavailable, report availability diagnostics instead of estimates. P2 failures must not break P0 MVP scenes.

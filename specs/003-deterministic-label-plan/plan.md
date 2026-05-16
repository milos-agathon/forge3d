# Implementation Plan: Deterministic LabelPlan

**Branch**: `003-deterministic-label-plan` | **Date**: 2026-05-15 | **Spec**: `specs/003-deterministic-label-plan/spec.md`
**Input**: Feature specification from `specs/003-deterministic-label-plan/spec.md`

## Summary

Create a deterministic offline label compiler contract that produces accepted labels, rejected labels with reason codes, diagnostics, bounds, seed, and render/export-ready payloads. Existing native substrate exists in `src/labels/layer.rs`, `src/labels/declutter.rs`, `src/labels/rtree.rs`, `src/labels/collision.rs`, `src/labels/projection.rs`, `src/labels/atlas.rs`, and `src/labels/typography.rs`, but no public `LabelPlan` exists. The new public Python module path is TBD until tasks inspect package layout and decide whether the compiler is Python-first, Rust-backed, or mixed.

## Technical Context

**Language/Version**: Python 3.x typed API with possible Rust/PyO3 acceleration from existing label modules.  
**Primary Dependencies**: feature `001` diagnostics, feature `002` truthful label primitives, native label layer/collision/atlas/typography/projection substrate.  
**Storage**: Deterministic in-memory plan plus JSON/bundle-ready serialization.  
**Testing**: pytest for compiler contracts and deterministic fixtures; Rust unit tests if native candidate/collision code is changed.  
**Target Platform**: Offline map production and render/export preparation.  
**Project Type**: Library API and compiler-like planning stage.  
**Performance Goals**: Stable repeated compile results for fixture-sized scenes; exact serialized comparison for fixed inputs.  
**Constraints**: Seeded behavior only; nondeterministic iteration normalized before scoring, diagnostics, bounds, or serialization.  
**Scale/Scope**: P0-R3, Section 14, Milestone 2, point and polygon labels, terrain sampling where available, keepouts, priorities, render/export payloads.

## Constitution Check

- [x] PRD traceability: P0-R3-AC1 through P0-R3-AC8 and Section 14.
- [x] Offline map scope: static/offline label planning only.
- [x] API truthfulness: successful compile returns real accepted/rejected plan data; unsupported render/export paths return diagnostics.
- [x] Determinism: seed, stable sort keys, normalized inputs, stable diagnostics, stable payloads.
- [x] Diagnostics first: rejected labels retained with reason codes and summary diagnostics.
- [x] Typed product contract: defines `LabelPlan` for later `MapScene` and bundle use.
- [x] Evidence plan: test/doc/diagnostic mapping below.
- [x] Support wording: advanced curved/repeated/non-Latin behavior remains non-goal or experimental.
- [x] Compatibility: existing label APIs remain; compiler consumes compatible source objects.
- [x] Continuity: implementation must update state artifacts.

## Project Structure

```text
specs/003-deterministic-label-plan/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── label-plan-contract.md
└── tasks.md
```

Existing paths to inspect during tasks:

```text
src/labels/layer.rs
src/labels/declutter.rs
src/labels/rtree.rs
src/labels/collision.rs
src/labels/projection.rs
src/labels/atlas.rs
src/labels/typography.rs
src/labels/line_label.rs
src/labels/curved.rs
python/forge3d/viewer_ipc.py
python/forge3d/export.py
python/forge3d/map_plate.py
tests/test_labels_pybindings.py
tests/test_api_contracts.py
```

**Structure Decision**: Public `LabelPlan` API location is TBD. The plan should prefer a typed Python wrapper around existing label/collision substrate and only add Rust bindings when needed for real compiler behavior or performance.

## Test Strategy First

1. Determinism test: compile identical inputs twice and compare accepted, rejected, diagnostics, bounds, seed, and serialized payload exactly.
2. Order normalization test: feed equivalent labels through different dict/set/filesystem orders and expect identical plan output.
3. Rejection reason fixtures for `collision`, `outside_view`, `missing_glyph`, `priority_lost`, `keepout_region`, `terrain_occluded`, `invalid_geometry`, `unsupported_geometry_type`, and `empty_text`.
4. Point candidate tests for center, above, below, left, right, and radial candidates; radial order must be seed-stable.
5. Polygon tests for centroid and visual-center/polylabel fallback; invalid geometry rejected with reason.
6. Terrain tests with known elevation sampler or typed terrain diagnostics.
7. Keepout tests for title, legend, scale bar, north arrow, and manual rectangles.
8. Render/export payload tests proving output is real payload data or typed diagnostic, never placeholder success.

## Implementation Strategy

1. Define source label, candidate, keepout, priority, terrain sampler, and output contracts before wiring implementation.
2. Normalize source labels by stable source ID, geometry type, coordinates, text, priority, and input index.
3. Generate point and polygon candidates with stable ordering; use seeded radial generation only through explicit seed.
4. Apply missing-glyph, empty-text, invalid-geometry, unsupported-geometry, outside-view, keepout, terrain, and collision checks while retaining rejected candidates.
5. Apply priority solve with deterministic tie-breaks.
6. Expose accepted/rejected/diagnostics/bounds/seed and deterministic render/export payloads.
7. Integrate diagnostics from feature `001`; keep feature `002` label APIs as inputs, not a replacement target.

## Acceptance Mapping

| PRD AC | Components | Tests | Docs | Diagnostics |
|---|---|---|---|---|
| P0-R3-AC1 | compiler/order/seed | repeated compile tests | LabelPlan guide | determinism metadata |
| P0-R3-AC2 | rejected labels | reason fixtures | rejection vocabulary | `label_rejection_summary`, `missing_glyphs` |
| P0-R3-AC3 | point candidates | candidate fixtures | candidate docs | invalid candidate diagnostics |
| P0-R3-AC4 | polygon candidates | centroid/polylabel tests | polygon docs | invalid geometry |
| P0-R3-AC5 | terrain sampler | sampler/diagnostic tests | terrain note | `terrain_occluded` or unavailable sampler |
| P0-R3-AC6 | keepouts | furniture/manual tests | keepout docs | `keepout_region` |
| P0-R3-AC7 | priority classes | collision priority tests | priority docs | `priority_lost` |
| P0-R3-AC8 | payloads | render/export payload tests | export note | unsupported backend diagnostics |

## Migration And Compatibility

No existing public `LabelPlan` exists, so this is additive. Existing label rendering paths remain available but must not be described as deterministic plan output unless they consume the compiler contract. Serialized plan payloads must include a version field so future advanced placement can evolve compatibly.

## Verification Matrix And Ledger

P0-R3 rows remain `Planned` until deterministic fixture tests, rejection reason tests, render/export payload tests, docs, and diagnostics are complete. Verification commands and evidence must be recorded in the matrix and ledger.

## Rollback And Safe Failure

If native substrate cannot supply terrain or glyph data, emit typed diagnostics and retain rejected labels. If render/export payload generation is unavailable, return diagnostics without reporting success. Deterministic plan content remains available even when rendering is blocked.

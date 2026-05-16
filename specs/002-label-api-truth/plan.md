# Implementation Plan: Label API Truth

**Branch**: `002-label-api-truth` | **Date**: 2026-05-15 | **Spec**: `specs/002-label-api-truth/spec.md`
**Input**: Feature specification from `specs/002-label-api-truth/spec.md`

## Summary

Make label public APIs truthful by ensuring create operations return stable IDs, configuration calls mutate real label state or fail with typed diagnostics, and line/curved label behavior renders real glyph instances or is explicitly classified as experimental/unsupported. The implementation wraps existing `python/forge3d/viewer_ipc.py`, `python/forge3d/viewer.py`, `src/viewer/cmd/labels_command.rs`, `src/viewer/ipc/protocol/response.rs`, and `src/labels/*`. T001 ownership inspection selected `ViewerHandle` in `python/forge3d/viewer.py` and public stubs in `python/forge3d/viewer.pyi` as the high-level Python API surface.

## Technical Context

**Language/Version**: Python 3.x with Rust/PyO3 label and viewer substrate.  
**Primary Dependencies**: `viewer_ipc`, `ViewerHandle`, native `LabelManager`, label command handlers, label style bindings, shared diagnostics from feature `001`.  
**Storage**: In-memory label state plus serializable IDs/configuration for later bundle use.  
**Testing**: pytest for public Python workflows and API contracts; Rust tests if native label command/manager behavior changes.  
**Target Platform**: Offline Python workflows and viewer/offscreen label rendering paths.  
**Project Type**: Library API and native bridge.  
**Performance Goals**: Stable ID allocation and batch result ordering deterministic for fixed inputs.  
**Constraints**: No raw IPC required by basic MVP label workflow; no create/config command may acknowledge success while doing nothing.  
**Scale/Scope**: P0-R1, P0-R2, Milestone 1, and label-related diagnostics.

## Constitution Check

- [x] PRD traceability: P0-R1-AC1 through P0-R1-AC6 and P0-R2-AC1 through P0-R2-AC6.
- [x] Offline map scope: limited to labels for offline map scenes.
- [x] API truthfulness: success means real state, renderable output, stable IDs, or structured diagnostics.
- [x] Determinism: stable IDs, batch ordering, diagnostics, and glyph instance ordering are fixed for fixed inputs.
- [x] Diagnostics first: unsupported/experimental/no-op paths report structured diagnostics.
- [x] Typed product contract: label APIs feed later `LabelPlan`, `ValidationReport`, and `Bundle`.
- [x] Evidence plan: tests/docs/diagnostics mapped below.
- [x] Support wording: line/curved/declutter/typography are not overclaimed.
- [x] Compatibility: raw IPC remains advanced/internal; high-level wrappers are additive unless a broken success path must fail honestly.
- [x] Continuity: update state artifacts during implementation.

## Project Structure

```text
specs/002-label-api-truth/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── label-api-contract.md
└── tasks.md
```

Existing implementation paths to inspect:

```text
python/forge3d/viewer.py
python/forge3d/viewer.pyi
python/forge3d/viewer_ipc.py
python/forge3d/__init__.py
python/forge3d/__init__.pyi
src/labels/
src/viewer/cmd/labels_command.rs
src/viewer/ipc/protocol/
tests/test_viewer_ipc.py
tests/test_labels_pybindings.py
tests/test_api_contracts.py
examples/fuji_labels_demo.py
```

**Structure Decision**: Add high-level Python wrappers to the existing viewer API surface if inspection confirms `ViewerHandle` is the right public layer; otherwise mark the chosen wrapper path in tasks before coding. Native response shape changes must be reflected in Python wrappers and stubs.

## Test Strategy First

1. Public workflow test that uses no raw IPC: add labels, load atlas, enable/disable, set typography or receive diagnostic, set declutter or receive diagnostic, clear labels.
2. Stable ID tests for `add_label`, `add_labels`, line labels, callouts, and label-related overlays where later reference is required.
3. Batch result tests preserving per-input order with IDs and diagnostics for mixed valid/invalid labels.
4. No-op success regression tests for typography, declutter, atlas load, create, remove, clear, and enabled state.
5. Line-label fixtures for horizontal, vertical, and diagonal paths; curved path fixtures must render correctly or return typed experimental/unsupported diagnostics.
6. Terrain-elevated line-label fixture or diagnostic test proving terrain sampling is unavailable in this path.
7. Docs/support matrix audit for point, line, curved, callout, typography, decluttering, and atlas loading support levels.

## Implementation Strategy

1. Inspect native `LabelManager` method signatures and IPC response objects before changing Python wrappers.
2. Introduce stable ID response contracts for successful create paths; use native IDs when already available rather than inventing Python-only IDs.
3. Add `ViewerHandle` high-level label methods or an inspected equivalent public wrapper.
4. Wire typography and declutter to real state where substrate exists; otherwise return typed diagnostics from feature `001`.
5. Verify whether line labels produce renderable glyph instances with rotation; if not, mark the path experimental/unsupported through diagnostics and docs.
6. Preserve raw IPC helpers for advanced users but keep MVP examples on high-level methods.

## Acceptance Mapping

| PRD AC | Components | Tests | Docs | Diagnostics |
|---|---|---|---|---|
| P0-R1-AC1 | create responses/IDs | stable ID tests | label API docs | per-object create failures |
| P0-R1-AC2 | typography state | layout/render/serialization tests | typography support | unsupported/experimental if no state |
| P0-R1-AC3 | declutter state | placement change or diagnostic tests | declutter support | unsupported/experimental |
| P0-R1-AC4 | high-level wrappers | no raw IPC workflow test | quickstart/example | operation diagnostics |
| P0-R1-AC5 | support matrix | docs audit | label support matrix | N/A |
| P0-R1-AC6 | no-op guards | no-op success tests | API failure behavior | `placeholder_fallback`, unsupported |
| P0-R2-AC1 | line glyph emission | line glyph tests | line support status | experimental/unsupported |
| P0-R2-AC2 | tangent rotation | rotation tests | tolerance note | experimental if absent |
| P0-R2-AC3 | upside-down behavior | handling or diagnostic test | support note | unsupported if absent |
| P0-R2-AC4 | curved labels | render or diagnostic tests | curved status | `experimental_feature` |
| P0-R2-AC5 | path fixtures | horizontal/vertical/diagonal/curved tests | fixture note | path diagnostics |
| P0-R2-AC6 | terrain label path | terrain test or diagnostic | terrain note | unavailable terrain diagnostic |

## Migration And Compatibility

Existing `viewer_ipc` functions remain available. High-level APIs should preserve current behavior only where it is truthful; previously successful no-op paths may become typed failures or diagnostics. Stubs must be updated with support-level failure behavior. Existing demos should be migrated to high-level wrappers only when behavior is supported or diagnostically honest.

## Verification Matrix And Ledger

Do not mark P0-R1 or P0-R2 rows beyond `Planned` until ID tests, no-op tests, line/curved status tests, docs, and diagnostics evidence exist. Implementation updates must record exact verification commands in the matrix and ledger.

## Rollback And Safe Failure

If stable native IDs cannot be returned for a path, the wrapper must return a typed diagnostic and avoid claiming success. If a setter cannot mutate state, it must fail honestly. If line/curved labels cannot render, classify as `experimental` or `unsupported` and keep point-label workflows unaffected.

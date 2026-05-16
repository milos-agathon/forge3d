# Implementation Plan: Diagnostics and Support Matrices

**Branch**: `001-diagnostics-support-matrices` | **Date**: 2026-05-15 | **Spec**: `specs/001-diagnostics-support-matrices/spec.md`
**Input**: Feature specification from `specs/001-diagnostics-support-matrices/spec.md`

## Summary

Define the product-level structured diagnostics contract and support-matrix documentation that all later map-rendering features consume. The plan wraps existing substrate in `python/forge3d/style.py`, `python/forge3d/buildings.py`, `python/forge3d/tiles3d.py`, `python/forge3d/bundle.py`, `python/forge3d/crs.py`, `python/forge3d/terrain_params.py`, `python/forge3d/mem.py`, `python/forge3d/viewer_ipc.py`, and native label/VT/building/tile paths. The public diagnostics contract is planned for `python/forge3d/diagnostics.py`, exported through `python/forge3d/__init__.py` and typed in `python/forge3d/__init__.pyi`; strict audit remediation additionally persists diagnostic reports through the existing scene-bundle state, adds public non-style diagnostic validators, and proves style output handoff shapes without implementing full `MapScene`, `LabelPlan`, or render-path ownership.

## Technical Context

**Language/Version**: Python `>=3.10` from `pyproject.toml`; Rust 2021 crate with PyO3 `0.21.2` and maturin `>=1.5,<2.0` from `Cargo.toml` and `pyproject.toml`.
**Primary Dependencies**: Existing forge3d Python modules, PyO3 extension, pytest, native device/memory diagnostics, style/building/tile/bundle/label/VT substrate.
**Storage**: Filesystem docs and bundle-ready JSON-serializable diagnostic payloads.
**Testing**: pytest for Python contracts and negative paths; cargo tests only if native diagnostic hooks are changed.
**Target Platform**: Offline Python library workflows on supported forge3d desktop/server platforms.
**Project Type**: Python/Rust library with documentation.
**Performance Goals**: Validation ordering deterministic; diagnostics should be cheap enough to run before render and must avoid GPU work unless explicitly needed for memory/device summaries.
**Constraints**: No raw IPC required for MVP workflows; diagnostics must be structured, serializable, deterministic, and must not turn unsupported paths into success.
**Scale/Scope**: P0 diagnostics inventory plus support matrices for style, labels, buildings, 3D Tiles, VT, and diagnostics reference.

## Constitution Check

- [x] PRD traceability: in scope P0-R5-AC1 through P0-R5-AC6, P0-R6-AC1 through P0-R6-AC6, PRD Sections 13 and 16, Appendix B, Milestone 0.
- [x] Offline map scope: limited to offline 3D map rendering diagnostics and docs.
- [x] API truthfulness: unsupported, experimental, Pro-gated, placeholder/fallback, and underdeveloped paths produce typed diagnostics or documented failures.
- [x] Determinism: reports sort diagnostics and summaries by severity rank, code, layer ID, object ID, and stable message/detail keys.
- [x] Diagnostics first: validation surfaces issues before render where enough input information exists.
- [x] Typed product contract: defines `Diagnostic`, `ValidationReport`, `LayerSummary`, `SupportMatrixEntry`, `RenderFailurePolicy`, and `SeverityPolicy` for later `MapScene`, `LabelPlan`, and `Bundle` workflows.
- [x] Evidence plan: each acceptance criterion maps to components, tests, docs, diagnostics, and matrix updates below.
- [x] Support wording: uses only PRD Appendix B classifications.
- [x] Compatibility: existing APIs remain available; new diagnostics wrap or augment behavior rather than removing current helpers.
- [x] Continuity: tasks must update `docs/superpowers/state/current-context-pack.md`, `docs/superpowers/state/implementation-ledger.md`, and the verification matrix when statuses change.

## Hostile Review Risk Table

| Area | Status | Risk after amendment | Required disposition |
|---|---|---|---|
| 1. PRD acceptance-criterion coverage | Green | P0-R5 and P0-R6 acceptance criteria are mapped below; Section 13, Section 16, Appendix B, and Milestone 0 are in scope. | Generated tasks must include every mapped PRD AC tag and must not mark rows beyond `Planned` without evidence. |
| 2. Test coverage | Green | Contract, negative, style, policy, serialization, determinism, and docs-audit tests are explicitly required. | Generated tasks must create test files named in this plan or amend the plan first. |
| 3. Diagnostics/support honesty | Green | Required diagnostic codes, severity floors, support levels, remediation, and affected IDs are part of the contract. | Unsupported, underdeveloped, Pro-gated, placeholder/fallback, and experimental paths must be diagnosed before render where inputs are knowable. |
| 4. Determinism/reproducibility | Green | Report ordering and serialization use stable sort keys and sorted JSON output. | Tasks must include repeated-run comparison tests. |
| 5. No-op success prevention | Green | Known silent-ignore and placeholder paths are required negative-test targets. | A path that cannot be wired must return a typed diagnostic or typed failure, not success. |
| 6. Raw IPC avoidance | Green | `viewer_ipc` may be inspected as substrate only; no canonical MVP validation workflow may require raw IPC. | Tasks must keep public validation helpers typed and Python-level. |
| 7. Bundle/documentation requirements | Green | Bundle-ready serialization, diagnostic-only scene-bundle persistence, and exact docs/support-matrix paths are listed below. | Full map-scene review-bundle round-trip remains later-feature scope, but diagnostic payloads must be lossless and embeddable. |
| 8. New-chat continuity artifacts | Green | Ledger, context pack, and verification matrix rules are explicit. | Every implementation/task session must update continuity artifacts before completion. |
| 9. P0/P1/P2 scope boundaries | Yellow | Feature `001` may reference P1/P2-owned paths only to classify and diagnose them, not to implement them. | Accepted risk: support matrices mention later features as `underdeveloped`, `missing`, `experimental`, `Pro-gated`, `placeholder/fallback`, or `unsupported`; implementation tasks must not pull P1/P2 rendering work into P0. |
| 10. File-path accuracy | Yellow | Proposed new code/docs/test paths are fixed by this plan, but task inspection may reveal a stronger local convention. | Accepted risk: any path change must update `plan.md`, `contracts/diagnostics-contract.md`, and generated tasks in the same planning pass before code edits. |

## Project Structure

### Documentation

```text
specs/001-diagnostics-support-matrices/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── diagnostics-contract.md
└── tasks.md                    # generated later by /speckit-tasks
```

### Existing Source Paths To Inspect During Tasks

```text
python/forge3d/__init__.py
python/forge3d/__init__.pyi
python/forge3d/style.py
python/forge3d/buildings.py
python/forge3d/tiles3d.py
python/forge3d/bundle.py
python/forge3d/crs.py
python/forge3d/terrain_params.py
python/forge3d/mem.py
python/forge3d/viewer_ipc.py
src/labels/
src/viewer/cmd/labels_command.rs
src/terrain/render_params/native_vt.rs
src/terrain/renderer/virtual_texture.rs
src/import/
src/tiles3d/
src/bundle/
tests/
docs/
```

### Planned Product Source And Test Paths

```text
python/forge3d/diagnostics.py
python/forge3d/__init__.py
python/forge3d/__init__.pyi
tests/test_diagnostics_contract.py
tests/test_diagnostics_style_support.py
tests/test_diagnostics_support_paths.py
tests/test_diagnostics_bundle_serialization.py
tests/test_bundle_roundtrip.py
tests/test_support_matrices_docs.py
```

### Planned Documentation Paths

```text
docs/guides/offline_3d_map_rendering.md
docs/guides/diagnostics_reference.md
docs/guides/style_support_matrix.md
docs/guides/label_support_matrix.md
docs/guides/building_support_matrix.md
docs/guides/tiles3d_support_matrix.md
docs/guides/virtual_texturing_support_matrix.md
docs/guides/competitive_positioning.md
docs/index.rst
```

Generated `docs/_build/` output is not a source path and must not be edited by this feature. Documentation tasks must update `docs/index.rst` to include any new source docs that should be built.

**Structure Decision**: Add the product diagnostics contract in `python/forge3d/diagnostics.py` and re-export public classes from `python/forge3d/__init__.py` with matching type stubs in `python/forge3d/__init__.pyi`. Support matrices live as source docs under `docs/guides/`; the listed paths are the default and may change only through an explicit plan amendment before implementation.

## Test Strategy First

1. Contract tests for `Diagnostic`, severity values, support-level vocabulary, serialization, deterministic ordering, and report status calculation.
2. Negative tests for `crs_mismatch`, `missing_glyphs`, `unsupported_style_field`, `unsupported_style_layer_type`, `pro_gated_path`, `placeholder_fallback`, `experimental_feature`, `vt_unsupported_family`, `python_public_3dtiles_incomplete`, `estimated_gpu_memory`, and `label_rejection_summary`.
3. Render policy tests showing warnings continue by default, fail when fail-on-warning is selected, and errors/fatals always block successful completion.
4. Style diagnostics tests against known supported `fill`, `line`, and `circle` paths plus unsupported layer/field fixtures after auditing `python/forge3d/style.py`.
5. Serialization tests proving diagnostics can be embedded in bundle-ready structures and preserved through the existing scene-bundle state without losing code, severity, affected IDs, support level, or remediation.
6. Docs audit tests or review checklist proving no public docs claim full Mapbox Style Specification, production 3D Tiles, VT normal/mask runtime, textured PBR buildings, or production line/curved labels.
7. No-op success regression tests for known risky paths from `research.md`: style fields dropped by parsing/application, building zero-geometry fallbacks, 3D Tiles incomplete decode/render paths, VT non-albedo family requests, and label command truthfulness diagnostics.
8. Determinism tests that build the same report twice and compare diagnostic order, support-summary order, and sorted JSON serialization byte-for-byte.
9. Raw IPC avoidance tests or docs audits proving public validation examples use `Diagnostic`/`ValidationReport` helpers and do not require users to call `python/forge3d/viewer_ipc.py`.

## Implementation Strategy

1. Confirm that `python/forge3d/diagnostics.py` fits existing export and typing conventions in `python/forge3d/__init__.py`, `python/forge3d/__init__.pyi`, and nearby dataclass modules; if not, amend this plan before selecting a different path.
2. Add Python dataclass or typed wrapper models for diagnostics and validation reports with deterministic `to_dict()`/`from_dict()` behavior and sorted JSON-friendly output.
3. Add support-level and severity validation helpers; reject unknown support terms rather than silently accepting them.
4. Add style-support audit logic around existing `StyleLayer`, `PaintProps`, and `LayoutProps` parsing without claiming streamed MVT support; unsupported fields and layer types must produce structured diagnostics, not dropped fields.
5. Add adapter functions or pure validation helpers for known placeholder/fallback/pro-gated paths so later `MapScene` and bundle features can report them before render. These adapters must not make the underlying render path look supported.
6. Preserve `ValidationReport` in the existing scene-bundle `SceneState` and `LoadedBundle` path as a diagnostic-only payload without claiming full `MapScene` bundle reconstruction.
7. Add docs support matrices and diagnostics reference using the PRD classifications exactly, with a source-doc audit that fails on forbidden overclaims.
8. Keep raw IPC as an inspected implementation substrate only. Public examples and tests for this feature must use typed diagnostics/report objects or future `MapScene.validate()`-compatible shapes.

## Acceptance Mapping

| PRD AC | Components | Tests | Docs | Diagnostics |
|---|---|---|---|---|
| P0-R5-AC1 | `Diagnostic`, `ValidationReport` | object-field contract tests | Diagnostics Reference | all required codes |
| P0-R5-AC2 | severity enum/policy | severity coverage tests | Diagnostics Reference | `info`, `warning`, `error`, `fatal` |
| P0-R5-AC3 | affected ID fields | layer/object ID propagation tests | Diagnostics Reference | affected layer/object IDs |
| P0-R5-AC4 | `to_dict`/`from_dict`, `SceneState.validation_report`, `LoadedBundle.validation_report` | serialization and bundle round-trip tests | Bundle diagnostics note | serializable payload |
| P0-R5-AC5 | `RenderFailurePolicy`, `SeverityPolicy` | warning-policy tests | Quickstart | blocking status |
| P0-R5-AC6 | support adapters and public validators | negative no-silent-ignore tests | support matrices | unsupported/fallback/pro-gated codes |
| P0-R6-AC1 | style docs | docs audit | Style Support Matrix | N/A |
| P0-R6-AC2 | style audit inventory | docs/test audit | Style Support Matrix | N/A |
| P0-R6-AC3 | style layer support mapping | fill/line/circle tests | Style Support Matrix | unsupported codes for gaps |
| P0-R6-AC4 | style diagnostics | unsupported style tests | Style Support Matrix | `unsupported_style_*` |
| P0-R6-AC5 | style output contract | vector/label-consumable shape tests | Style Support Matrix | unsupported fields |
| P0-R6-AC6 | docs wording audit | overclaim audit | all public docs | N/A |

## Required Diagnostic Coverage

| Code | Minimum severity when output-affecting | Support-level requirement | Test requirement |
|---|---|---|---|
| `crs_mismatch` | `error` | `unsupported` until an explicit compatible transform/policy exists | Mismatched scene/layer CRS fixture emits affected layer ID and remediation. |
| `missing_glyphs` | `warning` | `underdeveloped` or `unsupported` according to atlas/path context | Missing glyph fixture emits code before render and preserves missing glyph details. |
| `unsupported_style_field` | `warning` | `unsupported` for the field, with layer still classified by actual renderability | Style fixture with unknown paint/layout fields emits layer ID and field names. |
| `unsupported_style_layer_type` | `error` | `unsupported` for requested layer rendering | Style fixture with unsupported layer type emits layer ID and does not claim MVT parity. |
| `pro_gated_path` | `error` | `Pro-gated` | Building/style/bundle path without Pro/native availability emits diagnostic before success. |
| `placeholder_fallback` | `error` | `placeholder/fallback` | Zero-geometry or non-renderable fallback emits diagnostic and blocks success. |
| `experimental_feature` | `warning` | `experimental` | Line/curved label or other experimental path reports honest status. |
| `vt_unsupported_family` | `error` | `unsupported` or `missing` for non-albedo runtime family | Normal/mask VT request emits diagnostic instead of log-only warning. |
| `python_public_3dtiles_incomplete` | `error` | `underdeveloped` or `unsupported` according to requested workflow | Public 3D Tiles path that cannot complete render workflow emits diagnostic. |
| `estimated_gpu_memory` | `warning` | `supported` for estimate availability; `underdeveloped` if unavailable for a layer | Budget-risk fixture emits deterministic estimate or honest unavailable detail. |
| `label_rejection_summary` | `warning` | `supported` only when reason-coded plan data exists, otherwise `underdeveloped` | Synthetic rejected-label summary serializes counts/reasons deterministically. |

## Future Task Generation Requirements

Tasks generated later must include, at minimum:

1. Create or amend `python/forge3d/diagnostics.py` plus exports/stubs for `Diagnostic`, `ValidationReport`, `LayerSummary`, `SupportMatrixEntry`, `RenderFailurePolicy`, and `SeverityPolicy` with PRD AC tags `P0-R5-AC1` through `P0-R5-AC5`.
2. Add contract tests in `tests/test_diagnostics_contract.py` for field validation, unknown severity/support rejection, status calculation, render policy, and deterministic serialization.
3. Add style diagnostics tasks in `tests/test_diagnostics_style_support.py` and docs tasks for `docs/guides/style_support_matrix.md`, covering `P0-R6-AC1` through `P0-R6-AC6`.
4. Add support-path negative tests in `tests/test_diagnostics_support_paths.py` for Pro-gated, placeholder/fallback, experimental, VT, 3D Tiles, CRS, glyph, memory, and label-rejection diagnostics.
5. Add bundle-ready serialization tests in `tests/test_diagnostics_bundle_serialization.py` and diagnostic-only scene-bundle persistence tests in `tests/test_bundle_roundtrip.py`; full `MapScene` review-bundle reconstruction remains feature `005`.
6. Add docs/source audit coverage in `tests/test_support_matrices_docs.py` or an equivalent explicit verification command for the required docs paths and forbidden overclaim wording.
7. Update `docs/superpowers/state/current-context-pack.md` and `docs/superpowers/state/implementation-ledger.md` in every implementation session; update `docs/superpowers/state/requirements-verification-matrix.md` only when evidence justifies a status change.
8. Preserve P0 scope: do not implement `LabelPlan`, `MapScene`, buildings, 3D Tiles runtime rendering, or VT normal/mask runtime in feature `001`; only define diagnostics/support contracts, public validation helpers, diagnostic-only bundle persistence, and honest reporting for those paths.

## Migration And Compatibility

Existing public helpers remain importable. New diagnostics must not change success behavior silently; where current helpers are known to over-acknowledge or drop fields, the first compatible step is to expose validation/reporting helpers and later wire fail-fast behavior in owning features. Bundle schema additions must be additive and tolerate missing diagnostics for older bundles. Any compatibility shim that cannot prove real state mutation or renderability must surface `unsupported`, `experimental`, `Pro-gated`, or `placeholder/fallback` diagnostics rather than returning unqualified success.

## Verification Matrix And Ledger

Tasks must move P0-R5 and P0-R6 rows from `Planned` only when code, tests, docs, diagnostics, and command evidence exist. This plan does not change matrix status. Implementation must append continuity notes to `docs/superpowers/state/implementation-ledger.md` and refresh `docs/superpowers/state/current-context-pack.md`. If a future session changes a requirement status, it must update `docs/superpowers/state/requirements-verification-matrix.md` in the same session with commands and evidence.

## Rollback And Safe Failure

If the shared diagnostics contract cannot be wired to a path, that path must report an `unsupported` or feature-specific diagnostic rather than success. If bundle serialization fails, keep the original render/report result but return an error diagnostic and avoid writing a misleading bundle.

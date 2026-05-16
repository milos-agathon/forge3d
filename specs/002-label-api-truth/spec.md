# Feature Specification: Label API Truth

**Feature Branch**: `002-label-api-truth`  
**Created**: 2026-05-14  
**Status**: Draft  
**Input**: User description: "Create feature 002-label-api-truth for forge3d offline 3D map rendering."  
**Source of Truth**: `docs/superpowers/plans/prd.md` and `.specify/memory/constitution.md`  
**PRD Coverage**: P0-R1, P0-R2, Milestone 1, label-related diagnostics from P0-R5 where needed  
**MVP Blocking**: Yes - P0 label API truth and line/curved label honesty are release-blocking unless a documented human decision reclassifies them

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Create Public Labels Without Raw IPC (Priority: P1)

A geospatial Python user creates point labels, multiple labels, line labels, callouts, and label-related overlays through public high-level methods and receives stable IDs for every created object that may need later update, inspection, removal, export, or review.

**Why this priority**: This is the primary Milestone 1 outcome. Users must be able to write a basic public label workflow without raw IPC and without ambiguous create results.

**Independent Test**: Can be fully tested by creating labels through the public workflow, asserting returned IDs are stable and reusable, then updating, inspecting, or clearing the created objects without any raw IPC calls.

**Acceptance Scenarios**:

1. **Given** a public label workflow, **When** the user calls `add_label`, `add_labels`, line-label creation, callout creation, or overlay creation where the created object needs later reference, **Then** each successful create operation returns stable created IDs.
2. **Given** a user has created labels through public methods, **When** the user calls `clear_labels` or removes labels by returned IDs where supported, **Then** the target label state changes and later inspection or rendering does not report stale label objects as still active.
3. **Given** a basic offline map label example, **When** it creates labels, loads an atlas, enables labels, and renders or validates the scene, **Then** the example uses public high-level methods only and does not require raw IPC.

---

### User Story 2 - Configure Labels Truthfully (Priority: P1)

A cartographic production user configures label visibility, atlas loading, typography, and decluttering and can tell whether each setter changed real state or failed honestly with typed diagnostics.

**Why this priority**: P0-R1 forbids commands that acknowledge success while doing nothing. Typography and declutter controls must either have visible, measurable, or serializable effects, or return typed unsupported/experimental diagnostics.

**Independent Test**: Can be tested by applying each public setter to a controlled label scene and verifying state snapshots, layout metrics, render-visible differences, serialized state, or typed diagnostics.

**Acceptance Scenarios**:

1. **Given** labels are present, **When** the user calls `set_labels_enabled(False)` and then `set_labels_enabled(True)`, **Then** validation, inspection, or render output reflects the enabled state change.
2. **Given** label text requires glyphs from an atlas, **When** the user calls `load_label_atlas`, **Then** the active atlas state changes or a typed diagnostic explains why the atlas cannot be loaded or used.
3. **Given** a label scene with measurable typography-dependent layout, **When** the user calls `set_label_typography`, **Then** typography changes are visible, measurable, or serializable; otherwise the call returns a typed unsupported or experimental diagnostic instead of success.
4. **Given** a label scene with competing labels, **When** the user calls `set_declutter_algorithm`, **Then** placement behavior changes measurably; otherwise the call returns a typed unsupported or experimental diagnostic instead of success.

---

### User Story 3 - Render Or Honestly Classify Line And Curved Labels (Priority: P1)

A cartographic user creates line and curved labels and either sees renderable glyph instances placed along the requested path or receives explicit experimental/unsupported diagnostics before mistaking the path for production-ready behavior.

**Why this priority**: P0-R2 blocks shipping an API that accepts line or curved labels while silently rendering nothing or storing unused metadata.

**Independent Test**: Can be tested with horizontal, vertical, diagonal, and curved path fixtures that verify renderable glyph instances, tangent-following rotation where supported, upside-down behavior or documented unsupported status, and terrain-elevated behavior or diagnostics.

**Acceptance Scenarios**:

1. **Given** a horizontal, vertical, or diagonal line path with valid label text, **When** the user creates a line label on a supported path, **Then** renderable glyph instances are emitted along the path and can be measured or inspected.
2. **Given** line glyph rendering is supported, **When** a path changes tangent direction, **Then** glyph rotation follows the path tangent within documented tolerance.
3. **Given** curved labels are not production-ready, **When** the user attempts curved label creation, **Then** the operation returns or raises a typed `experimental_feature` or unsupported diagnostic rather than reporting successful production rendering.
4. **Given** terrain-elevated line labels are requested, **When** terrain sampling is unavailable in this feature path, **Then** validation or creation reports a typed diagnostic instead of silently flattening, ignoring, or misplacing the label.

---

### User Story 4 - Audit Label Support Claims (Priority: P2)

An internal forge3d developer or documentation reviewer can inspect public label docs and see exact support levels for point labels, line labels, curved labels, callouts, typography, decluttering, atlas loading, missing glyph behavior, and experimental/unsupported paths.

**Why this priority**: P0-R1 requires documentation to state support level and prevents overclaiming line/curved labels, typography, or declutter behavior before tests prove production support.

**Independent Test**: Can be tested by reviewing the label support matrix and API docs against the PRD taxonomy and verifying each documented support claim has matching tests or typed diagnostics.

**Acceptance Scenarios**:

1. **Given** public docs for label APIs, **When** a reviewer audits them, **Then** every PRD-scoped label operation uses the exact support-level terms `supported`, `underdeveloped`, `missing`, `Pro-gated`, `placeholder/fallback`, `experimental`, `unsupported`, or `non-goal`.
2. **Given** docs mention line or curved labels, **When** glyph emission, rotation, and path coverage tests do not prove production behavior, **Then** the docs classify those paths as `experimental` or `unsupported` and identify typed diagnostics.

### Edge Cases

- A create call succeeds but no stable ID is produced: the operation is treated as failed for this feature and must return a typed diagnostic or fail a regression test.
- `add_labels` receives a mixture of valid and invalid labels: successful labels return stable IDs and invalid entries return typed diagnostics without shifting or losing per-label correspondence.
- A label setter receives a valid value that is not yet supported by the active backend: the call returns a typed unsupported or experimental diagnostic instead of mutating unrelated state.
- `load_label_atlas` receives text coverage gaps, missing atlas data, or invalid atlas metadata: missing glyph or atlas diagnostics are reported before render where the input makes them knowable.
- `set_labels_enabled` is called before labels exist: the enabled state is recorded for future labels or the call returns a typed diagnostic; it must not report success while discarding the request.
- Line labels contain too few path points, zero-length segments, empty text, or unsupported geometry: creation or validation returns typed diagnostics instead of storing unused metadata.
- Horizontal, vertical, diagonal, and curved paths produce different orientation and placement expectations: tests must cover each path class separately.
- Terrain-elevated line labels are requested without terrain sampling availability: the behavior is diagnosed explicitly rather than silently falling back to incorrect elevation.
- `clear_labels` is called repeatedly: the operation remains idempotent and does not mask earlier no-op success in create or setter commands.
- Existing raw IPC paths remain available for internal or advanced use: the MVP user workflow must not require them.

### Support-Level Classification *(mandatory)*

Use the PRD Appendix B terms exactly: `supported`, `underdeveloped`, `missing`, `Pro-gated`, `placeholder/fallback`, `experimental`, `unsupported`, and `non-goal`.

- **Supported in this feature**: public high-level label creation and configuration paths that return stable IDs where needed, mutate real label state, emit renderable glyph instances, or return typed diagnostics instead of false success.
- **Underdeveloped**: existing label substrate where public workflow, stable IDs, state mutation, diagnostics, or end-to-end rendering is incomplete before this feature is implemented.
- **Missing**: deterministic `LabelPlan` compiler, full offline label placement planning, and advanced label placement behavior that belongs to feature `003` or later work.
- **Pro-gated**: any label behavior that depends on native or Pro-only paths not available in the public workflow.
- **Placeholder/fallback**: any label path that would acknowledge success while creating no renderable glyphs, storing only unused metadata, or producing non-renderable placeholder output.
- **Experimental**: line labels, curved labels, declutter modes, typography controls, or terrain-elevated label behavior that exist but are incomplete, unverified, or not safe to present as production-ready.
- **Unsupported**: label inputs, geometry types, typography settings, declutter algorithms, atlas formats, or terrain elevation modes that the product cannot render, validate, or apply truthfully in this feature.
- **Non-goal**: full deterministic `LabelPlan`, full non-Latin text shaping, advanced repeated/road/river label placement, full data-driven `LabelLayer` ingestion, and general non-map rendering workflows.

### Diagnostics & Failure Behavior *(mandatory)*

- **Diagnostic objects**: Label diagnostics must include code, severity, message, remediation, support level where applicable, affected label ID where known, affected layer or overlay ID where known, and enough context to identify the failed create or setter operation.
- **Required label-related diagnostic codes**: This feature must use `missing_glyphs`, `experimental_feature`, and `placeholder_fallback` where applicable. It may also use typed unsupported diagnostics from the shared diagnostics contract when a label operation cannot be performed honestly.
- **Pre-render validation**: Missing glyphs, unsupported atlas coverage, unsupported line or curved label behavior, terrain-elevated line-label unavailability, placeholder/fallback paths, and no-op setter behavior must be detectable before render where enough input information exists.
- **No-op prevention**: Public label create and setter commands must fail tests if they return success while omitting IDs, mutating no state, emitting no renderable output, or producing only unused metadata.
- **Determinism**: Stable IDs, per-label results from batch creation, label inspection order, diagnostics ordering, and line-label glyph instance ordering must be deterministic for fixed inputs.
- **Failure behavior**: Unsupported or experimental paths must return typed diagnostics or typed errors. Successful responses must mean the requested label state, renderable output, or serializable configuration actually changed.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001 [PRD: P0-R1-AC1]**: `add_label` MUST return a stable created label ID when the label is accepted for real state mutation or rendering.
- **FR-002 [PRD: P0-R1-AC1]**: `add_labels` MUST return stable IDs for each accepted label and typed per-label diagnostics for each rejected label without losing input order correspondence.
- **FR-003 [PRD: P0-R1-AC1, P0-R2-AC1]**: Line-label creation MUST return stable IDs for accepted line labels and must either emit renderable glyph instances along the path or return typed experimental/unsupported diagnostics.
- **FR-004 [PRD: P0-R1-AC1]**: Callout creation MUST return stable IDs where users need to update, inspect, remove, export, or review the callout.
- **FR-005 [PRD: P0-R1-AC1]**: Overlay creation used by label workflows MUST return stable IDs where users need to update, inspect, remove, export, or review the overlay.
- **FR-006 [PRD: P0-R1-AC4]**: A basic label workflow MUST be possible through public high-level methods without raw IPC.
- **FR-007 [PRD: P0-R1-AC4]**: `clear_labels` MUST be exposed as a public high-level method and must clear real label state or return a typed diagnostic explaining why it cannot.
- **FR-008 [PRD: P0-R1-AC4]**: `set_labels_enabled` MUST be exposed as a public high-level method and must change real enabled state, affect future label behavior, or return a typed diagnostic explaining why it cannot.
- **FR-009 [PRD: P0-R1-AC4]**: `load_label_atlas` MUST be exposed as a public high-level method where atlas loading is part of the label workflow and must change active atlas state or return typed diagnostics.
- **FR-010 [PRD: P0-R1-AC2]**: `set_label_typography` MUST change actual label typography state and produce a visible, measurable, or serializable effect, or fail honestly with typed unsupported/experimental diagnostics.
- **FR-011 [PRD: P0-R1-AC3]**: `set_declutter_algorithm` MUST change real placement behavior or fail honestly with typed unsupported/experimental diagnostics.
- **FR-012 [PRD: P0-R1-AC6]**: Tests MUST fail when any label create or configuration command returns success while doing nothing.
- **FR-013 [PRD: P0-R1-AC5]**: Public API documentation MUST state support level for point labels, line labels, curved labels, callouts, typography, decluttering, and atlas loading.
- **FR-014 [PRD: P0-R2-AC1]**: `add_line_label` MUST emit renderable glyph instances along horizontal, vertical, and diagonal supported paths, or return typed diagnostics that classify the path as experimental or unsupported.
- **FR-015 [PRD: P0-R2-AC2]**: Where line glyph rendering is supported, glyph rotation MUST follow the path tangent within documented tolerance.
- **FR-016 [PRD: P0-R2-AC3]**: Upside-down avoidance MUST be implemented or explicitly documented and diagnosed as unsupported.
- **FR-017 [PRD: P0-R2-AC4]**: `add_curved_label` MUST either render correctly or raise/return a typed experimental/unsupported status.
- **FR-018 [PRD: P0-R2-AC5]**: Tests MUST cover horizontal, vertical, diagonal, and curved label paths.
- **FR-019 [PRD: P0-R2-AC6]**: Tests MUST cover terrain-elevated line labels when terrain sampling is enabled; if terrain sampling is unavailable, creation or validation MUST return typed diagnostics.
- **FR-020 [PRD: P0-R5 label diagnostics]**: Missing glyphs and label-related unsupported, experimental, and placeholder/fallback paths MUST be reported as structured diagnostics, not only printed strings or logs.
- **FR-021 [PRD: Milestone 1]**: The feature MUST include at least one public basic label workflow example or test that creates labels, configures atlas or typography where supported, toggles labels, and clears labels without raw IPC.
- **FR-022 [PRD: Constitution IV]**: Stable ID allocation, batch result ordering, diagnostic ordering, and emitted glyph instance ordering MUST be reproducible for fixed inputs.

### Key Entities *(include if feature involves data)*

- **Label ID**: Stable public identifier returned by label creation for update, inspection, removal, export, review, and diagnostics.
- **Label Batch Result**: Ordered result from creating multiple labels, preserving per-input correspondence between returned IDs and typed diagnostics.
- **Label Configuration State**: Publicly observable state for enabled/disabled labels, active atlas, typography settings, and declutter selection.
- **Line Label Path**: Path geometry used to place glyph instances for line labels, including horizontal, vertical, diagonal, curved, invalid, and terrain-elevated variants.
- **Glyph Instance**: Renderable label element with position, glyph identity, and rotation or tangent information where supported.
- **Callout**: Label-related annotation with an ID where users need update, inspect, remove, export, or review workflows.
- **Label Overlay**: Overlay involved in label workflows with a stable ID where users need update, inspect, remove, export, or review workflows.
- **Label Diagnostic**: Structured diagnostic for missing glyphs, unsupported or experimental behavior, placeholder/fallback paths, no-op prevention, or terrain sampling unavailability.
- **Public API contract**: This feature defines truthful high-level label workflow behavior used by `MapScene`, `SceneRecipe`, `LabelPlan`, `ValidationReport`, `Diagnostic`, and `Bundle`; it does not implement the full deterministic `LabelPlan` compiler.

## PRD Traceability & Evidence *(mandatory)*

| PRD AC ID | Requirement in this spec | Evidence required | Verification target |
|---|---|---|---|
| P0-R1-AC1 | FR-001, FR-002, FR-003, FR-004, FR-005 | Public create contracts plus tests asserting stable IDs for each create path | Public label create tests |
| P0-R1-AC2 | FR-010 | State mutation evidence plus render, layout metric, or serialization tests for typography changes | Typography state tests |
| P0-R1-AC3 | FR-011 | Placement behavior tests or typed unsupported/experimental diagnostic tests | Declutter behavior tests |
| P0-R1-AC4 | FR-006, FR-007, FR-008, FR-009, FR-021 | Public workflow example or test using no raw IPC and high-level methods | Basic label workflow test |
| P0-R1-AC5 | FR-013 | Label support matrix and API docs with support levels | Label docs audit |
| P0-R1-AC6 | FR-012, FR-020 | Regression tests that detect success without state, IDs, renderable output, or diagnostics | No-op success tests |
| P0-R2-AC1 | FR-003, FR-014 | Glyph instance emission evidence for supported line paths or typed diagnostics | Line-label glyph tests |
| P0-R2-AC2 | FR-015 | Rotation or tangent measurement tests for supported line paths | Line-label rotation tests |
| P0-R2-AC3 | FR-016 | Upside-down handling tests or documented unsupported diagnostic | Upside-down behavior review |
| P0-R2-AC4 | FR-017 | Curved label render tests or typed experimental/unsupported diagnostics with docs | Curved label tests |
| P0-R2-AC5 | FR-018 | Horizontal, vertical, diagonal, and curved path fixtures | Path coverage tests |
| P0-R2-AC6 | FR-019 | Terrain-elevated label tests or diagnostics proving terrain sampling is unavailable | Terrain line-label tests |
| P0-R5 label diagnostics | FR-020 | Structured label diagnostics for missing glyphs, experimental, unsupported, and placeholder/fallback paths | Label diagnostic tests |
| Milestone 1 | FR-021 | Basic public label example and tests proving no raw IPC | Milestone exit review |
| Constitution IV | FR-022 | Stable ID, batch result, diagnostic, and glyph ordering checks | Determinism tests or review |

## Explicit Non-Goals *(mandatory)*

- Full deterministic `LabelPlan` compiler; that belongs to feature `003`.
- Full non-Latin text shaping.
- Advanced repeated label placement, road-label placement, river-label placement, or complex route/river-specific cartographic placement.
- Full data-driven label ingestion from geospatial tables or style expressions except where needed to prove the public API contract.
- Full `MapScene` render recipe and bundle workflow; this feature supplies truthful label primitives and diagnostics needed by later features.
- Treating line or curved labels as production-ready unless renderable glyph emission, rotation, and path coverage tests prove the claim.
- Treating typography or declutter setters as supported when they merely acknowledge commands without mutating real state.
- Browser delivery, hosted tile services, general DCC workflows, game/editor tooling, or non-map rendering features.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of P0-R1 and P0-R2 acceptance criteria have explicit requirements, evidence expectations, and verification targets in this specification.
- **SC-002**: A basic public label workflow can create, configure, enable or disable, and clear labels without raw IPC.
- **SC-003**: 100% of successful label create operations in scope return stable IDs where later update, inspection, removal, export, or review needs them.
- **SC-004**: 100% of in-scope label setters either produce visible, measurable, or serializable state changes or return typed diagnostics.
- **SC-005**: Regression tests fail for any in-scope command that returns success while doing nothing.
- **SC-006**: Horizontal, vertical, diagonal, and curved path fixtures are each covered by tests or typed experimental/unsupported diagnostics.
- **SC-007**: Terrain-elevated line-label behavior is either tested when terrain sampling is available or diagnosed when unavailable.
- **SC-008**: Public label docs state support level for point labels, line labels, curved labels, callouts, typography, decluttering, atlas loading, missing glyph behavior, and experimental or unsupported label paths.
- **SC-009**: Missing glyphs and label-related unsupported, experimental, and placeholder/fallback paths are surfaced as structured diagnostics before render where inputs make them knowable.
- **SC-010**: Repeated execution with fixed inputs produces stable IDs, stable batch result ordering, stable diagnostics ordering, and stable emitted glyph instance ordering.

## Assumptions

- Feature `001` defines the shared diagnostic taxonomy, but this feature may use or require compatible label diagnostics where label API truth depends on them.
- Stable IDs are required for created labels, line labels, callouts, and overlays only where users need update, inspect, remove, export, or review workflows; purely ephemeral internal objects do not need public IDs.
- Public high-level label methods may wrap existing lower-level substrate, but the MVP user workflow must not require raw IPC.
- Typography support can be proven by visible render output, measurable layout metrics, or serializable state; if none is available, the behavior must be diagnosed rather than reported successful.
- Declutter algorithm support can be proven by changed placement behavior; if no alternate algorithm is actually implemented, the setter must return typed unsupported or experimental diagnostics.
- Curved labels may be classified as `experimental` or `unsupported` if production-correct rendering is not available in this feature.
- Terrain-elevated line-label behavior may depend on terrain sampling not owned by this feature; if unavailable, explicit diagnostics satisfy the honesty requirement until a later feature implements the sampler.
- Existing dirty workspace changes are user-owned and are not part of this feature unless explicitly touched by this specification workflow.

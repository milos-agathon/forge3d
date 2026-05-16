# Feature Specification: Diagnostics and Support Matrices

**Feature Branch**: `001-diagnostics-support-matrices`
**Created**: 2026-05-14
**Status**: Draft
**Input**: User description: "Create feature 001-diagnostics-support-matrices for forge3d offline 3D map rendering."
**Source of Truth**: `docs/superpowers/plans/prd.md` and `.specify/memory/constitution.md`
**PRD Coverage**: P0-R5, P0-R6, Section 13, Section 16, Appendix B, Milestone 0
**MVP Blocking**: Yes - P0 diagnostics and support-matrix honesty are release-blocking unless a documented human decision reclassifies them

## Clarifications

### Session 2026-05-14

- Safe default: Error and fatal diagnostics always block successful render completion; warning diagnostics block only when fail-on-warning behavior is selected; informational diagnostics never block by themselves.
- Safe default: CRS mismatches without an explicit transform are diagnosed as `crs_mismatch`; this feature does not decide the minimum supported CRS transform set for later render workflows.
- Safe default: Building, 3D Tiles, VT, and label paths are classified from observable public workflow behavior: existing but incomplete workflows are `underdeveloped`, native-only requirements are `Pro-gated`, non-renderable fallback output is `placeholder/fallback`, unverified APIs are `experimental`, and unavailable product capability is `unsupported` or `missing` according to Appendix B.
- Safe default: Bundle-ready diagnostics must be serializable without information loss and must be included by any scene bundle path that accepts validation metadata; full bundle round-trip remains outside this feature.
- Safe default: Validation diagnostics, layer summaries, supported feature summaries, and unsupported feature summaries must use deterministic ordering for fixed inputs so review bundles and tests are reproducible.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Validate Before Render (Priority: P1)

A geospatial Python user validates an offline 3D map-rendering workflow before rendering and receives a structured report that identifies unsupported, risky, incomplete, or fallback paths.

**Why this priority**: This is the primary user-visible value of P0-R5. Users must know before render when output would be incomplete, misleading, or unsupported.

**Independent Test**: Can be fully tested by validating scenes or bundle-ready scene descriptions that contain known CRS, glyph, style, Pro-gated, placeholder/fallback, experimental, VT, 3D Tiles, memory, and label-rejection conditions, then inspecting structured diagnostics.

**Acceptance Scenarios**:

1. **Given** a map-rendering workflow with a layer CRS that differs from the scene or terrain CRS and no transform is provided, **When** validation runs, **Then** the report includes a `crs_mismatch` diagnostic with severity, affected layer ID where known, and remediation.
2. **Given** labels containing glyphs absent from the active atlas, **When** validation runs, **Then** the report includes `missing_glyphs` diagnostics before render instead of relying on printed warnings or visual failure.
3. **Given** a workflow that requests a Pro-gated or placeholder/fallback path, **When** validation runs, **Then** the report includes `pro_gated_path` or `placeholder_fallback` diagnostics and does not describe the path as supported.

---

### User Story 2 - Configure Render Failure Policy (Priority: P1)

A production user chooses whether rendering should stop on warnings or continue while preserving diagnostics for review.

**Why this priority**: P0-R5 requires configurable render behavior so strict production runs can fail early while exploratory workflows can continue honestly.

**Independent Test**: Can be tested with a workflow that produces at least one warning-level diagnostic and verifying both fail-on-warning and continue-on-warning outcomes.

**Acceptance Scenarios**:

1. **Given** validation produces warnings and render policy is fail-on-warning, **When** render is requested, **Then** rendering stops before output is treated as successful and returns the blocking diagnostics.
2. **Given** validation produces warnings and render policy is continue-on-warning, **When** render is requested, **Then** rendering may proceed while the report and bundle-ready metadata retain the warnings.

---

### User Story 3 - Understand Style Support (Priority: P1)

A developer or map author checks style support and sees which local/provided feature styling paths are supported, which fields or layer types are unsupported, and that streamed MVT rendering is not claimed.

**Why this priority**: P0-R6 prevents users from mistaking limited offline feature styling for full Mapbox Style Specification or streamed vector-tile support.

**Independent Test**: Can be tested by reviewing the support matrix and validating style inputs with supported and unsupported layer types and fields.

**Acceptance Scenarios**:

1. **Given** documentation for style support, **When** a user reads it, **Then** it states that P0 support applies to local/provided features and not streamed MVT tiles.
2. **Given** a style using supported `fill`, `line`, or `circle` layer types as confirmed by the implementation audit, **When** support is documented, **Then** the support matrix explicitly lists those layer types and their supported fields.
3. **Given** a style using an unsupported layer type or unsupported paint/layout field, **When** validation runs, **Then** diagnostics identify `unsupported_style_layer_type` or `unsupported_style_field` with affected style layer ID where known.

---

### User Story 4 - Audit Public Support Claims (Priority: P2)

An internal forge3d developer reviews docs and release materials to ensure map-rendering support is classified using the PRD taxonomy.

**Why this priority**: Milestone 0 requires support matrix cleanup so users and future contributors do not overclaim incomplete substrate.

**Independent Test**: Can be tested by auditing required documentation and release-facing support tables against Appendix B classifications and the diagnostics reference.

**Acceptance Scenarios**:

1. **Given** public documentation for labels, buildings, 3D Tiles, virtual texturing, diagnostics, and style support, **When** reviewed, **Then** it distinguishes `supported`, `underdeveloped`, `missing`, `Pro-gated`, `placeholder/fallback`, `experimental`, `unsupported`, and `non-goal` paths.
2. **Given** docs or support matrices mention 3D Tiles, buildings, VT normal/mask, line or curved labels, or Pro-gated paths, **When** reviewed, **Then** they do not describe incomplete or gated paths as fully supported.

### Edge Cases

- Validation receives a condition with no known layer or object ID: the diagnostic still exists and records unknown affected IDs without blocking serialization.
- Multiple diagnostics apply to the same layer or object: all diagnostics are preserved in deterministic, reviewable order.
- A diagnostic is informational but render policy is fail-on-warning: informational diagnostics do not block by themselves.
- Validation produces error or fatal diagnostics while continue-on-warning is selected: render completion is still blocked because the policy only changes warning behavior.
- Unsupported style fields appear inside otherwise supported layer types: the layer is not silently accepted as fully supported.
- A style layer has an ID but the affected feature object does not: the diagnostic includes the layer ID and leaves object ID unset.
- Memory cannot be estimated because required inputs are absent: validation reports the unavailable estimate honestly rather than inventing precision.
- Bundle-ready validation data contains diagnostics from paths whose final rendering implementation belongs to later features: the diagnostic still records support level and remediation without claiming render support.

### Support-Level Classification *(mandatory)*

Use the PRD Appendix B terms exactly: `supported`, `underdeveloped`, `missing`, `Pro-gated`, `placeholder/fallback`, `experimental`, `unsupported`, and `non-goal`.

- **Supported in this feature**: structured diagnostics, severity classification, affected ID reporting where known, bundle-ready diagnostic serialization, render warning policy, style support matrix, diagnostics reference, and support-level documentation cleanup.
- **Underdeveloped**: existing label, 3D Tiles, building, style, and VT substrate where public workflow, integration, diagnostics, or end-to-end rendering is incomplete.
- **Missing**: product capabilities that do not yet exist for the scoped map-rendering workflow, including deterministic `LabelPlan`, typed `MapScene` recipe, and VT normal/mask runtime support until implemented by their owning features.
- **Pro-gated**: capabilities that require native/Pro symbols or packaging and therefore cannot be presented as public/open supported paths.
- **Placeholder/fallback**: any fallback path that would return zero geometry, non-renderable geometry, placeholder output, or incomplete output while appearing successful.
- **Experimental**: APIs or substrate that exist but are incomplete, unverified, or not safe to present as production-ready, including line/curved label behavior until proven by feature `002`.
- **Unsupported**: product inputs or workflows that cannot be rendered or validated as requested in PRD-scoped workflows.
- **Non-goal**: streamed MVT rendering, full Mapbox Style Specification parity, full MapScene rendering in this feature, label placement implementation, building implementation, production 3D Tiles runtime, VT normal/mask runtime implementation, browser delivery, hosted tile services, general DCC, and game/editor tooling.

### Diagnostics & Failure Behavior *(mandatory)*

- **Diagnostic objects**: Diagnostics must include code, severity, message, remediation, support level where applicable, affected layer ID where known, affected object ID where known, and bundle-ready serialization data.
- **Required diagnostic codes**: `crs_mismatch`, `missing_glyphs`, `unsupported_style_field`, `unsupported_style_layer_type`, `pro_gated_path`, `placeholder_fallback`, `experimental_feature`, `vt_unsupported_family`, `python_public_3dtiles_incomplete`, `estimated_gpu_memory`, and `label_rejection_summary`.
- **Severity behavior**: Allowed severity values are `info`, `warning`, `error`, and `fatal`. Validation reports must expose the highest applicable status across diagnostics. Error and fatal diagnostics always prevent successful render completion; warning diagnostics prevent successful render completion only when fail-on-warning behavior is selected; informational diagnostics never block by themselves.
- **Severity floors**: `crs_mismatch` without transform, unsupported style layer types, Pro-gated required paths, placeholder/fallback required paths, VT unsupported families, incomplete public 3D Tiles paths, and unsupported product inputs must be at least `error` when they affect requested output. `missing_glyphs`, unsupported style fields, experimental features, estimated GPU memory budget risk, and label rejection summaries must be at least `warning` when they can affect output. Pure inventory or summary diagnostics may be `info` only when they do not affect requested output.
- **Pre-render validation**: Validation must detect PRD-scoped unsupported, experimental, Pro-gated, placeholder/fallback, style, glyph, CRS, VT, 3D Tiles, memory, and label rejection conditions before render where enough input information exists.
- **No-op prevention**: Unsupported features must produce typed diagnostics or typed failure behavior; they must not be silently ignored or represented only by logs, printed strings, or successful no-op outcomes.
- **Bundle readiness**: Diagnostics must have a serializable representation suitable for scene bundles and bundle-ready structures. Any bundle path that records validation metadata must preserve diagnostics without information loss; full bundle load/render round-trip remains outside this feature.
- **Determinism**: For fixed inputs, diagnostics and support summaries must be stable for serialized comparison, documentation evidence, and bundle review. Ordering must be deterministic by severity rank, diagnostic code, affected layer ID, affected object ID, and message or equivalent stable tie-breakers.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001 [PRD: P0-R5-AC1]**: The system MUST return diagnostics as structured objects for PRD-scoped map-rendering workflows, not only as printed strings, log lines, or warnings.
- **FR-002 [PRD: P0-R5-AC2]**: Each diagnostic MUST include one of the severity values `info`, `warning`, `error`, or `fatal`.
- **FR-003 [PRD: P0-R5-AC3]**: Each diagnostic MUST include affected layer and object identifiers where the workflow input makes them knowable.
- **FR-004 [PRD: P0-R5-AC4]**: Diagnostic reports MUST be serializable in scene bundles or bundle-ready structures without losing code, severity, affected IDs, remediation, or support-level information.
- **FR-005 [PRD: P0-R5-AC5]**: Users MUST be able to choose render behavior that either fails on warning-level diagnostics or continues while preserving diagnostics; this choice MUST NOT allow error or fatal diagnostics to be treated as successful render completion.
- **FR-006 [PRD: P0-R5-AC6]**: PRD-scoped unsupported features MUST never be silently ignored; validation or render preparation must return structured diagnostics or typed failure behavior.
- **FR-007 [PRD: Section 13]**: Validation reports MUST expose an overall status, diagnostics, layer summaries, GPU memory estimate when known, supported feature summary, and unsupported feature summary.
- **FR-008 [PRD: Section 13]**: The diagnostic inventory MUST include `crs_mismatch`, `missing_glyphs`, `unsupported_style_field`, `unsupported_style_layer_type`, `pro_gated_path`, `placeholder_fallback`, `experimental_feature`, `vt_unsupported_family`, `python_public_3dtiles_incomplete`, `estimated_gpu_memory`, and `label_rejection_summary`.
- **FR-009 [PRD: P0-R6-AC1]**: Documentation MUST state that P0 style support applies to local or provided features, not streamed MVT tiles.
- **FR-010 [PRD: P0-R6-AC2]**: Documentation MUST explicitly list supported style layer types.
- **FR-011 [PRD: P0-R6-AC3]**: The support matrix MUST include `fill`, `line`, and `circle` layers when the implementation audit confirms they are supported.
- **FR-012 [PRD: P0-R6-AC4]**: Unsupported style layer types and unsupported paint or layout fields MUST produce `unsupported_style_layer_type` or `unsupported_style_field` diagnostics.
- **FR-013 [PRD: P0-R6-AC5]**: Style application outputs MUST be defined in a way that can feed `VectorOverlay` workflows and future `LabelLayer` workflows without claiming streamed MVT support.
- **FR-014 [PRD: P0-R6-AC6]**: Public documentation MUST NOT claim full Mapbox Style Specification support.
- **FR-015 [PRD: Section 16]**: Documentation MUST include or update the Offline 3D Map Rendering Quickstart, LabelPlan Guide references, Style Support Matrix, Building Layer Support Matrix, 3D Tiles Support Matrix, Virtual Texturing Support Matrix, Diagnostics Reference, and Competitive Positioning Note as needed for this feature's scope.
- **FR-016 [PRD: Appendix B]**: Support classifications in specs, docs, diagnostics, release notes, and support matrices MUST use only `supported`, `underdeveloped`, `missing`, `Pro-gated`, `placeholder/fallback`, `experimental`, `unsupported`, and `non-goal`.
- **FR-017 [PRD: Milestone 0]**: Public support matrices MUST distinguish supported, underdeveloped, missing, Pro-gated, placeholder/fallback, experimental, unsupported, and non-goal map-rendering features.
- **FR-018 [PRD: P0-R5-AC6]**: Diagnostics for Pro-gated, placeholder/fallback, experimental, VT unsupported family, incomplete public 3D Tiles, missing glyph, CRS mismatch, GPU memory, and label rejection paths MUST be represented even when final render implementations belong to later features.
- **FR-019 [PRD: Constitution IV]**: Validation reports, diagnostics, layer summaries, supported feature summaries, unsupported feature summaries, and serialized diagnostic payloads MUST be deterministic for fixed inputs.
- **FR-020 [PRD: Section 16, Appendix B]**: Required documentation MUST avoid umbrella wording such as "partial support" unless it is paired with the exact PRD classification and a concrete diagnostic or limitation.

### Key Entities *(include if feature involves data)*

- **ValidationReport**: A structured validation result for a map-rendering workflow. Key attributes are overall status, diagnostics, layer summaries, estimated GPU memory when known, supported feature summary, and unsupported feature summary.
- **Diagnostic**: A structured validation finding. Key attributes are code, severity, message, remediation, support level where applicable, affected layer ID where known, affected object ID where known, and serialization-ready data.
- **LayerSummary**: A per-layer validation summary that records layer identity, support status, relevant feature support, and diagnostic references.
- **SupportMatrixEntry**: A documentation and review entry for a feature, layer type, style field, or workflow path. Key attributes are support classification, scope, limitations, diagnostics emitted, and remediation or next step.
- **RenderFailurePolicy**: User-selected behavior for warning-level diagnostics, distinguishing fail-on-warning from continue-with-diagnostics.
- **SeverityPolicy**: User-visible blocking rules that determine report status and render completion behavior from diagnostic severities.
- **Public API contract**: This feature defines behavior for `ValidationReport` and `Diagnostic` and their bundle-ready use by `MapScene`, `SceneRecipe`, `LabelPlan`, and `Bundle`; it does not implement the full render path.

## PRD Traceability & Evidence *(mandatory)*

| PRD AC ID | Requirement in this spec | Evidence required | Verification target |
|---|---|---|---|
| P0-R5-AC1 | FR-001 | Diagnostic data model and tests asserting object fields | Diagnostic object tests |
| P0-R5-AC2 | FR-002 | Severity coverage for `info`, `warning`, `error`, `fatal` | Severity validation tests |
| P0-R5-AC3 | FR-003 | Fixtures showing layer and object ID propagation where possible | Affected ID diagnostics tests |
| P0-R5-AC4 | FR-004 | Scene-bundle and bundle-ready serialization evidence preserving diagnostic fields | Serialization and bundle persistence tests |
| P0-R5-AC5 | FR-005 | Fail-on-warning and continue-on-warning behavior evidence | Render policy tests |
| P0-R5-AC6 | FR-006, FR-018 | Negative tests proving unsupported paths are not silently ignored | Unsupported/fallback diagnostic tests |
| P0-R6-AC1 | FR-009 | Style docs state local/provided feature scope and exclude streamed MVT | Style support docs audit |
| P0-R6-AC2 | FR-010 | Supported layer types are listed | Style support matrix audit |
| P0-R6-AC3 | FR-011 | `fill`, `line`, and `circle` status is audited and documented | Style support matrix plus implementation audit |
| P0-R6-AC4 | FR-012 | Unsupported style type and field fixtures emit diagnostics | Style diagnostics tests |
| P0-R6-AC5 | FR-013 | Style output contract is consumable by vector workflows and future labels | Style workflow contract tests or review |
| P0-R6-AC6 | FR-014 | Public docs do not claim full Mapbox Style Specification support | Docs wording audit |
| Section 13 | FR-007, FR-008 | Validation report fields and required diagnostic inventory | Validation report contract tests |
| Section 16 | FR-015 | Required docs and wording discipline updates | Docs review checklist |
| Appendix B | FR-016, FR-020 | Support classifications use only PRD terms and avoid ambiguous umbrella wording | Support taxonomy audit |
| Milestone 0 | FR-017 | Support matrices and diagnostics taxonomy exist | Milestone exit review |
| Constitution IV | FR-019 | Stable report ordering and serialized diagnostic comparison evidence | Determinism tests or review |

## Explicit Non-Goals *(mandatory)*

- Implementing the full `MapScene` render path.
- Implementing a streamed vector-tile renderer.
- Implementing label placement.
- Implementing buildings, 3D Tiles, or VT normal/mask runtime.
- Implementing full Mapbox Style Specification support.
- Implementing deterministic `LabelPlan` placement or label rendering/export behavior.
- Implementing browser delivery, hosted tile services, general DCC workflows, game/editor tooling, or non-map rendering features.
- Treating Pro-gated, placeholder/fallback, experimental, or underdeveloped paths as public supported features.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of P0-R5 and P0-R6 acceptance criteria have explicit requirements, evidence expectations, and verification targets in this specification.
- **SC-002**: 100% of required MVP diagnostic codes are represented in the validation/reporting contract and diagnostics reference scope.
- **SC-003**: Validation for known unsupported or incomplete map-rendering inputs returns structured diagnostics with code, severity, remediation, and affected IDs where known.
- **SC-004**: Render policy can be demonstrated in both fail-on-warning and continue-on-warning modes for the same warning-producing workflow.
- **SC-005**: Public style documentation distinguishes local/provided feature styling from streamed MVT and contains no full Mapbox Style Specification support claim.
- **SC-006**: Support matrices cover style, labels, buildings, 3D Tiles, virtual texturing, and diagnostics taxonomy using only PRD-approved support classifications.
- **SC-007**: Diagnostic serialization preserves all required diagnostic fields in scene-bundle or bundle-ready review data.
- **SC-008**: Documentation review finds zero overclaims that classify underdeveloped, Pro-gated, placeholder/fallback, experimental, unsupported, or non-goal paths as supported.
- **SC-009**: Repeated validation of the same fixed workflow produces the same diagnostic codes, severities, affected IDs, support summaries, and serialized ordering.
- **SC-010**: Every support matrix entry that is not `supported` includes the exact PRD classification, diagnostic behavior, and user-facing limitation or remediation.

## Assumptions

- This feature may define bundle-ready diagnostic structures before full bundle round-trip behavior is implemented by later features.
- CRS transform implementation scope belongs to later `MapScene` work; this feature requires honest `crs_mismatch` diagnostics when a mismatch is detectable and no transform is provided.
- The minimum supported CRS transform set remains a later product decision; until it is decided, the safe behavior is to diagnose mismatches rather than silently transform or silently continue.
- Style support for `fill`, `line`, and `circle` must be documented according to the actual implementation audit rather than assumed universally supported.
- Label placement, buildings, 3D Tiles, and VT runtime work remain outside this feature, but their incomplete or unsupported paths must be diagnosable and documented.
- Public/open versus Pro-gated building packaging remains a later product decision; this feature classifies paths by observable availability and whether native/Pro capability is required.
- Existing dirty workspace changes are user-owned and are not part of this feature unless explicitly touched by this specification workflow.

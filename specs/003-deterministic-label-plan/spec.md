# Feature Specification: Deterministic LabelPlan

**Feature Branch**: `003-deterministic-label-plan`
**Created**: 2026-05-14
**Status**: Draft
**Input**: User description: "Create feature 003-deterministic-label-plan for forge3d offline 3D map rendering."
**Source of Truth**: `docs/superpowers/plans/prd.md` and `.specify/memory/constitution.md`
**PRD Coverage**: P0-R3, Section 14, Milestone 2, related `missing_glyphs` and `label_rejection_summary` diagnostics from P0-R5
**MVP Blocking**: Yes - deterministic label compilation is release-blocking unless a documented human decision reclassifies it

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compile A Reproducible Label Plan (Priority: P1)

A cartographic production user compiles offline map labels before rendering and receives the same accepted and rejected label sets whenever the source labels, camera, viewport, terrain transform, typography, glyph atlas, keepouts, priority rules, and seed are unchanged.

**Why this priority**: This is the core P0-R3 outcome. Offline map production depends on reviewable and reproducible label placement rather than live, order-dependent placement behavior.

**Independent Test**: Can be fully tested by compiling the same controlled label layer multiple times with fixed camera, viewport, terrain transform, keepouts, priority rules, glyph atlas, typography, and seed, then comparing accepted labels, rejected labels, diagnostics, bounds, and seed.

**Acceptance Scenarios**:

1. **Given** identical label inputs, camera, viewport, terrain transform, keepouts, typography, glyph atlas, priority rules, and seed, **When** the user compiles a `LabelPlan` twice, **Then** the accepted labels, rejected labels, diagnostics, bounds, and seed are identical in content and deterministic order.
2. **Given** a label set with competing candidates, **When** the plan is compiled with a fixed seed, **Then** any seeded tie-breaking produces the same winner and loser set on repeated runs.
3. **Given** label source inputs arrive in different but equivalent map, set, or filesystem iteration orders, **When** the plan is compiled, **Then** normalized ordering preserves the same accepted and rejected plan.

---

### User Story 2 - Inspect Rejected Labels With Reasons (Priority: P1)

A map author can inspect labels that were not accepted and see explicit reason codes such as collision, outside view, missing glyph, lower priority, keepout conflict, terrain occlusion, invalid geometry, unsupported geometry type, or empty text.

**Why this priority**: Rejected labels must not disappear silently. Reason codes are required for map QA, diagnostics, support, and repeatable production review.

**Independent Test**: Can be tested with fixtures that trigger each required rejection reason and verify every rejected label or rejected candidate has a reason code and stable affected label identity where known.

**Acceptance Scenarios**:

1. **Given** two labels collide and one has lower priority, **When** the plan is compiled, **Then** the loser is rejected with `priority_lost` or `collision` according to the documented precedence rules.
2. **Given** label text contains glyphs absent from the active atlas, **When** the plan is compiled, **Then** the affected label is rejected or diagnosed with `missing_glyph` and the plan diagnostics include `missing_glyphs`.
3. **Given** labels fall outside the viewport, overlap a keepout region, are terrain-occluded, have invalid geometry, use unsupported geometry, or contain empty text, **When** the plan is compiled, **Then** each rejected item records the matching reason code.

---

### User Story 3 - Generate Candidate Positions For Points And Polygons (Priority: P1)

A map author uses point and polygon labels and expects the compiler to evaluate standard cartographic candidate positions before choosing accepted labels.

**Why this priority**: Point and polygon candidate generation is the minimum useful placement surface for the deterministic compiler in Milestone 2.

**Independent Test**: Can be tested by compiling point and polygon fixtures and inspecting candidate generation, accepted placements, and rejected candidates without needing full `MapScene` orchestration.

**Acceptance Scenarios**:

1. **Given** a point label, **When** candidates are generated, **Then** center, above, below, left, right, and radial candidates are available according to the label's configured candidate policy.
2. **Given** a valid polygon label, **When** candidates are generated, **Then** the polygon centroid is considered and a visual-center or polylabel fallback is used where centroid placement is unsuitable.
3. **Given** invalid polygon geometry or unsupported geometry type, **When** compilation runs, **Then** the label is rejected with `invalid_geometry` or `unsupported_geometry_type` instead of being silently skipped.

---

### User Story 4 - Respect Terrain, Keepouts, And Priorities (Priority: P1)

A cartographic production user compiles labels against terrain elevation, map furniture keepout regions, manual keepout rectangles, and priority classes so the accepted label plan is ready for rendering or export.

**Why this priority**: P0-R3 requires terrain sampling, keepouts, and priorities for offline label placement to be useful in composed map plates.

**Independent Test**: Can be tested with known elevation samples, keepout rectangles for each map furniture type, and collision fixtures where priority classes determine winners.

**Acceptance Scenarios**:

1. **Given** an active DEM or terrain transform exposes elevation for label positions, **When** terrain labels are compiled, **Then** accepted label positions use sampled elevation or rejected labels report `terrain_occluded` where the terrain blocks placement.
2. **Given** keepout regions for title boxes, legends, scale bars, north arrows, and manual rectangles, **When** labels intersect those regions, **Then** affected candidates are rejected with `keepout_region`.
3. **Given** colliding labels from different priority classes, **When** the compiler solves placement, **Then** priority classes influence deterministic winners and lower-priority candidates are rejected with stable reason codes.

---

### User Story 5 - Render Or Export The Accepted Plan (Priority: P2)

A user can take the accepted labels from a compiled plan and pass them to a render or export workflow without recompiling placement or losing diagnostics.

**Why this priority**: P0-R3 requires the compiled plan to be renderable and exportable, but full `MapScene` orchestration belongs to feature `004`.

**Independent Test**: Can be tested by compiling a plan, exporting or serializing the accepted payload, and verifying the payload preserves accepted labels, rejected labels, diagnostics, bounds, and seed.

**Acceptance Scenarios**:

1. **Given** a compiled plan with accepted labels, **When** the user requests a render payload or export payload, **Then** accepted labels are exposed in deterministic order with enough placement, bounds, typography, and glyph information for downstream rendering or export.
2. **Given** render or export cannot be completed by the active backend, **When** the user requests it, **Then** the plan returns structured diagnostics instead of reporting a successful no-op.

### Edge Cases

- A label has empty or whitespace-only text: it is rejected with `empty_text` and does not enter collision solving.
- Label text contains one or more glyphs absent from the active atlas: the label or affected candidate is rejected with `missing_glyph`, and diagnostics summarize missing glyph coverage.
- A feature has invalid coordinates, zero-area polygon geometry, self-intersection, or non-finite bounds: it is rejected with `invalid_geometry`.
- A feature geometry type is outside the feature scope: it is rejected with `unsupported_geometry_type`.
- A label candidate is completely outside the viewport or outside the camera-visible area: it is rejected with `outside_view`.
- A label is partially visible but all candidates collide, lose priority, intersect keepouts, or are terrain-occluded: rejected candidates retain the reason that caused final exclusion.
- Multiple rejection reasons apply to one label or candidate: the plan records a deterministic primary reason and may include secondary details in diagnostics.
- A terrain sampler is unavailable: terrain elevation is not invented; the plan either uses documented flat placement where allowed by the input contract or emits typed diagnostics for terrain-dependent labels.
- Keepout rectangles overlap each other or overlap the viewport boundary: each keepout remains active and deterministic ordering decides diagnostic ordering only, not whether the keepout applies.
- Randomized radial candidate ordering or jitter is requested: randomness is seeded and the generated candidate order is reproducible for the same seed.
- Input label order is nondeterministic due to dictionaries, sets, filesystem enumeration, or parallel ingestion: the compiler normalizes iteration before scoring and serialization.

### Support-Level Classification *(mandatory)*

Use the PRD Appendix B terms exactly: `supported`, `underdeveloped`, `missing`, `Pro-gated`, `placeholder/fallback`, `experimental`, `unsupported`, and `non-goal`.

- **Supported in this feature**: deterministic `LabelPlan.compile`, accepted and rejected plan inspection, required rejection reason codes, point and polygon candidate generation, terrain sampling where available, keepout regions, priority classes, deterministic diagnostics, bounds, seed exposure, and render/export payload exposure.
- **Underdeveloped**: existing label substrate where public label creation, line/curved labels, typography, atlas state, or diagnostics are not yet fully productized by features `001` and `002`.
- **Missing**: the deterministic `LabelPlan` product feature before this specification is implemented.
- **Pro-gated**: any terrain, atlas, rendering, or export capability that requires native or Pro-only functionality outside the public workflow.
- **Placeholder/fallback**: any plan compile, render, or export path that would return empty placement, placeholder output, or non-renderable data while appearing successful.
- **Experimental**: candidate policies, terrain occlusion behavior, or render/export backends that exist but are not yet verified as production-ready.
- **Unsupported**: advanced geometry types, unavailable render/export paths, unsupported glyph coverage, invalid geometry, or terrain-dependent behavior that cannot be compiled truthfully.
- **Non-goal**: advanced repeated labels along long lines, full curved text shaping, non-Latin shaping, full `MapScene` render orchestration, streamed MVT rendering, browser delivery, hosted tile services, general DCC workflows, game/editor tooling, and non-map rendering features.

### Diagnostics & Failure Behavior *(mandatory)*

- **Diagnostic objects**: Plan diagnostics must include code, severity, message, remediation, support level where applicable, affected label ID where known, affected layer or source ID where known, affected candidate ID where available, and deterministic serialization data.
- **Plan outputs**: `LabelPlan` must expose `accepted`, `rejected`, `diagnostics`, `bounds`, and `seed`.
- **Required rejection reasons**: Rejected labels or candidates must use `collision`, `outside_view`, `missing_glyph`, `priority_lost`, `keepout_region`, `terrain_occluded`, `invalid_geometry`, `unsupported_geometry_type`, and `empty_text`.
- **Required related diagnostic codes**: The compiler must emit or contribute to `missing_glyphs` and `label_rejection_summary` diagnostics where applicable.
- **Pre-render validation**: Missing glyphs, empty text, invalid geometry, unsupported geometry, outside-view placement, keepout conflicts, priority losses, terrain occlusion, and unsupported render/export payloads must be detectable before final render where enough input information exists.
- **No-op prevention**: Successful compile, render-payload, and export-payload results must contain real accepted/rejected plan data or structured diagnostics. Empty placeholder success is not allowed.
- **Determinism**: The compiler must normalize input ordering, candidate ordering, scoring tie-breaks, collision solving, diagnostics ordering, bounds ordering, and serialized payload ordering for fixed inputs and seed.
- **Failure behavior**: Unsupported or experimental paths return typed diagnostics or typed errors. Rejected labels are retained with reason codes rather than dropped.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001 [PRD: P0-R3-AC1, Section 14.4]**: `LabelPlan.compile` MUST produce the same accepted labels, rejected labels, diagnostics, bounds, and seed for identical labels, camera, viewport, terrain transform, typography, glyph atlas, keepouts, priority rules, and seed.
- **FR-002 [PRD: P0-R3-AC1, Section 14.4]**: The compiler MUST normalize any iteration order that could vary across maps, sets, filesystem traversal, parallel ingestion, or backend enumeration before candidate scoring, collision solving, diagnostics, and serialization.
- **FR-003 [PRD: P0-R3-AC2, Section 14.3]**: Rejected labels or rejected candidates MUST include one of the required reason codes: `collision`, `outside_view`, `missing_glyph`, `priority_lost`, `keepout_region`, `terrain_occluded`, `invalid_geometry`, `unsupported_geometry_type`, or `empty_text`.
- **FR-004 [PRD: P0-R3-AC2, P0-R5 label diagnostics]**: The plan MUST expose structured diagnostics for missing glyphs and label rejection summaries, including affected label or layer identifiers where known.
- **FR-005 [PRD: P0-R3-AC3]**: Point labels MUST support center, above, below, left, right, and radial placement candidates.
- **FR-006 [PRD: P0-R3-AC3]**: Radial candidate generation MUST be deterministic for fixed input and seed, including any angle order, jitter, or tie-break behavior.
- **FR-007 [PRD: P0-R3-AC4]**: Polygon labels MUST support centroid candidate placement and a visual-center or polylabel fallback when centroid placement is unsuitable.
- **FR-008 [PRD: P0-R3-AC4]**: Polygon candidate generation MUST reject invalid polygon geometry with `invalid_geometry` rather than silently skipping it.
- **FR-009 [PRD: P0-R3-AC5, Section 14.1]**: Terrain labels MUST sample elevation from the active DEM, terrain transform, or elevation sampler where available.
- **FR-010 [PRD: P0-R3-AC5]**: Terrain-dependent labels MUST report `terrain_occluded` or a typed diagnostic when terrain visibility or sampling prevents truthful placement.
- **FR-011 [PRD: P0-R3-AC6]**: Keepout regions MUST support title boxes, legends, scale bars, north arrows, and manual rectangles.
- **FR-012 [PRD: P0-R3-AC6]**: Candidates intersecting active keepout regions MUST be rejected with `keepout_region`.
- **FR-013 [PRD: P0-R3-AC7]**: Label priority classes MUST be supported and MUST influence deterministic collision winners and losers.
- **FR-014 [PRD: P0-R3-AC7]**: Priority tie-breaks MUST be deterministic and documented through stable sort keys or equivalent user-visible behavior.
- **FR-015 [PRD: P0-R3-AC8, Section 14.2]**: The compiled plan MUST expose accepted labels in a render/export-ready payload without requiring placement recompilation.
- **FR-016 [PRD: P0-R3-AC8]**: Exported or serialized plan payloads MUST preserve accepted labels, rejected labels, diagnostics, bounds, and seed in deterministic order.
- **FR-017 [PRD: Section 14.1]**: `LabelPlan.compile` MUST accept label source features, camera, viewport or output size, terrain transform or elevation sampler, keepout regions, priority rules, typography settings, active glyph atlas, and deterministic seed.
- **FR-018 [PRD: Section 14.2]**: `LabelPlan` MUST expose `accepted`, `rejected`, `diagnostics`, `bounds`, and `seed`.
- **FR-019 [PRD: P0-R5 label diagnostics]**: Missing glyph handling MUST be available before final render and must not rely only on printed warnings or visual failure.
- **FR-020 [PRD: Constitution III]**: Compile, render-payload, and export-payload operations MUST NOT report success while doing nothing or returning non-renderable placeholder output.
- **FR-021 [PRD: Constitution IV]**: Plan comparison for fixed inputs MUST be stable enough for automated regression evidence without depending on platform-specific map iteration.
- **FR-022 [PRD: Milestone 2]**: The feature MUST include tests or verification artifacts demonstrating fixed inputs produce identical accepted/rejected label plans and that every rejected label has a reason code.

### Key Entities *(include if feature involves data)*

- **LabelPlan**: A compiled, deterministic plan containing accepted labels, rejected labels, diagnostics, bounds, and the seed used for placement.
- **Accepted Label**: A label selected for rendering or export, including stable label identity where known, source reference, placement candidate, screen or world bounds where applicable, typography reference, glyph coverage status, and priority class.
- **Rejected Label**: A source label or candidate not selected for rendering, including stable label identity where known, reason code, candidate information where applicable, affected bounds where known, and diagnostic references.
- **Label Candidate**: A proposed label placement derived from source geometry, including anchor, candidate type, score inputs, bounds, terrain sample where applicable, and deterministic ordering key.
- **Point Candidate Policy**: Candidate set for point labels, including center, above, below, left, right, and radial candidates.
- **Polygon Candidate Policy**: Candidate set for polygon labels, including centroid and visual-center or polylabel fallback.
- **Keepout Region**: A screen-space or map-layout exclusion area for title boxes, legends, scale bars, north arrows, or manual rectangles.
- **Priority Class**: A deterministic ordering and scoring category used to decide collision winners and losers.
- **Terrain Sample**: Elevation or visibility information taken from the active DEM, terrain transform, or elevation sampler where available.
- **Plan Bounds**: Screen and world bounds exposed where applicable for accepted labels, rejected candidates, and the overall plan.
- **Public API contract**: This feature defines `LabelPlan` behavior used by `MapScene`, `SceneRecipe`, `ValidationReport`, `Diagnostic`, `Bundle`, render, and export workflows; it does not implement full `MapScene` orchestration.

## PRD Traceability & Evidence *(mandatory)*

| PRD AC ID | Requirement in this spec | Evidence required | Verification target |
|---|---|---|---|
| P0-R3-AC1 | FR-001, FR-002, FR-021, FR-022 | Repeated compile tests comparing accepted, rejected, diagnostics, bounds, and seed for fixed inputs | LabelPlan determinism tests |
| P0-R3-AC2 | FR-003, FR-004, FR-022 | Rejection fixtures covering every required reason code and diagnostics summary | Rejection reason tests |
| P0-R3-AC3 | FR-005, FR-006 | Point candidate fixtures for center, above, below, left, right, and radial candidates | Point candidate tests |
| P0-R3-AC4 | FR-007, FR-008 | Polygon fixtures proving centroid and visual-center/polylabel fallback plus invalid geometry rejection | Polygon candidate tests |
| P0-R3-AC5 | FR-009, FR-010 | Terrain sampler tests with known DEM/transform values or typed terrain diagnostics | Terrain label tests |
| P0-R3-AC6 | FR-011, FR-012 | Keepout fixtures for title boxes, legends, scale bars, north arrows, and manual rectangles | Keepout tests |
| P0-R3-AC7 | FR-013, FR-014 | Collision fixtures proving priority classes determine deterministic winners and losers | Priority solve tests |
| P0-R3-AC8 | FR-015, FR-016, FR-020 | Render/export payload tests preserving plan contents or typed diagnostics for unsupported paths | Plan render/export tests |
| Section 14.1 | FR-017 | Compile contract accepts required inputs | API contract tests |
| Section 14.2 | FR-018 | Plan exposes accepted, rejected, diagnostics, bounds, and seed | Plan output contract tests |
| Section 14.3 | FR-003 | Required rejection reason vocabulary is preserved | Rejection vocabulary tests |
| Section 14.4 | FR-001, FR-002, FR-021 | Seed and order normalization evidence | Deterministic ordering tests |
| P0-R5 label diagnostics | FR-004, FR-019 | `missing_glyphs` and `label_rejection_summary` diagnostics are structured | Diagnostic integration tests |
| Milestone 2 | FR-022 | Exit evidence for deterministic accepted/rejected plans with reason codes | Milestone exit review |
| Constitution III | FR-020 | Negative tests for no-op compile/render/export success | No-op success tests |

## Explicit Non-Goals *(mandatory)*

- Advanced repeated labels along long lines.
- Full curved text shaping.
- Non-Latin shaping.
- Full `MapScene` render orchestration.
- Replacing feature `002` public label API truth work.
- Full data-driven `LabelLayer` ingestion from geospatial tables or style expressions beyond what is needed for the label source contract.
- Full streamed MVT renderer or full Mapbox Style Specification support.
- Building, 3D Tiles, VT normal/mask runtime, browser delivery, hosted tile services, general DCC workflows, game/editor tooling, or non-map rendering features.
- Treating line labels, curved labels, Pro-gated paths, placeholder/fallback paths, or unsupported render/export backends as fully supported without tests and diagnostics.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of P0-R3 acceptance criteria have explicit requirements, evidence expectations, and verification targets in this specification.
- **SC-002**: Repeated compilation of the same fixed fixture produces identical accepted labels, rejected labels, diagnostics, bounds, and seed.
- **SC-003**: 100% of rejected labels or candidates in required fixtures include one of the required reason codes.
- **SC-004**: Point label fixtures demonstrate center, above, below, left, right, and radial candidate support.
- **SC-005**: Polygon label fixtures demonstrate centroid placement and visual-center or polylabel fallback.
- **SC-006**: Terrain label fixtures either sample known elevation values from the active DEM/terrain transform or return typed diagnostics when terrain sampling is unavailable.
- **SC-007**: Keepout fixtures cover title boxes, legends, scale bars, north arrows, and manual rectangles.
- **SC-008**: Priority fixtures demonstrate that higher-priority labels win collisions and lower-priority labels receive stable rejection reasons.
- **SC-009**: The plan exposes `accepted`, `rejected`, `diagnostics`, `bounds`, and `seed` and can produce deterministic render/export payload data.
- **SC-010**: Missing glyphs and rejection summaries are exposed as structured diagnostics before final render where inputs make them knowable.
- **SC-011**: Automated checks confirm no unresolved template placeholders or clarification markers remain in this specification.

## Assumptions

- Feature `001` defines the shared diagnostic taxonomy; this feature must use compatible `missing_glyphs` and `label_rejection_summary` diagnostics.
- Feature `002` supplies truthful label primitives where needed; this feature may define a label source contract without reimplementing all public label creation APIs.
- Terrain sampling uses the active DEM, terrain transform, or elevation sampler when available; where no sampler exists, terrain-dependent placement must be diagnosed honestly.
- The render/export requirement is satisfied by exposing deterministic accepted-label payloads and typed diagnostics for unsupported backend paths; full `MapScene.render()` orchestration belongs to feature `004`.
- Bundle default behavior for storing compiled plans versus recompiling source labels remains a later product decision, but exported plan payloads must be deterministic and bundle-ready.
- Pixel-level render tolerance is not decided in this feature; this feature verifies deterministic plan content and payload ordering.
- Existing dirty workspace changes are user-owned and are not part of this feature unless explicitly touched by this specification workflow.

# Feature Specification: Material, VT, and Large-Scene P2

**Feature Branch**: `006-material-vt-large-scene-p2`  
**Created**: 2026-05-15  
**Status**: Draft  
**Input**: User description: "Create feature 006-material-vt-large-scene-p2 for forge3d offline 3D map rendering. Scope: P2-R1 Virtual Texture Normal/Mask Runtime Support; P2-R2 Textured PBR Building Materials End to End; P2-R3 Advanced Static Label Placement; P2-R4 Large-Scene Map Rendering Maturity; Milestone 5 Material, VT, and Large-Scene Polish."  
**Source of Truth**: `docs/superpowers/plans/prd.md` and `.specify/memory/constitution.md`  
**PRD Coverage**: P2-R1, P2-R2, P2-R3, P2-R4, Milestone 5, and related diagnostics/support-level docs needed to avoid overclaiming P2 parity gaps  
**MVP Blocking**: No - this is P2 polish and parity work; it must not block MVP unless a documented human decision expands release scope

## Clarifications

### Session 2026-05-15

- Q: Can a diagnosed unsupported or fallback P2 path be classified as `supported`? -> A: No. Only real render output, real placement results, or real diagnostics from available scene data are `supported`; diagnosed unavailable paths keep their exact PRD support-level term.
- Q: What is the default user-visible behavior when a requested P2 capability is unavailable before render? -> A: Emit structured pre-render diagnostics. Error and fatal diagnostics block successful render by default; warning diagnostics follow the configured render policy.
- Q: Which diagnostic codes make texture/material and large-scene unavailable cases testable? -> A: Add feature-local structured codes for `missing_texture_path`, `missing_uvs`, `unsupported_texture_format`, `unavailable_cache_lod_stats`, and `unsupported_instancing_path`, alongside the PRD diagnostic inventory.
- Q: What deterministic comparison default applies to P2 validation and render fixtures? -> A: Use stable ordering and exact fixture comparisons for deterministic fixtures unless the plan records a numeric tolerance before verification.
- Q: Does this P2 feature block MVP readiness if a P2 path is deferred? -> A: No. Deferred P2 gaps require explicit diagnostics and documentation, not MVP blocking, unless a documented human decision expands release scope.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Validate Virtual Texture Families Before Render (Priority: P1)

A geospatial Python user requests virtual texture inputs beyond albedo and receives either real rendered normal/mask behavior or a structured pre-render diagnostic that states the exact support status.

**Why this priority**: VT normal/mask families are a named P2 parity gap. Silent skipping would make rendered maps misleading, so truthful validation is the highest-value slice.

**Independent Test**: Can be fully tested by validating scenes that request albedo-only and albedo plus normal/mask virtual texture families, then confirming rendered support or `vt_unsupported_family` diagnostics are deterministic and inspectable before render.

**Acceptance Scenarios**:

1. **Given** a scene requests only supported albedo virtual texture data, **When** validation runs, **Then** the report states the albedo family support status and does not emit non-albedo diagnostics.
2. **Given** a scene requests normal or mask virtual texture families, **When** validation runs, **Then** those families are either marked renderable or reported with `vt_unsupported_family` before render.
3. **Given** a non-albedo family cannot be rendered in the active runtime, **When** render is requested, **Then** the system does not silently skip that family while reporting success.

---

### User Story 2 - Use Textured Building Materials Honestly (Priority: P1)

A cartographic production user adds textured building materials to a `MapScene` building layer and receives either albedo-textured building output or precise diagnostics for missing UVs, missing texture paths, unsupported texture use, or explicit material fallback.

**Why this priority**: Building material textures are a visible parity gap versus higher-end offline renderers, but the product must preserve trust when fixtures lack texture prerequisites.

**Independent Test**: Can be tested with a textured building fixture, a fixture without UVs, and a missing-texture fixture, then validating that each path renders or diagnoses the exact material status before render.

**Acceptance Scenarios**:

1. **Given** a building layer has valid albedo texture data and UVs, **When** the user validates and renders the scene, **Then** the building material uses the albedo texture or reports a typed diagnostic explaining why it cannot.
2. **Given** a building layer requests a texture but UV data is absent, **When** validation runs, **Then** the report identifies the affected layer or object and explains the fallback or unsupported status.
3. **Given** a texture path is missing or unreadable, **When** validation runs, **Then** the report surfaces the missing path before render and does not present fallback material as textured PBR success.

---

### User Story 3 - Compile Advanced Static Labels (Priority: P2)

A cartographic production user prepares polished static labels for long roads, rivers, landmarks, peaks, cities, and annotations while preserving deterministic placement and rejected-label diagnostics.

**Why this priority**: Advanced placement improves map polish after the P0 deterministic label plan exists, but it remains secondary to truthful VT/material behavior in this P2 feature.

**Independent Test**: Can be tested by compiling fixed label fixtures for repeated line labels, curved paths, road/river rule presets, leader lines, and priority presets, then comparing accepted/rejected results across repeated runs.

**Acceptance Scenarios**:

1. **Given** a long line feature with a configured repeat distance, **When** label planning runs, **Then** repeated labels are placed deterministically or rejected with reason codes.
2. **Given** a curved road or river feature, **When** curved text is enabled, **Then** accepted glyph placement follows the path where supported or returns an explicit `experimental` or `unsupported` diagnostic.
3. **Given** labels for capitals, cities, rivers, peaks, roads, and annotations compete for space, **When** priority presets are applied, **Then** accepted and rejected labels follow deterministic priority rules.
4. **Given** complex-script shaping is not prioritized for the release, **When** labels require that shaping path, **Then** validation documents the deferral without blocking other supported advanced label placement.

---

### User Story 4 - Diagnose Large-Scene Resource Risk (Priority: P2)

A scene reviewer or internal developer validates a large offline map scene and receives memory budget estimates, cache/LOD statistics where available, instancing support status, and bottleneck layer diagnostics before committing to expensive output.

**Why this priority**: Large-scene maturity is product polish for offline map rendering. Users need actionable risk information without converting forge3d into a live globe streaming engine.

**Independent Test**: Can be tested with large terrain, point cloud, building, and tile fixtures that expose known counts or metadata, then checking validation reports for memory estimates, cache/LOD stats, instancing status, and bottleneck layer summaries.

**Acceptance Scenarios**:

1. **Given** a large scene has enough metadata to estimate resource use, **When** validation runs, **Then** the report includes memory budget estimates and identifies whether the scene is within configured limits.
2. **Given** terrain, point cloud, building, or tile layers expose cache or LOD statistics, **When** validation or render diagnostics run, **Then** those statistics appear in deterministic layer summaries.
3. **Given** a repeated building or object workflow can use instancing through map-scene layers, **When** validation runs, **Then** the report identifies whether instancing is used, unavailable, or unsupported.
4. **Given** a render is dominated by a specific layer type, **When** diagnostics are reviewed, **Then** bottleneck layer types are identified where data is available.

### Edge Cases

- A scene requests normal or mask virtual texture families while the runtime only supports albedo: validation emits `vt_unsupported_family` before render.
- A scene mixes supported albedo VT data with unsupported non-albedo families: validation reports only the unsupported families while preserving the supported albedo path.
- Virtual texture family names are duplicated or inconsistently ordered: validation normalizes family order for deterministic diagnostics.
- A building material references a missing texture path: validation reports the missing path and affected layer or object before render.
- Building geometry lacks UVs for a requested texture: validation reports UV absence and explicit material fallback or unsupported status.
- A textured building fixture renders with scalar material fallback: diagnostics and docs make the fallback explicit.
- Advanced line labels produce overlapping repeats on short segments: rejected labels include deterministic reason codes.
- Curved text is requested on geometry that cannot support it: validation reports `experimental_feature` or `unsupported` status instead of claiming render readiness.
- Road/river placement rules conflict with user-provided priorities: deterministic priority and rule ordering decide the accepted/rejected label set.
- Large-scene metadata is incomplete: validation states which estimates or stats are unavailable rather than inventing precision.
- Resource estimates exceed a configured budget: validation reports `estimated_gpu_memory` with severity appropriate to render policy.
- Cache/LOD stats are available for some layer types but not others: diagnostics distinguish available, unavailable, and unsupported stats per layer.

### Support-Level Classification *(mandatory)*

Use the PRD Appendix B terms exactly: `supported`, `underdeveloped`, `missing`, `Pro-gated`, `placeholder/fallback`, `experimental`, `unsupported`, and `non-goal`.

- **Supported in this feature**: user-visible P2 paths that render real VT normal/mask data, render albedo-textured building fixtures through `MapScene`, compile advanced static labels deterministically, or report large-scene memory/cache/LOD diagnostics from available scene data. Support applies only to the capability that actually works; unavailable underlying cache, LOD, instancing, material, or VT paths remain `underdeveloped`, `placeholder/fallback`, `experimental`, or `unsupported` as applicable.
- **Underdeveloped**: existing VT family declarations, building texture infrastructure, line/curved label substrate, instancing, cache, LOD, and large-scene stats where substrate exists but public `MapScene` integration or end-to-end diagnostics are incomplete.
- **Missing**: VT normal/mask runtime rendering, textured building material workflow, advanced static label placement, or large-scene diagnostics when no public/product path or diagnostic exists yet.
- **Pro-gated**: building or renderer paths that require unavailable native/Pro capabilities before textured building materials or large-scene stats can be rendered.
- **Placeholder/fallback**: scalar material fallback, zero geometry, skipped VT families, omitted cache stats, or non-renderable advanced labels when they would otherwise be mistaken for successful output.
- **Experimental**: curved text, complex-script shaping, advanced line placement, or large-scene diagnostic paths that exist but are incomplete or not production-ready.
- **Unsupported**: requested VT families, texture formats, missing UV workflows, advanced label modes, instancing paths, or cache/LOD stats the product cannot render or validate in the active configuration.
- **Non-goal**: web-first globe engine parity, hosted tile-provider ecosystem, Blender/Unreal-style general tooling, live global streaming, general DCC editing, game/editor tooling, animation/cinematic production, and non-map rendering expansion.

### Diagnostics & Failure Behavior *(mandatory)*

- **Diagnostic objects**: P2 diagnostics must include code, severity, message, remediation, support level where applicable, affected layer ID where known, affected object ID where known, and deterministic serialization data.
- **Required diagnostic coverage**: `vt_unsupported_family` for unsupported non-albedo VT requests, `estimated_gpu_memory` for large-scene memory risk, `placeholder_fallback` for skipped families or material fallbacks that would otherwise look successful, `pro_gated_path` for unavailable native/pro building paths, `experimental_feature` for incomplete advanced labels, and feature-local structured diagnostics for `missing_texture_path`, `missing_uvs`, `unsupported_texture_format`, `unavailable_cache_lod_stats`, and `unsupported_instancing_path`.
- **Pre-render validation**: VT family support, missing texture paths, UV absence, material fallback, advanced label mode support, memory budget estimates, cache/LOD stat availability, instancing support, and bottleneck layer summaries must be detectable before render where input metadata allows.
- **No-op prevention**: The system must not report textured PBR, VT normal/mask, advanced label, instancing, cache/LOD, or large-scene support when the requested path was skipped, dropped, replaced by scalar fallback, or unavailable.
- **Determinism**: Diagnostics, advanced label accepted/rejected sets, resource summaries, cache/LOD summaries, bottleneck layer ordering, and support-status reports must be stable for fixed inputs, seed, camera, output size, terrain transform, assets, and configured budgets.
- **Failure behavior**: Error and fatal diagnostics block successful render completion. Warning diagnostics block when fail-on-warning behavior is selected. Unsupported P2 paths must fail with typed diagnostics or documented deferral rather than silent fallback.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001 [PRD: P2-R1-AC1]**: The system MUST either render requested `normal` and `mask` virtual texture families in the native runtime or flag each unsupported family at validation time.
- **FR-002 [PRD: P2-R1-AC2]**: The system MUST NOT silently skip requested non-albedo virtual texture families without structured diagnostics.
- **FR-003 [PRD: P2-R1-AC3]**: Validation and verification MUST cover both albedo-only and albedo plus normal/mask virtual texture configurations.
- **FR-004 [PRD: P2-R1-AC4]**: Documentation MUST state the exact runtime support status for albedo, normal, and mask virtual texture families.
- **FR-005 [PRD: P2-R2-AC1]**: Building layers MUST support albedo texture materials at minimum or diagnose why the requested albedo texture path is unavailable.
- **FR-006 [PRD: P2-R2-AC2]**: Validation MUST report UV presence or absence for textured building material workflows.
- **FR-007 [PRD: P2-R2-AC3]**: Validation MUST report missing or unreadable building texture paths before render.
- **FR-008 [PRD: P2-R2-AC4]**: Material fallback from textured PBR intent to scalar or placeholder material MUST be explicit in diagnostics and documentation.
- **FR-009 [PRD: P2-R2-AC5]**: Verification MUST include a textured building fixture rendered or diagnosed through `MapScene`.
- **FR-010 [PRD: P2-R3-AC1]**: Advanced label placement MUST support repeated labels along long lines with configurable repeat distance or diagnose the mode as unsupported.
- **FR-011 [PRD: P2-R3-AC2]**: Advanced label placement MUST support curved text along paths where renderable, or return an explicit `experimental` or `unsupported` diagnostic.
- **FR-012 [PRD: P2-R3-AC3]**: Advanced label placement MUST include road and river placement rules or diagnose those presets as unsupported.
- **FR-013 [PRD: P2-R3-AC4]**: Landmark and callout leader-line placement MUST be supported or diagnosed with explicit support status.
- **FR-014 [PRD: P2-R3-AC5]**: Multi-class priority presets MUST cover capitals, cities, rivers, peaks, roads, and annotations with deterministic conflict resolution.
- **FR-015 [PRD: P2-R3-AC6]**: A HarfBuzz-compatible complex-script shaping path MUST be included only if prioritized; otherwise the feature MUST document non-blocking deferral and emit honest diagnostics where needed.
- **FR-016 [PRD: P2-R4-AC1]**: Large-scene validation MUST include memory budget estimates where input metadata supports estimation.
- **FR-017 [PRD: P2-R4-AC2]**: Terrain, point cloud, building, and tile layers MUST expose cache or LOD statistics where available and distinguish unavailable stats from unsupported stats.
- **FR-018 [PRD: P2-R4-AC3]**: Instancing workflows MUST be surfaced through map-scene layer APIs where relevant or diagnosed as unavailable or unsupported.
- **FR-019 [PRD: P2-R4-AC4]**: Render diagnostics MUST identify bottleneck layer types where timing, memory, count, or cache data is available.
- **FR-020 [PRD: P2-R4-AC5]**: Documentation MUST differentiate offline large-scene rendering maturity from live globe streaming or hosted tile-provider parity.
- **FR-021 [PRD: Milestone 5]**: The feature MUST ensure each major remaining gap in scope is either implemented end to end or explicitly diagnosed before render.
- **FR-022 [PRD: Appendix B, Constitution VIII]**: Public docs, examples, support matrices, and diagnostics MUST use only PRD Appendix B support-level terms and must not overclaim P2 paths.
- **FR-023 [PRD: Constitution III]**: P2 APIs and workflows MUST either produce real render output, real placement results, real diagnostics, or real documentation of unsupported status; they MUST NOT report no-op success.
- **FR-024 [PRD: Constitution IV]**: P2 validation reports, diagnostics, advanced label plans, resource summaries, and support matrices MUST use deterministic ordering for fixed inputs.
- **FR-025 [PRD: Explicit non-goals]**: The feature MUST preserve offline 3D map rendering scope and exclude web-first globe, hosted tile ecosystem, general DCC, game/editor, and non-map rendering expansion.
- **FR-026 [PRD: Section 13, Section 18]**: Feature-local diagnostic codes for missing texture paths, missing UVs, unsupported texture formats, unavailable cache/LOD stats, and unsupported instancing paths MUST be structured, serializable, documented, and covered by negative tests.
- **FR-027 [PRD: Section 8.2, Constitution IV]**: Deterministic P2 validation and render fixtures MUST use exact comparison for stable serialized outputs and deterministic render fixtures unless a numeric tolerance is recorded in the plan before verification.
- **FR-028 [PRD: Section 21, Section 17]**: Deferred or diagnosed P2 paths MUST NOT block P0 MVP readiness unless a documented human product decision explicitly expands MVP scope.

### Key Entities *(include if feature involves data)*

- **Virtual Texture Family Request**: User intent for albedo, normal, and mask virtual texture families, including requested family names, source identity, support status, and diagnostics.
- **VT Family Support Report**: Validation summary that states whether each requested VT family is `supported`, `unsupported`, `underdeveloped`, or `placeholder/fallback`.
- **Textured Building Material**: Building material intent including albedo texture reference, UV availability, scalar fallback, support status, and affected building layer or object identity where known.
- **Building Texture Diagnostic**: Structured finding for missing texture paths, missing UVs, unavailable material support, Pro-gated paths, or explicit fallback.
- **Advanced Label Rule Set**: Static label placement rules for repeat distance, curved text, road/river presets, leader lines, priority presets, and optional shaping policy.
- **Advanced Label Plan Result**: Deterministic accepted/rejected label output for advanced placement modes, including rejection reasons and support diagnostics.
- **Large-Scene Resource Summary**: Pre-render summary of memory estimates, cache/LOD stats where available, instancing status, and bottleneck layer types.
- **Public API contract**: This feature extends `MapScene`, `SceneRecipe`, `LabelPlan`, `ValidationReport`, `Diagnostic`, and `Bundle` behavior for P2 workflows without changing MVP scope or requiring raw IPC.

## PRD Traceability & Evidence *(mandatory)*

| PRD AC ID | Requirement in this spec | Evidence required | Verification target |
|---|---|---|---|
| P2-R1-AC1 | FR-001 | Runtime evidence for rendered normal/mask families or validation evidence for `vt_unsupported_family` | VT family validation/render tests |
| P2-R1-AC2 | FR-002, FR-023 | Negative tests proving requested non-albedo families are not silently skipped | VT no-silent-skip tests |
| P2-R1-AC3 | FR-003 | Fixtures for albedo-only and albedo plus normal/mask configurations | VT configuration tests |
| P2-R1-AC4 | FR-004, FR-022 | Virtual Texturing Support Matrix with exact runtime support status | VT docs audit |
| P2-R2-AC1 | FR-005, FR-009 | Textured building fixture render or typed diagnostic evidence | Textured building MapScene test |
| P2-R2-AC2 | FR-006 | Validation report showing UV presence and absence | Building UV diagnostics tests |
| P2-R2-AC3 | FR-007 | Pre-render diagnostics for missing texture paths | Missing texture tests |
| P2-R2-AC4 | FR-008, FR-022 | Material fallback diagnostics and documentation | Material fallback docs and tests |
| P2-R2-AC5 | FR-009 | End-to-end textured building fixture through `MapScene` | Textured building fixture test |
| P2-R3-AC1 | FR-010 | Tests for repeated labels along long lines with repeat distance | Repeated line label tests |
| P2-R3-AC2 | FR-011 | Curved text render/layout tests or typed experimental/unsupported diagnostics | Curved text tests |
| P2-R3-AC3 | FR-012 | Road and river placement rule tests | Road/river rule tests |
| P2-R3-AC4 | FR-013 | Landmark/callout leader-line placement tests | Leader-line placement tests |
| P2-R3-AC5 | FR-014 | Priority preset tests for capitals, cities, rivers, peaks, roads, and annotations | Priority preset tests |
| P2-R3-AC6 | FR-015 | Product decision and design/tests if prioritized, otherwise docs and diagnostics for deferral | Shaping path decision review |
| P2-R4-AC1 | FR-016 | Large-scene validation tests for memory budget estimates | Memory budget validation tests |
| P2-R4-AC2 | FR-017 | Diagnostics tests for cache/LOD stats by layer type where available | Cache/LOD diagnostics tests |
| P2-R4-AC3 | FR-018 | API contract/tests for map-scene instancing integration or diagnostics | Instancing workflow tests |
| P2-R4-AC4 | FR-019 | Render diagnostics identifying bottleneck layer types | Bottleneck diagnostics tests |
| P2-R4-AC5 | FR-020, FR-025 | Docs differentiating offline large-scene rendering from live globe streaming | Large-scene docs audit |
| Milestone 5 | FR-021 | Exit review showing all P2 gaps implemented or diagnosed before render | Milestone 5 exit review |
| Constitution III | FR-002, FR-008, FR-023 | Negative tests for no-op success and silent fallback | No-op success tests |
| Constitution IV | FR-014, FR-024, FR-027 | Deterministic ordering and exact fixture comparisons for diagnostics, label plans, resource summaries, and deterministic render fixtures unless a recorded tolerance exists | Determinism tests |
| Section 13, Section 18 | FR-026 | Feature-local diagnostic codes are structured, serializable, documented, and covered by negative tests | Diagnostic code tests |
| Section 17, Section 21 | FR-028 | P2 deferrals remain non-MVP-blocking unless a documented product decision expands scope | MVP scope review |

## Explicit Non-Goals *(mandatory)*

- Web-first globe engine parity.
- Hosted tile-provider ecosystem.
- Blender/Unreal-style general tooling.
- Non-map rendering expansion.
- Full streamed MVT renderer or full Mapbox Style Specification parity.
- Full 3D Tiles or Cesium-grade global runtime parity.
- Browser delivery, live global streaming, general DCC editing, gameplay, animation, simulation, or cinematic production features.
- Treating Pro-gated, placeholder/fallback, experimental, underdeveloped, or unsupported P2 paths as `supported`.
- Making this P2 feature a blocker for P0 MVP completion unless a documented human decision expands release scope.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of P2-R1 through P2-R4 acceptance criteria have explicit requirements, evidence expectations, and verification targets in this specification.
- **SC-002**: 100% of requested non-albedo VT family scenarios either render real normal/mask behavior or emit `vt_unsupported_family` before render.
- **SC-003**: Albedo-only and albedo plus normal/mask VT configurations are both represented in validation or render verification.
- **SC-004**: Textured building fixtures either render with albedo texture through `MapScene` or produce typed diagnostics for missing UVs, missing texture paths, unavailable support, or explicit fallback.
- **SC-005**: 100% of missing building texture path fixtures report the missing path before render.
- **SC-006**: Advanced label fixtures for repeated line labels, curved text, road/river rules, leader lines, and priority presets produce deterministic accepted/rejected results or typed diagnostics.
- **SC-007**: Large-scene validation reports memory budget estimates whenever enough metadata exists and distinguishes unavailable estimates from unsupported estimates.
- **SC-008**: Cache/LOD stats are surfaced for terrain, point cloud, building, and tile layers where available, with unavailable stats explicitly identified.
- **SC-009**: Render diagnostics identify bottleneck layer types where timing, count, cache, or memory data is available.
- **SC-010**: Required P2 docs and support matrices contain zero overclaims against the PRD Appendix B taxonomy.
- **SC-011**: Repeated validation from fixed inputs produces stable diagnostic ordering, label result ordering, resource summary ordering, and support-status ordering.
- **SC-012**: Automated checks confirm no unresolved template placeholders or clarification markers remain in this specification.
- **SC-013**: Negative diagnostics tests cover `missing_texture_path`, `missing_uvs`, `unsupported_texture_format`, `unavailable_cache_lod_stats`, and `unsupported_instancing_path`.
- **SC-014**: P2 support matrices and release notes distinguish implemented P2 support from diagnosed deferral and state that deferred P2 gaps do not block P0 MVP readiness.
- **SC-015**: Before completion, `docs/superpowers/state/current-context-pack.md` and `docs/superpowers/state/implementation-ledger.md` record feature `006` continuity evidence; `docs/superpowers/state/requirements-verification-matrix.md` is updated only when requirement status changes.

## Assumptions

- Features `001` through `004` provide the diagnostic, label-plan, and `MapScene` foundations needed for this P2 feature; this feature does not redefine the MVP.
- Feature `005` building, 3D Tiles, and bundle integrations may be prerequisites for some textured material and large-scene workflows; unavailable dependencies must be diagnosed rather than silently skipped.
- The acceptable outcome for each P2 gap is either real end-to-end support or explicit pre-render diagnostics with exact support status.
- Albedo texture support is the minimum textured building material target; additional PBR texture channels may be planned later if they remain within map-rendering scope.
- Complex-script shaping is optional unless explicitly prioritized by a later product decision; non-prioritized shaping remains a documented deferral with diagnostics where applicable.
- Large-scene diagnostics rely on available metadata; missing metadata must be reported as unavailable rather than estimated imprecisely.
- This feature preserves the PRD's offline-first product identity and does not introduce hosted/live globe delivery requirements.
- Existing dirty workspace changes are user-owned and are not part of this feature unless explicitly touched by this specification workflow.

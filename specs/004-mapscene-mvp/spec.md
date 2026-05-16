# Feature Specification: MapScene MVP

**Feature Branch**: `004-mapscene-mvp`  
**Created**: 2026-05-14  
**Status**: Draft  
**Input**: User description: "Create feature 004-mapscene-mvp for forge3d offline 3D map rendering."  
**Source of Truth**: `docs/superpowers/plans/prd.md` and `.specify/memory/constitution.md`  
**PRD Coverage**: P0-R4, Section 15, Section 21, Section 16 docs required for MVP workflows, Milestone 3, and MVP integration dependencies from P0-R5/P0-R6/P0-R3  
**MVP Blocking**: Yes - typed MapScene validation, PNG render, review bundle save, canonical examples, and Section 21 MVP items are release-blocking unless a documented human decision reclassifies them

## Clarifications

### Session 2026-05-15

- Q: What is the default render behavior when validation returns warnings? -> A: Continue on warnings by default; opt-in fail-on-warning blocks warnings, and errors or fatal diagnostics always block successful render completion.
- Q: What CRS transform behavior is required for MVP? -> A: No implicit CRS transforms; layers must match scene or terrain CRS, provide an explicit compatible transform or policy, or validation emits `crs_mismatch`.
- Q: Should MVP bundles persist compiled `LabelPlan` output by default? -> A: Store the compiled label plan plus deterministic source references for supported label-bearing scenes.
- Q: What reproducibility evidence is required for PNG tests? -> A: Use a fixed reproducibility profile; non-image artifacts must be exact-stable, and PNG fixtures must use exact pixel comparison unless a documented numeric tolerance is recorded before verification.
- Q: How should unavailable P1/P2 layer capabilities appear in the P0 typed recipe? -> A: Represent user intent only and validate unavailable capabilities as `Pro-gated`, `placeholder/fallback`, `experimental`, or `unsupported` without claiming render success.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Define And Validate One Typed Map Scene (Priority: P1)

A geospatial Python user defines an offline 3D map scene with one typed recipe containing terrain, camera, lighting, layers, map furniture, and output settings, then validates it before rendering.

**Why this priority**: This is the primary P0-R4 outcome. The product gap is a reliable typed scene contract that replaces stitched low-level calls and raw IPC for MVP workflows.

**Independent Test**: Can be fully tested by constructing a typed scene recipe with each MVP recipe component, calling `MapScene.validate()`, and inspecting a structured validation report before any render output is produced.

**Acceptance Scenarios**:

1. **Given** a typed recipe containing terrain, raster overlay, vector overlay, label layer, point cloud layer, camera, lighting, output spec, and map furniture, **When** the user constructs a `MapScene`, **Then** each component is represented in the public recipe without requiring raw IPC.
2. **Given** a scene with detectable issues, **When** the user calls `MapScene.validate()`, **Then** validation returns a structured report with diagnostics, layer summaries, support status, render-blocking status, and memory estimate where enough information exists before rendering.
3. **Given** a recipe includes an unsupported or Pro-gated path, **When** validation runs, **Then** the report includes typed diagnostics rather than silently omitting the feature or substituting placeholder output.

---

### User Story 2 - Render A Reproducible PNG (Priority: P1)

A cartographic production user validates a supported typed scene, compiles deterministic labels, and renders a PNG output that can be reproduced from the same recipe and inputs.

**Why this priority**: PRD Section 21 requires PNG render output, deterministic labels, precise diagnostics, and reproducible offline rendering as the MVP user-visible outcome.

**Independent Test**: Can be tested with a fixed terrain+raster or terrain+vector+labels fixture that validates successfully, compiles a deterministic `LabelPlan`, renders PNG output, and compares pixels exactly unless planning has recorded a numeric tolerance before verification.

**Acceptance Scenarios**:

1. **Given** a valid MVP-supported scene, **When** the user calls `MapScene.render("map.png")`, **Then** a PNG output is written and linked to the validation report and deterministic label plan used for the render.
2. **Given** labels are present, **When** render preparation runs, **Then** deterministic point and polygon label placement is compiled before final render and label diagnostics remain inspectable.
3. **Given** validation reports diagnostics, **When** the user requests render, **Then** warnings continue by default, opt-in fail-on-warning blocks warning diagnostics, and error or fatal diagnostics always prevent successful PNG completion.

---

### User Story 3 - Save A Reproducible Review Bundle (Priority: P1)

A scene reviewer saves a review bundle for a supported MVP scene and receives enough reproducible scene intent to review inputs, diagnostics, label plans, camera, lighting, output, and supported layer metadata later.

**Why this priority**: The MVP target is not just an image; it is a reproducible map-production workflow with a review bundle for supported layer types.

**Independent Test**: Can be tested by saving the same validated scene twice and comparing bundle manifest contents, recipe data, diagnostics, label plan references or payloads, and supported layer metadata in deterministic order.

**Acceptance Scenarios**:

1. **Given** a supported scene has been validated and rendered or prepared for render, **When** the user calls `MapScene.save_bundle("map.forge3d")`, **Then** a review bundle is written with deterministic recipe, diagnostics, compiled label plan plus deterministic label source references, camera, lighting, output, and supported layer metadata.
2. **Given** unsupported, Pro-gated, placeholder/fallback, or experimental features are present, **When** a bundle is saved, **Then** their diagnostics and support classifications are preserved rather than represented as successful rendered assets.
3. **Given** external assets are referenced by the recipe, **When** the bundle is saved, **Then** supported asset metadata is recorded deterministically and missing or unsupported asset handling is reported through typed diagnostics.

---

### User Story 4 - Use Canonical MVP Examples (Priority: P1)

A Python user learns the MVP workflow from canonical examples for terrain plus raster, terrain plus vector plus labels, and terrain plus buildings plus labels or an honest building diagnostic path when building availability is not present.

**Why this priority**: P0-R4 requires at least three canonical examples and the user's request requires the building example to be honest when building support is Pro-gated or unavailable.

**Independent Test**: Can be tested by running or smoke-validating the three examples and confirming they use `MapScene` typed recipes, produce expected validation/render/bundle behavior, and do not require raw IPC.

**Acceptance Scenarios**:

1. **Given** the terrain+raster example, **When** a user runs or validates it, **Then** it demonstrates typed terrain, raster overlay, camera, lighting, output, validation, PNG render, and bundle save.
2. **Given** the terrain+vector+labels example, **When** a user runs or validates it, **Then** it demonstrates vector overlay styling diagnostics, deterministic point and polygon labels, validation, PNG render, and bundle save.
3. **Given** building ingestion is available in the current packaging, **When** the terrain+buildings+labels example runs, **Then** it renders or prepares supported building content through `MapScene`; otherwise the example demonstrates typed `pro_gated_path`, `placeholder_fallback`, or unsupported diagnostics without claiming successful building rendering.

---

### User Story 5 - Read MVP Workflow Documentation (Priority: P2)

A user or internal developer reads documentation for the MVP workflow and sees exact support levels, diagnostics, examples, and non-goals without overclaims about web delivery, full 3D Tiles, textured PBR buildings, VT normal/mask runtime, or full Mapbox style support.

**Why this priority**: Section 16 documentation is required for MVP workflows, and support-level honesty is a constitutional gate.

**Independent Test**: Can be tested by auditing the required docs against the PRD taxonomy and verifying each MVP workflow doc links to diagnostics, style support, label plan behavior, bundle behavior, and support matrices.

**Acceptance Scenarios**:

1. **Given** the Offline 3D Map Rendering Quickstart, **When** a user follows it, **Then** it shows a typed `MapScene` workflow from recipe to validate, render PNG, and save bundle.
2. **Given** support matrices and diagnostics docs, **When** a reviewer audits them, **Then** style, building, 3D Tiles, virtual texturing, diagnostics, and competitive positioning use only PRD-approved support classifications.
3. **Given** unsupported or deferred features appear in docs or examples, **When** the text is reviewed, **Then** it classifies them as `underdeveloped`, `Pro-gated`, `placeholder/fallback`, `experimental`, `unsupported`, or `non-goal` as applicable.

### Edge Cases

- A layer CRS differs from the terrain or scene CRS and no explicit compatible transform or policy is provided: validation returns `crs_mismatch` before render.
- CRS metadata is missing from a layer: validation reports an explicit diagnostic or unknown CRS status rather than assuming compatibility or applying an implicit transform.
- GPU memory can be estimated from dimensions, point counts, raster sizes, or geometry counts: validation includes `estimated_gpu_memory`; if information is insufficient, the report states that the estimate is unavailable without inventing precision.
- Labels contain missing glyphs: validation and label-plan compilation surface `missing_glyphs` before final render.
- Style uses unsupported layer types or paint/layout fields: validation reports `unsupported_style_layer_type` or `unsupported_style_field`.
- A requested building layer depends on Pro/native availability or falls back to zero geometry: validation reports `pro_gated_path` or `placeholder_fallback`, and examples remain honest.
- A point cloud source has insufficient metadata for memory estimate or render preparation: validation reports the known limits and does not silently drop the layer.
- A map furniture element creates keepout regions for labels: the keepouts are included in label planning and bundle metadata.
- Render is requested before explicit validation: render performs or requires validation and exposes the report before treating output as successful.
- Bundle save is requested for a scene with blocking diagnostics: the bundle records diagnostics and support status and does not imply the scene was successfully rendered.
- Repeated validation, label compilation, bundle save, and example output from fixed inputs must use deterministic ordering for reports and serialized review data.
- A label-bearing supported scene is bundled: the bundle records the compiled label plan used for rendering and deterministic source references sufficient to detect stale or changed source inputs.
- A render fixture is used for reproducibility evidence: the fixture records a fixed reproducibility profile covering seed, camera, output size, terrain transform, style data, asset hashes or stable fixture IDs, renderer/backend identity where known, and the exact pixel comparison or documented numeric tolerance.

### Support-Level Classification *(mandatory)*

Use the PRD Appendix B terms exactly: `supported`, `underdeveloped`, `missing`, `Pro-gated`, `placeholder/fallback`, `experimental`, `unsupported`, and `non-goal`.

- **Supported in this feature**: typed `MapScene` and `SceneRecipe` user contract, `TerrainSource`, `RasterOverlay`, `VectorOverlay`, `LabelLayer`, `PointCloudLayer` where the current public API can load and render the source honestly, `MapFurnitureLayer`, building layer intent with honest diagnostics, camera, lighting, output spec, `MapScene.validate()`, PNG render for MVP-supported scenes, `MapScene.save_bundle()` for supported layer types, deterministic label-plan integration for point and polygon labels, and canonical examples.
- **Underdeveloped**: existing building, 3D Tiles, style, label, point-cloud, bundle, and render substrate where public integration, support matrices, or full production workflow remains incomplete before this feature is implemented.
- **Missing**: the typed `MapScene` / `SceneRecipe` product contract and end-to-end MVP workflow before this feature is implemented.
- **Pro-gated**: building or other paths requiring native/Pro symbols or packaging not available in the public workflow.
- **Placeholder/fallback**: building, layer, render, or bundle paths that would return zero geometry, non-renderable data, placeholder output, or incomplete saved state while appearing successful.
- **Experimental**: line or curved labels, 3D Tiles public scene workflow, render backends, or asset integrations that exist but are incomplete, unverified, or not production-ready.
- **Unsupported**: full streamed MVT rendering, unsupported style layer types or fields, unavailable CRS transforms, VT normal/mask runtime, textured PBR building workflows, unsupported point cloud formats, unsupported building paths, and unsupported render or bundle paths.
- **Non-goal**: full streamed MVT renderer, full 3D Tiles production workflow, textured PBR buildings, VT normal/mask runtime, advanced curved label placement, full non-Latin shaping, browser delivery, general DCC editing, hosted tile services, game/editor tooling, and non-map rendering features.

### Diagnostics & Failure Behavior *(mandatory)*

- **Diagnostic objects**: `MapScene.validate()` diagnostics must include code, severity, message, remediation, support level where applicable, affected layer ID where known, affected object ID where known, and deterministic serialization data.
- **ValidationReport**: Validation must return structured overall status, diagnostics, layer summaries, support summaries, estimated GPU memory where enough information exists, and render blocking status before rendering.
- **Required MVP diagnostic codes**: The MVP scene workflow must surface `crs_mismatch`, `missing_glyphs`, `unsupported_style_field`, `unsupported_style_layer_type`, `pro_gated_path`, `placeholder_fallback`, `experimental_feature`, `vt_unsupported_family`, `python_public_3dtiles_incomplete`, `estimated_gpu_memory`, and `label_rejection_summary` where applicable.
- **Pre-render validation**: CRS mismatches, missing glyphs, unsupported style fields, unsupported style layer types, unsupported or Pro-gated layer paths, placeholder/fallback behavior, estimated GPU memory risk, incomplete public 3D Tiles paths, VT unsupported families, and label rejection summaries must be available before render where enough input information exists.
- **No-op prevention**: `validate`, `render`, and `save_bundle` must not report success when no real report, PNG output, bundle artifact, renderable layer data, or typed diagnostic was produced.
- **Determinism**: Validation report ordering, label-plan compilation, PNG fixture pixel comparisons, bundle manifests, layer summaries, diagnostics, and example outputs must be reproducible for fixed inputs, seed, camera, output size, terrain transform, style data, and assets. PNG fixture checks must be exact unless planning records a numeric tolerance before verification.
- **Failure behavior**: Warning diagnostics continue by default. A user-selected fail-on-warning policy blocks warning diagnostics. Error and fatal diagnostics always block successful render completion. Unsupported features fail with typed diagnostics rather than silent fallback.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001 [PRD: P0-R4-AC4, Section 21]**: The system MUST provide a typed `MapScene` / `SceneRecipe` user contract for defining an offline 3D map scene from one recipe.
- **FR-002 [PRD: P0-R4-AC4, Section 15]**: The typed recipe MUST represent `TerrainSource` with source identity, CRS, terrain metadata, and elevation sampling availability where known.
- **FR-003 [PRD: P0-R4-AC4, Section 15]**: The typed recipe MUST represent `RasterOverlay` with source identity, layer ID, CRS where known, opacity or basic visual intent, and validation metadata.
- **FR-004 [PRD: P0-R4-AC4, Section 15]**: The typed recipe MUST represent `VectorOverlay` with source identity or provided features, layer ID, CRS where known, supported styling intent, and unsupported style diagnostics.
- **FR-005 [PRD: P0-R4-AC4, Section 15, Section 21]**: The typed recipe MUST represent `LabelLayer` and integrate deterministic `LabelPlan` compilation for point and polygon labels.
- **FR-006 [PRD: P0-R4-AC4, Section 15]**: The typed recipe MUST represent `PointCloudLayer` with source identity, CRS where known, point count or metadata where available, and validation support status.
- **FR-007 [PRD: P0-R4-AC4, Section 15]**: The typed recipe MUST represent `BuildingLayer` or equivalent building layer intent so building availability, Pro-gated paths, placeholder/fallback paths, and unsupported paths can be diagnosed honestly.
- **FR-008 [PRD: P0-R4-AC4, Section 15, Section 21]**: The typed recipe MUST represent camera, lighting, output spec, and `MapFurnitureLayer` including title, legend, scale bar, north arrow, and label keepout definitions.
- **FR-009 [PRD: P0-R4-AC1, Section 21]**: `MapScene.validate()` MUST return a structured `ValidationReport` before rendering.
- **FR-010 [PRD: P0-R4-AC5]**: Validation MUST detect CRS mismatches when a layer CRS differs from the scene or terrain CRS and no transform or compatible policy is provided.
- **FR-011 [PRD: P0-R4-AC6]**: Validation MUST include a GPU memory estimate where enough information exists, including known raster dimensions, output size, point counts, terrain metadata, or geometry counts.
- **FR-012 [PRD: P0-R4-AC7]**: Unsupported, experimental, Pro-gated, and placeholder/fallback features MUST fail with typed diagnostics rather than silent fallback.
- **FR-013 [PRD: Section 21]**: Validation MUST include missing-glyph diagnostics, style support diagnostics, CRS mismatch diagnostics, and label rejection summaries for MVP workflows where applicable.
- **FR-014 [PRD: P0-R4-AC2, Section 21]**: `MapScene.render()` MUST support PNG output for MVP-supported scenes.
- **FR-015 [PRD: P0-R4-AC2, Constitution IV]**: PNG render workflows MUST use fixed recipe inputs, deterministic label plans, camera, lighting, output size, and exact pixel comparison unless planning records a numeric tolerance before verification.
- **FR-016 [PRD: P0-R4-AC3, Section 21]**: `MapScene.save_bundle()` MUST write a reproducible review bundle for supported MVP layer types.
- **FR-017 [PRD: P0-R4-AC3]**: Bundles MUST include enough scene intent for review, including recipe data, source metadata, camera, lighting, output spec, map furniture, diagnostics, compiled label plan data plus deterministic label source references for supported label-bearing scenes, and supported layer metadata.
- **FR-018 [PRD: P0-R4-AC8]**: The feature MUST include a canonical terrain+raster `MapScene` example that validates, renders PNG output, and saves a review bundle.
- **FR-019 [PRD: P0-R4-AC8, Section 21]**: The feature MUST include a canonical terrain+vector+labels `MapScene` example that demonstrates deterministic point and polygon label planning, style diagnostics, validation, PNG output, and bundle save.
- **FR-020 [PRD: P0-R4-AC8]**: The feature MUST include a canonical terrain+buildings+labels example; if building rendering is unavailable, Pro-gated, placeholder/fallback, or unsupported in the current packaging, the example MUST demonstrate typed diagnostics honestly instead of claiming successful building rendering.
- **FR-021 [PRD: Section 16]**: MVP workflow documentation MUST include or update Offline 3D Map Rendering Quickstart, LabelPlan Guide references, Style Support Matrix, Building Layer Support Matrix, 3D Tiles Support Matrix, Virtual Texturing Support Matrix, Diagnostics Reference, and Competitive Positioning Note.
- **FR-022 [PRD: Section 16, Appendix B]**: Documentation and examples MUST avoid overclaiming full streamed MVT rendering, full 3D Tiles production workflow, textured PBR buildings, VT normal/mask runtime, advanced curved labels, full non-Latin shaping, browser delivery, or general DCC editing.
- **FR-023 [PRD: Section 21]**: The MVP workflow MUST allow a Python user to define one typed scene, validate it, receive precise diagnostics, compile deterministic labels, render PNG output, and save a review bundle without raw IPC for canonical examples.
- **FR-024 [PRD: Constitution III]**: `MapScene.validate()`, `MapScene.render()`, and `MapScene.save_bundle()` MUST either produce real structured reports, real PNG output, real bundle artifacts, or typed diagnostics/errors; they MUST NOT report no-op success.
- **FR-025 [PRD: Constitution IV]**: Recipe serialization, validation reports, diagnostics, label plan integration, bundle manifests, layer summaries, and example outputs MUST use deterministic ordering for fixed inputs.
- **FR-026 [PRD: Milestone 3]**: The feature MUST demonstrate that common offline 3D maps can be rendered with one typed recipe and that validation catches CRS mismatch, missing glyphs, unsupported style fields, and memory warnings.
- **FR-027 [PRD: P0-R5-AC5, P0-R4-AC7]**: The default diagnostics policy MUST continue on warning diagnostics, MUST allow an explicit fail-on-warning policy, and MUST block successful render completion for error or fatal diagnostics.
- **FR-028 [PRD: P0-R4-AC5, Constitution IV]**: MVP validation MUST NOT apply implicit CRS transforms; compatibility requires matching CRS metadata or an explicit compatible transform or policy.
- **FR-029 [PRD: P0-R4-AC2, Constitution IV]**: Render reproducibility tests MUST define a fixed reproducibility profile and compare PNG fixtures exactly unless a documented numeric pixel tolerance is recorded before verification.
- **FR-030 [PRD: P0-R4-AC7, Appendix B]**: P1/P2 or unavailable layer capabilities included in the typed recipe MUST represent user intent only and MUST validate as `Pro-gated`, `placeholder/fallback`, `experimental`, or `unsupported` when they cannot honestly render in the current packaging.

### Key Entities *(include if feature involves data)*

- **MapScene**: User-facing scene object created from a typed recipe. It validates scene intent, prepares deterministic labels, renders supported PNG output, and saves review bundles.
- **SceneRecipe**: Typed declaration of terrain, layers, camera, lighting, output, map furniture, render policy, diagnostics policy, and reproducibility settings.
- **TerrainSource**: Terrain input with source identity, CRS, dimensions or metadata where known, elevation sampling availability, and memory-estimate inputs.
- **RasterOverlay**: Raster imagery or overlay layer with source identity, CRS where known, visual intent, metadata, and diagnostics.
- **VectorOverlay**: Vector feature layer with source identity or provided features, CRS where known, style intent, supported style fields, and unsupported style diagnostics.
- **LabelLayer**: Label source layer that feeds deterministic `LabelPlan` compilation for point and polygon labels and surfaces missing-glyph and rejection diagnostics.
- **PointCloudLayer**: Point cloud source layer with source identity, CRS where known, point count or metadata where available, support status, and memory-estimate inputs.
- **BuildingLayer**: Building source or intent represented in the recipe so supported, `Pro-gated`, `placeholder/fallback`, `experimental`, or `unsupported` paths can be diagnosed.
- **MapFurnitureLayer**: Title, legend, scale bar, north arrow, and keepout definitions that affect map composition and label planning.
- **Camera**: Typed view definition for reproducible scene framing.
- **Lighting**: Typed lighting preset or settings used for reproducible offline render intent.
- **OutputSpec**: Output size, format, path intent, and reproducibility metadata for PNG render workflows.
- **ReproducibilityProfile**: Deterministic render-test context containing seed, camera, output size, terrain transform, style data, relevant asset hashes or stable fixture IDs, renderer/backend identity where known, and exact pixel comparison or documented numeric tolerance.
- **ValidationReport**: Structured pre-render result containing overall status, diagnostics, layer summaries, support summaries, memory estimate where known, and render blocking status.
- **Diagnostic**: Structured finding with code, severity, message, remediation, support level where applicable, affected IDs where known, and deterministic serialization data.
- **ReviewBundle**: Saved scene review artifact containing reproducible recipe intent, diagnostics, label plan data or deterministic label sources, supported layer metadata, and output metadata.
- **Public API contract**: This feature integrates `MapScene`, `SceneRecipe`, `LabelPlan`, `ValidationReport`, `Diagnostic`, and `Bundle` for MVP workflows while preserving honest diagnostics for P1/P2 or unavailable paths.

## PRD Traceability & Evidence *(mandatory)*

| PRD AC ID | Requirement in this spec | Evidence required | Verification target |
|---|---|---|---|
| P0-R4-AC1 | FR-009, FR-013 | API contract and tests for structured `ValidationReport` before render | MapScene validation report tests |
| P0-R4-AC2 | FR-014, FR-015, FR-029 | PNG render integration test with deterministic fixture behavior and fixed reproducibility profile | MapScene PNG render tests |
| P0-R4-AC3 | FR-016, FR-017 | Bundle save tests showing stable serialized scene intent, compiled label plan data, and deterministic source references for supported layer types | Review bundle save tests |
| P0-R4-AC4 | FR-001 through FR-008 | Typed API/model evidence and construction tests for recipe components | SceneRecipe construction tests |
| P0-R4-AC5 | FR-010, FR-028 | Validation fixtures emitting `crs_mismatch` for incompatible CRS without explicit compatible transform or policy | CRS mismatch validation tests |
| P0-R4-AC6 | FR-011 | Validation fixtures for GPU memory estimates where enough metadata exists | GPU memory estimate tests |
| P0-R4-AC7 | FR-012, FR-024, FR-027, FR-030 | Negative tests for unsupported, Pro-gated, experimental, placeholder/fallback, warning-policy, and unavailable P1/P2 paths | Typed diagnostic failure tests |
| P0-R4-AC8 | FR-018, FR-019, FR-020 | Three canonical examples plus smoke tests or documented validation | Canonical example tests |
| Section 15 | FR-002 through FR-008 | Layer recipe coverage for TerrainSource, RasterOverlay, VectorOverlay, LabelLayer, BuildingLayer, PointCloudLayer, MapFurnitureLayer, camera, lighting, and output | SceneRecipe layer contract review |
| Section 16 | FR-021, FR-022 | Required MVP docs and support matrices updated with honest support classifications | MVP docs audit |
| Section 21 | FR-001, FR-005, FR-009, FR-013 through FR-023, FR-026 | End-to-end MVP workflow evidence covering typed recipe, validation, diagnostics, LabelPlan, PNG, bundle, style/glyph/CRS diagnostics | MVP workflow review |
| P0-R5 diagnostic dependencies | FR-012, FR-013, FR-024 | Structured diagnostics for unsupported, glyph, style, CRS, memory, fallback, and label rejection paths | Diagnostics integration tests |
| P0-R6 style dependencies | FR-004, FR-013, FR-019, FR-021 | Style support diagnostics feed vector and label workflows without claiming streamed MVT | Style workflow validation tests |
| Constitution III | FR-024 | Negative tests proving no no-op success for validate, render, and bundle save | No-op success tests |
| Constitution IV | FR-015, FR-025, FR-028, FR-029 | Deterministic ordering and reproducibility evidence for reports, label plans, bundles, examples, CRS policy, and render fixtures | Determinism tests or review |
| Milestone 3 | FR-023, FR-026 | One typed recipe renders common offline 3D maps and validation catches required issues | Milestone exit review |

## Explicit Non-Goals *(mandatory)*

- Full streamed MVT renderer.
- Full 3D Tiles production workflow.
- Textured PBR buildings.
- VT normal/mask runtime.
- Advanced curved label placement.
- Full non-Latin shaping.
- Browser delivery.
- General DCC editing.
- Hosted tile-provider ecosystems, web-first globe runtime, game/editor tooling, animation/cinematic production, or non-map rendering features.
- Treating Pro-gated building ingestion, placeholder/fallback building output, incomplete public 3D Tiles paths, VT normal/mask placeholders, unsupported style fields, or experimental line/curved labels as fully supported.
- Implementing P1 data-driven `LabelLayer.from_geodataframe`, full building and 3D Tiles round-trip production workflows, textured building materials, or P2 advanced label placement unless required only as typed diagnostics for MVP honesty.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of P0-R4 acceptance criteria and PRD Section 21 MVP must-include items have explicit requirements, evidence expectations, and verification targets in this specification.
- **SC-002**: A user can define a supported offline 3D map scene with one typed recipe containing terrain, camera, lighting, output, map furniture, and supported layer types.
- **SC-003**: `MapScene.validate()` returns a structured report before render for 100% of canonical examples and known invalid fixtures.
- **SC-004**: Validation fixtures detect CRS mismatch, missing glyphs, unsupported style fields or layer types, unsupported or Pro-gated features, placeholder/fallback paths, label rejection summaries, and estimated GPU memory where metadata is sufficient.
- **SC-005**: At least one MVP-supported scene renders PNG output through `MapScene.render()` with a fixed reproducibility profile and exact pixel comparison unless a documented numeric tolerance is recorded before verification.
- **SC-006**: `MapScene.save_bundle()` writes deterministic review bundle metadata for supported MVP layer types, stores compiled label plan data plus deterministic source references for supported label-bearing scenes, and preserves diagnostics without information loss.
- **SC-007**: Three canonical examples exist and are runnable or smoke-validated: terrain+raster, terrain+vector+labels, and terrain+buildings+labels or honest building diagnostics where building availability does not permit rendering.
- **SC-008**: Canonical examples require zero raw IPC calls for the MVP workflow.
- **SC-009**: Deterministic point and polygon label plans are compiled and inspectable before final render for label-bearing MVP examples.
- **SC-010**: Required MVP docs exist or are updated and contain zero support overclaims against the PRD Appendix B taxonomy.
- **SC-011**: Repeated validation and bundle save from fixed inputs produce the same diagnostic ordering, layer summary ordering, label-plan references or payload ordering, and serialized bundle manifest ordering.
- **SC-012**: Automated checks confirm no unresolved template placeholders or clarification markers remain in this specification.
- **SC-013**: Validation policy tests prove warnings continue by default, fail-on-warning blocks warnings when selected, and error or fatal diagnostics always block successful render completion.
- **SC-014**: CRS tests prove no implicit CRS transforms are applied; compatible rendering requires matching CRS metadata or an explicit compatible transform or policy.

## Assumptions

- `MapScene` is specified as the public top-level user contract; implementation planning may decide whether it wraps existing `Scene`, offscreen rendering, viewer snapshot substrate, or other internals as long as raw IPC is not required for MVP workflows.
- PNG output is the minimum required render format for MVP; other export formats remain outside this feature unless they are needed for docs or bundle metadata.
- Building layer representation is required in the typed recipe for P0, but successful building rendering may be diagnosed as `Pro-gated`, `placeholder/fallback`, `experimental`, or `unsupported` when public availability does not permit honest rendering.
- CRS transform implementation scope remains conservative for MVP: no implicit CRS transforms are applied, and mismatches without a provided compatible transform or policy must be diagnosed before render.
- GPU memory estimates are required where metadata is available; absent metadata must be reported honestly rather than converted into a false estimate.
- Deterministic label planning depends on feature `003`; this feature integrates point and polygon label plans rather than expanding into advanced curved label placement.
- Style support diagnostics depend on feature `001`; this feature integrates those diagnostics into `MapScene.validate()` and examples.
- Pixel-level render verification defaults to exact pixel comparison for deterministic fixtures. If platform or backend variation makes exact comparison unsuitable, planning must document the numeric tolerance before any verification row can be marked `Verified`.
- Warning diagnostics continue by default so users can inspect non-blocking issues without losing MVP ergonomics; fail-on-warning remains available for stricter CI and production workflows.
- Existing dirty workspace changes are user-owned and are not part of this feature unless explicitly touched by this specification workflow.

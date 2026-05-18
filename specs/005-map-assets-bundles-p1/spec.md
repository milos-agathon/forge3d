# Feature Specification: Map Assets and Bundle Round-Trip P1

**Feature Branch**: `005-map-assets-bundles-p1`  
**Created**: 2026-05-15  
**Status**: Draft  
**Input**: User description: "Create feature 005-map-assets-bundles-p1 for forge3d offline 3D map rendering. Scope: P1-R1 LabelLayer Data Ingestion; P1-R2 Typography and Font Handling Upgrade; P1-R3 Integrated Building Layer Workflow; P1-R4 Public 3D Tiles Scene Integration; P1-R5 Bundle Round-Trip for Map Scenes; Milestone 4 Integrated Geospatial Assets."  
**Source of Truth**: `docs/superpowers/plans/prd.md` and `.specify/memory/constitution.md`  
**PRD Coverage**: P1-R1, P1-R2, P1-R3, P1-R4, P1-R5, Milestone 4, and P1 diagnostic dependencies from P0-R5/P0-R6  
**MVP Blocking**: No - this is P1 follow-on work after the P0 MVP unless a documented human decision expands MVP scope

## Clarifications

### Session 2026-05-15

- Q: What bundle label persistence policy best preserves deterministic review intent? -> A: Store both source labels and compiled `LabelPlan` payloads where available; if either cannot be persisted or replayed, emit structured diagnostics.
- Q: Which diagnostic codes make P1 negative paths testable? -> A: Add feature-local structured codes for `missing_label_field`, `unicode_coverage_gap`, `unsupported_tile_format`, `unsupported_tile_feature`, `missing_external_asset`, and `unavailable_terrain_sampler`, alongside the PRD diagnostic inventory.
- Q: Can validation support be described as support for unavailable building or 3D Tiles rendering? -> A: No. The diagnostic capability can be `supported`, but unavailable rendering or ingestion keeps its exact `Pro-gated`, `placeholder/fallback`, `experimental`, or `unsupported` classification.
- Q: What deterministic comparison default applies to bundle and label ingestion fixtures? -> A: Use stable ordering and exact fixture comparisons for deterministic manifests, diagnostics, label plans, and round-trip outputs unless the plan records a numeric tolerance before verification.
- Q: Does this P1 feature block MVP readiness? -> A: No. P1 deferrals require explicit diagnostics and documentation, not MVP blocking, unless a documented human decision expands release scope.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ingest Geospatial Labels Into MapScene (Priority: P1)

A geospatial Python user creates a `LabelLayer` from geospatial feature data, transforms it into the scene CRS, samples terrain height where available, evaluates simple label text expressions, and compiles it into a deterministic `LabelPlan`.

**Why this priority**: Data-driven labels are the highest leverage P1 asset workflow because they connect real map data to the deterministic label compiler and typed `MapScene` workflow.

**Independent Test**: Can be fully tested by ingesting point, line, and polygon label fixtures with known CRS, text properties, and glyph coverage, then compiling a `LabelPlan` and inspecting accepted/rejected labels plus diagnostics.

**Acceptance Scenarios**:

1. **Given** point, line, and polygon source features with text fields and CRS metadata, **When** a user creates a `LabelLayer`, **Then** all supported geometry types produce label candidates or typed diagnostics with affected feature IDs.
2. **Given** label features use a CRS different from the scene CRS, **When** the layer is validated or compiled, **Then** forge3d CRS utilities transform positions or report `crs_mismatch` with remediation before rendering.
3. **Given** label text uses `{name}`, `get`, `concat`, `coalesce`, or casing transforms, **When** labels are compiled, **Then** text values are evaluated deterministically or missing fields are reported before render.
4. **Given** terrain sampling is available, **When** label world positions are prepared, **Then** terrain-height sampling contributes to the label positions; otherwise terrain sampling availability is reported honestly.

---

### User Story 2 - Use Production Typography Basics (Priority: P1)

A cartographic production user renders labels with a documented default Latin atlas, can generate or load atlas data from fonts, declares fallback ranges, and sees Unicode coverage diagnostics before output.

**Why this priority**: P1 labels must move beyond demo atlas behavior while avoiding overclaims about full complex-script shaping.

**Independent Test**: Can be tested by preparing label fixtures that cover supported Latin glyphs, unsupported Unicode ranges, multiline labels, callouts, tracking, line-height, and fallback declarations, then validating layout metrics and diagnostics.

**Acceptance Scenarios**:

1. **Given** a scene uses the bundled default Latin atlas, **When** validation runs, **Then** the atlas coverage is documented and missing glyphs are reported before render.
2. **Given** a user provides a TTF or OTF font for atlas generation, **When** atlas generation succeeds, **Then** labels can use the generated coverage; if generation cannot complete, the user receives a typed diagnostic.
3. **Given** font fallback ranges are declared, **When** label text spans multiple ranges, **Then** fallback selection is deterministic and Unicode coverage gaps are diagnosed before render.
4. **Given** labels use kerning, tracking, line-height, multiline text, or callouts where those controls are exposed, **When** layout runs, **Then** the controls have measurable layout effects or return typed unsupported diagnostics.

---

### User Story 3 - Integrate Building Layers Honestly (Priority: P1)

A user adds GeoJSON, CityJSON, or mesh-derived building content to a `MapScene` and can tell before render whether the path is available, `Pro-gated`, `placeholder/fallback`, `experimental`, or `unsupported`.

**Why this priority**: Building ingestion substrate exists, but users must not confuse native availability or zero-geometry fallback with successful public integration.

**Independent Test**: Can be tested by validating and rendering known CityJSON and GeoJSON building fixtures where native support is available, plus negative fixtures for Pro-gated, fallback, unsupported, geometry-count, bounding-box, scalar PBR, and textured PBR status.

**Acceptance Scenarios**:

1. **Given** a building fixture uses an available native path, **When** `MapScene.validate()` and render preparation run, **Then** geometry counts, bounding boxes, support status, and renderability are reported.
2. **Given** a building fixture requires unavailable Pro/native support, **When** validation runs, **Then** the user receives `pro_gated_path` or `placeholder_fallback` diagnostics before render.
3. **Given** building material data includes scalar PBR settings, **When** docs and validation are reviewed, **Then** scalar support is documented honestly.
4. **Given** building material data requires textured PBR support, **When** validation runs, **Then** textured PBR is either supported end to end or explicitly reported as `unsupported`.

---

### User Story 4 - Load Supported 3D Tiles Into MapScene (Priority: P1)

A user adds supported 3D Tiles fixtures to a `MapScene` for offline rendering or review and receives clear diagnostics for unsupported tile formats, unsupported B3DM/GLB features, cache information, and LOD configuration availability.

**Why this priority**: 3D Tiles infrastructure exists, but the P1 product outcome is public scene integration with honest boundaries, not full Cesium runtime parity.

**Independent Test**: Can be tested by loading supported tileset and B3DM fixtures through public `Tiles3DLayer` construction, validating support status, and exercising unsupported-format and unsupported-feature diagnostics.

**Acceptance Scenarios**:

1. **Given** a supported local 3D Tiles fixture, **When** a user adds it to a `MapScene`, **Then** public validation identifies it as renderable or reviewable through the supported path.
2. **Given** an unsupported tile format or unsupported B3DM/GLB feature, **When** validation runs, **Then** the scene reports typed diagnostics instead of silently dropping tile content.
3. **Given** cache statistics or LOD configuration are available from the tile workflow, **When** validation or render preparation runs, **Then** those values appear in diagnostics or layer summaries; if unavailable, that absence is reported honestly.
4. **Given** documentation describes 3D Tiles support, **When** it is audited, **Then** it states that this feature is not full Cesium runtime parity.

---

### User Story 5 - Round-Trip Map Scene Bundles (Priority: P1)

A reviewer saves a map scene bundle containing terrain metadata, layer metadata, camera, lighting, output spec, label plan data or label sources, diagnostics, and supported export settings, then loads it later to reconstruct a renderable scene where assets are available.

**Why this priority**: P1 integrated geospatial assets must support review and reproducibility, including honest diagnostics for missing external assets and unsupported paths.

**Independent Test**: Can be tested by saving and loading bundles for scenes with labels, buildings, and 3D Tiles, then comparing reconstructed scene intent, diagnostics, missing-asset behavior, and renderability where assets exist.

**Acceptance Scenarios**:

1. **Given** a validated scene with supported terrain, labels, buildings, or 3D Tiles, **When** the user saves a bundle, **Then** all required scene intent and diagnostics are persisted deterministically.
2. **Given** the bundle is loaded on a machine where referenced assets are present, **When** reconstruction runs, **Then** the scene can validate and render or prepare supported content with equivalent intent.
3. **Given** referenced assets are missing, **When** the bundle is loaded or validated, **Then** missing assets are reported with structured diagnostics and affected layer IDs.
4. **Given** unsupported, Pro-gated, placeholder/fallback, or experimental content is included, **When** the bundle round-trips, **Then** its support status and diagnostics are preserved without implying successful ingestion.

### Edge Cases

- Source label features have missing text fields: validation reports missing fields before rendering and does not synthesize misleading labels.
- Source label geometries are empty, invalid, unsupported, or mixed: supported geometries proceed deterministically and unsupported geometries emit diagnostics with affected feature IDs where possible.
- CRS metadata is absent or incompatible: validation reports unknown or mismatched CRS before render instead of assuming compatibility.
- Terrain sampling is requested but no active terrain sampler is available: validation reports sampling unavailable and uses a documented fallback only if the user selected that policy.
- Text expressions reference missing properties or incompatible value types: expression diagnostics identify the layer and property path where possible.
- Labels contain glyphs outside the active atlas or fallback ranges: Unicode coverage and missing glyph diagnostics appear before render.
- Font atlas generation cannot load a TTF or OTF: the operation reports a typed diagnostic and does not pretend the atlas is usable.
- Building native support varies by package: validation distinguishes available native path, `Pro-gated`, `placeholder/fallback`, and `unsupported`.
- Building ingestion returns zero geometry: validation reports `placeholder_fallback` or equivalent support status and render is not treated as successful building output.
- 3D Tiles fixtures include unsupported formats, extensions, or B3DM/GLB features: validation reports typed diagnostics and does not overclaim support.
- 3D Tiles cache or LOD stats are unavailable: diagnostics state unavailable rather than inventing values.
- Bundle save references external assets that are later moved or deleted: bundle load reports missing external asset diagnostics and preserves review metadata.
- Bundle policy must choose between storing compiled label plans and source labels: the bundle records enough deterministic information to preserve render intent, and any recompilation policy is documented.
- Repeated bundle save, load, validation, and label compilation from fixed inputs must produce deterministic ordering for manifests, diagnostics, layer summaries, and label-plan references or payloads.

### Support-Level Classification *(mandatory)*

Use the PRD Appendix B terms exactly: `supported`, `underdeveloped`, `missing`, `Pro-gated`, `placeholder/fallback`, `experimental`, `unsupported`, and `non-goal`.

- **Supported in this feature**: data-driven `LabelLayer` construction from supported geospatial features, P1 simple label expressions, CRS transform through forge3d CRS utilities where available, terrain-height sampling where available, default Latin atlas coverage reporting, font atlas generation where supported, `BuildingLayer` and `Tiles3DLayer` validation diagnostics through `MapScene`, supported fixture rendering where native/public paths exist, and bundle save/load round-trip for supported map-scene assets. Diagnostic support does not make unavailable rendering or ingestion paths `supported`; those paths keep their exact PRD support-level classification.
- **Underdeveloped**: existing label, font, building, 3D Tiles, and bundle substrate before this feature completes public integration, diagnostics, docs, and tests.
- **Missing**: the complete P1 integrated workflow before this feature is implemented, including data-driven label layer ingestion, typography upgrade, public 3D Tiles scene integration, and asset-aware bundle round-trip.
- **Pro-gated**: building ingestion or other native paths that require unavailable Pro/native symbols or packaging.
- **Placeholder/fallback**: building, 3D Tiles, label, font, or bundle paths that would return zero geometry, non-renderable content, generated placeholder data, or incomplete review state while appearing successful.
- **Experimental**: public 3D Tiles scene workflows, line or curved label ingestion, optional shaping paths, or rendering paths that exist but are incomplete, unverified, or not production-ready.
- **Unsupported**: full Cesium runtime parity, full streamed MVT rendering, unsupported 3D Tiles formats or B3DM/GLB features, textured PBR buildings unless implemented end to end, unavailable CRS transforms, unavailable complex-script shaping, and unsupported font formats.
- **Non-goal**: web-first globe hosting, hosted tile services, browser delivery, general DCC editing, game/editor tooling, non-map rendering, VT normal/mask runtime, and textured PBR buildings beyond explicit P1 status diagnostics.

### Diagnostics & Failure Behavior *(mandatory)*

- **Diagnostic objects**: P1 diagnostics must include code, severity, message, remediation, support level where applicable, affected layer ID, affected object or feature ID where known, and deterministic serialization data suitable for bundles.
- **Pre-render validation**: `MapScene.validate()` must surface missing label fields, missing glyphs, Unicode coverage gaps, CRS issues, unavailable terrain sampling, building support status, building zero-geometry fallback, unsupported textured PBR status, 3D Tiles support status, unsupported B3DM/GLB features, cache/LOD availability, estimated memory where known, and missing bundle assets where applicable.
- **Required diagnostic codes and families**: The feature must use existing P0 diagnostic codes where applicable, including `crs_mismatch`, `missing_glyphs`, `pro_gated_path`, `placeholder_fallback`, `experimental_feature`, `python_public_3dtiles_incomplete`, and `estimated_gpu_memory`, and must add feature-local structured diagnostics for `missing_label_field`, `unicode_coverage_gap`, `unsupported_tile_format`, `unsupported_tile_feature`, `missing_external_asset`, and `unavailable_terrain_sampler`.
- **No-op prevention**: `LabelLayer` ingestion, atlas generation, building ingestion, 3D Tiles loading, bundle save, and bundle load must either produce real user-visible state/artifacts or typed diagnostics/errors; they must not report success while producing no labels, no glyph coverage, zero building geometry, no tile content, or unreconstructable bundle state.
- **Determinism**: Feature order, expression evaluation order, CRS transform reporting, terrain sampling decisions, fallback selection, diagnostics ordering, bundle manifests, label-plan references or payloads, and round-trip comparisons must be deterministic for fixed inputs and documented policy.
- **Failure behavior**: Error and fatal diagnostics block successful render completion. Warning diagnostics can block or continue according to the configured render policy while remaining visible in validation and bundles.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001 [PRD: P1-R1-AC1]**: `LabelLayer` ingestion MUST support point, line, and polygon source geometries by producing supported label candidates or typed diagnostics for unsupported geometry conditions.
- **FR-002 [PRD: P1-R1-AC2]**: `LabelLayer` ingestion MUST support automatic CRS transform through forge3d CRS utilities where available and MUST report incompatible or unavailable CRS transforms before render.
- **FR-003 [PRD: P1-R1-AC3]**: `LabelLayer` ingestion MUST support terrain-height sampling for world positions when a terrain sampler is available and MUST report unavailable sampling when requested but unavailable.
- **FR-004 [PRD: P1-R1-AC4]**: `LabelLayer` ingestion MUST support simple label text expressions: `{name}`, `get`, `concat`, `coalesce`, and casing transforms.
- **FR-005 [PRD: P1-R1-AC5]**: Data-driven `LabelLayer` workflows MUST produce a deterministic `LabelPlan` with accepted and rejected labels.
- **FR-006 [PRD: P1-R1-AC6]**: `LabelLayer` validation MUST report missing text fields and missing glyphs before rendering.
- **FR-007 [PRD: P1-R2-AC1]**: The product MUST provide a bundled default Latin atlas with documented glyph coverage.
- **FR-008 [PRD: P1-R2-AC2]**: The product MUST support runtime or build-time atlas generation from TTF or OTF fonts, or return typed diagnostics when generation cannot complete for a given font.
- **FR-009 [PRD: P1-R2-AC3]**: Typography configuration MUST support font fallback range declarations with deterministic fallback behavior.
- **FR-010 [PRD: P1-R2-AC4]**: Validation MUST provide Unicode coverage diagnostics before rendering.
- **FR-011 [PRD: P1-R2-AC5]**: Kerning, tracking, and line-height MUST be applied where exposed or explicitly reported as unsupported where unavailable.
- **FR-012 [PRD: P1-R2-AC6]**: Multiline labels and callouts MUST be supported for P1 typography workflows or return typed diagnostics for unsupported cases.
- **FR-013 [PRD: P1-R2-AC7]**: Optional non-Latin shaping MAY be designed, but it MUST NOT be treated as P1-blocking unless a documented product decision prioritizes it.
- **FR-014 [PRD: P1-R3-AC1]**: `MapScene.validate()` MUST distinguish available native building paths, `Pro-gated` paths, `placeholder/fallback` paths, and `unsupported` paths.
- **FR-015 [PRD: P1-R3-AC2]**: CityJSON and GeoJSON building fixtures MUST render through `MapScene` where the native path is available.
- **FR-016 [PRD: P1-R3-AC3]**: Building diagnostics MUST include geometry count and bounding boxes where building data can be inspected.
- **FR-017 [PRD: P1-R3-AC4]**: Scalar PBR building material support MUST be documented with exact support status.
- **FR-018 [PRD: P1-R3-AC5]**: Textured PBR building support MUST either be implemented end to end or explicitly marked `unsupported` through diagnostics and documentation.
- **FR-019 [PRD: P1-R3-AC6]**: Fallback zero-geometry building behavior MUST NOT be mistaken for successful ingestion and MUST emit typed diagnostics.
- **FR-020 [PRD: P1-R4-AC1]**: The public Python API MUST load supported 3D Tiles fixtures into `MapScene` through a typed `Tiles3DLayer` workflow.
- **FR-021 [PRD: P1-R4-AC2]**: Validation MUST distinguish supported 3D Tiles formats from unsupported formats.
- **FR-022 [PRD: P1-R4-AC3]**: 3D Tiles cache statistics MUST appear in diagnostics or layer summaries where available.
- **FR-023 [PRD: P1-R4-AC4]**: Screen-space-error or equivalent LOD configuration MUST be exposed if supported by the renderer, or diagnosed as unsupported if unavailable.
- **FR-024 [PRD: P1-R4-AC5]**: Unsupported B3DM or GLB features MUST produce typed diagnostics.
- **FR-025 [PRD: P1-R4-AC6]**: 3D Tiles documentation MUST state that this feature is not full Cesium runtime parity.
- **FR-026 [PRD: P1-R5-AC1]**: `MapScene.save_bundle()` MUST include terrain source metadata, layer metadata, camera, lighting, output spec, label plans or label sources, diagnostics, and supported export settings.
- **FR-027 [PRD: P1-R5-AC2]**: Loading a bundle MUST reconstruct a renderable scene where assets are available and support status permits rendering.
- **FR-028 [PRD: P1-R5-AC3]**: Missing external assets MUST be reported with structured diagnostics during bundle load or validation.
- **FR-029 [PRD: P1-R5-AC4]**: Bundles MUST include enough information for review workflows, including support status, diagnostics, layer summaries, and scene intent.
- **FR-030 [PRD: P1-R5-AC5]**: Bundle round-trip behavior MUST be covered by tests for supported assets and missing external assets.
- **FR-031 [PRD: Milestone 4]**: Building and 3D Tiles capabilities MUST be exposed through `MapScene` where available and diagnosed honestly where unavailable.
- **FR-032 [PRD: Constitution III]**: Public ingestion, typography, building, 3D Tiles, bundle save, and bundle load APIs MUST either produce real state or typed diagnostics/errors and MUST NOT report no-op success.
- **FR-033 [PRD: Constitution IV]**: Label ingestion, typography selection, building summaries, 3D Tiles summaries, diagnostics, bundle save, and bundle load MUST use deterministic ordering for fixed inputs.
- **FR-034 [PRD: Section 16, Appendix B]**: Documentation and examples MUST use PRD support-level terms exactly and MUST avoid overclaiming full Cesium runtime parity, full Mapbox GL parity, textured PBR buildings, VT normal/mask runtime, or complex-script shaping.
- **FR-035 [PRD: Section 13, Section 18]**: Feature-local diagnostic codes for missing label fields, Unicode coverage gaps, unsupported tile formats, unsupported tile features, missing external assets, and unavailable terrain samplers MUST be structured, serializable, documented, and covered by negative tests.
- **FR-036 [PRD: P1-R5-AC1, P1-R5-AC2]**: Bundles MUST persist both source labels and compiled `LabelPlan` payloads where available; inability to persist or replay either form MUST be represented by structured diagnostics.
- **FR-037 [PRD: Section 8.2, Constitution IV]**: Deterministic P1 bundle, label ingestion, and validation fixtures MUST use exact comparison for stable serialized outputs unless a numeric tolerance is recorded in the plan before verification.
- **FR-038 [PRD: Section 17, Section 21]**: Deferred or diagnosed P1 paths MUST NOT block P0 MVP readiness unless a documented human product decision explicitly expands MVP scope.

### Key Entities *(include if feature involves data)*

- **LabelLayer**: Data-driven label source created from geospatial features, feature properties, style-derived text rules, CRS metadata, terrain sampling policy, typography settings, and diagnostics.
- **LabelTextExpression**: Supported expression intent for label text, including `{name}`, `get`, `concat`, `coalesce`, and casing transforms, with deterministic evaluation and missing-field diagnostics.
- **FontAtlas**: Glyph atlas data, source font metadata, coverage ranges, generation status, and diagnostics for missing glyphs or unsupported font inputs.
- **FontFallbackRange**: Declared Unicode range and preferred font or atlas source used to select glyph coverage deterministically.
- **TypographySettings**: Cartographic label controls such as kerning, tracking, line-height, multiline behavior, callout behavior, and support status for optional shaping paths.
- **BuildingLayer**: Scene layer intent for GeoJSON, CityJSON, or mesh-derived buildings, including support status, geometry count, bounding boxes, material status, and renderability.
- **Tiles3DLayer**: Scene layer intent for 3D Tiles sources, including source identity, supported format status, unsupported feature diagnostics, cache stats where available, and LOD configuration where supported.
- **BundleManifest**: Deterministic record of scene intent, terrain metadata, layers, camera, lighting, output spec, labels or label plans, diagnostics, supported export settings, external asset references, and review metadata.
- **MissingAssetDiagnostic**: Structured diagnostic for bundle assets that are referenced but unavailable at load or validation time.
- **Public API contract**: This feature extends the `MapScene`, `SceneRecipe`, `LabelPlan`, `ValidationReport`, `Diagnostic`, and `Bundle` product contract with P1 integrated geospatial assets and bundle round-trip behavior.

## PRD Traceability & Evidence *(mandatory)*

| PRD AC ID | Requirement in this spec | Evidence required | Verification target |
|---|---|---|---|
| P1-R1-AC1 | FR-001 | Point, line, and polygon ingestion tests with supported and unsupported geometry fixtures | LabelLayer geometry ingestion tests |
| P1-R1-AC2 | FR-002 | CRS transform tests using forge3d CRS utilities plus mismatch diagnostics | LabelLayer CRS transform tests |
| P1-R1-AC3 | FR-003 | Terrain-height sampling tests or diagnostics proving unavailable sampling | LabelLayer terrain sampling tests |
| P1-R1-AC4 | FR-004 | Expression fixtures for `{name}`, `get`, `concat`, `coalesce`, and casing transforms | Label expression tests |
| P1-R1-AC5 | FR-005 | Integration test from ingested `LabelLayer` to accepted/rejected `LabelPlan` | LabelLayer to LabelPlan tests |
| P1-R1-AC6 | FR-006 | Missing text field and missing glyph pre-render diagnostics | LabelLayer diagnostic tests |
| P1-R2-AC1 | FR-007 | Default Latin atlas artifact, coverage docs, and coverage tests | Default atlas coverage tests |
| P1-R2-AC2 | FR-008 | TTF/OTF atlas generation tests or typed generation diagnostics | Font atlas generation tests |
| P1-R2-AC3 | FR-009 | Font fallback declaration tests and docs | Font fallback tests |
| P1-R2-AC4 | FR-010 | Unicode coverage diagnostics for supported and unsupported ranges | Unicode coverage tests |
| P1-R2-AC5 | FR-011 | Layout metric tests for kerning, tracking, and line-height, or unsupported diagnostics | Typography metric tests |
| P1-R2-AC6 | FR-012 | Multiline label and callout layout/render tests or unsupported diagnostics | Multiline and callout tests |
| P1-R2-AC7 | FR-013 | Design note or explicit product decision showing optional shaping is non-blocking | Shaping deferral review |
| P1-R3-AC1 | FR-014 | Building validation tests for native, Pro-gated, placeholder/fallback, and unsupported paths | Building support-status tests |
| P1-R3-AC2 | FR-015 | CityJSON and GeoJSON fixture render tests where native path is available | Building fixture render tests |
| P1-R3-AC3 | FR-016 | Diagnostics tests for building geometry counts and bounding boxes | Building diagnostics tests |
| P1-R3-AC4 | FR-017 | Building support matrix documenting scalar PBR support | Building docs audit |
| P1-R3-AC5 | FR-018 | Textured PBR end-to-end tests or typed unsupported diagnostics plus docs | Textured PBR status tests |
| P1-R3-AC6 | FR-019 | Negative tests proving zero-geometry fallback emits `placeholder/fallback` diagnostics | Building fallback tests |
| P1-R4-AC1 | FR-020 | Public API tests loading supported 3D Tiles fixtures into `MapScene` | Tiles3DLayer load tests |
| P1-R4-AC2 | FR-021 | Validation tests for supported and unsupported tile formats | 3D Tiles format tests |
| P1-R4-AC3 | FR-022 | Diagnostics or layer summary tests for cache stats where available | 3D Tiles cache diagnostics tests |
| P1-R4-AC4 | FR-023 | LOD configuration tests or typed unsupported diagnostics | 3D Tiles LOD config tests |
| P1-R4-AC5 | FR-024 | Negative tests for unsupported B3DM/GLB features | 3D Tiles unsupported feature tests |
| P1-R4-AC6 | FR-025 | 3D Tiles support matrix wording audit | 3D Tiles docs audit |
| P1-R5-AC1 | FR-026 | Bundle schema/tests proving each required scene field is persisted | Bundle manifest tests |
| P1-R5-AC2 | FR-027 | Bundle load/render round-trip tests where assets are available | Bundle reconstruction tests |
| P1-R5-AC3 | FR-028 | Missing external asset diagnostics during load or validation | Missing asset tests |
| P1-R5-AC4 | FR-029 | Review bundle fixture and docs showing review state coverage | Review workflow tests |
| P1-R5-AC5 | FR-030 | Automated bundle save/load/validate round-trip tests | Bundle round-trip tests |
| Milestone 4 | FR-031 | Integrated label, building, 3D Tiles, and bundle workflow evidence | Milestone 4 exit review |
| Constitution III | FR-032 | Negative tests for no-op success across ingestion, typography, buildings, tiles, and bundles | No-op success tests |
| Constitution IV | FR-033, FR-037 | Deterministic ordering and exact fixture comparisons for diagnostics, manifests, labels, summaries, and round-trip comparisons unless a recorded tolerance exists | Determinism tests |
| Section 16 / Appendix B | FR-034 | Docs and examples use only PRD support classifications and preserve non-goals | P1 docs audit |
| Section 13 / Section 18 | FR-035 | Feature-local diagnostic codes are structured, serializable, documented, and covered by negative tests | Diagnostic code tests |
| P1-R5-AC1 / P1-R5-AC2 | FR-036 | Bundle fixtures persist source labels and compiled label plans where available and diagnose replay gaps | Bundle label persistence tests |
| Section 17 / Section 21 | FR-038 | P1 deferrals remain non-MVP-blocking unless a documented product decision expands scope | MVP scope review |

## Explicit Non-Goals *(mandatory)*

- Blocking MVP completion unless a documented human decision expands MVP scope.
- Full Cesium runtime parity or live global 3D Tiles streaming.
- Full Mapbox GL parity or streamed MVT rendering.
- Full complex-script shaping unless separately prioritized by documented product decision.
- Textured PBR building implementation beyond P1's requirement to implement it end to end or mark it `unsupported`.
- VT normal/mask runtime support.
- Browser delivery, hosted tile services, general DCC editing, game/editor tooling, animation/cinematic production, or non-map rendering.
- Treating Pro-gated native paths, zero-geometry fallback, incomplete 3D Tiles support, missing fonts, missing glyphs, or unsupported bundle assets as successful supported output.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of P1-R1 through P1-R5 acceptance criteria have explicit requirements, evidence expectations, and verification targets in this specification.
- **SC-002**: Users can create a `LabelLayer` from supported point, line, and polygon geospatial data and receive a deterministic accepted/rejected `LabelPlan`.
- **SC-003**: Label ingestion diagnostics identify missing text fields, missing glyphs, CRS issues, and unavailable terrain sampling before render.
- **SC-004**: Typography validation reports default Latin atlas coverage, Unicode coverage gaps, font fallback behavior, and atlas generation status before render.
- **SC-005**: Building validation distinguishes available native path, `Pro-gated`, `placeholder/fallback`, and `unsupported` for all building fixtures used by tests or examples.
- **SC-006**: Building diagnostics include geometry count and bounding boxes for inspectable building data.
- **SC-007**: Supported 3D Tiles fixtures load through public `MapScene` workflows, while unsupported formats and unsupported B3DM/GLB features produce typed diagnostics.
- **SC-008**: 3D Tiles docs contain zero claims of full Cesium runtime parity.
- **SC-009**: Bundle save persists all required scene intent fields, diagnostics, and supported export settings in deterministic order.
- **SC-010**: Bundle load reconstructs a renderable scene where assets are available and emits structured missing-asset diagnostics where assets are unavailable.
- **SC-011**: Repeated bundle save/load/validate from fixed inputs produces stable manifest ordering, diagnostics ordering, layer summaries, and label-plan references or payloads.
- **SC-012**: P1 docs and examples contain zero support overclaims against the PRD Appendix B taxonomy.
- **SC-013**: Automated checks confirm no unresolved template placeholders or clarification markers remain in this specification.
- **SC-014**: Negative diagnostics tests cover `missing_label_field`, `unicode_coverage_gap`, `unsupported_tile_format`, `unsupported_tile_feature`, `missing_external_asset`, and `unavailable_terrain_sampler`.
- **SC-015**: Bundle round-trip fixtures persist both source labels and compiled `LabelPlan` payloads where available, with diagnostics for any unsupported replay or persistence gap.
- **SC-016**: Before completion, `docs/superpowers/state/current-context-pack.md` and `docs/superpowers/state/implementation-ledger.md` record feature `005` continuity evidence; `docs/superpowers/state/requirements-verification-matrix.md` is updated only when requirement status changes.

## Assumptions

- P0 features `001` through `004` define stable diagnostics, label planning, and `MapScene` contracts before P1 implementation depends on them.
- P1 label expression support includes all forms listed in the PRD: `{name}`, `get`, `concat`, `coalesce`, and casing transforms.
- Bundle round-trip preserves render intent by storing both source labels and compiled `LabelPlan` payloads where available; unsupported persistence or replay gaps must emit structured diagnostics.
- Building workflows may vary by package. The spec requires native-path rendering where available and typed `Pro-gated`, `placeholder/fallback`, or `unsupported` diagnostics where unavailable.
- 3D Tiles support is scoped to local supported fixtures and offline review/render preparation, not large production tile hierarchies or full Cesium runtime parity.
- Textured PBR building support can satisfy P1 by being explicitly marked `unsupported` with diagnostics and docs if it is not implemented end to end.
- Non-Latin shaping is non-blocking for P1 unless a documented product decision changes priority.
- Existing dirty workspace changes are user-owned and are not part of this feature unless explicitly touched by this specification workflow.

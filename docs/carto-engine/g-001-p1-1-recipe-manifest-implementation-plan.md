# G-001 / P1-1 Recipe Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small evidence-derived recipe-family manifest schema for existing example workflows without replacing `SceneRecipe` or `MapScene`.

**Architecture:** `RecipeManifest` is a thin metadata layer that records provenance, inputs, defaults, support status, diagnostics, and fixture intent. It serializes to deterministic JSON, can reference or feed existing `SceneRecipe` / `MapScene` where applicable, and contains no rendering behavior.

**Tech Stack:** Python dataclasses, stdlib `json`, stdlib `pathlib`, deterministic dict validation, existing MapScene/diagnostics concepts, and pytest. No Rust, PyO3, shader, renderer, network, or optional GIS backend changes are in P1-1.

---

## Canonical Identity

The canonical audit identity is `G-001`: recipe-family manifests and provenance. `P1-1` is the Phase 1 task that adds a small recipe manifest schema for existing examples. `G-003` is a separate gap for reusable pass compositing and blend modes, and is explicitly out of scope for this plan.

## Prerequisite Verification

Result: PASS. Verification was performed against freshly fetched `origin/main` at `f45ad42` (`Merge pull request #98 from milos-agathon/codex/g002-later-domain-remote-helpers`). Before this planning change, the current branch tree matched `origin/main` (`git diff --name-status origin/main HEAD` was empty). Local branch `main` is stale at `8fa1893`, so implementation agents must use fetched `origin/main` or update local `main` before starting runtime work.

GitHub/main history evidence:

- PR #93 merged G-002c C1 vector metadata/read.
- PR #94 merged G-002c C2 vector reprojection.
- PR #95 merged G-002c C3 vector geometry APIs.
- C4 is present in main history as commits `34513b0`, `47e1602`, `f2ebeac`, `47d6e7c`, `7a7d687`, `7f87efe`, `04b1f0e`, and `8166616`.
- PR #96 merged G-002c C5 rasterization and masks.
- PR #97 merged G-002c C6 thematic raster.
- PR #98 merged the later GIS helper surface.

File-level G-002 evidence:

- G-002a1 raster public surface exists in `src/gis/raster_info.rs`, `src/gis/raster_write.rs`, `src/gis/types.rs`, `src/gis/mod.rs`, `python/forge3d/gis.py`, `python/forge3d/gis.pyi`, `tests/test_gis_raster.py`, and `tests/test_gis_read_raster.py`.
- G-002b CRS, affine, nodata, windowing, alignment, and reprojection exist in `src/gis/crs.rs`, `src/gis/affine.rs`, `src/gis/warp.rs`, `src/gis/mod.rs`, `python/forge3d/gis.py`, `python/forge3d/gis.pyi`, `tests/test_gis_crs_affine.py`, `tests/test_gis_alignment_windowing.py`, and `tests/test_gis_resample_reproject.py`.
- G-002c C1-C4 vector IO, CRS, geometry, and overlay exist in `src/gis/vector.rs`, `src/gis/geometry.rs`, `src/gis/geometry/`, `python/forge3d/gis.py`, `python/forge3d/gis.pyi`, `tests/test_gis_vector_io.py`, `tests/test_gis_vector_crs.py`, `tests/test_gis_vector_geom.py`, and `tests/test_gis_vector_overlay.py`.
- G-002c C5 exists in `src/gis/rasterize.rs`, PyO3 registrations in `src/py_module/functions/gis.rs`, wrappers/stubs in `python/forge3d/gis.py` and `python/forge3d/gis.pyi`, and `tests/test_gis_rasterize_mask.py`.
- G-002c C6 exists in `src/gis/thematic.rs`, PyO3 registrations in `src/py_module/functions/gis.rs`, wrappers/stubs in `python/forge3d/gis.py` and `python/forge3d/gis.pyi`, and `tests/test_gis_thematic.py`.
- Later helpers exist in `src/gis/remote.rs`, `src/gis/tiles.rs`, `src/gis/osm.rs`, `src/gis/domain.rs`, `src/gis/terrarium.rs`, wrapper/stub exports in `python/forge3d/gis.py` and `python/forge3d/gis.pyi`, and focused tests `tests/test_gis_remote.py`, `tests/test_gis_cog_tiles.py`, `tests/test_gis_osm.py`, and `tests/test_gis_domain.py`.
- GIS classes are registered in `src/py_module/classes.rs` as `RasterInfo`, `VectorInfo`, `AffineTransform`, and `CrsTransform`.
- `tests/test_api_contracts.py` includes G-002 functions through `fetch_remote_geodata`, `cache_geodata`, `prepare_osm_scene`, and `estimate_local_utm`; `LATER_GIS_FUNCTIONS = []` after PR #98, confirming no intentionally absent later helpers remain.

File-level MapScene/SceneRecipe evidence:

- Public exports exist in `python/forge3d/__init__.py` and `python/forge3d/__init__.pyi` for `MapScene`, `SceneRecipe`, `TerrainSource`, `RasterOverlay`, `VectorOverlay`, `LabelLayer`, `PointCloudLayer`, `MapSceneBuildingLayer`, `MapFurnitureLayer`, `OrbitCamera`, `LightingPreset`, and `OutputSpec`.
- Typed models and behavior exist in `python/forge3d/map_scene.py`: layer dataclasses, `SceneRecipe`, `MapScene`, validation, render, bundle load/save, and support diagnostics.
- Contract tests exist in `tests/test_mapscene_recipe_contract.py`.
- Render tests exist in `tests/test_mapscene_render_png.py`.
- Bundle tests exist in `tests/test_mapscene_save_bundle.py`.
- Support-status diagnostics tests exist in `tests/test_mapscene_support_status.py`.
- Terrain visual golden coverage exists in `tests/test_terrain_visual_goldens.py`.
- Vector drape, map plate, furniture, labels, COG, point cloud, and building primitives exist in `python/forge3d/style.py`, `python/forge3d/terrain_params.py`, `python/forge3d/label_plan.py`, `python/forge3d/legend.py`, `python/forge3d/scale_bar.py`, `python/forge3d/north_arrow.py`, `python/forge3d/buildings.py`, `python/forge3d/cog.py`, `python/forge3d/pointcloud.py`, `tests/test_vector_drape.py`, and `tests/test_map_plate_layout.py`.

## Source Documents

- `docs/carto-engine/golden-map-recipe-capability-audit.md` is the canonical recipe-family evidence source for this phase.
- `docs/carto-engine/rust-gis-implementation-plan.md`, `docs/carto-engine/gis-operation-api-crosswalk.md`, `docs/carto-engine/gis-contract-evidence.md`, `docs/carto-engine/g-002b-support-matrix.md`, and the G-002c phase plans are the GIS context sources.
- `docs/carto-engine/g-002-later-domain-remote-helpers-implementation-plan.md` documents the later helper contracts that are now implemented.

No prior `G-001`/`P1-1`, recipe-manifest, or recipe-family manifest planning file exists under `docs/`, so this phase-specific file is the least disruptive planning location.

## P1-1 Scope

Plan implementation of a small manifest schema that records evidence-derived metadata for existing examples. The manifest must describe:

- `schema_version`
- `recipe_family`
- `recipe_id`
- `status`
- `source_examples`
- `source_evidence`
- `required_inputs`
- `optional_inputs`
- `produced_outputs`
- `layers`
- `alignment`
- `preprocessing`
- `camera_defaults`
- `lighting_defaults`
- `styling_defaults`
- `annotations_defaults`
- `render_export_defaults`
- `support_status`
- `diagnostics`
- `tests`
- `golden_fixture_intent`
- `non_goals`
- `open_questions`

The first MapScene-oriented manifest batch is:

- `terrain_demo`
- `terrain_label`
- `landcover_esri_terrain_viewer`
- `climate_bivariate`
- `hydrology_river`
- `mapscene_showcases`

P1-1 also records how the same schema later covers the next MapScene batch:

- `terrain_relief_rem`
- `population_spike_worldpop`
- `population_ghsl_3d`
- `builtup_cover_3d`
- `pointcloud_cog`
- `urban_osm_city`
- `luxembourg_rail_overlay`

Removed from the MapScene recipe roadmap:

- `labels_styles_picking`, because `terrain_label` covers the needed label-first workflow.
- `wildfire_smoke`, `satellite_timelapse`, `osm_city_flood_daycycle`, and `humanity_globe_video`, because video/temporal recipes are out of scope for this MapScene-first plan.

After P1-1, MapScene capability work must follow `docs/carto-engine/mapscene-enrichment-capability-ranking.md`: manifest/provenance first, then alignment diagnostics, map plate composition, population surface styling, multi-pass compositing, recipe fixtures, terrain camera/lighting/relief presets, categorical raster style, vector line draping, bivariate raster style, building adapter, docs-to-test traceability, urban context builder, label defaults, and PointCloud/COG adapters. Update that ranking with evidence before changing roadmap order.

## Manifest Format Decision

Use a hybrid format: focused Python dataclasses in a new `python/forge3d/recipe_manifest.py` module that serialize to stable JSON-compatible dicts.

Reasons:

- Easy to diff: fixtures are deterministic JSON with sorted keys and stable list ordering.
- Source evidence friendly: `source_examples` and `source_evidence` can cite repo paths and line ranges.
- Validation is lightweight: stdlib-only dataclass and dict validation, no optional GIS/rendering dependencies.
- Future MapScene bridge is clean: manifest may contain a `mapscene` reference block without importing or replacing `MapScene`.
- Docs and tests can consume the same JSON fixtures.
- It avoids growing `python/forge3d/map_scene.py`, which already owns typed scene/render/bundle behavior.

No top-level `forge3d.__all__` export is required in P1-1. Users and tests import `forge3d.recipe_manifest` directly. Add `__init__.py` and `__init__.pyi` exports only in a later API-stabilization change if project maintainers want `from forge3d import RecipeManifest`.

## Schema Decisions

- `schema_version`: string, initially `"1"`. Any other major version emits `recipe_manifest_schema_version_unsupported`.
- `recipe_family`: one of the known family tokens listed in this plan.
- `recipe_id`: stable lowercase identifier, unique within family.
- `status`: one of the audit vocabulary values.
- `support_status`: dict keyed by capability or surface, with values from the same audit vocabulary.
- `source_examples`: list of repo-relative file paths. Validation checks local existence only.
- `source_evidence`: list of dicts with `path`, optional `line_start`, optional `line_end`, and `note`.
- `required_inputs` / `optional_inputs`: list of input descriptors with `name`, `kind`, optional `crs`, optional `shape`, optional `format`, and optional `role`.
- `produced_outputs`: list of output descriptors with `kind`, `format`, optional `path`, and optional `deterministic`.
- `layers`: ordered list of layer descriptors with `layer_id`, `layer_type`, `role`, `required`, and optional `style_ref`.
- `alignment`: declarative assumptions only, such as `crs`, `extent`, `resolution`, `transform`, and `notes`. P1-1 does not compare data.
- `preprocessing`, `camera_defaults`, `lighting_defaults`, `styling_defaults`, `annotations_defaults`, and `render_export_defaults`: JSON-compatible dicts; use `"example_defined"` when exact reusable defaults are still script-local.
- `diagnostics`: list of stable diagnostic tokens expected from manifest validation.
- `tests`: list of test names or fixture references proving manifest validation, not runtime rendering.
- `golden_fixture_intent`: dict with `status` in `exists`, `missing`, or `deferred`, plus optional source test/file evidence.
- `non_goals`: list of explicit exclusions for the manifest.
- `open_questions`: list only for implementation blockers. P1-1 should use `[]`.

Allowed `status` vocabulary:

- `proven_in_forge3d`
- `partially_proven`
- `exists_only_as_example_or_script_logic`
- `exists_but_not_exposed_as_public_api`
- `exists_but_not_tested`
- `not_found`
- `evidence_missing`
- `unclear_requires_human_confirmation`

Allowed P1-1 `layer_type` tokens:

- `terrain_dem`
- `raster_continuous`
- `raster_categorical`
- `raster_bivariate`
- `vector_polygon`
- `vector_line`
- `label_annotation`
- `map_furniture`
- `mapscene_recipe`
- `pointcloud`
- `cog_raster`
- `building_footprint`
- `temporal_sequence`
- `globe_raster`

P1-1 validates layer token membership only. It does not claim render support for every token.

## Stable Diagnostic Tokens

Validation must use stable lowercase tokens:

- `recipe_manifest_missing_field`
- `recipe_manifest_invalid_field`
- `recipe_manifest_invalid_status`
- `recipe_manifest_unknown_family`
- `recipe_manifest_missing_source`
- `recipe_manifest_unsupported_layer`
- `recipe_manifest_alignment_unspecified`
- `recipe_manifest_render_path_unspecified`
- `recipe_manifest_golden_not_selected`
- `recipe_manifest_mapscene_partial`
- `recipe_manifest_example_only`
- `recipe_manifest_schema_version_unsupported`

Token meanings:

- `recipe_manifest_missing_field`: a required manifest key is absent.
- `recipe_manifest_invalid_field`: a field has the wrong shape or type.
- `recipe_manifest_invalid_status`: `status` or `support_status` uses a value outside the allowed vocabulary.
- `recipe_manifest_unknown_family`: `recipe_family` is outside the known registry.
- `recipe_manifest_missing_source`: a `source_examples` or `source_evidence.path` file does not exist in the repo.
- `recipe_manifest_unsupported_layer`: a layer uses an unknown `layer_type`.
- `recipe_manifest_alignment_unspecified`: a multi-spatial-input recipe omits CRS/alignment assumptions.
- `recipe_manifest_render_path_unspecified`: a render/export output omits a path or explicit `"example_defined"` path policy.
- `recipe_manifest_golden_not_selected`: no canonical tiny recipe-level golden fixture is selected.
- `recipe_manifest_mapscene_partial`: a manifest can reference MapScene but cannot be fully represented by current MapScene behavior.
- `recipe_manifest_example_only`: support exists only in examples or script logic.
- `recipe_manifest_schema_version_unsupported`: manifest version is not supported by the validator.

## File Deltas For P1-1 Implementation

Create:

- `python/forge3d/recipe_manifest.py`
  - Dataclasses: `RecipeManifest`, `RecipeInput`, `RecipeOutput`, `RecipeLayer`, `SourceEvidence`, `GoldenFixtureIntent`.
  - Constants: allowed families, statuses, layer types, diagnostic tokens, schema version.
  - Helpers: `manifest_from_dict`, `manifest_to_dict`, `manifest_to_json`, `manifest_from_json`, `validate_manifest`, `load_manifest`, `save_manifest`.
  - Deterministic serialization: sorted keys, two-space indentation, trailing newline.
  - No imports from `rasterio`, `geopandas`, `shapely`, `rioxarray`, `xarray`, `terra`, rendering modules, or network clients.
- `python/forge3d/recipe_manifest.pyi`
  - Matching public dataclass and helper signatures.
- `tests/test_recipe_manifest.py`
  - Focused construction, serialization, diagnostics, no-network, and no-optional-backend tests.
- `tests/fixtures/recipe_manifests/terrain_demo.json`
- `tests/fixtures/recipe_manifests/landcover_esri_terrain_viewer.json`
- `tests/fixtures/recipe_manifests/climate_bivariate.json`
- `tests/fixtures/recipe_manifests/terrain_label.json`
- `tests/fixtures/recipe_manifests/hydrology_river.json`
- `tests/fixtures/recipe_manifests/mapscene_showcases.json`
- `docs/carto-engine/recipe-family-manifest-schema.md`
  - User-facing schema reference copied from the implemented constants and fixture examples.

Modify:

- `docs/carto-engine/g-001-p1-1-recipe-manifest-implementation-plan.md`
  - Add final implementation notes only if implementation discovers a factual mismatch.

Do not modify in P1-1:

- `python/forge3d/map_scene.py`
- `src/gis/`
- `src/py_module/functions/gis.rs`
- `src/py_module/classes.rs`
- `python/forge3d/gis.py`
- `python/forge3d/gis.pyi`
- rendering, shader, COG, pointcloud, terrain, building, label, legend, scale bar, or north arrow modules

Conditional changes:

- `python/forge3d/__init__.py` and `python/forge3d/__init__.pyi`: change only if maintainers explicitly decide P1-1 should export manifest classes at top level.
- `tests/test_api_contracts.py`: change only if top-level public exports are added.

## Initial Family Fixture Requirements

| Family | Source examples | Required inputs | Layer stack | Defaults | Outputs | Current support | MapScene compatibility | Missing reusable abstraction | Future task | Expected diagnostics | Golden intent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `terrain_demo` | `examples/terrain_viewer/terrain_demo.py`, `python/forge3d/terrain_demo.py` | DEM/heightfield | `terrain_dem` with continuous colormap styling | camera, lighting, and styling may be `example_defined` with evidence lines | PNG/RGBA | `proven_in_forge3d` | `partially_proven` because Scene/terrain support exists and MapScene terrain/raster support exists, but the terrain demo workflow is not a MapScene manifest yet | recipe metadata around DEM, colormap, z scale, camera, lighting | P1-4 canonical recipe golden selection | `recipe_manifest_golden_not_selected` if no recipe-level golden is selected | record existing terrain visual golden coverage in `tests/test_terrain_visual_goldens.py`; recipe-level golden deferred |
| `landcover_esri_terrain_viewer` | `examples/landcover_esri/bosnia_terrain_landcover_viewer.py` plus sibling landcover viewers | DEM, categorical raster, class palette | `terrain_dem`, `raster_categorical` terrain drape | camera, lighting, categorical class palette, alpha may be `example_defined` | PNG/snapshot | `exists_only_as_example_or_script_logic` | `partially_proven` because raster drape exists but categorical style object and tiny canonical asset are missing | public categorical raster style object and manifest fixture asset | P2-1 categorical style | `recipe_manifest_example_only`, `recipe_manifest_mapscene_partial`, `recipe_manifest_golden_not_selected` | missing tiny canonical asset |
| `climate_bivariate` | `examples/climate_bivariate/europe_bivariate_climate_map.py`, Belgium/France variants | DEM, two rasters, bivariate matrix | `terrain_dem`, two `raster_continuous` inputs, derived `raster_bivariate`, `map_furniture` legend intent | camera, lighting, bivariate matrix, legend may be `example_defined` | PNG/poster | `exists_only_as_example_or_script_logic` | `partially_proven` because terrain/raster/furniture primitives exist but bivariate style/legend are script-local | public bivariate raster style and legend | P2-1 bivariate style/legend | `recipe_manifest_example_only`, `recipe_manifest_mapscene_partial`, `recipe_manifest_golden_not_selected` | missing tiny canonical asset |
| `terrain_label` | `examples/labels_styles_picking/fuji_labels_demo.py` and MapScene label examples | DEM/heightfield, label points/lines/callouts, font/style defaults | `terrain_dem`, `label_annotation` | camera, lighting, label style, halo, and placement may be `example_defined` | PNG/snapshot | `proven_in_forge3d` for label primitives; manifest support is new | `partially_proven` because labels exist but recipe-level terrain-label defaults are not declared | manifest metadata around terrain labels and style defaults | P1-1 manifest fixture, then P2 label integration only if needed | `recipe_manifest_golden_not_selected` | missing tiny canonical terrain-label asset |
| `hydrology_river` | `examples/hydrology_river_basins/poland_river_basins_forge3d.py`, `pnoa_river_showcase.py`, `turkiye_river_basins_3d.py` | DEM, river lines, optional basin/context polygons | `terrain_dem`, `vector_line`, optional `vector_polygon` context | camera, lighting, river width/color rules may be `example_defined` | PNG/poster | `exists_only_as_example_or_script_logic` | `partially_proven` because vector overlay and rasterization exist, but reusable line styling/drape helper is missing | reusable river line styling and drape helper | P2-2 line stroke/drape | `recipe_manifest_example_only`, `recipe_manifest_mapscene_partial`, `recipe_manifest_golden_not_selected` | missing tiny canonical asset |
| `mapscene_showcases` | `examples/mapscene_terrain_raster.py`, `examples/mapscene_vector_labels.py`, `examples/mapscene_bundled_datasets_showcase.py`, `examples/mapscene_p1_assets_bundle_showcase.py`, `examples/mapscene_buildings_labels.py` | synthetic DEM/raster/vector/labels/building inputs depending on fixture | `mapscene_recipe` referencing existing `SceneRecipe` structure; do not duplicate MapScene schema | use existing `SceneRecipe` camera, lighting, output, validation, render, and bundle defaults by reference | PNG and bundle outputs | `partially_proven` as a family; core `SceneRecipe`/`MapScene` behavior is `proven_in_forge3d` | `proven_in_forge3d` for terrain/raster/vector/label/render/bundle; partial for building/point-cloud adapters | manifest provenance layer only | P2-4 adapter hardening, if needed | `recipe_manifest_mapscene_partial` only for partial showcase rows; no schema duplication diagnostic | `mapscene_bundle_tiny` fixture exists through synthetic examples; recipe-level visual golden deferred |

Each fixture must include `alignment` and `render_export_defaults`. If a source example defines those values only in script-local code, record `"example_defined"` plus `source_evidence`; do not infer or invent numeric defaults.

## Later Family Coverage Boundary

P1-1 records these families in the allowed-family registry and schema docs, but does not create fixtures for them unless the implementation keeps a single compact documented row per family:

| Family | Schema coverage later | P1-1 boundary |
| --- | --- | --- |
| `terrain_relief_rem` | DEM, mask, two-pass relief intent, overlay blend metadata | no G-003 pass compositing helper, no visual golden |
| `population_spike_worldpop` | population raster, optional DEM, spike/height-shade intent | no spike renderer or fixture asset |
| `population_ghsl_3d` | GHSL raster, boundaries, labels, extrusion/spike intent | no new population rendering behavior |
| `builtup_cover_3d` | built-up raster, DEM, mask, color/height intent | no categorical/built-up style implementation |
| `pointcloud_cog` | point cloud and COG sources, cache/memory report intent | no MapScene point-cloud adapter hardening |
| `urban_osm_city` | OSM-derived roads/water/buildings/labels provenance | no OSM querying or urban context helper |
| `luxembourg_rail_overlay` | rail vector lines, DEM, drape/stroke intent | no line stroke helper |

## Backward Compatibility Constraints

P1-1 must preserve:

- all G-002 GIS APIs and aliases
- all `python/forge3d/gis.py` wrappers and `python/forge3d/gis.pyi` stubs
- all PyO3 registrations in `src/py_module/functions/gis.rs` and `src/py_module/classes.rs`
- all `MapScene` and `SceneRecipe` public behavior
- all rendering APIs
- all existing example scripts
- all existing tests unless an implementation note explains an unavoidable doc-only expectation change

The manifest schema must not:

- replace `SceneRecipe` or `MapScene`
- move rendering behavior into manifests
- introduce a new rendering engine path
- import optional GIS backend libraries at runtime
- fetch data, query OSM, use slippy tiles, fetch Terrarium, or create hidden caches
- add shaders
- add visual golden images
- add non-fixture examples

## Non-Goals

- Raster/vector/terrain alignment diagnostics implementation from P1-3.
- G-003 pass compositing or blend-mode implementation.
- Canonical tiny golden fixture selection from P1-4.
- Categorical or bivariate raster style implementation from P2-1.
- Line stroke/drape helper implementation from P2-2.
- Map plate helper implementation from P2-3.
- MapScene adapter hardening from P2-4.
- Urban context layer helper implementation from P2-5.
- Recipe-level visual goldens from P3-1.
- Video/temporal recipe support for `wildfire_smoke`, `satellite_timelapse`, `osm_city_flood_daycycle`, and `humanity_globe_video`.
- Support matrices per recipe from P3-3.
- Gallery sidecar metadata from P3-4.
- New shader/rendering behavior.
- New remote data fetching or cache behavior.
- New OSM query behavior.
- New routing, travel-time, or network analysis.
- New examples that depend on external services.

## T-Recipe-Manifest Test Plan

Create `tests/test_recipe_manifest.py` with these focused tests:

- [ ] Construct a valid manifest for `terrain_demo`.
- [ ] Construct a valid manifest for `landcover_esri_terrain_viewer`.
- [ ] Construct a valid manifest for `climate_bivariate`.
- [ ] Construct a valid manifest for `terrain_label`.
- [ ] Construct a valid manifest for `hydrology_river`.
- [ ] Construct a valid manifest for `mapscene_showcases`.
- [ ] Deterministic serialization round-trip returns the same dict and JSON text.
- [ ] Missing required field emits `recipe_manifest_missing_field`.
- [ ] Invalid status emits `recipe_manifest_invalid_status`.
- [ ] Unknown recipe family emits `recipe_manifest_unknown_family`.
- [ ] Missing source example emits `recipe_manifest_missing_source`.
- [ ] Unsupported layer emits `recipe_manifest_unsupported_layer`.
- [ ] MapScene-compatible manifest reports compatibility without replacing or importing `MapScene`.
- [ ] Validation requires no network access.
- [ ] Validation requires no optional Python GIS backend libraries.
- [ ] Validation requires no runtime rendering.

Contract/API tests:

- [ ] Do not modify `tests/test_api_contracts.py` if `RecipeManifest` is not exported from `forge3d.__all__`.
- [ ] If top-level exports are added, add only the new public classes/functions to `tests/test_api_contracts.py`.
- [ ] If `python/forge3d/__init__.py` is touched, mirror typing in `python/forge3d/__init__.pyi`.
- [ ] If GIS-facing imports are touched, add a no-Python-GIS-runtime-backend guard test.

## Implementation Checklist

### Planning

- [ ] Reconfirm `origin/main` contains G-002 and MapScene evidence before runtime work.
- [ ] Re-read `docs/carto-engine/golden-map-recipe-capability-audit.md` Sections 3, 5, 6, 7, 8, and 9.
- [ ] Keep this plan as the phase boundary; do not expand into later phases.

### Implementation

- [ ] Create `python/forge3d/recipe_manifest.py` with stdlib-only dataclasses, constants, conversion, deterministic serialization, and validation.
- [ ] Create `python/forge3d/recipe_manifest.pyi` with matching signatures.
- [ ] Create the six JSON fixtures under `tests/fixtures/recipe_manifests/`.
- [ ] Create `docs/carto-engine/recipe-family-manifest-schema.md`.
- [ ] Leave `python/forge3d/map_scene.py` unchanged unless a factual import boundary forces a tiny compatibility note.
- [ ] Leave Rust and PyO3 files unchanged.

### Tests

- [ ] Add `tests/test_recipe_manifest.py` with the T-recipe-manifest cases.
- [ ] Keep tests stdlib-only and rendering-free.
- [ ] Keep fixture paths repo-relative.

### Validation

- [ ] `git status --short`
- [ ] `git diff --name-status`
- [ ] `git diff --stat`
- [ ] `git diff`
- [ ] `git ls-files --others --exclude-standard`
- [ ] `python -m py_compile python/forge3d/map_scene.py`
- [ ] `python -m py_compile python/forge3d/__init__.py`
- [ ] `python -m py_compile python/forge3d/recipe_manifest.py`
- [ ] `python -m pytest tests/test_recipe_manifest.py -v`
- [ ] `python -m pytest tests/test_mapscene_recipe_contract.py -v`
- [ ] `python -m pytest tests/test_mapscene_support_status.py -v`
- [ ] `python -m pytest tests/test_api_contracts.py -v` only if public API exports are touched.
- [ ] `cargo fmt --check` only if Rust files are touched.
- [ ] `cargo check` only if Rust files are touched.
- [ ] Broader cargo/python tests only if touched files justify them.

### Review Bundle

- [ ] Generate a fresh temporary review bundle after validation.
- [ ] Include git status, diff name-status, diff stat, full diff, untracked files, validation logs, command metadata and exit codes, branch, merge base, current commit, timestamp, and cwd.
- [ ] Keep the bundle outside the repo or ignored.
- [ ] Do not stage, commit, or include the review bundle in the PR.

## Open Questions

None blocking. Conservative P1-1 decision: keep manifests as metadata-only JSON-backed dataclasses in a new module, avoid top-level exports, and use audit vocabulary unchanged.

# Feature Research: Map Assets and Bundle Round-Trip P1

**Feature directory**: `specs/005-map-assets-bundles-p1`  
**Date**: 2026-05-15  
**Scope**: implementation inventory before technical planning; no code implementation.

## Source Artifacts Inspected

- `docs/superpowers/plans/prd.md`
- `.specify/memory/constitution.md`
- `specs/005-map-assets-bundles-p1/spec.md`
- `docs/superpowers/state/current-context-pack.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/open-blockers.md`

## Commands Used For Inspection

```powershell
Get-ChildItem -Path specs -Directory | Select-Object -ExpandProperty Name
Get-Content -Path docs/superpowers/plans/prd.md
Get-Content -Path .specify/memory/constitution.md
Get-ChildItem -Path specs -Recurse -Filter spec.md | Select-Object -ExpandProperty FullName
git status --short --branch
Get-Content -Path docs/superpowers/state/current-context-pack.md
Get-Content -Path docs/superpowers/state/implementation-ledger.md
Get-Content -Path docs/superpowers/state/open-blockers.md
Get-Content -Path docs/superpowers/state/requirements-verification-matrix.md
Get-Content -Path specs/005-map-assets-bundles-p1/spec.md
if (Test-Path specs/005-map-assets-bundles-p1/research.md) { Get-Content -Path specs/005-map-assets-bundles-p1/research.md } else { Write-Output '<missing>' }
rg --files | rg "(buildings|tiles3d|label|bundle|map_scene|mapscene|style|crs|terrain|pointcloud|font|atlas|fixture|test|example|docs/)"
rg -n "^(class|def) |^    def |^async def |^@dataclass|^class " python/forge3d/buildings.py python/forge3d/tiles3d.py python/forge3d/bundle.py python/forge3d/crs.py python/forge3d/style.py python/forge3d/style_expressions.py python/forge3d/viewer.py python/forge3d/viewer_ipc.py python/forge3d/pointcloud.py python/forge3d/map_plate.py python/forge3d/export.py python/forge3d/__init__.py python/forge3d/__init__.pyi
rg -n "\b(MapScene|SceneRecipe|LabelPlan|ValidationReport|Diagnostic|LabelLayer|Tiles3DLayer|BuildingLayer)\b" python src tests docs examples specs -g "!docs/_build/**"
rg -n "(fallback|placeholder|stub|no.?op|Pro|pro_|license|gated|HAS_|hasattr|warnings.warn|return \[\]|return None|return \{\}|NotImplemented|unsupported|experimental)" python/forge3d/buildings.py python/forge3d/tiles3d.py python/forge3d/bundle.py python/forge3d/viewer_ipc.py src/tiles3d src/labels src/bundle src/import tests/test_pro_gating.py tests/test_buildings_cityjson.py tests/test_3dtiles_parse.py
rg --files tests assets examples docs src python | rg "(3dtiles|tiles3d|building|cityjson|geojson|label|font|atlas|bundle|crs|style|mapscene|map_scene|pointcloud)"
rg -n -C 3 "placeholder|_check_pro_access|import_osm_buildings_from_geojson_py|parse_cityjson_py|add_buildings_3dtiles|return BuildingLayer" python/forge3d/buildings.py
rg -n -C 3 "NotImplementedError|decode_b3dm|decode_pnts|load_tileset|cache_stats|SseParams|Tiles3dRenderer" python/forge3d/tiles3d.py
rg -n -C 2 "class BundleManifest|class SceneBaseState|class SceneState|class LoadedBundle|def save_bundle|def load_bundle|scene_state|labels|vector_overlays|raster_overlays|_check_pro_access|missing|exists" python/forge3d/bundle.py
rg -n -C 2 "class EvalContext|def evaluate|_eval_get|_eval_concat|_eval_coalesce|_eval_downcase|_eval_upcase|def load_style|def apply_style|class FontStack|class GlyphRange|get_required_glyph_ranges|layout_to_label_style" python/forge3d/style_expressions.py python/forge3d/style.py
rg -n -C 2 "pub struct LabelLayer|pub struct LabelFeature|pub enum FeatureGeometry|pub enum PlacementStrategy|generate_labels|features_from_points|features_from_lines|label_layer_from_mapbox_style|TypographySettings|KerningTable|apply_text_case|MsdfAtlas|load_from_files|Space fallback|basic ASCII" src/labels/layer.rs src/labels/typography.rs src/labels/atlas.rs src/labels/py_bindings.rs src/labels/mod.rs src/labels/types.rs
rg -n -C 2 "register_py_functions|register_py_classes|import_osm_buildings_from_geojson_py|import_osm_buildings_extrude_py|material_from_tags_py|material_from_name_py|parse_cityjson_py|PyLabelStyle|PyLabelFlags" src/py_module/functions.rs src/py_module/functions/io_import.rs src/py_module/classes.rs
rg -n "TTF|OTF|fontdue|ab_glyph|rusttype|ttf|otf|generate.*atlas|atlas generation|msdf" python src tests docs assets -g "!docs/_build/**"
```

## Phase 0 Planning Decisions

- Decision: P1 implementation extends the P0 product API paths created by features `001` through `004`; it must not create a parallel P1-only scene model.
  Rationale: The PRD product contract centers `MapScene`, `LabelPlan`, `ValidationReport`, `Diagnostic`, and `Bundle`.
  Alternatives considered: Implement standalone P1 wrappers, rejected because it would fragment the typed workflow.

- Decision: Bundles store both source labels and compiled `LabelPlan` payloads where available.
  Rationale: This preserves review intent while allowing stale/missing/replay gaps to be diagnosed.
  Alternatives considered: Store only sources and recompile, rejected because exact review reproduction could drift; store only compiled plans, rejected because source review would be incomplete.

- Decision: Diagnostic capability does not upgrade unavailable building or 3D Tiles rendering to `supported`.
  Rationale: Support status must describe real render/ingestion capability, not merely the ability to report a limitation.
  Alternatives considered: Treat diagnosed paths as supported validation, rejected as potentially misleading in support matrices.

## 1. Existing Modules, Classes, And Functions Relevant To This Feature

### Python modules

- `python/forge3d/buildings.py`
  - `BuildingMaterial`, `Building`, and `BuildingLayer` dataclasses.
  - `BuildingLayer.building_count`, `total_vertices`, `total_triangles`, `max_lod`, and `bounds`.
  - `infer_roof_type`, `material_from_tags`, and `material_from_name`.
  - `add_buildings`, `add_buildings_cityjson`, and `add_buildings_3dtiles`.
  - Native probes call `_NATIVE.import_osm_buildings_from_geojson_py`, `_NATIVE.parse_cityjson_py`, `_NATIVE.infer_roof_type_py`, `_NATIVE.material_from_tags_py`, and `_NATIVE.material_from_name_py` when present.

- `python/forge3d/tiles3d.py`
  - `BoundingVolume`, `TileContent`, `Tile`, `Tileset`, `VisibleTile`, `SseParams`, and `Tiles3dRenderer`.
  - `load_tileset`, `decode_b3dm`, and `decode_pnts`.
  - `Tiles3dRenderer.get_visible_tiles`, `compute_sse`, and `cache_stats`.
  - No public `Tiles3DLayer` class found. `Tiles3dRenderer` is a lower-level parser/traversal helper.

- `python/forge3d/bundle.py`
  - `TerrainMeta`, `CameraBookmark`, `RasterOverlaySpec`, `SceneBaseState`, `ReviewLayer`, `SceneVariant`, `SceneState`, `BundleManifest`, and `LoadedBundle`.
  - `save_bundle`, `load_bundle`, and `is_bundle`.
  - `SceneBaseState` and `ReviewLayer` already carry `raster_overlays`, `vector_overlays`, `labels`, and `scatter_batches`.
  - No `MapScene.save_bundle()` method found; bundle API is module-level and `ViewerHandle.load_bundle()` exists separately.

- `python/forge3d/style.py`
  - `VectorStyle`, `LabelStyle`, `PaintProps`, `LayoutProps`, `StyleLayer`, `StyleSpec`, `SpriteEntry`, `SpriteAtlas`, `GlyphRange`, and `FontStack`.
  - `load_style`, `parse_style`, `paint_to_vector_style`, `layout_to_label_style`, `layer_to_vector_style`, `layer_to_label_style`, `apply_style`, `load_sprite_atlas`, `parse_glyph_url`, and `get_required_glyph_ranges`.
  - `load_style` and `apply_style` are Pro-gated.

- `python/forge3d/style_expressions.py`
  - `EvalContext` and `evaluate`.
  - Existing expression support includes `get`, `concat`, `coalesce`, `downcase`, and `upcase`, which are relevant to P1 label text expressions.
  - Also includes broader Mapbox expression helpers such as comparison, control flow, math, and color/number evaluation.

- `python/forge3d/crs.py`
  - `proj_available`, `transform_coords`, `reproject_geom`, `parse_crs_from_wkt`, `crs_to_epsg`, `get_crs_from_rasterio`, and `get_crs_from_geopandas`.
  - This is the existing CRS utility surface for P1-R1 automatic CRS transforms.

- `python/forge3d/viewer_ipc.py`
  - Raw IPC functions exist for labels and review bundles: `add_label`, `add_line_label`, `add_curved_label`, `add_callout`, `remove_label`, `clear_labels`, `set_labels_enabled`, `load_label_atlas`, `set_label_typography`, `set_declutter_algorithm`, `save_bundle`, and `load_bundle`.
  - This is relevant as existing substrate, but the P1 feature should not rely on raw IPC as the public typed workflow.

- `python/forge3d/viewer.py`
  - `ViewerHandle.load_bundle`, `load_overlay`, `load_point_cloud`, `list_scene_variants`, `list_review_layers`, `apply_scene_variant`, `set_review_layer_visible`, and `snapshot`.
  - No `ViewerHandle.add_label` high-level wrapper found in the inspected output; existing label examples use `viewer_ipc`.

- `python/forge3d/__init__.py`
  - Re-exports `save_bundle`, `load_bundle`, `BuildingLayer`, `add_buildings`, `add_buildings_cityjson`, and `add_buildings_3dtiles`.
  - No `MapScene`, `SceneRecipe`, `LabelPlan`, `ValidationReport`, `Diagnostic`, `LabelLayer`, or `Tiles3DLayer` exports found.

- `python/forge3d/__init__.pyi`
  - Contains native `LabelFlags` and `LabelStyle` stubs.
  - Contains `Scene` native text methods including `set_native_text_atlas` and text mesh methods, but no P1 map-scene product classes found.

### Rust/native modules

- `src/labels/layer.rs`
  - Existing native substrate includes `LabelFeature`, `FeatureGeometry`, `PlacementStrategy`, `LabelLayerConfig`, `GeneratedLabel`, `LabelPlacementType`, and `LabelLayer`.
  - Functions include `LabelLayer::new`, `add_feature`, `add_features`, `generate_labels`, `feature_count`, `label_count`, `features_from_points`, `features_from_lines`, `label_layer_from_mapbox_style`, and `apply_mapbox_text_style`.
  - This looks relevant to `LabelLayer` data ingestion, but no PyO3 registration for this `LabelLayer` was found during inspection.

- `src/labels/typography.rs`
  - `TypographySettings`, `KerningTable`, `compute_advances_with_typography`, `TextCase`, and `apply_text_case`.
  - Relevant to kerning, tracking, line-height, and casing transforms.

- `src/labels/atlas.rs`
  - `MsdfAtlas`, `MsdfAtlas::load`, `MsdfAtlas::load_from_files`, `get_glyph`, `measure_text`, and `layout_text`.
  - Contains a fallback path that creates a basic ASCII set if parsed glyphs are empty.

- `src/labels/py_bindings.rs`
  - PyO3 classes `PyLabelFlags` and `PyLabelStyle`.
  - Registered in `src/py_module/classes.rs` via `m.add_class::<crate::labels::py_bindings::PyLabelStyle>()?` and `m.add_class::<crate::labels::py_bindings::PyLabelFlags>()?`.

- `src/import/osm_buildings.rs`
  - `RoofType`, `infer_roof_type`, `infer_roof_type_from_json`, `import_osm_buildings_extrude_py`, `import_osm_buildings_from_geojson_py`, and `infer_roof_type_py`.
  - Python registration occurs in `src/py_module/functions/io_import.rs`.

- `src/import/building_materials.rs`
  - `BuildingMaterial`, material constants, `material_from_tags`, `material_from_name`, `roof_material_from_tags`, `parse_css_color`, `material_from_tags_py`, and `material_from_name_py`.
  - Scalar material support exists. No textured building material path was identified here.

- `src/import/cityjson/`
  - `parser.rs`: `parse_cityjson`, CRS/transform/extent parsing, building parsing, height field extraction, material/roof inference.
  - `types.rs`: `BuildingGeom`, `CityJsonMeta`.
  - `geometry.rs`: `parse_geometry`, `compute_normals`.
  - `bindings.rs`: `parse_cityjson_py`.
  - Python registration occurs in `src/py_module/functions/io_import.rs`.

- `src/tiles3d/`
  - `tileset.rs`: `TilesetAsset`, `TilesetJson`, `Tileset`, `load`, `from_json`, `resolve_uri`, `tile_count`, `max_depth`, `has_required_extensions`, and `required_extensions`.
  - `tile.rs`: `Tile`, `TileContent`, `TileRefine`.
  - `bounds.rs`: `BoundingVolume` variants and bounding helpers.
  - `sse.rs`: `SseParams`, `compute_sse`, `should_refine`, and distance helpers.
  - `traversal.rs`: `TilesetTraverser`, `VisibleTile`, and `TraversalStats`.
  - `b3dm.rs`: `decode_b3dm`, `load_b3dm`, and GLB mesh parsing helpers.
  - `pnts.rs`: `decode_pnts`, `load_pnts`, and point payload extraction.
  - `renderer.rs`: `Tiles3dRenderer`, `TileContent`, `MeshData`, `PointData`, `cache_stats`, `BuildingRenderData`, `prepare_buildings`, and `get_visible_buildings`.

- `src/bundle/`
  - `manifest.rs`: Rust `BundleManifest`, `TerrainMeta`, and `CameraBookmark` plus checksum helpers.
  - `mod.rs`: `is_bundle` and `manifest_path`.
  - Python bundle behavior appears richer than this Rust manifest substrate.

- `src/core/text_mesh/` and `src/scene/py_api/native_text.rs`
  - `src/core/text_mesh/builder.rs` uses `ttf_parser` to build text mesh geometry from font bytes.
  - `src/scene/py_api/native_text.rs` exposes native text atlas upload methods on `Scene`.
  - These may be relevant to font/typography inventory, but they are not obviously wired to P1 `LabelLayer` atlas generation.

## 2. Existing Tests And Fixtures Relevant To This Feature

### Tests

- `tests/test_pro_gating.py`
  - Verifies Pro gates for `add_buildings`, `add_buildings_cityjson`, `add_buildings_3dtiles`, `load_style`, `apply_style`, `save_bundle`, and `load_bundle`.

- `tests/test_buildings_extrude.py`
  - Tests `Building`, `BuildingLayer`, GeoJSON building import, height keys, missing file behavior, multipolygon input, and vertex count.
  - Uses `pytestmark = pytest.mark.usefixtures("pro_license")`.

- `tests/test_buildings_cityjson.py`
  - Tests CityJSON import, CRS metadata, multiple buildings, LOD selection, building parts, invalid input, transform application, and the sample CityJSON dataset.
  - Contains comments acknowledging that native parser extracts height where fallback may not.
  - Uses `pytestmark = pytest.mark.usefixtures("pro_license")`.

- `tests/test_buildings_materials.py`
  - Tests building material inference from names and tags plus dataclass serialization.

- `tests/test_buildings_roof.py`
  - Tests roof type inference.

- `tests/test_3dtiles_parse.py`
  - Tests Python `tiles3d` parsing for bounding volumes, tile trees, tileset loading, URI resolution, tile count, and max depth.

- `tests/test_3dtiles_sse.py`
  - Tests `SseParams`, SSE behavior, traversal, max depth, monotonicity, and initial cache stats.

- `tests/test_bundle_roundtrip.py`
  - Tests `BundleManifest`, minimal save/load, presets, camera bookmarks, checksum verification, scene state v2 review layers/assets, effective state behavior, invalid variants/layers, and legacy v1 synthesis.
  - Uses `pytestmark = pytest.mark.usefixtures("pro_license")`.

- `tests/test_bundle_render.py`
  - Tests bundle manifest version, preset round-trip, and image hash verification.
  - Uses `pytestmark = pytest.mark.usefixtures("pro_license")`.

- `tests/test_bundle_cli.py`
  - Tests terrain demo CLI flags and bundle directory structure/version behavior.
  - Uses `pytestmark = pytest.mark.usefixtures("pro_license")`.

- `tests/test_style_parser.py`, `tests/test_style_render.py`, `tests/test_style_visual.py`, `tests/test_style_pixel_diff.py`, and `tests/test_mapbox_streets_fixture.py`
  - Relevant to style parsing/application and Mapbox streets fixture behavior.
  - Style tests are Pro-gated where style loading/application is used.

- `tests/test_crs_reproject.py` and `tests/test_crs_auto.py`
  - Relevant to CRS utility behavior and terrain CRS metadata.

- `tests/test_labels_pybindings.py` and `tests/test_api_contracts.py::TestLabelBindings`
  - Verify native `LabelStyle` and `LabelFlags` PyO3 classes.
  - Do not appear to cover native `LabelLayer` ingestion.

- `tests/conftest.py`
  - Defines `pro_license` fixture using signed test key support.

### Fixtures and assets

- `assets/fonts/default_atlas.json`
- `assets/fonts/default_atlas.png`
- `assets/geojson/sample_buildings.city.json`
- `assets/geojson/mount_fuji_buildings.geojson`
- `assets/geojson/10-270-592.city.json`
- `python/forge3d/data/sample_boundaries.geojson`
- `tests/fixtures/mapbox_streets_v8.json`

No checked-in `.b3dm`, `.pnts`, `.glb`, or `tileset.json` fixture file was found by the file search. Existing 3D Tiles tests synthesize `tileset.json` data in temporary files.

## 3. Existing Docs And Examples Relevant To This Feature

- `docs/guides/feature_map.md`
  - Lists vector overlays/labels via raw IPC, Pro workflows for buildings and scene bundles, and production-oriented scene assets including `forge3d.buildings`, `forge3d.style`, `forge3d.bundle`, `forge3d.map_plate`, and `forge3d.export`.

- `docs/guides/data_and_scene_workflows.md`
  - Documents viewer overlays/labels through `forge3d.viewer_ipc`, building workflows through `forge3d.buildings`, lower-level 3D Tiles through `forge3d.tiles3d`, and bundle save/load workflows.

- `docs/guides/output_and_integration.md`
  - Mentions `save_bundle`, `load_bundle`, and `ViewerHandle.load_bundle`.

- `docs/start/architecture.md`
  - Describes `forge3d.pointcloud`, `forge3d.buildings`, `forge3d.tiles3d`, `forge3d.style`, `forge3d.map_plate`, `forge3d.export`, and `forge3d.bundle`; notes vector overlays and labels use raw IPC or `forge3d.viewer_ipc`.

- `docs/examples/index.md`
  - Lists `fuji_labels_demo.py` for labels and `buildings_viewer_interactive.py` for building imports from GeoJSON, CityJSON, and 3D Tiles-backed sources.

- `docs/gallery/02-mount-fuji-labels.md`
  - Demonstrates label commands through viewer IPC.

- `docs/gallery/05-3d-buildings.md`
  - Marks buildings as Pro and uses `forge3d.add_buildings_cityjson("assets/geojson/10-270-592.city.json")`.

- `docs/tutorials/gis-track/04-3d-buildings.md`
  - Marks the tutorial as Pro and uses `add_buildings_cityjson`.
  - Explicitly says vector overlay IPC is not for volumetric CityJSON building meshes.

- `docs/tutorials/python-track/04-scene-bundles.md`
  - Marks bundle tutorial as Pro.
  - Shows `f3d.save_bundle`, `f3d.load_bundle`, and `ViewerHandle.load_bundle`.

- `docs/api/api_reference.rst`
  - Includes automodule entries for `forge3d.tiles3d`, `forge3d.buildings`, and `forge3d.bundle`.

- `examples/fuji_labels_demo.py`
  - Uses raw IPC functions for label atlas, point labels, line labels, curved labels, callouts, typography, zoom, and max visible labels.

- `examples/buildings_viewer_interactive.py`
  - Imports `BuildingLayer` and converts building layers to vector overlay-like preview data.

- `examples/style_viewer_interactive.py`
  - Uses style parsing/application paths.

## 4. Native/Rust Paths Relevant To This Feature

- `src/labels/layer.rs`
- `src/labels/typography.rs`
- `src/labels/atlas.rs`
- `src/labels/mod.rs`
- `src/labels/types.rs`
- `src/labels/line_label.rs`
- `src/labels/curved.rs`
- `src/labels/callout.rs`
- `src/labels/leader.rs`
- `src/labels/declutter.rs`
- `src/labels/collision.rs`
- `src/labels/rtree.rs`
- `src/labels/projection.rs`
- `src/labels/py_bindings.rs`
- `src/viewer/cmd/labels_command.rs`
- `src/viewer/ipc/protocol/translate/labels.rs`
- `src/viewer/state/labels.rs`
- `src/viewer/scene_review.rs`
- `src/import/osm_buildings.rs`
- `src/import/building_materials.rs`
- `src/import/cityjson.rs`
- `src/import/cityjson/parser.rs`
- `src/import/cityjson/types.rs`
- `src/import/cityjson/geometry.rs`
- `src/import/cityjson/bindings.rs`
- `src/tiles3d/mod.rs`
- `src/tiles3d/tileset.rs`
- `src/tiles3d/tile.rs`
- `src/tiles3d/bounds.rs`
- `src/tiles3d/sse.rs`
- `src/tiles3d/traversal.rs`
- `src/tiles3d/b3dm.rs`
- `src/tiles3d/pnts.rs`
- `src/tiles3d/renderer.rs`
- `src/tiles3d/error.rs`
- `src/bundle/mod.rs`
- `src/bundle/manifest.rs`
- `src/py_module/functions/io_import.rs`
- `src/py_module/classes.rs`
- `src/core/text_mesh/builder.rs` (uncertain relevance to label atlas generation)
- `src/scene/py_api/native_text.rs` (uncertain relevance to label atlas generation)

## 5. Python Public API Paths Relevant To This Feature

- `python/forge3d/__init__.py`
  - Public exports include bundle and building APIs.
  - No public product-layer `MapScene`, `SceneRecipe`, `LabelPlan`, `ValidationReport`, `Diagnostic`, `LabelLayer`, or `Tiles3DLayer` found.

- `python/forge3d/__init__.pyi`
  - Public type stubs for native `Scene`, text methods, and native `LabelStyle`/`LabelFlags`.
  - No stubs for `MapScene`, `SceneRecipe`, `LabelPlan`, `ValidationReport`, `Diagnostic`, `LabelLayer`, or `Tiles3DLayer` found.

- `python/forge3d/buildings.py`
  - Public `BuildingLayer` exists today, but it is a dataclass layer returned by module-level building import helpers, not a `MapScene` layer contract.

- `python/forge3d/tiles3d.py`
  - Public lower-level `Tileset` and `Tiles3dRenderer` helpers exist.
  - Public `Tiles3DLayer` was not found.

- `python/forge3d/bundle.py`
  - Public module-level `save_bundle`, `load_bundle`, and manifest/state dataclasses exist.
  - Bundle payload supports generic `labels`, `vector_overlays`, and `raster_overlays`, but no typed `LabelPlan` payload was found.

- `python/forge3d/crs.py`
  - Existing CRS transform/reprojection functions.

- `python/forge3d/style.py` and `python/forge3d/style_expressions.py`
  - Existing style and expression utilities relevant to future `LabelLayer.from_style_layer` or expression evaluation.

- `python/forge3d/viewer.py` and `python/forge3d/viewer_ipc.py`
  - Existing raw IPC and viewer integration paths.
  - These are existing substrate, not the target typed P1 public workflow.

## 6. Suspected Placeholder, Fallback, Or No-Op Paths

- `python/forge3d/buildings.py`
  - `add_buildings` fallback branch creates `Building` entries with `positions=np.zeros((0, 3))`, `normals=np.zeros((0, 3))`, and `indices=np.zeros(0)` when native GeoJSON import is unavailable.
  - `add_buildings_cityjson` fallback branch creates placeholder zero-geometry buildings when native CityJSON parsing is unavailable.
  - `add_buildings_3dtiles` returns a `BuildingLayer` with `buildings=[]` and metadata only; comments say actual tile loading happens during rendering.
  - These are direct candidates for `placeholder_fallback` diagnostics.

- `python/forge3d/tiles3d.py`
  - `decode_b3dm` parses headers but raises `NotImplementedError` for glTF mesh extraction.
  - Existing Python `Tiles3dRenderer` cache stats are in-memory counters; no public `MapScene` render integration was found.

- `src/labels/atlas.rs`
  - If no glyphs are parsed, `MsdfAtlas` creates a basic ASCII fallback metrics set.
  - This may keep layout functional while hiding atlas parsing failure unless diagnosed.

- `python/forge3d/bundle.py`
  - `load_bundle` ignores missing terrain files by leaving `dem_file` as `None` if the manifest-referenced file is absent.
  - `load_bundle` only verifies checksums for files that exist; missing checked files are not obviously reported as structured diagnostics.
  - No typed `missing_external_asset` diagnostic path was found.

- `python/forge3d/viewer_ipc.py`
  - `set_label_typography` and `set_declutter_algorithm` send commands through IPC. Whether the native side mutates real label manager state is uncertain here and is already identified as a P0 risk in the PRD/context artifacts.

- `src/labels/layer.rs`
  - Native `LabelLayer::generate_labels` exists, but product-level accepted/rejected `LabelPlan` behavior and Python exposure were not found.

- `MapScene`, `LabelPlan`, `ValidationReport`, and `Diagnostic`
  - Found only in specs/docs/state artifacts during this inspection. No implementation paths found under `python/forge3d`, `src`, `tests`, or `examples`.

## 7. Pro-Gated Paths

- `python/forge3d/buildings.py`
  - `_check_pro_access("GeoJSON building import")`
  - `_check_pro_access("CityJSON building import")`
  - `_check_pro_access("3D Tiles building import")`

- `python/forge3d/style.py`
  - `_check_pro_access("Mapbox style loading")`
  - `_check_pro_access("Mapbox style application")`

- `python/forge3d/bundle.py`
  - `_check_pro_access("Scene bundle save")`
  - `_check_pro_access("Scene bundle load")`

- `python/forge3d/map_plate.py`
  - `_check_pro_access("Map plate composition")`

- `python/forge3d/export.py`
  - `_check_pro_access("SVG export")`
  - `_check_pro_access("PDF export")`

- `tests/test_pro_gating.py`
  - Contains regression coverage for the Pro-gated paths above.

- `tests/conftest.py`
  - Defines `pro_license` fixture used by building, style, bundle, export, and map plate tests.

## 8. Unknowns That Need Verification

- `specs/005-map-assets-bundles-p1/plan.md` is missing. Technical planning has not yet started for this feature.

- P0 dependencies are not implemented in the current repo state:
  - No `MapScene`, `SceneRecipe`, `LabelPlan`, `ValidationReport`, or `Diagnostic` implementation found outside specs/docs/state.
  - This feature assumes features `001` through `004` provide stable product contracts before P1 implementation depends on them.

- Native `src/labels/layer.rs` has a real Rust `LabelLayer`, but Python exposure was not found. It is uncertain whether P1 should wrap this Rust type, implement Python-side ingestion, or wait for feature `003`/`004` product contracts.

- Runtime/build-time TTF/OTF atlas generation was not found for labels. `src/core/text_mesh/builder.rs` uses `ttf_parser` for text mesh geometry, and native text atlas upload exists on `Scene`, but a label atlas generator from TTF/OTF was not identified.

- 3D Tiles native Rust support is stronger than Python `decode_b3dm`, but no PyO3 registration for Rust `tiles3d` decode/traversal APIs was identified in the inspected registration files. Need verify whether 3D Tiles public Python integration should bind Rust paths or extend existing Python helpers.

- Building native paths are registered in `src/py_module/functions/io_import.rs`, but actual availability in installed wheels depends on build features/packaging. Planning must preserve `Pro-gated` and `placeholder/fallback` diagnostics.

- No persistent `LabelPlan` bundle schema was found. Existing `SceneBaseState.labels` is a generic list of dicts and does not distinguish source labels from compiled plans.

- Existing bundle load behavior does not obviously emit structured diagnostics for missing assets; missing-asset policy and serialization shape need design after `Diagnostic` exists.

- Existing style expression support includes the required P1 operations, but missing-field diagnostics are not present in the observed functions. `get` returns `None` for missing values, so P1 must verify how to surface `missing_label_field`.

- Existing docs mark buildings and bundles as Pro but do not yet provide the P1 support matrices requested by the PRD. Planning must decide whether this feature adds docs or depends on feature `001` support-matrix docs.

- Existing 3D Tiles tests synthesize tilesets but do not use checked-in B3DM/PNTS fixtures. P1 may need new fixtures or explicit diagnostic-only tests for unsupported payloads.

- `python/forge3d/buildings.py:add_buildings_3dtiles` returns empty metadata-only layers. It is uncertain whether any viewer/native render path consumes this later; verify before claiming renderability.

- Whether line/curved label ingestion can be treated as supported for P1 is uncertain. PRD/context classify line/curved label behavior as experimental until feature `002` proves renderability or typed unsupported/experimental diagnostics.

# Feature Research: MapScene MVP

**Feature directory**: `specs/004-mapscene-mvp`
**Date**: 2026-05-15
**Scope**: feature-specific implementation inventory before technical planning; no code implementation.

## Source Artifacts Inspected

- `docs/superpowers/plans/prd.md`
- `.specify/memory/constitution.md`
- `specs/004-mapscene-mvp/spec.md`
- `docs/superpowers/state/current-context-pack.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/open-blockers.md`
- `AGENTS.md`
- `git status --short --branch`

## Commands Used For Inspection

```powershell
Get-Content -Path docs/superpowers/plans/prd.md
Get-Content -Path .specify/memory/constitution.md
Get-Content -Path specs/004-mapscene-mvp/spec.md
if (Test-Path specs/004-mapscene-mvp/research.md) { Get-Content -Path specs/004-mapscene-mvp/research.md } else { Write-Output '<missing>' }
Get-Content -Path docs/superpowers/state/current-context-pack.md
Get-Content -Path docs/superpowers/state/implementation-ledger.md
Get-Content -Path docs/superpowers/state/requirements-verification-matrix.md
Get-Content -Path docs/superpowers/state/open-blockers.md
git status --short --branch
rg -n "\b(MapScene|SceneRecipe|ValidationReport|Diagnostic|LabelPlan|TerrainSource|RasterOverlay|VectorOverlay|LabelLayer|PointCloudLayer|BuildingLayer|MapFurnitureLayer|OutputSpec|OrbitCamera|LightingPreset|ReviewBundle)\b" -S .
rg --files -g "*.py" -g "*.pyi" -g "*.rs" -g "*.md" -g "*.json" -g "*.toml" | rg "(forge3d|src|tests|examples|docs|specs|assets|fixtures)"
rg -n "def (render|render_.*|save_bundle|load_bundle|validate|load_overlay|load_point_cloud|add_.*label|add_vector_overlay)|class .*Scene|class .*Renderer|class .*Bundle|struct .*Scene|pub struct .*Scene|pyclass|add_class|add_function|render_rgba|snapshot|png|PNG" python src tests examples docs -S
rg -n "\b(MapScene|SceneRecipe|ValidationReport|Diagnostic|LabelPlan|TerrainSource|RasterOverlay|VectorOverlay|LabelLayer|PointCloudLayer|BuildingLayer|MapFurnitureLayer|OutputSpec|OrbitCamera|LightingPreset|ReviewBundle)\b" python src tests examples docs --glob '!docs/_build/**' -S
rg -n "^(class|def|@dataclass)|^    def |__all__|_check_pro_access|warnings.warn|NotImplemented|fallback|placeholder|np.zeros|return \[\]|return None|return \{\}" python/forge3d/__init__.py python/forge3d/__init__.pyi python/forge3d/viewer.py python/forge3d/viewer.pyi python/forge3d/viewer_ipc.py python/forge3d/bundle.py python/forge3d/buildings.py python/forge3d/tiles3d.py python/forge3d/style.py python/forge3d/crs.py python/forge3d/pointcloud.py python/forge3d/map_plate.py python/forge3d/lighting.py python/forge3d/offline.py python/forge3d/terrain_params.py
rg -n "^(pub struct|pub enum|impl |pub fn|fn |#\[pyclass\]|#\[pymethods\]|#\[pyfunction\])|m\.add_class|m\.add_function|register_py|render_rgba|render_png|snapshot|save_bundle|load_bundle|label|pointcloud|building|tiles3d|overlay|diagnostic|memory" src/lib.rs src/py_module src/scene src/terrain src/viewer src/labels src/bundle src/import src/tiles3d src/pointcloud src/render src/core src/lighting -S
rg --files tests assets examples docs python src | rg -i "(mapscene|map_scene|scene.*recipe|validation.*report|diagnostic|terrain|overlay|vector|style|label|bundle|building|pointcloud|3dtiles|tiles3d|crs|memory|gpu|map_plate|lighting|camera|snapshot|offline|fixture|golden|png)"
rg -n "def test_|class Test|pytestmark|pro_license|render_png|render_rgba|snapshot|bundle|overlay|label|building|pointcloud|style|crs_mismatch|memory|diagnostic|MapScene|SceneRecipe|ValidationReport" tests/test_terrain_renderer.py tests/test_terrain_overlay_stack.py tests/test_vector_overlay_rendering.py tests/test_vector_overlay_drape.py tests/test_style_parser.py tests/test_bundle_roundtrip.py tests/test_bundle_render.py tests/test_buildings_extrude.py tests/test_buildings_cityjson.py tests/test_pointcloud_gpu_integration.py tests/test_pointcloud_lod.py tests/test_viewer_ipc.py tests/test_labels_pybindings.py tests/test_api_contracts.py tests/test_crs_reproject.py tests/test_crs_auto.py tests/test_pro_gating.py tests/test_lighting_preset.py tests/test_camera_rigs.py tests/test_map_plate_layout.py tests/test_3dtiles_parse.py tests/test_3dtiles_sse.py -S
rg -n "MapScene|SceneRecipe|ValidationReport|Diagnostic|LabelPlan|MapScene|save_bundle|load_bundle|snapshot|load_overlay|load_point_cloud|viewer_ipc|raw IPC|building|3D Tiles|style|labels|point cloud|terrain|offline|MapPlate|bundle|Pro|unsupported|fallback|experimental" docs/guides docs/tutorials docs/gallery docs/start docs/viewer docs/terrain examples README.md --glob '!docs/_build/**' -S
rg -n "render_png|render_rgba|raster_overlay|native_text|text_atlas|add_text|set_native|Overlay|Scene\(" src/scene python/forge3d/__init__.pyi tests/test_api_contracts.py -S
rg -n "memory_metrics|global_memory_metrics|MemoryReport|cache_stats|gpu_byte_size|estimated|memory_budget|utilization|report" python/forge3d src tests -S
rg -n "Diagnostic|ValidationReport|validate\(|validate_for_terrain|MeshValidationReport|Validation|report_device|device_probe|global_memory_metrics" python/forge3d src tests scripts -S
if (Test-Path specs/004-mapscene-mvp/plan.md) { Get-Content -Path specs/004-mapscene-mvp/plan.md } else { Write-Output 'NO_PLAN_MD' }
Get-ChildItem -Path specs/004-mapscene-mvp -Force | Select-Object Name,Length,LastWriteTime
rg -n "MapScene|SceneRecipe|ValidationReport|Diagnostic|LabelPlan|MapScene.validate|MapScene.render|MapScene.save_bundle" specs/004-mapscene-mvp docs/superpowers/state -S
Get-Content -Path src/py_module/classes.rs
Get-Content -Path src/py_module/functions.rs
Get-Content -Path src/py_module/functions/io_import.rs
```

## Phase 0 Planning Decisions

- Decision: `MapScene` is planned as a top-level typed product API that wraps existing `Scene`, offscreen, viewer, terrain, style, label, building, point-cloud, CRS, and bundle substrate internally.
  Rationale: The product gap is scattered primitives; exposing raw IPC or low-level stitched calls would not satisfy PRD Section 21.
  Alternatives considered: Make users compose existing viewer/IPC helpers, rejected by P0-R4 and the no-raw-IPC MVP requirement.

- Decision: MVP render planning prefers the native/offscreen PNG path; viewer snapshot support is optional or diagnostic unless task inspection proves it is required and reliable.
  Rationale: P0 requires PNG output, not two render backends. The offscreen/native path better matches reproducible offline output.
  Alternatives considered: Require both offscreen and viewer snapshot in MVP, rejected as unnecessary scope expansion.

- Decision: No implicit CRS transforms in MVP; matching CRS or explicit compatible policy is required.
  Rationale: This is the clarified safe default and prevents silent geospatial misalignment.
  Alternatives considered: Auto-transform whenever CRS differs, rejected until minimum CRS transform support is defined and tested.

## 1. Existing Modules, Classes, And Functions Relevant To This Feature

### Product-level MapScene surface

- No implemented public `MapScene`, `SceneRecipe`, `TerrainSource`, `RasterOverlay`, `VectorOverlay`, product-level `LabelLayer`, `PointCloudLayer`, `MapFurnitureLayer`, `OutputSpec`, `ValidationReport`, product-level `Diagnostic`, or `ReviewBundle` class/function was found under `python/forge3d`, `src`, `tests`, `examples`, or non-built docs outside specs/state references.
- `specs/004-mapscene-mvp/plan.md` does not exist yet. Technical planning has not started.
- `python/forge3d/__init__.py` exports existing lower-level objects such as `Scene`, `TerrainRenderer`, `OverlayLayer`, `ViewerHandle`, `BuildingLayer`, `save_bundle`, `load_bundle`, style helpers, CRS helpers, and terrain/offline helpers. It does not export the MapScene MVP product classes above.
- `python/forge3d/__init__.pyi` stubs native `Scene`, `TerrainRenderer`, `OverlayLayer`, bundle imports, native text/raster overlay methods, and point-cloud bindings. It does not stub the MapScene MVP product classes above.

### Existing render and scene substrate

- `python/forge3d/viewer.py`
  - `ViewerHandle` provides current high-level viewer workflow helpers: `load_terrain`, `load_overlay`, `load_point_cloud`, `set_camera_lookat`, `set_orbit_camera`, `set_sun`, `set_ibl`, `set_z_scale`, `snapshot`, `load_bundle`, scene-review query methods, and raw `send_ipc`.
  - `snapshot()` writes PNG through the viewer IPC path and waits for file creation.
  - `send_ipc()` remains the escape hatch for unsupported or advanced commands; this is relevant because the feature requires canonical MVP workflows without raw IPC.

- `python/forge3d/viewer_ipc.py`
  - Existing raw command helpers cover labels, vector overlays, scene review state, bundle save/load requests, terrain controls, and snapshots.
  - Relevant helpers include `add_label`, `add_line_label`, `add_curved_label`, `add_callout`, `clear_labels`, `set_labels_enabled`, `load_label_atlas`, `set_label_typography`, `set_declutter_algorithm`, `add_vector_overlay`, `save_bundle`, and `load_bundle`.

- `src/scene/*`
  - Native `Scene` is registered as a PyO3 class in `src/py_module/classes.rs`.
  - `src/scene/py_api/base.rs` exposes `Scene.__new__`, `set_camera_look_at`, and terrain-height upload.
  - `src/scene/py_api/bloom.rs` exposes `render_png` and `render_rgba` as instance methods.
  - `src/scene/render_paths/png.rs` and `src/scene/render_paths/rgba.rs` implement native offscreen render outputs.
  - `src/scene/py_api/raster_overlay.rs`, `src/scene/py_api/native_overlays.rs`, `src/scene/py_api/native_text.rs`, and `src/scene/py_api/text_mesh.rs` are relevant lower-level substrate for raster overlays and text, but they are not the typed map-scene product contract.

- `src/terrain/renderer/*`
  - Native `TerrainRenderer` is registered in `src/py_module/classes.rs`.
  - `src/terrain/renderer/py_api.rs` exposes `render_terrain_pbr_pom`, `render_with_aov`, offline accumulation methods, scatter memory reports, and probe memory reports.
  - `python/forge3d/offline.py` and docs in `docs/terrain/offline-render-quality.md` describe the existing offline terrain accumulation wrapper.

### Terrain, raster, vector, and style substrate

- `python/forge3d/terrain_params.py`
  - Relevant dataclasses include `LightSettings`, `IblSettings`, `ShadowSettings`, `FogSettings`, `BloomSettings`, `OfflineQualitySettings`, `OverlayLayerConfig`, `OverlaySettings`, `VectorOverlaySettings`, `VectorVertex`, and `VectorOverlayConfig`.
  - `VectorOverlayConfig.to_ipc_dict()` converts vector overlay intent to raw IPC payload shape.
  - `TerrainVTSettings` and `VTLayerFamily` accept `albedo`, `normal`, and `mask` families, but comments state native runtime currently pages only `albedo`.

- `python/forge3d/style.py`
  - Dataclasses: `VectorStyle`, `LabelStyle`, `PaintProps`, `LayoutProps`, `StyleLayer`, `StyleSpec`, `SpriteEntry`, `SpriteAtlas`, `GlyphRange`, and `FontStack`.
  - Functions: `load_style`, `parse_style`, `_parse_layer`, `paint_to_vector_style`, `layout_to_label_style`, `layer_to_vector_style`, `layer_to_label_style`, `apply_style`, `parse_color`, sprite/glyph helpers.
  - `apply_style()` currently filters render application to visible `fill`, `line`, and `circle` layers.
  - `load_style()` and `apply_style()` are Pro-gated.

- `python/forge3d/crs.py`
  - Public CRS utilities include `proj_available`, `transform_coords`, `reproject_geom`, `parse_crs_from_wkt`, `crs_to_epsg`, `get_crs_from_rasterio`, `get_crs_from_geopandas`, and `_crs_equal`.
  - These are relevant to `crs_mismatch` validation, but no product-level MapScene validation wrapper was found.

### LabelPlan and label substrate

- No product-level public `LabelPlan` implementation was found. Feature `004` depends on feature `003` or equivalent plan integration.
- `src/labels/*` provides native label substrate:
  - `src/labels/mod.rs`: `LabelManager`, point/line label state, atlas loading, camera update, collision, visible instance upload.
  - `src/labels/layer.rs`: native `LabelLayer`, `LabelFeature`, `FeatureGeometry`, `PlacementStrategy`, `GeneratedLabel`, and Mapbox symbol-layer conversion helpers. This `LabelLayer` is not registered as a Python class in `src/py_module/classes.rs`.
  - `src/labels/declutter.rs`, `src/labels/rtree.rs`, `src/labels/collision.rs`, `src/labels/projection.rs`, `src/labels/atlas.rs`, `src/labels/typography.rs`, `src/labels/line_label.rs`, `src/labels/curved.rs`, `src/labels/callout.rs`, and `src/labels/leader.rs` are relevant lower-level pieces.
- `src/labels/py_bindings.rs` exposes only `LabelStyle` and `LabelFlags` through PyO3; registration is in `src/py_module/classes.rs`.

### Buildings, 3D Tiles, and point clouds

- `python/forge3d/buildings.py`
  - Public dataclasses: `BuildingMaterial`, `Building`, and `BuildingLayer`.
  - Public functions: `infer_roof_type`, `material_from_tags`, `material_from_name`, `add_buildings`, `add_buildings_cityjson`, and `add_buildings_3dtiles`.
  - `BuildingLayer` is an existing building import result, not a typed MapScene layer contract.

- `python/forge3d/tiles3d.py`
  - Public classes/dataclasses: `BoundingVolume`, `TileContent`, `Tile`, `Tileset`, `VisibleTile`, `SseParams`, and `Tiles3dRenderer`.
  - Functions: `load_tileset`, `decode_b3dm`, and `decode_pnts`.
  - No public `Tiles3DLayer` or MapScene integration was found. This is P1-adjacent but feature `004` must represent unavailable or P1/P2 intent honestly.

- `python/forge3d/pointcloud.py`
  - Public point-cloud datasets and helpers: `LazDataset`, `CopcDataset`, `EptDataset`, `open_laz`, `open_copc`, `open_ept`, `open_pointcloud`, `PointCloudRenderer`, and LOD traversal types.
  - `ViewerHandle.load_point_cloud()` provides a viewer integration path.
  - Native point-cloud support exists in `src/pointcloud/*`, and `PyPointBuffer` is registered in `src/py_module/classes.rs`.

### Bundle and map furniture substrate

- `python/forge3d/bundle.py`
  - Public dataclasses: `TerrainMeta`, `CameraBookmark`, `RasterOverlaySpec`, `SceneBaseState`, `ReviewLayer`, `SceneVariant`, `SceneState`, `BundleManifest`, and `LoadedBundle`.
  - Public functions: `save_bundle`, `load_bundle`, and `is_bundle`.
  - `SceneBaseState` and `ReviewLayer` already serialize generic `raster_overlays`, `vector_overlays`, `labels`, and `scatter_batches`.
  - No explicit `diagnostics`, `ValidationReport`, `SceneRecipe`, or compiled `LabelPlan` field was found.

- `python/forge3d/map_plate.py`
  - Public map furniture/composition classes include `BBox`, `MapPlateConfig`, `PlateRegion`, `TitleElement`, `LegendElement`, `ScaleBarElement`, `InsetElement`, `NorthArrowElement`, and `MapPlate`.
  - `MapPlate` is Pro-gated and works on already-rendered image arrays, not a typed MapScene layer contract.

- `python/forge3d/legend.py`, `python/forge3d/scale_bar.py`, and `python/forge3d/north_arrow.py`
  - Existing cartographic furniture render helpers are relevant to `MapFurnitureLayer` and LabelPlan keepouts, but no MapScene integration was found.

## 2. Existing Tests And Fixtures Relevant To This Feature

### Tests

- API contract / public exports:
  - `tests/test_api_contracts.py`
  - `tests/test_install_smoke.py`

- Viewer and IPC:
  - `tests/test_viewer_ipc.py`
  - `tests/test_vector_overlay_rendering.py`
  - `tests/test_vector_overlay_drape.py`
  - `tests/test_vector_drape.py`

- Terrain render and offline output:
  - `tests/test_terrain_renderer.py`
  - `tests/test_terrain_runtime.py`
  - `tests/test_terrain_render_color_space.py`
  - `tests/test_terrain_visual_goldens.py`
  - `tests/test_tv12_offline_quality.py`
  - `tests/test_tv12_offline_architecture.py`
  - `tests/test_aov.py`
  - `tests/test_denoise_settings.py`
  - `tests/test_exr_output.py`
  - `tests/test_png_io_fallback.py`

- Style and vector styling:
  - `tests/test_style_parser.py`
  - `tests/test_style_render.py`
  - `tests/test_style_visual.py`
  - `tests/test_style_pixel_diff.py`
  - `tests/test_mapbox_streets_fixture.py`

- Labels:
  - `tests/test_labels_pybindings.py`
  - `tests/test_api_contracts.py::TestLabelBindings`
  - `tests/test_export_svg.py`
  - `tests/test_export_projection.py`

- Bundles:
  - `tests/test_bundle_roundtrip.py`
  - `tests/test_bundle_render.py`
  - `tests/test_bundle_cli.py`

- Buildings:
  - `tests/test_buildings_extrude.py`
  - `tests/test_buildings_cityjson.py`
  - `tests/test_buildings_materials.py`
  - `tests/test_buildings_roof.py`

- Point clouds:
  - `tests/test_pointcloud_gpu_integration.py`
  - `tests/test_pointcloud_lod.py`
  - `tests/test_copc_laz_fixture.py`

- CRS:
  - `tests/test_crs_reproject.py`
  - `tests/test_crs_auto.py`

- Camera and lighting:
  - `tests/test_camera_rigs.py`
  - `tests/test_lighting_preset.py`
  - `tests/test_lighting_alignment.py`
  - `tests/test_sun_ephemeris.py`

- Map furniture:
  - `tests/test_map_plate_layout.py`
  - `tests/test_export_svg.py`

- 3D Tiles:
  - `tests/test_3dtiles_parse.py`
  - `tests/test_3dtiles_sse.py`

- Pro gating:
  - `tests/test_pro_gating.py`
  - `tests/test_license.py`
  - `tests/conftest.py` fixture `pro_license`

No tests for public `MapScene`, `SceneRecipe`, `ValidationReport`, `Diagnostic`, typed recipe construction, `MapScene.validate()`, `MapScene.render()`, or `MapScene.save_bundle()` were found.

### Fixtures and assets

- Terrain/raster:
  - `assets/tif/dem_rainier.tif`
  - `assets/tif/Mount_Fuji_30m.tif`
  - `assets/tif/switzerland_dem.tif`
  - `assets/tif/switzerland_land_cover.tif`
  - `python/forge3d/data/mini_dem.npy`
  - `tests/golden/terrain/*`

- Vector/style:
  - `python/forge3d/data/sample_boundaries.geojson`
  - `tests/fixtures/mapbox_streets_v8.json`
  - `examples/sample_style.json`

- Labels/fonts:
  - `assets/fonts/default_atlas.json`
  - `assets/fonts/default_atlas.png`
  - `assets/gpkg/Mount_Fuji_places.gpkg`
  - `docs/gallery/images/02-mount-fuji-labels.png`

- Buildings:
  - `assets/geojson/sample_buildings.city.json`
  - `assets/geojson/10-270-592.city.json`
  - `assets/geojson/mount_fuji_buildings.geojson`

- Point clouds:
  - `assets/lidar/MtStHelens.laz`

- 3D Tiles:
  - No checked-in `.b3dm`, `.pnts`, `.glb`, or `tileset.json` fixture was found by file search. Existing 3D Tiles tests synthesize tileset JSON in temporary files.

## 3. Existing Docs And Examples Relevant To This Feature

- `README.md`
  - Documents terrain snapshots, overlays, labels, point clouds, buildings, Mapbox-style import, scene bundles, and Pro boundaries.

- `docs/start/quickstart.md`
  - Current workflow uses `open_viewer_async`, `ViewerHandle.load_overlay`, `ViewerHandle.snapshot`, and raw `viewer.send_ipc` / `forge3d.viewer_ipc` for lower-level vector overlays and labels.

- `docs/start/architecture.md`
  - Describes existing modules and states that raster overlays load through `ViewerHandle.load_overlay`, while vector overlays and labels use raw IPC or `forge3d.viewer_ipc`.

- `docs/viewer/index.md`
  - Documents `ViewerHandle` workflows: `load_overlay`, `load_point_cloud`, `snapshot`, and raw `send_ipc` for advanced commands.

- `docs/guides/feature_map.md`
  - Lists current substrate for native/offscreen rendering, terrain, overlays, labels, point clouds, buildings, map plates, export, style, and bundles.

- `docs/guides/data_and_scene_workflows.md`
  - Documents current data/scene workflows through viewer APIs, `viewer_ipc`, buildings, lower-level 3D Tiles, and bundle workflows.

- `docs/guides/rendering_and_analysis.md`
  - Documents `TerrainRenderer`, `TerrainRenderParams`, `render_offline`, and device diagnostics.

- `docs/guides/output_and_integration.md`
  - Documents viewer snapshots, map plates, vector export, scene bundles, and scripted pipelines.

- `docs/terrain/offline-render-quality.md`
  - Documents existing offline terrain render flow and fallback denoise behavior.

- `docs/tutorials/python-track/01-your-first-3d-terrain.md`
- `docs/tutorials/python-track/02-camera-lighting-and-animation.md`
- `docs/tutorials/python-track/03-point-clouds.md`
- `docs/tutorials/python-track/04-scene-bundles.md`
- `docs/tutorials/gis-track/02-drape-overlays-on-terrain.md`
- `docs/tutorials/gis-track/03-build-a-map-plate.md`
- `docs/tutorials/gis-track/04-3d-buildings.md`

- Examples relevant to canonical MapScene MVP examples:
  - `examples/terrain_viewer_interactive.py`
  - `examples/swiss_terrain_landcover_viewer.py`
  - `examples/bosnia_terrain_landcover_viewer.py`
  - `examples/luxembourg_rail_overlay.py`
  - `examples/style_viewer_interactive.py`
  - `examples/fuji_labels_demo.py`
  - `examples/buildings_viewer_interactive.py`
  - `examples/pointcloud_viewer_interactive.py`
  - `examples/terrain_atmosphere_path_demo.py`
  - `examples/triangle_png.py`
  - `examples/png_numpy_roundtrip.py`

No Offline 3D Map Rendering Quickstart using `MapScene`, no MapScene canonical examples, and no MapScene API docs were found.

## 4. Native/Rust Paths Relevant To This Feature

- PyO3 registration:
  - `src/lib.rs`
  - `src/py_module/classes.rs`
  - `src/py_module/functions.rs`
  - `src/py_module/functions/io_import.rs`
  - `src/py_module/functions/diagnostics.rs`
  - `src/py_module/functions/rendering.rs`

- Native scene/offscreen rendering:
  - `src/scene/mod.rs`
  - `src/scene/core.rs`
  - `src/scene/core/constructor.rs`
  - `src/scene/py_api.rs`
  - `src/scene/py_api/base.rs`
  - `src/scene/py_api/bloom.rs`
  - `src/scene/py_api/raster_overlay.rs`
  - `src/scene/py_api/native_overlays.rs`
  - `src/scene/py_api/native_text.rs`
  - `src/scene/py_api/text_mesh.rs`
  - `src/scene/render_paths/png.rs`
  - `src/scene/render_paths/rgba.rs`

- Terrain rendering and overlays:
  - `src/terrain/renderer.rs`
  - `src/terrain/renderer/core.rs`
  - `src/terrain/renderer/py_api.rs`
  - `src/terrain/render_params.rs`
  - `src/terrain/render_params/py_api.rs`
  - `src/terrain/render_params/native_overlays.rs`
  - `src/terrain/render_params/native_vt.rs`
  - `src/terrain/renderer/virtual_texture.rs`
  - `src/viewer/terrain/overlay.rs`
  - `src/viewer/terrain/overlay/stack.rs`
  - `src/viewer/terrain/vector_overlay.rs`
  - `src/viewer/cmd/terrain_command.rs`
  - `src/viewer/cmd/vector_overlay_command.rs`

- Labels and label IPC:
  - `src/labels/mod.rs`
  - `src/labels/types.rs`
  - `src/labels/layer.rs`
  - `src/labels/declutter.rs`
  - `src/labels/rtree.rs`
  - `src/labels/collision.rs`
  - `src/labels/projection.rs`
  - `src/labels/atlas.rs`
  - `src/labels/typography.rs`
  - `src/labels/py_bindings.rs`
  - `src/viewer/state/labels.rs`
  - `src/viewer/cmd/labels_command.rs`
  - `src/viewer/ipc/protocol/translate/labels.rs`
  - `src/viewer/ipc/protocol/response.rs`

- Bundles and scene review:
  - `src/bundle/mod.rs`
  - `src/bundle/manifest.rs`
  - `src/viewer/scene_review.rs`
  - `src/viewer/cmd/scene_review_command.rs`
  - `src/viewer/ipc/protocol/translate/scene_review.rs`
  - `src/viewer/ipc/protocol/payloads.rs`

- Buildings:
  - `src/import/osm_buildings.rs`
  - `src/import/building_materials.rs`
  - `src/import/cityjson.rs`
  - `src/import/cityjson/bindings.rs`
  - `src/import/cityjson/parser.rs`
  - `src/import/cityjson/geometry.rs`
  - `src/import/cityjson/types.rs`

- 3D Tiles:
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

- Point clouds:
  - `src/pointcloud/mod.rs`
  - `src/pointcloud/renderer.rs`
  - `src/pointcloud/copc.rs`
  - `src/pointcloud/copc_decode.rs`
  - `src/pointcloud/ept.rs`
  - `src/pointcloud/octree.rs`
  - `src/viewer/pointcloud.rs`
  - `src/viewer/cmd/pointcloud_command.rs`
  - `src/py_types/pointcloud.rs`

- Diagnostics and memory:
  - `src/py_functions/diagnostics.rs`
  - `src/core/memory_tracker.rs`
  - `src/core/memory_tracker/*`
  - `src/render/memory_budget.rs`
  - `src/util/memory_budget.rs`
  - `src/geometry/validate.rs`
  - `src/vector/data.rs`

- Camera/lighting:
  - `src/camera/mod.rs`
  - `src/camera/validation.rs`
  - `src/viewer/camera_controller.rs`
  - `src/terrain/camera.rs`
  - `src/lighting/*`
  - `src/lighting/py_bindings/*`

## 5. Python Public API Paths Relevant To This Feature

- Top-level/public aggregation:
  - `python/forge3d/__init__.py`
  - `python/forge3d/__init__.pyi`

- Current rendering and viewer paths:
  - `python/forge3d/viewer.py`
  - `python/forge3d/viewer.pyi`
  - `python/forge3d/viewer_ipc.py`
  - `python/forge3d/offline.py`
  - `python/forge3d/helpers/offscreen.py`
  - `python/forge3d/_png.py`

- Terrain, overlays, camera, lighting:
  - `python/forge3d/terrain_params.py`
  - `python/forge3d/terrain_demo.py`
  - `python/forge3d/camera_rigs.py`
  - `python/forge3d/camera_rigs.pyi`
  - `python/forge3d/lighting.py`
  - `python/forge3d/presets.py`

- Style/vector/labels:
  - `python/forge3d/style.py`
  - `python/forge3d/style_expressions.py`
  - `python/forge3d/vector.py`
  - `python/forge3d/export.py`

- Bundles/review:
  - `python/forge3d/bundle.py`

- Buildings, 3D Tiles, point clouds:
  - `python/forge3d/buildings.py`
  - `python/forge3d/tiles3d.py`
  - `python/forge3d/pointcloud.py`

- CRS/data/device/memory:
  - `python/forge3d/crs.py`
  - `python/forge3d/datasets.py`
  - `python/forge3d/_gpu.py`
  - `python/forge3d/_memory.py`
  - `python/forge3d/mem.py`

- Map furniture/output:
  - `python/forge3d/map_plate.py`
  - `python/forge3d/legend.py`
  - `python/forge3d/scale_bar.py`
  - `python/forge3d/north_arrow.py`

Not found as implemented public APIs outside specs/docs/state: `MapScene`, `SceneRecipe`, `TerrainSource`, `RasterOverlay`, `VectorOverlay`, product-level `LabelLayer`, `PointCloudLayer`, `MapFurnitureLayer`, `OutputSpec`, `ValidationReport`, product-level `Diagnostic`, `LabelPlan`, and `ReviewBundle`.

## 6. Suspected Placeholder, Fallback, Or No-Op Paths

- `MapScene`, `SceneRecipe`, `ValidationReport`, product-level `Diagnostic`, and typed recipe layer classes are missing. Any current code claiming those APIs would be invented.

- `python/forge3d/__init__.py::Renderer.render_triangle_rgba`
  - This is a fallback triangle pattern using `np.zeros` and simple pixel fill. It is useful as a smoke path but not a MapScene renderer.

- `python/forge3d/viewer.py`
  - Current MVP-like workflows require separate calls to `load_terrain`, `load_overlay`, `load_point_cloud`, camera/lighting setters, `snapshot`, and raw `send_ipc` for labels/vector overlays. This is the integration gap, not a typed scene recipe.

- `python/forge3d/viewer_ipc.py`
  - Raw label/vector/bundle IPC helpers may acknowledge commands without returning the stable product-level objects required by features `002` through `004`.
  - `save_bundle` and `load_bundle` IPC paths are request/async poll helpers, not `MapScene.save_bundle()`.

- `python/forge3d/buildings.py`
  - `add_buildings()` fallback creates `Building` entries with `positions=np.zeros((0, 3))` and empty indices when native GeoJSON import is unavailable.
  - `add_buildings_cityjson()` fallback creates zero-geometry placeholder buildings when native CityJSON parsing is unavailable.
  - `add_buildings_3dtiles()` returns an empty metadata-only `BuildingLayer`.

- `python/forge3d/tiles3d.py`
  - `decode_b3dm()` validates headers/embedded GLB then raises `NotImplementedError` for glTF mesh extraction.
  - `decode_pnts()` creates zero positions if `POSITION` is absent.
  - `Tiles3dRenderer` exposes traversal/cache helpers but no public MapScene render integration.

- `python/forge3d/style.py`
  - `_parse_layer()` preserves known paint/layout fields only; unsupported fields appear likely to be dropped without structured diagnostics.
  - `apply_style()` ignores non-`fill`/`line`/`circle` layers for application without a structured diagnostic in the inspected path.

- `src/viewer/cmd/labels_command.rs`
  - Prior inventories flagged `SetLabelTypography`, `SetDeclutterAlgorithm`, and curved/line-label paths as requiring truthfulness verification before MapScene examples depend on them.

- `src/labels/layer.rs`
  - Native `LabelLayer::generate_labels` skips empty text and does not expose product-level rejected label reason objects.

- `python/forge3d/terrain_params.py`, `src/terrain/render_params/native_vt.rs`, and `src/terrain/renderer/virtual_texture.rs`
  - Non-albedo VT families are accepted in Python but current native runtime pages only `albedo`.

- `python/forge3d/bundle.py`
  - Existing bundle schema has generic `labels`, `vector_overlays`, and `raster_overlays`, but no explicit diagnostics payload, compiled label plan payload, typed recipe payload, render policy, or output spec field.
  - `load_bundle()` verifies checksums only for files that exist; missing manifest-referenced files are not obviously surfaced as structured diagnostics.

- `python/forge3d/map_plate.py`
  - `LegendElement`, `ScaleBarElement`, and `NorthArrowElement` are placeholders inside the plate composer; they may be useful for future keepout/furniture representation but are not a MapScene furniture layer.

## 7. Pro-Gated Paths

- `python/forge3d/_license.py`
  - Defines `LicenseError`, `set_license_key`, and `_check_pro_access`.

- `python/forge3d/style.py`
  - `load_style()` gates "Mapbox style loading".
  - `apply_style()` gates "Mapbox style application".

- `python/forge3d/buildings.py`
  - `add_buildings()` gates "GeoJSON building import".
  - `add_buildings_cityjson()` gates "CityJSON building import".
  - `add_buildings_3dtiles()` gates "3D Tiles building import".

- `python/forge3d/bundle.py`
  - `save_bundle()` gates "Scene bundle save".
  - `load_bundle()` gates "Scene bundle load".

- `python/forge3d/map_plate.py`
  - `MapPlate.__init__()` gates "Map plate composition".

- `python/forge3d/export.py`
  - SVG/PDF export paths are Pro-gated.

- Tests:
  - `tests/test_pro_gating.py` covers Pro gates for style, buildings, bundles, export, and map plate paths.
  - `tests/conftest.py` defines `pro_license` used by building, bundle, style, export, and map-plate tests.

Uncertain: `MapScene.render()` and `MapScene.save_bundle()` MVP behavior must decide whether to wrap Pro-gated module-level bundle/style/building paths, provide public/open alternatives for P0, or emit `pro_gated_path`/`placeholder_fallback` diagnostics where packaging requires it.

## 8. Unknowns That Need Verification

- Whether `MapScene` should wrap native `Scene`, `TerrainRenderer`/offline render, viewer snapshot, or a hybrid. `docs/superpowers/state/open-blockers.md` keeps B-001 and B-004 open for feature `004`.

- Whether MVP `MapScene.render()` should use native `Scene.render_png`, `TerrainRenderer.render_terrain_pbr_pom` plus `numpy_to_png`, `ViewerHandle.snapshot`, or multiple backends selected by recipe capabilities.

- Whether `MapScene.save_bundle()` can satisfy P0 while `python/forge3d/bundle.py::save_bundle()` is Pro-gated, or whether P0 needs a non-Pro typed review-bundle save path with limited supported layer metadata.

- Whether feature `004` should introduce product `Diagnostic` and `ValidationReport` if feature `001` is not implemented first, or depend on those contracts being available from feature `001`.

- Whether feature `004` should introduce only recipe intent for `BuildingLayer` and `Tiles3DLayer` while validation emits diagnostics, or expose compatibility wrappers around existing `python/forge3d.buildings.BuildingLayer` and `python/forge3d.tiles3d.Tiles3dRenderer`.

- Exact recipe serialization shape for deterministic ordering: layer IDs, diagnostics, memory estimates, label-plan references/payloads, camera, lighting, map furniture, and output settings need design.

- Exact CRS validation behavior for mixed layer CRS metadata. Feature spec defaults to no implicit transforms, but current examples such as `examples/terrain_viewer_interactive.py` can reproject vectors by CLI options. MapScene needs a typed explicit policy.

- Exact GPU memory estimate source. Existing options include static estimates from recipe dimensions/point counts, `python/forge3d/mem.py`, native `global_memory_metrics`, point-cloud `MemoryReport`, terrain scatter/probe reports, and 3D Tiles cache stats. No MapScene-level estimate exists.

- How to integrate deterministic `LabelPlan` from feature `003` into a MapScene render payload. No public `LabelPlan` implementation currently exists.

- How to convert vector/style/label recipes into renderable payloads without raw IPC in canonical examples. Existing `VectorOverlayConfig.to_ipc_dict()` and `viewer_ipc.add_vector_overlay()` are IPC-shaped.

- Whether canonical examples can be smoke-validated without GPU/viewer availability or need a split between construction/validation tests and GPU-marked render tests.

- Whether existing fixtures are enough for exact PNG comparison. The spec requires exact pixel comparison unless planning records a numeric tolerance.

- Whether docs under `docs/_build/` should be ignored for support wording audits as generated output.

- Current branch and continuity files point at feature `005-map-assets-bundles-p1` or earlier feature inventory in places, while this request targets `004-mapscene-mvp`. This inventory treats `004` as the target by explicit user request.

- No product tests were run in this inventory. Findings are from static inspection only.

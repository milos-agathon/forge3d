# Feature Research: Material, VT, and Large-Scene P2

**Feature directory**: `specs/006-material-vt-large-scene-p2`  
**Date**: 2026-05-15  
**Scope**: feature-specific implementation inventory before technical planning; no code implementation.

## Source Artifacts Inspected

- `docs/superpowers/plans/prd.md`
- `.specify/memory/constitution.md`
- `specs/006-material-vt-large-scene-p2/spec.md`
- `docs/superpowers/state/current-context-pack.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/open-blockers.md`
- `AGENTS.md`

## Commands Used For Inspection

```powershell
Get-Content -Path C:\Users\milos\.codex\plugins\cache\openai-curated\superpowers\1b89ff49\skills\using-superpowers\SKILL.md
Get-Content -Path C:\Users\milos\forge3d\.agents\skills\speckit-plan\SKILL.md
Get-Content -Path docs/superpowers/plans/prd.md
Get-Content -Path .specify/memory/constitution.md
Get-Content -Path specs/006-material-vt-large-scene-p2/spec.md
if (Test-Path specs/006-material-vt-large-scene-p2/research.md) { Get-Content -Path specs/006-material-vt-large-scene-p2/research.md } else { Write-Output '<missing>' }
git status --short
Get-Content -Path docs/superpowers/state/current-context-pack.md
Get-Content -Path docs/superpowers/state/implementation-ledger.md
Get-Content -Path docs/superpowers/state/requirements-verification-matrix.md
Get-Content -Path docs/superpowers/state/open-blockers.md
git status --short --branch
rg -n "VirtualTexture|virtual_texture|vt_|Virtual Texture|normal.*mask|albedo|mask|normal" python src tests docs examples -S
rg -n "Building|buildings|CityJSON|GeoJSON|texture|material|PBR|uv|UV" python src tests docs examples -S
rg -n "LabelPlan|advanced label|curved|repeat|leader|callout|priority|declutter|HarfBuzz|shaping|glyph|Label" python src tests docs examples specs -S
rg -n "large[-_ ]scene|memory budget|estimated_gpu_memory|MemoryReport|cache|LOD|lod|instanc|bottleneck|screen-space|screen space|statistics|stats" python src tests docs examples -S
rg --files python src tests docs examples | rg "(virtual|terrain_params|building|material|texture|label|point|tiles3d|bundle|diagnostic|mem|geometry|instanc|lod|clipmap|cog|large|feature_map|data_and_scene|output_and_integration|offline-render-quality)"
rg -n "class .*Scene|class MapScene|SceneRecipe|ValidationReport|Diagnostic|LabelPlan|VirtualTexture|Building|PointCloud|Tiles3D|OutputSpec|Layer" python\forge3d tests specs -S
rg -n "pub struct|struct|enum|impl|fn |pyclass|pyfunction|add_function|add_class" src\terrain src\labels src\import src\tiles3d src\scene src\pointcloud src\py_functions src\io src\bundle src\geometry -S
Get-Content -Path specs/005-map-assets-bundles-p1/research.md -TotalCount 220
Get-Content -Path specs/003-deterministic-label-plan/research.md -TotalCount 220
rg -n "class VTLayerFamily|class VirtualTextureSettings|family|allowed|albedo|normal|mask|register_material_vt_source|get_material_vt_stats|material_vt" python/forge3d/terrain_params.py src/terrain/renderer/virtual_texture.rs src/terrain/renderer/py_api.rs src/terrain/render_params/native_vt.rs tests/test_tv20_virtual_texturing.py -S
rg -n "class BuildingMaterial|class Building|class BuildingLayer|texture|uv|material|fallback|placeholder|_check_pro_access|add_buildings|parse_cityjson|import_osm|material_from" python/forge3d/buildings.py src/import/building_materials.rs src/import/cityjson src/import/osm_buildings.rs src/io/gltf_read.rs src/io/obj_read.rs tests/test_buildings_materials.py tests/test_buildings_cityjson.py tests/test_buildings_extrude.py -S
rg -n "repeat_distance|Curved|curved|leader|callout|priority|road|river|placement|HarfBuzz|shaping|glyph|line label|LineLabel|layout_curved_text|compute_line_label" src/labels python/forge3d/viewer_ipc.py src/viewer/cmd/labels_command.rs examples/fuji_labels_demo.py tests/test_labels_pybindings.py tests/test_api_contracts.py -S
rg -n "MemoryReport|RenderStats|cache_stats|TraversalStats|LOD|lod|SseParams|instanc|scatter|memory_report|get_scatter_stats|get_scatter_memory_report|get_material_vt_stats|global_memory_metrics|estimated_gpu_memory|bottleneck|budget" python/forge3d/pointcloud.py python/forge3d/tiles3d.py python/forge3d/geometry.py src/pointcloud src/tiles3d src/terrain/renderer src/scene/py_api src/py_functions/diagnostics.rs tests/test_pointcloud_lod.py tests/test_pointcloud_gpu_integration.py tests/test_3dtiles_sse.py tests/test_tv20_virtual_texturing.py -S
rg -n "MapScene|SceneRecipe|ValidationReport|Diagnostic|LabelPlan|missing_texture_path|missing_uvs|unsupported_texture_format|unavailable_cache_lod_stats|unsupported_instancing_path|vt_unsupported_family|estimated_gpu_memory" python/forge3d src tests docs examples --glob '!docs/_build/**' -S
rg --files docs examples assets tests | rg -i "(virtual|vt|building|cityjson|3dtiles|tiles3d|label|fuji|pointcloud|point_cloud|large|lod|instanc|material|texture|terrain|offline|feature_map|data_and_scene|output_and_integration|mapbox|atlas)"
rg --files tests | rg -i "(tv20|virtual|building|cityjson|3dtiles|tiles3d|label|pointcloud|point_cloud|geometry|instanc|terrain_material|terrain_tv|terrain_overlay|api_contract|style|bundle|crs|viewer_ipc)"
rg -n "enable-gpu-instancing|enable-tbn|features =|\[features\]|copc_laz|material|vt|virtual" Cargo.toml pyproject.toml -S
rg -n "add_function|add_class|register|import_osm_buildings|parse_cityjson|material_from|LabelStyle|LabelFlags|gpu_instancing|TerrainRenderer|Point|tiles3d|diagnostic|memory" src/lib.rs src/py_module -S
rg -n "BuildingLayer|BuildingMaterial|add_buildings|VTLayerFamily|VirtualTextureSettings|TerrainRenderer|PointCloudRenderer|Tiles3dRenderer|gpu_instancing_available|MapScene|LabelPlan|Diagnostic|ValidationReport|LabelStyle|LabelFlags" python/forge3d/__init__.py python/forge3d/__init__.pyi -S
rg -n "class PointBuffer|class PointCloudRenderer|def memory_report|cache|point_budget|def load_copc|def load_ept|def visible" python/forge3d/pointcloud.py -S
rg -n "def cache_stats|class Tiles3dRenderer|class SseParams|load_tileset|decode_b3dm|decode_pnts|NotImplemented|unsupported|TODO|placeholder" python/forge3d/tiles3d.py -S
rg -n "class Texture|def load_texture|texture|albedo|normal|roughness|metallic|PBR|material" python/forge3d/textures.py python/forge3d/materials.py src/render src/io -S
```

## 1. Existing Modules, Classes, And Functions Relevant To This Feature

### Product-level P2 scene surface

- No implemented public `MapScene`, `SceneRecipe`, `ValidationReport`, `Diagnostic`, or `LabelPlan` class/function was found under `python/forge3d`, `src`, `tests`, `examples`, or non-built docs outside specs/state references.
- No implemented feature-local diagnostic codes were found for `missing_texture_path`, `missing_uvs`, `unsupported_texture_format`, `unavailable_cache_lod_stats`, `unsupported_instancing_path`, or `vt_unsupported_family` in product code.
- `specs/006-material-vt-large-scene-p2/plan.md` does not exist yet. Technical planning has not started.

### Virtual texturing

- `python/forge3d/terrain_params.py`
  - Defines `VTLayerFamily` and accepts only `albedo`, `normal`, and `mask` family names.
  - The class docstring states that `normal` and `mask` are forward-compatible placeholders while the current native runtime pages only `albedo`.
  - Defines `VirtualTextureSettings` with `layers`, duplicate-family validation, `residency_budget_mb`, and `actual_mip_count()`.

- `src/terrain/render_params/native_vt.rs`
  - Native comments state that Python accepts `albedo`, `normal`, and `mask`, but the current terrain VT runtime pages only `albedo` and ignores the others.

- `src/terrain/renderer/virtual_texture.rs`
  - Defines `TERRAIN_VT_SUPPORTED_FAMILY: &str = "albedo"`.
  - `register_source()` stores non-albedo families for forward compatibility but records that native runtime will ignore them.
  - Runtime page-table and atlas preparation select only the `albedo` family and skip non-albedo sources.
  - Exposes internal stats such as `total_pages`, `resident_pages`, `cache_budget_pages`, and `cache_budget_mb`.

- `src/terrain/renderer/py_api.rs`
  - `TerrainRenderer.register_material_vt_source(...)`, `clear_material_vt_sources()`, and `get_material_vt_stats()` expose material VT source registration and stats to Python.
  - These are renderer-level methods, not typed `MapScene` validation diagnostics.

### Building materials and textures

- `python/forge3d/buildings.py`
  - Defines `BuildingMaterial`, `Building`, and `BuildingLayer`.
  - `BuildingMaterial` is scalar PBR data: base color/albedo, roughness, metallic, emissive, and related scalar fields. No texture path or UV support field was found in this dataclass.
  - `BuildingLayer` exposes `building_count`, `total_vertices`, `total_triangles`, `max_lod`, and `bounds`; this is useful for large-scene summaries.
  - `add_buildings()`, `add_buildings_cityjson()`, and `add_buildings_3dtiles()` are Pro-gated through `_check_pro_access(...)`.
  - Native probes call `_NATIVE.import_osm_buildings_from_geojson_py` and `_NATIVE.parse_cityjson_py` when available.
  - Fallback paths create placeholder buildings with scalar materials.

- `src/import/building_materials.rs`
  - Defines native scalar PBR `BuildingMaterial`, material presets, `material_from_tags()`, `material_from_name()`, `roof_material_from_tags()`, and PyO3 wrappers `material_from_tags_py()` and `material_from_name_py()`.
  - No building texture path, texture format validation, or UV diagnostics object was found here.

- `src/import/osm_buildings.rs`
  - OSM building import/extrusion includes positions, normals, UVs, indices, roof type, materials, and LOD-related behavior.
  - PyO3 wrappers `import_osm_buildings_extrude_py()` and `import_osm_buildings_from_geojson_py()` return mesh-like data consumed by Python building wrappers.

- `src/import/cityjson/`
  - `parser.rs`, `types.rs`, `geometry.rs`, and `bindings.rs` provide CityJSON parsing, geometry extraction, CRS/transform/extent metadata, scalar material inference, normals, and Python binding `parse_cityjson_py()`.
  - No end-to-end textured building material workflow was found.

- `python/forge3d/textures.py` and `python/forge3d/materials.py`
  - `textures.py` defines `Tex`, `PbrTexSet`, `load_texture()`, `build_pbr_textures()`, and `extract_mr_channels()`.
  - `materials.py` defines `PbrMaterial` with optional `PbrTexSet`.
  - These are general PBR texture helpers, not wired to `BuildingLayer` or `MapScene` building materials in inspected paths.

- `src/io/gltf_read.rs` and `src/io/obj_read.rs`
  - Existing mesh import can read normals and UVs; OBJ import also records `diffuse_texture`.
  - This is relevant substrate for UV/texture support, but no building-layer texture validation workflow was found.

- `src/render/material_set/`
  - Terrain material sets support albedo texture paths and memory-budget fallback behavior for terrain material textures.
  - This is terrain material infrastructure, not building material integration.

### Advanced static labels

- `python/forge3d/viewer_ipc.py`
  - Existing raw IPC helpers include `add_line_label(..., repeat_distance=...)`, `add_curved_label(...)`, `add_callout(...)`, `set_label_typography(...)`, and `set_declutter_algorithm(...)`.
  - These are not `LabelPlan` compile APIs and do not provide deterministic accepted/rejected advanced label plans.

- `src/viewer/cmd/labels_command.rs`
  - Handles `AddLineLabel`, `AddCurvedLabel`, `AddCallout`, `SetLabelTypography`, and `SetDeclutterAlgorithm`.
  - `AddCurvedLabel` routes to `LabelManager.add_line_label(..., LineLabelPlacement::Along, 0.0)`.
  - Typography and declutter commands require verification because earlier inventories flagged possible no-op/truthfulness gaps.

- `src/labels/types.rs`
  - Defines `LineLabelPlacement`, `LabelFlags`, `LabelStyle`, `LabelData`, `LineLabelData`, `GlyphPlacement`, and `LeaderLine`.
  - `LineLabelData.repeat_distance` exists, but feature-level repeated-label plan behavior is not implemented as a public `LabelPlan` contract.

- `src/labels/line_label.rs`
  - Defines `compute_line_label_placement()` and `compute_glyph_advances()`.
  - Computes glyph positions along a line path for center/along placement.
  - No product-level repeated-label acceptance/rejection plan or reason-code model was found.

- `src/labels/curved.rs`
  - Defines `SampledPath`, `CurvedGlyphInstance`, `CurvedTextLayout`, `layout_curved_text()`, and `project_curved_layout()`.
  - Relevant curved-text substrate exists, but public rendering/plan integration remains uncertain.

- `src/labels/layer.rs`
  - Defines `LabelLayerConfig.allow_curved`, `repeat_distance`, `PlacementStrategy`, `LabelPlacementType`, and generated labels from features.
  - Existing `generate_labels()` is useful substrate, but prior inventory found no retained rejection reasons, no `LabelPlan` compile API, and no public PyO3 registration for native `LabelLayer`.

- `src/labels/declutter.rs`, `src/labels/rtree.rs`, and `src/labels/collision.rs`
  - Provide collision and priority substrate.
  - Current declutter results do not expose rejected-label reason codes.

- `src/labels/callout.rs` and `src/labels/leader.rs`
  - Provide callout box and leader-line geometry helpers.
  - No advanced static `LabelPlan` leader-line placement contract was found.

- No HarfBuzz-compatible shaping path was found in the inspected commands. This is uncertain because only repository text search was performed; absence should be verified during planning before declaring it definitively missing.

### Large-scene, cache, LOD, memory, and instancing substrate

- `python/forge3d/pointcloud.py`
  - Defines `PointCloudRenderer` with `point_budget`, LOD traversal, visible-node selection, and an internal `_cache`.
  - Relevant to large-scene point-cloud summaries, but no typed `MapScene` resource summary integration was found.

- `src/pointcloud/renderer.rs`
  - Defines `PointBuffer`, `RenderStats`, `MemoryReport`, and `PointCloudRenderer`.
  - `MemoryReport` includes `cache_used`, `cache_budget`, `utilization`, and `entry_count`.
  - This is useful substrate for `estimated_gpu_memory` or large-scene diagnostics but is not exposed as a product-level `ValidationReport` in inspected paths.

- `src/pointcloud/traversal.rs`
  - Defines `TraversalParams`, `PointCloudTraverser`, `VisibleNode`, and `TraversalStats` with point-budget usage.

- `python/forge3d/tiles3d.py`
  - Defines `SseParams`, `Tiles3dRenderer`, `load_tileset()`, `decode_b3dm()`, and `decode_pnts()`.
  - `Tiles3dRenderer.cache_stats()` exists.
  - `decode_b3dm()` raises `NotImplementedError` for glTF mesh extraction.

- `src/tiles3d/`
  - `sse.rs`, `traversal.rs`, `tileset.rs`, `tile.rs`, `bounds.rs`, and `renderer.rs` provide screen-space-error traversal, tile counts, max depth, cache stats, and building render data.
  - `renderer.rs` includes `CacheStats`, `BuildingRenderData`, and `BuildingInstance`.

- `python/forge3d/geometry.py`
  - `instance_mesh()` provides CPU mesh expansion.
  - `gpu_instancing_available()` probes the native GPU instancing feature.
  - `instance_mesh_gpu_render()` raises when the native GPU instancing renderer is unavailable.
  - `generate_lod_chain()` provides mesh LOD-chain generation.

- `src/scene/py_api/instanced_mesh.rs`
  - `Scene.add_instanced_mesh()`, `clear_instanced_meshes()`, and `update_instanced_transforms()` are gated by Cargo feature `enable-gpu-instancing`.

- `src/terrain/renderer/py_api.rs`
  - Feature-gated scatter/instancing APIs: `set_scatter_batches()`, `clear_scatter_batches()`, `get_scatter_stats()`, and `get_scatter_memory_report()`.
  - General memory/stat helpers: `get_probe_memory_report()`, `get_reflection_probe_memory_report()`, and `get_material_vt_stats()`.

- `src/py_functions/diagnostics.rs`
  - `global_memory_metrics()` returns memory metrics including `within_budget`.
  - This is an engine diagnostic helper, not the PRD `Diagnostic` object model.

## 2. Existing Tests And Fixtures Relevant To This Feature

### Virtual texturing

- `tests/test_tv20_virtual_texturing.py`
  - Tests `VTLayerFamily` defaults, rejects unknown families, accepts forward-compatible `normal` and `mask`, and explicitly asserts the native runtime is currently albedo-only.
  - Tests `register_material_vt_source()`, `get_material_vt_stats()`, disabled VT baseline behavior, albedo VT output changes, and residency budget enforcement.

### Buildings and materials

- `tests/test_buildings_materials.py`
  - Tests scalar building material presets, tag inference, dataclass serialization, and material differentiation.

- `tests/test_buildings_extrude.py`
  - Tests Pro-gated GeoJSON building import, fallback/native behavior comments, missing file errors, multipolygon input, and vertex counts.

- `tests/test_buildings_cityjson.py`
  - Tests Pro-gated CityJSON import, metadata, multiple buildings, LOD selection, building parts, invalid input, transform application, and sample fixture loading.

- `tests/test_buildings_roof.py`
  - Tests roof type inference.

- No dedicated textured-building fixture test, missing texture path test, missing UV diagnostic test, or unsupported texture format test was found.

### Advanced labels

- `tests/test_labels_pybindings.py` and `tests/test_api_contracts.py::TestLabelBindings`
  - Test native `LabelStyle` and `LabelFlags` classes.

- Rust inline tests exist in `src/labels/line_label.rs`, `src/labels/curved.rs`, `src/labels/leader.rs`, `src/labels/callout.rs`, `src/labels/declutter.rs`, `src/labels/layer.rs`, `src/labels/rtree.rs`, and `src/labels/collision.rs`.

- No `LabelPlan` tests, repeated line-label plan fixtures, road/river placement-rule tests, advanced priority-preset tests, or HarfBuzz shaping tests were found.

### Large scenes, cache, LOD, memory, and instancing

- `tests/test_pointcloud_lod.py`
  - Tests point-cloud LOD traversal and point-budget behavior.

- `tests/test_pointcloud_gpu_integration.py`
  - Tests point-buffer GPU buffer conversion and Python LOD traversal with fixtures.

- `tests/test_3dtiles_parse.py` and `tests/test_3dtiles_sse.py`
  - Test tileset parsing, SSE behavior, traversal, max depth, and initial cache stats.

- `tests/test_gpu_lod_selection.py` and `tests/test_terrain_tv13_lod_pipeline.py`
  - Relevant to terrain/GPU LOD behavior.

- `tests/test_terrain_scatter.py`
  - Relevant to terrain scatter/instancing substrate.

- No product-level large-scene `ValidationReport` tests, bottleneck layer diagnostic tests, unavailable cache/LOD diagnostic tests, or unsupported instancing path diagnostics were found.

### Fixtures and assets

- `assets/geojson/sample_buildings.city.json`
- `assets/geojson/mount_fuji_buildings.geojson`
- `assets/gpkg/Mount_Fuji_places.gpkg`
- `assets/tif/Mount_Fuji_30m.tif`
- `assets/fonts/default_atlas.json`
- `assets/fonts/default_atlas.png`
- `tests/fixtures/mapbox_streets_v8.json`

No checked-in textured building fixture with texture files and UV expectations was found. No checked-in `.b3dm`, `.pnts`, `.glb`, or `tileset.json` fixture file was found by file search; 3D Tiles tests appear to synthesize temporary tilesets.

## 3. Existing Docs And Examples Relevant To This Feature

- `docs/superpowers/plans/prd.md`
  - Source of truth for P2-R1 through P2-R4, diagnostics, support-level terms, and Milestone 5.

- `specs/006-material-vt-large-scene-p2/spec.md`
  - Feature contract for VT normal/mask behavior, textured PBR buildings, advanced labels, and large-scene diagnostics.

- `docs/guides/feature_map.md`
  - Existing feature overview, including terrain, overlays, labels, bundles, and rendering capabilities.

- `docs/guides/data_and_scene_workflows.md`
  - Existing workflows for vector overlays, labels, styles, point clouds, and scenes.

- `docs/guides/output_and_integration.md`
  - Existing bundle/repeatable-scene and output integration docs.

- `docs/terrain/offline-render-quality.md`
  - Existing offline rendering quality guidance.

- `docs/gallery/02-mount-fuji-labels.md` and `examples/fuji_labels_demo.py`
  - Existing label demo material including priorities, leader lines, line labels, curved labels, callouts, typography, and declutter through raw IPC.

- `docs/gallery/05-3d-buildings.md`, `docs/tutorials/gis-track/04-3d-buildings.md`, and `examples/buildings_viewer_interactive.py`
  - Existing building docs/examples. They are useful context but should be audited for support-level wording before P2 claims.

- `examples/pointcloud_viewer_interactive.py`
  - Existing point-cloud viewer example relevant to large-scene point-cloud substrate.

- `docs/assets/thumbnails/f16_instancing.svg`
  - Existing docs asset related to instancing; not a product API contract.

- No dedicated Virtual Texturing Support Matrix, Building Texture Support Matrix, Advanced Label Placement Guide, or Large-Scene Diagnostics Reference was found outside specs/state references.

## 4. Native/Rust Paths Relevant To This Feature

- `src/terrain/render_params/native_vt.rs`
- `src/terrain/renderer/virtual_texture.rs`
- `src/terrain/renderer/py_api.rs`
- `src/import/building_materials.rs`
- `src/import/osm_buildings.rs`
- `src/import/cityjson/parser.rs`
- `src/import/cityjson/types.rs`
- `src/import/cityjson/geometry.rs`
- `src/import/cityjson/bindings.rs`
- `src/io/gltf_read.rs`
- `src/io/obj_read.rs`
- `src/io/obj_write.rs`
- `src/render/material_set/core.rs`
- `src/render/material_set/gpu.rs`
- `src/render/material_set/py_api.rs`
- `src/labels/types.rs`
- `src/labels/mod.rs`
- `src/labels/layer.rs`
- `src/labels/line_label.rs`
- `src/labels/curved.rs`
- `src/labels/declutter.rs`
- `src/labels/rtree.rs`
- `src/labels/collision.rs`
- `src/labels/callout.rs`
- `src/labels/leader.rs`
- `src/labels/atlas.rs`
- `src/labels/typography.rs`
- `src/pointcloud/renderer.rs`
- `src/pointcloud/traversal.rs`
- `src/pointcloud/octree.rs`
- `src/tiles3d/sse.rs`
- `src/tiles3d/traversal.rs`
- `src/tiles3d/tileset.rs`
- `src/tiles3d/tile.rs`
- `src/tiles3d/renderer.rs`
- `src/scene/py_api/instanced_mesh.rs`
- `src/terrain/renderer/scatter.rs`
- `src/py_functions/diagnostics.rs`
- `src/py_module/classes.rs`
- `src/py_module/functions.rs`
- `src/py_module/functions/io_import.rs`
- `src/py_module/functions/geometry.rs`
- `src/py_module/functions/diagnostics.rs`

## 5. Python Public API Paths Relevant To This Feature

- `python/forge3d/terrain_params.py`
  - `VTLayerFamily`, `VirtualTextureSettings`, and `TerrainRenderParams` fields related to VT.

- `python/forge3d/__init__.py` and `python/forge3d/__init__.pyi`
  - Re-export `VTLayerFamily`, `TerrainRenderer`, `BuildingLayer`, `BuildingMaterial`, `add_buildings`, `add_buildings_cityjson`, and `add_buildings_3dtiles`.
  - No `MapScene`, `LabelPlan`, `Diagnostic`, or `ValidationReport` export found.

- Native `TerrainRenderer` methods exposed through `_forge3d`
  - `register_material_vt_source`, `clear_material_vt_sources`, and `get_material_vt_stats`.

- `python/forge3d/buildings.py`
  - `BuildingMaterial`, `Building`, `BuildingLayer`, `add_buildings`, `add_buildings_cityjson`, `add_buildings_3dtiles`, `material_from_tags`, and `material_from_name`.

- `python/forge3d/textures.py` and `python/forge3d/materials.py`
  - `Tex`, `PbrTexSet`, `load_texture`, `build_pbr_textures`, `extract_mr_channels`, and `PbrMaterial`.

- `python/forge3d/viewer_ipc.py`
  - Raw label helpers for line labels, curved labels, callouts, typography, and declutter.

- `python/forge3d/geometry.py`
  - `instance_mesh`, `gpu_instancing_available`, `instance_mesh_gpu_render`, and `generate_lod_chain`.

- `python/forge3d/pointcloud.py`
  - `PointCloudRenderer` and point-cloud dataset/LOD helper classes.

- `python/forge3d/tiles3d.py`
  - `Tileset`, `Tile`, `SseParams`, `Tiles3dRenderer`, `load_tileset`, `decode_b3dm`, and `decode_pnts`.

- `python/forge3d/bundle.py`
  - Existing bundle manifest/state APIs that future P2 diagnostics and support summaries may need to serialize through once product diagnostics exist.

## 6. Suspected Placeholder, Fallback, Or No-Op Paths

- `python/forge3d/terrain_params.py`
  - `VTLayerFamily` accepts `normal` and `mask` as forward-compatible families; comments say they remain placeholders until native extension.

- `src/terrain/render_params/native_vt.rs` and `src/terrain/renderer/virtual_texture.rs`
  - Non-albedo VT families are stored/accepted but ignored or skipped by current runtime. This is a direct P2 target and must produce `vt_unsupported_family` diagnostics unless implemented.

- `python/forge3d/buildings.py`
  - GeoJSON and CityJSON fallback branches create placeholder building objects when native paths are unavailable.
  - Textured material fallback to scalar material is likely because `BuildingMaterial` lacks texture-path/UV fields; exact behavior needs implementation planning.

- `python/forge3d/tiles3d.py`
  - `decode_b3dm()` raises `NotImplementedError` for glTF mesh extraction.
  - Public 3D Tiles scene integration remains underdeveloped from prior feature inventory.

- `src/viewer/cmd/labels_command.rs`
  - `SetLabelTypography` and `SetDeclutterAlgorithm` require verification for real state mutation.
  - `AddCurvedLabel` currently routes into line-label storage; production curved text rendering remains uncertain.

- `src/labels/mod.rs`
  - Line labels compute glyph placements, but prior inventory found they do not emit visible glyph instances with rotation applied yet.

- `src/labels/layer.rs`
  - `LabelLayer::generate_labels()` skips empty text rather than retaining a rejected entry with reason code.

- `python/forge3d/geometry.py`
  - `instance_mesh_gpu_render()` explicitly raises when the GPU instanced renderer is not available.

- `src/terrain/renderer/py_api.rs`
  - Scatter/instancing APIs are feature-gated; absence must be diagnosed as unavailable or unsupported in a product-level scene report.

- `src/py_functions/diagnostics.rs`
  - Current memory/device diagnostics are plain dict-style reports, not PRD diagnostic objects with support levels/remediation.

## 7. Pro-Gated Paths

- `python/forge3d/buildings.py`
  - `_check_pro_access("GeoJSON building import")` in `add_buildings()`.
  - `_check_pro_access("CityJSON building import")` in `add_buildings_cityjson()`.
  - `_check_pro_access("3D Tiles building import")` in `add_buildings_3dtiles()`.

- `tests/test_pro_gating.py`
  - Existing test coverage for Pro gates includes building import, style APIs, and bundle APIs.

- `python/forge3d/style.py`
  - `load_style()` and `apply_style()` were identified as Pro-gated in prior feature inventories. They may affect label/style inputs used by P2 advanced labels.

- `python/forge3d/bundle.py`
  - `save_bundle()` and `load_bundle()` were identified as Pro-gated in prior feature inventories. They may affect serialization of P2 diagnostics once product diagnostics exist.

- Cargo/native feature gates:
  - `Cargo.toml` defines `enable-gpu-instancing`.
  - `pyproject.toml` includes `enable-gpu-instancing` in maturin build features.
  - `src/scene/py_api/instanced_mesh.rs` and terrain scatter APIs are guarded by `#[cfg(feature = "enable-gpu-instancing")]`.

## 8. Unknowns That Need Verification

- Whether P2 will implement real native VT `normal`/`mask` paging or only product diagnostics in this feature.
- Whether non-albedo VT requests should be rejected during `MapScene.validate()` before reaching `TerrainRenderer.register_material_vt_source()`.
- Whether a building texture workflow should extend `BuildingMaterial`, add a separate material-intent object, or rely on existing `PbrTexSet`/`PbrMaterial` helpers.
- Whether current OSM/CityJSON building geometry preserves enough UV metadata for textured buildings, and how missing UVs can be detected per building/object.
- Whether checked-in assets need new textured building fixtures, because none were found during static inspection.
- Which texture formats should be allowed for `unsupported_texture_format` tests.
- Whether advanced static label placement depends on feature `003` `LabelPlan` being implemented first, or whether P2 only layers new modes and diagnostics onto it.
- Whether `repeat_distance` in existing line-label data is currently used beyond storage/IPC routing; exact rendering behavior needs code-level verification.
- Whether curved text substrate in `src/labels/curved.rs` is connected to viewer rendering or only exists as isolated layout helpers.
- Whether road/river placement rules and multi-class priority presets should be native Rust, Python product-layer policy, or both.
- Whether any HarfBuzz-compatible shaping crate or optional dependency exists outside the text search terms used here; current search found no obvious path.
- Whether large-scene bottleneck diagnostics should use validation-time metadata only, render-time timing counters, or both.
- Whether point-cloud `MemoryReport`, 3D Tiles cache stats, terrain scatter stats, material VT stats, and global memory metrics should be normalized into one product-level large-scene summary model.
- Whether unavailable cache/LOD stats should be a warning or info diagnostic by default for P2.
- Whether GPU instancing availability should be reported per layer type or as one scene-level support summary.
- Whether docs generated under `docs/_build/` are stale and should be ignored for support wording audits; source docs should be treated as authoritative.

## Phase 0 Planning Decisions

- Decision: P2 paths are either real end-to-end support or explicit pre-render diagnostics; diagnosed deferrals do not block P0 MVP.
  Rationale: The PRD classifies these items as P2 polish and requires honesty rather than MVP gating.
  Alternatives considered: Block MVP on VT normal/mask, textured buildings, advanced labels, or large-scene polish, rejected without a documented human scope change.

- Decision: VT normal/mask planning keeps both options open: native runtime implementation or validation-time `vt_unsupported_family`.
  Rationale: The current runtime pages albedo only, and implementing normal/mask may require native resource/shader changes beyond diagnostics.
  Alternatives considered: Claim normal/mask support because Python settings accept those families, rejected because runtime support is not present.

- Decision: Large-scene diagnostics normalize only available metadata and explicitly mark unavailable cache/LOD/instancing stats.
  Rationale: The repo has fragmented stats across point cloud, 3D Tiles, terrain scatter, material VT, and memory helpers; invented estimates would violate diagnostic honesty.
  Alternatives considered: Produce scene-level estimates from incomplete metadata, rejected unless exact assumptions are documented and tested.

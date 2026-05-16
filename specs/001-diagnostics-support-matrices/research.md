# Diagnostics and Support Matrices Implementation Inventory

Created: 2026-05-15
Feature: `001-diagnostics-support-matrices`
Scope: pre-plan repository inventory only. No implementation decisions or code changes are made here.

## Source Artifacts Inspected

- `docs/superpowers/plans/prd.md`
- `.specify/memory/constitution.md`
- `specs/001-diagnostics-support-matrices/spec.md`
- `docs/superpowers/state/current-context-pack.md`
- `docs/superpowers/state/implementation-ledger.md`
- `docs/superpowers/state/requirements-verification-matrix.md`
- `docs/superpowers/state/open-blockers.md`
- `git status --short --branch`

## 1. Existing Modules, Classes, and Functions Relevant To This Feature

### Diagnostics-adjacent substrate

- `src/py_module/functions/diagnostics.rs` registers native utility diagnostics: `enumerate_adapters`, `device_probe`, `global_memory_metrics`, `render_debug_pattern_frame`, `engine_info`, `report_device`, and framegraph/demo diagnostics.
- `python/forge3d/_gpu.py` exposes `enumerate_adapters()`, `device_probe()`, and `has_gpu()`.
- `python/forge3d/mem.py` exposes `memory_metrics()`, `update_memory_usage()`, `enforce_memory_budget()`, and `override_memory_limit()`.
- `src/geometry/validate.rs` contains `MeshValidationReport`, `validate_mesh()`, and mesh validation issues. This is not the PRD `ValidationReport`, but it is an existing validation-report pattern.
- No implemented product-level `Diagnostic`, `ValidationReport`, `LayerSummary`, `SupportMatrixEntry`, `RenderFailurePolicy`, or `SeverityPolicy` was found outside specs/docs/state. This finding is based on `rg` searches and should be rechecked before planning.

### Style support

- `python/forge3d/style.py`
  - Dataclasses: `VectorStyle`, `LabelStyle`, `PaintProps`, `LayoutProps`, `StyleLayer`, `StyleSpec`, `SpriteEntry`, `SpriteAtlas`, `GlyphRange`, `FontStack`.
  - Functions: `load_style()`, `parse_style()`, `_parse_layer()`, `evaluate_color_expr()`, `evaluate_number_expr()`, `paint_to_vector_style()`, `layout_to_label_style()`, `layer_to_vector_style()`, `layer_to_label_style()`, `apply_style()`, `parse_color()`, `_evaluate_filter()`, sprite/glyph helpers.
  - `load_style()` and `apply_style()` are Pro-gated through `_check_pro_access()`.
  - `apply_style()` filters render application to visible `fill`, `line`, and `circle` layers.
  - Uncertain: unsupported paint/layout fields appear to be ignored during `_parse_layer()` because only known fields are copied into dataclasses.
- `python/forge3d/style_expressions.py`
  - `EvalContext` and `evaluate()` plus expression helpers including `get`, `interpolate`, `step`, `match`, `case`, `coalesce`, arithmetic/comparison/logical expressions, `concat`, `downcase`, `upcase`, and color conversions.
- `src/style/converters.rs` contains Rust-side style-to-vector/label conversion helpers, including `layout_to_label_style()` and `layer_to_label_style()`.

### Buildings and Pro/fallback paths

- `python/forge3d/buildings.py`
  - Dataclasses: `BuildingMaterial`, `Building`, `BuildingLayer`.
  - Functions: `infer_roof_type()`, `_parse_roof_tag()`, `material_from_tags()`, `material_from_name()`, `add_buildings()`, `add_buildings_cityjson()`, `add_buildings_3dtiles()`.
  - `add_buildings()`, `add_buildings_cityjson()`, and `add_buildings_3dtiles()` are Pro-gated through `_check_pro_access()`.
  - Native fast paths probe `_NATIVE.import_osm_buildings_from_geojson_py` and `_NATIVE.parse_cityjson_py`.
  - Fallback paths construct zero-geometry `Building` entries with `np.zeros((0, 3))` positions and empty indices.
  - `add_buildings_3dtiles()` currently validates tileset metadata and returns an empty `BuildingLayer` with `source_format="3dtiles"`.
- `src/import/osm_buildings.rs`, `src/import/cityjson/*`, and `src/import/building_materials.rs` contain native building import, CityJSON parsing, material inference, and PyO3 functions.

### 3D Tiles

- `python/forge3d/tiles3d.py`
  - Dataclasses/classes: `BoundingVolume`, `TileContent`, `Tile`, `Tileset`, `VisibleTile`, `SseParams`, `Tiles3dRenderer`.
  - Functions: `load_tileset()`, `decode_b3dm()`, `decode_pnts()`.
  - `decode_b3dm()` validates B3DM and embedded GLB presence, then raises `NotImplementedError` for glTF mesh extraction.
  - `decode_pnts()` creates zero positions when `POSITION` is absent.
  - No public Python `Tiles3DLayer` was found.
- `src/tiles3d/*`
  - Native modules include `b3dm.rs`, `pnts.rs`, `tileset.rs`, `tile.rs`, `traversal.rs`, `sse.rs`, `renderer.rs`, `bounds.rs`, and `error.rs`.
  - `src/tiles3d/renderer.rs` has `Tiles3dRenderer`, `TileContent`, `CacheStats`, cache budgeting, SSE configuration, and unsupported-format errors.
  - Uncertain: Rust 3D Tiles APIs are crate-internal substrate; no PyO3 registration for a public Python 3D Tiles renderer was found during this inventory.

### Labels and glyph-related substrate

- `python/forge3d/viewer_ipc.py`
  - Label/vector IPC helpers: `add_label()`, `remove_label()`, `clear_labels()`, `set_labels_enabled()`, `load_label_atlas()`, `add_line_label()`, `add_curved_label()`, `add_callout()`, `set_label_typography()`, `set_declutter_algorithm()`, `add_vector_overlay()`.
- `src/labels/*`
  - `src/labels/mod.rs`: `LabelManager`, `add_label()`, `add_line_label()`, `remove_label()`, `set_label_style()`, `clear()`, `set_enabled()`, `set_zoom()`, `set_max_visible()`, `update_with_camera()`, `upload_to_renderer()`.
  - `src/labels/layer.rs`: native `LabelLayer`, `LabelLayerConfig`, `LabelFeature`, `FeatureGeometry`, `PlacementStrategy`, `GeneratedLabel`, `MapboxSymbolLayer`, `features_from_points()`, `features_from_lines()`, `label_layer_from_mapbox_style()`, `apply_mapbox_text_style()`.
  - `src/labels/declutter.rs`, `src/labels/line_label.rs`, `src/labels/curved.rs`, `src/labels/rtree.rs`, `src/labels/typography.rs`, `src/labels/projection.rs`, `src/labels/callout.rs`, and `src/labels/leader.rs` are relevant to support classification and future diagnostic triggers.
  - `src/labels/py_bindings.rs` exposes only `LabelStyle` and `LabelFlags` to Python.
- `src/viewer/cmd/labels_command.rs`
  - `SetLabelTypography` and `SetDeclutterAlgorithm` print success-like messages and return `true` without visible state mutation in the inspected handler.
  - `AddCurvedLabel` maps to `add_line_label(..., LineLabelPlacement::Along, 0.0)`.
  - Line labels compute glyph placements, but `src/labels/mod.rs` notes that line label rotation is tracked but not applied to glyph rendering.

### Bundles and serializable state

- `python/forge3d/bundle.py`
  - Dataclasses: `TerrainMeta`, `CameraBookmark`, `RasterOverlaySpec`, `SceneBaseState`, `ReviewLayer`, `SceneVariant`, `SceneState`, `BundleManifest`, `LoadedBundle`.
  - Functions: `save_bundle()`, `load_bundle()`, `is_bundle()`, internal coercion/path/validation helpers.
  - `save_bundle()` and `load_bundle()` are Pro-gated through `_check_pro_access()`.
  - Existing state serializes terrain metadata, camera bookmarks, raster overlays, vector overlays, labels, scatter batches, review layers, variants, preset data, assets, and checksums.
  - No explicit diagnostics field was found in `SceneState`, `ReviewLayer`, `SceneVariant`, or `BundleManifest`.
- `src/bundle/*` contains native manifest helpers but does not appear to provide PRD diagnostics serialization.

### CRS, VT, memory, and public API aggregation

- `python/forge3d/crs.py`: `proj_available()`, `transform_coords()`, `reproject_geom()`, `parse_crs_from_wkt()`, `crs_to_epsg()`, `_crs_equal()`, `get_crs_from_rasterio()`, `get_crs_from_geopandas()`.
- `python/forge3d/terrain_params.py`: `VTLayerFamily` and `TerrainVTSettings` accept `albedo`, `normal`, and `mask`, with comments stating that current native runtime pages only `albedo`.
- `src/terrain/render_params/native_vt.rs` and `src/terrain/renderer/virtual_texture.rs` mirror the albedo-only VT runtime; native `register_source()` logs a warning for non-albedo families.
- `python/forge3d/__init__.py` exports/imports style, buildings, bundle, CRS, and VT settings. It does not appear to export `MapScene`, `SceneRecipe`, `LabelPlan`, `ValidationReport`, or `Diagnostic`.
- `python/forge3d/__init__.pyi` includes public stubs for `LabelStyle`/`LabelFlags`, terrain render APIs, memory/device helpers, and bundle imports, but no PRD product diagnostics objects were found.

## 2. Existing Tests and Fixtures Relevant To This Feature

### Tests

- Style:
  - `tests/test_style_parser.py`
  - `tests/test_style_render.py`
  - `tests/test_style_visual.py`
  - `tests/test_style_pixel_diff.py`
  - `tests/test_mapbox_streets_fixture.py`
- Buildings:
  - `tests/test_buildings_roof.py`
  - `tests/test_buildings_materials.py`
  - `tests/test_buildings_extrude.py`
  - `tests/test_buildings_cityjson.py`
- 3D Tiles:
  - `tests/test_3dtiles_parse.py`
  - `tests/test_3dtiles_sse.py`
- Labels and viewer IPC:
  - `tests/test_labels_pybindings.py`
  - `tests/test_api_contracts.py` section `TestLabelBindings`
  - `tests/test_viewer_ipc.py`
- Bundles:
  - `tests/test_bundle_roundtrip.py`
  - `tests/test_bundle_render.py`
  - `tests/test_bundle_cli.py`
- CRS:
  - `tests/test_crs_reproject.py`
  - `tests/test_crs_auto.py`
- VT:
  - `tests/test_tv20_virtual_texturing.py`
- Pro gating:
  - `tests/test_pro_gating.py`
  - `tests/test_license.py`
  - `tests/conftest.py` fixture `pro_license`

### Fixtures and assets

- `tests/fixtures/mapbox_streets_v8.json`
- `assets/geojson/sample_buildings.city.json`
- `assets/geojson/10-270-592.city.json`
- `assets/geojson/mount_fuji_buildings.geojson`
- `assets/fonts/default_atlas.json`
- `assets/tif/Mount_Fuji_30m.tif`
- `assets/tif/dem_rainier.tif`
- `assets/tif/switzerland_dem.tif`
- `assets/tif/switzerland_land_cover.tif`
- `assets/lidar/MtStHelens.laz`
- `python/forge3d/data/sample_boundaries.geojson`
- `tests/golden/terrain/*`

## 3. Existing Docs and Examples Relevant To This Feature

- `README.md` documents labels, overlays, Pro workflow, Pro-gated building imports, Mapbox-style import, and scene bundles.
- `docs/guides/feature_map.md` lists terrain, overlays, labels, device/memory diagnostics, and Pro workflows for buildings, Mapbox-style import, map plates, export, and bundles.
- `docs/guides/data_and_scene_workflows.md` documents viewer overlays/labels, style translation, buildings, and lower-level 3D Tiles parsing/traversal.
- `docs/guides/rendering_and_analysis.md` includes existing diagnostics/device utility guidance.
- `docs/viewer/index.md` documents raw/advanced viewer IPC use.
- `docs/terrain/offline-render-quality.md` documents offline render quality and fallback behavior for denoise.
- `docs/tutorials/python-track/04-scene-bundles.md`
- `docs/tutorials/gis-track/03-build-a-map-plate.md`
- `docs/tutorials/gis-track/04-3d-buildings.md`
- `docs/gallery/02-mount-fuji-labels.md`
- `docs/gallery/05-3d-buildings.md`
- `docs/gallery/08-vector-export.md`
- `docs/gallery/10-map-plate.md`
- Examples:
  - `examples/fuji_labels_demo.py`
  - `examples/style_viewer_interactive.py`
  - `examples/buildings_viewer_interactive.py`
  - `examples/terrain_viewer_interactive.py`
  - `examples/pointcloud_viewer_interactive.py`
  - `examples/sample_style.json`

No dedicated diagnostics reference or support matrices for style, labels, buildings, 3D Tiles, VT, or the PRD taxonomy were found during this inventory. This is uncertain until docs outside the searched paths are reviewed.

## 4. Native/Rust Paths Relevant To This Feature

- Diagnostics/device/memory:
  - `src/py_module/functions/diagnostics.rs`
  - `src/core/memory_tracker/*`
  - `src/core/virtual_texture/*`
  - `src/core/render_bundles.rs`
  - `src/core/render_bundles_types.rs`
- PyO3 registration:
  - `src/lib.rs`
  - `src/py_module/classes.rs`
  - `src/py_module/functions.rs`
  - `src/py_module/functions/io_import.rs`
- Labels:
  - `src/labels/mod.rs`
  - `src/labels/types.rs`
  - `src/labels/py_bindings.rs`
  - `src/labels/layer.rs`
  - `src/labels/declutter.rs`
  - `src/labels/line_label.rs`
  - `src/labels/curved.rs`
  - `src/labels/rtree.rs`
  - `src/labels/typography.rs`
  - `src/labels/projection.rs`
  - `src/viewer/cmd/labels_command.rs`
  - `src/viewer/ipc/protocol/response.rs`
- Style:
  - `src/style/converters.rs`
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
  - `src/tiles3d/b3dm.rs`
  - `src/tiles3d/pnts.rs`
  - `src/tiles3d/tileset.rs`
  - `src/tiles3d/tile.rs`
  - `src/tiles3d/traversal.rs`
  - `src/tiles3d/sse.rs`
  - `src/tiles3d/renderer.rs`
  - `src/tiles3d/error.rs`
- VT:
  - `src/terrain/render_params/native_vt.rs`
  - `src/terrain/renderer/virtual_texture.rs`
- Bundles/review state:
  - `src/bundle/mod.rs`
  - `src/bundle/manifest.rs`
  - `src/viewer/scene_review.rs`
  - `src/viewer/ipc/protocol/request.rs`
  - `src/viewer/ipc/protocol/payloads.rs`
  - `src/viewer/ipc/protocol/translate/scene_review.rs`

## 5. Python Public API Paths Relevant To This Feature

- `python/forge3d/__init__.py`
- `python/forge3d/__init__.pyi`
- `python/forge3d/_license.py`
- `python/forge3d/_gpu.py`
- `python/forge3d/mem.py`
- `python/forge3d/style.py`
- `python/forge3d/style_expressions.py`
- `python/forge3d/buildings.py`
- `python/forge3d/tiles3d.py`
- `python/forge3d/bundle.py`
- `python/forge3d/crs.py`
- `python/forge3d/terrain_params.py`
- `python/forge3d/viewer.py`
- `python/forge3d/viewer.pyi`
- `python/forge3d/viewer_ipc.py`
- `python/forge3d/datasets.py`

Not found as implemented public APIs outside specs/docs/state: `MapScene`, `SceneRecipe`, `LabelPlan`, `ValidationReport`, `Diagnostic`, public Python `LabelLayer`, public Python `Tiles3DLayer`, `SupportMatrixEntry`, `RenderFailurePolicy`, and `SeverityPolicy`.

## 6. Suspected Placeholder, Fallback, or No-Op Paths

- `python/forge3d/buildings.py`
  - GeoJSON fallback creates placeholder buildings with zero positions and zero indices.
  - CityJSON fallback creates placeholder buildings with zero positions and zero indices.
  - `add_buildings_3dtiles()` returns an empty `BuildingLayer` after metadata validation.
- `python/forge3d/tiles3d.py`
  - `decode_b3dm()` raises `NotImplementedError` after validating embedded GLB presence.
  - `decode_pnts()` defaults to zero positions if `POSITION` is absent.
- `python/forge3d/style.py`
  - `_parse_layer()` only preserves known paint/layout keys; unsupported fields are likely dropped without a structured diagnostic.
  - `apply_style()` only applies `fill`, `line`, and `circle`; unsupported layer types and supported-but-non-vector `symbol` layers do not emit structured diagnostics.
- `src/viewer/cmd/labels_command.rs`
  - `SetLabelTypography` prints an update message and returns `true` while inspected arguments are ignored.
  - `SetDeclutterAlgorithm` prints the selected algorithm and returns `true` while inspected settings are not wired to placement.
  - `AddCurvedLabel` is routed as a line label with `LineLabelPlacement::Along`.
- `src/labels/mod.rs`
  - Line labels compute placements, but glyph rendering does not apply line-label rotation.
- `python/forge3d/terrain_params.py`, `src/terrain/render_params/native_vt.rs`, and `src/terrain/renderer/virtual_texture.rs`
  - `normal` and `mask` VT families are accepted as forward-compatible placeholders while native runtime pages only `albedo`.
  - Non-albedo VT registration logs a warning rather than returning a structured map-scene diagnostic.
- `python/forge3d/bundle.py`
  - Existing bundle state has no explicit diagnostics payload field; preserving PRD diagnostics may require schema extension or a separate bundle-ready metadata path.

## 7. Pro-Gated Paths

- `python/forge3d/_license.py`
  - `LicenseError`, `set_license_key()`, and `_check_pro_access()`.
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
- Pro-gating tests exist in `tests/test_pro_gating.py`, and Pro test execution uses the deterministic `pro_license` fixture in `tests/conftest.py`.
- `README.md`, `docs/guides/feature_map.md`, `docs/gallery/05-3d-buildings.md`, `docs/gallery/08-vector-export.md`, `docs/gallery/10-map-plate.md`, `docs/tutorials/python-track/04-scene-bundles.md`, and `docs/tutorials/gis-track/04-3d-buildings.md` mention Pro requirements.

## 8. Unknowns That Need Verification

- Whether feature `001` should introduce product diagnostics in a new module such as `python/forge3d/diagnostics.py`, fold them into future `MapScene`, or both. The spec defines behavior but no implemented product model exists yet.
- Whether bundle diagnostics should be added to `SceneState`, `BundleManifest`, a new sidecar file, or a bundle-ready validation structure that later `MapScene` writes.
- Exact support matrix source of truth: docs location and file names are not yet established for style, labels, buildings, 3D Tiles, VT, diagnostics, and competitive positioning.
- Exact list of supported and unsupported style paint/layout fields. Current `PaintProps`/`LayoutProps` reveal known fields, but unsupported-field detection needs a deliberate field inventory.
- Whether `background` and `symbol` style layers should be classified as supported for parsing only, underdeveloped for render/application, or unsupported for vector overlay output in feature `001`.
- Whether `parse_style()` should remain usable without Pro while `load_style()`/`apply_style()` are Pro-gated, and how diagnostics should report Pro-gated style application.
- Whether building native paths are always present in the open wheel or packaging-dependent. Current Python checks probe native symbols after Pro access succeeds.
- Whether zero-geometry building fallback is still intentional behavior or should be converted to typed diagnostics immediately in feature `001`.
- Whether native Rust 3D Tiles renderer APIs are intentionally not exposed to Python yet, or whether a registration path exists that was missed.
- Whether `estimated_gpu_memory` should use existing `python/forge3d/mem.py`, native `global_memory_metrics`, VT residency stats, 3D Tiles cache stats, or only static estimates for feature `001`.
- Whether missing glyph diagnostics can read `assets/fonts/default_atlas.json` directly or must integrate with native `MsdfAtlas` metrics.
- Whether `LabelLayer` should be public only later; native `src/labels/layer.rs` exists, but no public Python `LabelLayer` was found.
- Whether docs under `docs/_build/` should be ignored as generated output for support-matrix audits.

## Commands Used For Inspection

```powershell
Get-Content -Raw docs/superpowers/plans/prd.md
Get-Content -Raw .specify/memory/constitution.md
Get-Content -Raw specs/001-diagnostics-support-matrices/spec.md
git status --short --branch
Get-ChildItem specs/001-diagnostics-support-matrices -Force | Select-Object Name,Length,LastWriteTime
Get-Content -Raw docs/superpowers/state/current-context-pack.md
Get-Content -Raw docs/superpowers/state/implementation-ledger.md
Get-Content -Raw docs/superpowers/state/requirements-verification-matrix.md
Get-Content -Raw docs/superpowers/state/open-blockers.md
rg -n "class (Diagnostic|ValidationReport|LayerSummary|SupportMatrix|MapScene|SceneRecipe|LabelPlan)|def (validate|render|save_bundle|to_dict|from_dict)|Diagnostic\(|ValidationReport\(|support_level|unsupported_style|missing_glyph|crs_mismatch|estimated_gpu_memory|placeholder_fallback|pro_gated_path|vt_unsupported_family|python_public_3dtiles_incomplete|label_rejection_summary" python src tests docs examples specs/001-diagnostics-support-matrices
rg -n "class .*Style|def .*style|layers|paint|layout|fill|line|circle|unsupported|Mapbox|MVT|vector" python/forge3d/style.py python/forge3d/style_expressions.py tests docs examples
rg -n "class .*Building|def .*building|load_building|CityJSON|GeoJSON|fallback|_native|_forge3d|pro|Pro|zero|empty|placeholder|height" python/forge3d/buildings.py src tests docs examples
rg -n "tiles3d|3D Tiles|B3DM|GLB|class .*Tile|def .*tile|unsupported|fallback|cache|LOD|screen" python/forge3d/tiles3d.py src/tiles3d tests docs examples
rg -n "virtual_texture|VirtualTexture|albedo|normal|mask|family|vt_|page|unsupported" python/forge3d/terrain_params.py src/terrain tests docs examples
rg -n "^(class|def) |^@dataclass|_check_pro_access|fallback|placeholder|not implemented|Unsupported|warnings.warn|return .*(empty|None)|np.zeros|raise ValueError" python/forge3d/style.py python/forge3d/style_expressions.py python/forge3d/buildings.py python/forge3d/tiles3d.py python/forge3d/bundle.py python/forge3d/viewer_ipc.py python/forge3d/viewer.py python/forge3d/crs.py
rg -n "^(pub struct|pub enum|impl |pub fn|fn |#\[pyclass\]|#\[pymethods\]|#\[pyfunction\])|TODO|not implemented|Unsupported|fallback|ignore|warn|warning" src/labels src/viewer src/tiles3d src/import src/bundle src/terrain/render_params/native_vt.rs src/terrain/renderer/virtual_texture.rs src/lib.rs
rg --files tests | rg "(style|building|tiles3d|tile|label|glyph|bundle|crs|virtual|vt|diagnostic|support|matrix|validation|viewer_ipc|mapscene|scene)"
rg --files docs examples | rg "(style|building|tiles3d|tile|label|glyph|bundle|crs|virtual|vt|diagnostic|support|matrix|offline|map|viewer)"
rg -n "MapScene|SceneRecipe|LabelPlan|ValidationReport|Diagnostic|LayerSummary|SupportMatrixEntry|RenderFailurePolicy|SeverityPolicy|LabelLayer|Tiles3DLayer|BuildingLayer|RasterOverlay|VectorOverlay|OutputSpec|OrbitCamera|LightingPreset" python src tests docs examples specs
rg -n "def test_|class Test|pytest|add_buildings|BuildingLayer|placeholder|np.zeros|_check_pro_access|skip|raises|xfail" tests/test_buildings_roof.py tests/test_buildings_materials.py tests/test_buildings_extrude.py tests/test_buildings_cityjson.py
rg -n "def test_|class Test|parse_style|load_style|apply_style|fill|line|circle|symbol|unsupported|Mapbox|_check_pro_access|raises|xfail" tests/test_style_parser.py tests/test_style_render.py tests/test_style_visual.py tests/test_style_pixel_diff.py
rg -n "def test_|class Test|Tileset|load_tileset|decode_b3dm|decode_pnts|SSE|cache|unsupported|raises|xfail" tests/test_3dtiles_parse.py tests/test_3dtiles_sse.py
rg -n "def test_|class Test|Label|glyph|style|typography|declutter|line_label|curved|add_label|viewer_ipc|raises|xfail" tests/test_labels_pybindings.py tests/test_viewer_ipc.py
rg -n "def test_|class Test|Bundle|manifest|SceneState|RasterOverlaySpec|ReviewLayer|SceneVariant|label|vector|diagnostic|metadata|roundtrip|raises|xfail" tests/test_bundle_roundtrip.py tests/test_bundle_render.py tests/test_bundle_cli.py
rg -n "def test_|class Test|VTLayerFamily|TerrainVTSettings|normal|mask|albedo|virtual|unsupported|skip|raises|xfail" tests/test_tv20_virtual_texturing.py tests/test_crs_reproject.py tests/test_crs_auto.py
rg -n "class TestLabelBindings|PyLabelStyle|PyLabelFlags|LabelStyle|LabelFlags|EXPECTED_CLASSES|EXPECTED_FUNCTIONS|MapScene|LabelPlan|Diagnostic|ValidationReport" tests/test_api_contracts.py python/forge3d/__init__.py python/forge3d/__init__.pyi src/lib.rs
rg --files | rg "(fixtures|data|assets|sample|golden).*(geojson|cityjson|json|b3dm|pnts|glb|png|tif|laz|las)$|\.(geojson|cityjson|b3dm|pnts|glb|laz|las)$"
rg -n "diagnostic|support matrix|support status|Mapbox|style|building|3D Tiles|Virtual Texturing|labels|scene bundle|Pro|fallback|experimental|unsupported|offline 3D map" docs/guides docs/tutorials docs/gallery docs/terrain docs/viewer examples README.md pyproject.toml
rg -n "pub fn .*_py|m\.add_function|m\.add_class|building|cityjson|b3dm|pnts|tiles3d|label|bundle|material|vt|style" src/lib.rs src/import src/tiles3d src/labels src/bundle
rg -n "__all__|from \\.|import .*buildings|import .*tiles3d|BuildingLayer|load_tileset|add_buildings|bundle|style|crs|VTLayerFamily|TerrainVTSettings" python/forge3d/__init__.py python/forge3d/__init__.pyi
rg -n "SetLabelTypography|SetDeclutterAlgorithm|AddCurvedLabel|AddCallout|AddLabel|AddLineLabel|success|IpcResponse|TODO|placeholder|not implemented|ack" src/viewer/cmd/labels_command.rs src/viewer/ipc/protocol/response.rs python/forge3d/viewer_ipc.py
```

## Phase 0 Planning Decisions

- Decision: Define product diagnostics as typed Python objects first, with native/Rust hooks used only where existing substrate already exposes device, memory, label, VT, building, or tile facts.
  Rationale: The repository has diagnostic-adjacent substrate but no product-level `Diagnostic` or `ValidationReport`; a Python wrapper keeps MVP workflows typed and serializable without raw IPC.
  Alternatives considered: Rust-first diagnostic model, rejected for planning because several required inputs are currently Python public wrappers or docs/support classifications.

- Decision: Treat support matrices as documentation plus testable diagnostic inventory, not as implementation claims.
  Rationale: The PRD requires support honesty across underdeveloped, Pro-gated, placeholder/fallback, experimental, unsupported, and non-goal paths.
  Alternatives considered: Mark paths supported when a low-level substrate exists, rejected because it would overclaim buildings, 3D Tiles, VT normal/mask, and line/curved labels.

- Decision: Deterministic ordering is part of the validation/report contract.
  Rationale: Bundle-ready reports and review evidence must compare exactly for fixed inputs.
  Alternatives considered: Preserve discovery order, rejected because dict/set/filesystem/native enumeration can drift.

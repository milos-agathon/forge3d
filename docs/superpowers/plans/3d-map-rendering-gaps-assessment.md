# forge3d 3D Map Rendering Gaps Assessment

Date: 2026-05-14

This report assesses the claims in `docs/superpowers/plans/3d-map-rendering-gaps.md`
against the local forge3d codebase. It separates supported claims from claims
that need narrower wording, then adds missed strengths and concrete priorities
for label-engine depth and integration maturity.

## Executive Summary

The plan's main direction is right: forge3d is strongest as an offline,
Python-native terrain, cartographic, and analysis renderer, while the
end-to-end geospatial scene layer is still incomplete.

The report should not frame forge3d as trying to compete with web-first
delivery or hosted tile-provider ecosystems. The more accurate comparison axis
is offline map production: terrain rendering, reproducible snapshots, vector and
raster overlays, cartographic composition, labels, bundles, AOV/offline quality,
and scene-review workflows.

The sharpest framing is:

> forge3d has substantial terrain, geometry, CityJSON/GeoJSON, virtual
> texturing, instancing, cartographic output, and 3D Tiles infrastructure, but
> the public/open, end-to-end map-production layer is still uneven. The highest
> leverage improvements are a deterministic label compiler and a typed scene
> recipe API that unifies terrain, overlays, labels, buildings, point clouds,
> camera, lighting, export, and bundles.

## Supported Claims

### Terrain Strength

The terrain-strength claim is supported.

The docs list terrain viewing, DEM/COG inputs, overlays, vector overlays,
terrain quality controls, and native/offscreen rendering in
`docs/guides/feature_map.md:10`.

The Python type surface exposes major terrain and rendering capabilities:

- `Scene` starts at `python/forge3d/__init__.pyi:98`.
- `Scene` includes SSAO/SSGI/SSR, bloom, cloud shadows, and volumetric clouds
  around `python/forge3d/__init__.pyi:115`.
- `TerrainRenderer` starts at `python/forge3d/__init__.pyi:590`.
- Offline accumulation, HDR resolve, AOV output, and offline rendering are
  exposed around `python/forge3d/__init__.pyi:611` and
  `python/forge3d/__init__.pyi:664`.
- `PointBuffer` is exposed at `python/forge3d/__init__.pyi:1015`.

### Vector-Tile And Style-Engine Gap

The vector-tile/style-engine gap is supported.

`python/forge3d/style.py` parses Mapbox-style JSON, but `load_style()` and
`apply_style()` are Pro-gated at `python/forge3d/style.py:150` and
`python/forge3d/style.py:500`.

`apply_style()` only filters and applies `fill`, `line`, and `circle` layers to
already-provided feature dictionaries. It does not fetch or stream MVT tiles.
See `python/forge3d/style.py:531`.

The style viewer example reads local vectors with GeoPandas and sends
preconverted triangle/line overlays. It is not a streamed vector-tile renderer:

- Local vector read path: `examples/style_viewer_interactive.py:147`.
- Overlay send path: `examples/style_viewer_interactive.py:518`.

### Virtual Texturing Limitation

The VT limitation is exactly correct.

`VTLayerFamily` reserves `albedo`, `normal`, and `mask`, but documents that the
native runtime pages only `albedo`; `normal` and `mask` are forward-compatible
placeholders. See `python/forge3d/terrain_params.py:1249`.

`TerrainVTSettings` repeats that non-albedo layer families are accepted for
future compatibility at `python/forge3d/terrain_params.py:1314`.

Rust enforces the current native limit with
`TERRAIN_VT_SUPPORTED_FAMILY = "albedo"` at
`src/terrain/renderer/virtual_texture.rs:17`. Non-albedo families are skipped
at runtime around `src/terrain/renderer/virtual_texture.rs:529`.

### Textured-Building Material Gap

The textured-building-material gap is mostly correct.

`BuildingMaterial` is scalar PBR only: albedo, roughness, metallic,
emissive, and alpha. See `python/forge3d/buildings.py:42`.

There is general asset/material infrastructure:

- glTF mesh import extracts positions, normals, UVs, and indices in
  `src/io/gltf_read.rs:15` and `src/io/gltf_read.rs:27`.
- OBJ material metadata includes diffuse texture fields in
  `src/io/obj_read.rs:18` and `src/io/obj_read.rs:119`.
- Python PBR texture helpers exist in `python/forge3d/textures.py:69`.

However, that texture infrastructure is not integrated into the building
ingestion/material path. The accurate gap is not "forge3d has no texture
support"; it is "building and city-asset workflows do not yet expose textured
PBR materials end to end."

## Claims That Need Revision

### Building Ingestion Is Not Pure Placeholder

The plan overstates the placeholder claim for GeoJSON and CityJSON buildings.

`add_buildings()` is Pro-gated, but when the native symbol is available it calls
`import_osm_buildings_from_geojson_py` and returns real merged geometry. See
`python/forge3d/buildings.py:262` and `python/forge3d/buildings.py:296`.

`add_buildings_cityjson()` is also Pro-gated, but when the native symbol is
available it calls `parse_cityjson_py` and returns real building meshes. See
`python/forge3d/buildings.py:371` and `python/forge3d/buildings.py:404`.

The zero-geometry fallback branches are real, but they apply when the native
extension path is unavailable. See `python/forge3d/buildings.py:348` and
`python/forge3d/buildings.py:489`.

The correct claim is:

> Open/non-Pro building ingestion is blocked by licensing, and fallback paths
> remain placeholders, but Pro/native GeoJSON and CityJSON ingestion paths are
> real and tested. The remaining gap is public availability and integrated
> end-to-end rendering, not total absence of building geometry support.

### 3D Tiles Infrastructure Is Undercounted

The plan is directionally right that forge3d lacks a polished public 3D Tiles
scene-rendering workflow, but it undercounts Rust-side infrastructure.

Python `decode_b3dm()` validates the container then raises
`NotImplementedError`, so the pure Python public path is incomplete. See
`python/forge3d/tiles3d.py:297` and `python/forge3d/tiles3d.py:343`.

Rust has more substantial support:

- `Tiles3dRenderer::get_visible_tiles()` at `src/tiles3d/renderer.rs:88`.
- `Tiles3dRenderer::load_tile_content()` at `src/tiles3d/renderer.rs:98`.
- B3DM decode at `src/tiles3d/b3dm.rs:61`.
- B3DM GLB geometry parsing at `src/tiles3d/b3dm.rs:141`.
- Cache stats at `src/tiles3d/renderer.rs:182`.

The precise gap is Python/public scene integration and a production-quality
large-tile rendering workflow, not complete absence of 3D Tiles machinery.

### Instancing And LOD Are Understated

The claim that forge3d has no comparable instancing/LOD substrate is too broad.

Native `Scene.add_instanced_mesh()` exists at
`src/scene/py_api/instanced_mesh.rs:13`. Python geometry helpers also expose
native instancing paths at `python/forge3d/geometry.py:286` and
`python/forge3d/geometry.py:336`.

Terrain clipmap and GPU LOD infrastructure also exists:

- Clipmap module: `src/terrain/clipmap/mod.rs:1`.
- GPU LOD selector: `src/terrain/clipmap/gpu_lod.rs:188`.

The accurate gap is a higher-level geospatial scene graph and workflow, not
zero instancing or zero LOD capability.

## Verification

Native introspection was run locally and confirmed the installed extension
exposes the relevant native symbols, including:

- `parse_cityjson_py`
- `import_osm_buildings_from_geojson_py`
- `io_import_gltf_py`
- `geometry_instance_mesh_py`
- `geometry_instance_mesh_gpu_render_py`
- `Scene.add_instanced_mesh`

Focused tests passed with this command:

```powershell
python -m pytest tests/test_pro_gating.py tests/test_buildings_cityjson.py::test_sample_cityjson_dataset_contains_renderable_geometry tests/test_3dtiles_parse.py tests/test_3dtiles_sse.py tests/test_tv20_virtual_texturing.py::test_native_terrain_vt_runtime_is_currently_albedo_only -q
```

Result:

```text
39 passed in 1.09s
```

Bottom line: keep the plan's main direction, but revise the
building/3D Tiles/instancing sections. The sharpest accurate framing is that
forge3d has substantial terrain, geometry, CityJSON/GeoJSON, VT, instancing,
and 3D Tiles infrastructure, but the public/open, end-to-end map-engine layer is
still incomplete.

## Missed Strengths

### Cartographic Production Stack

forge3d has map-plate composition, legends, scale bars, north arrows, labels,
and SVG/PDF export paths. That matters when comparing against Blender,
rayshader, and static-map workflows, not just Cesium or Mapbox.

Relevant anchors:

- `docs/guides/feature_map.md:26`
- `python/forge3d/map_plate.py:116`
- `python/forge3d/export.py:486`
- `python/forge3d/export.py:523`

### Offline Render And Analysis Pipeline

AOVs, HDR/offline accumulation, denoise settings, and offline quality controls
are real differentiators versus browser map engines. The assessment should
compare this against Blender/rayshader/offline rendering workflows, not only
interactive GIS tools.

Relevant anchors:

- `docs/terrain/offline-render-quality.md`
- `tests/test_aov.py:1`
- `tests/test_denoise_settings.py:1`
- `python/forge3d/__init__.pyi:611`
- `python/forge3d/__init__.pyi:664`

### Interaction And Inspection Tooling

Picking, lasso/highlight, overlays, labels, and OIT/transparent rendering are
important for QA and analysis workflows and were not emphasized enough.

Relevant anchors:

- `docs/guides/feature_map.md:13`
- `docs/guides/feature_map.md:14`
- `src/scene/py_api/oit.rs`

### Terrain Streaming Substrate

COG window/tile reads, cache statistics, clipmap generation, GPU LOD pieces, and
memory-budgeted terrain infrastructure are stronger than the plan implies.

Caveat: this is not the same as a finished Cesium-grade global globe runtime.
For forge3d's offline goal, the important point is that the terrain streaming
substrate is more substantial than the plan gives credit for.

Relevant anchors:

- `tests/test_cog_streaming.py:1`
- `tests/test_cog_streaming.py:230`
- `src/terrain/clipmap/mod.rs:1`
- `src/terrain/clipmap/gpu_lod.rs:188`

### General Render Core

Native instancing, glTF/OBJ import, GPU polygon extrusion, BRDF shader work,
SDF/path-tracing modules, and PBR texture concepts exist. The missing nuance is
that many of these are not yet assembled into a polished geospatial city/map
runtime.

Relevant anchors:

- `src/scene/py_api/instanced_mesh.rs:13`
- `src/io/gltf_read.rs:15`
- `src/vector/api/extrusion.rs:10`
- `python/forge3d/textures.py:69`

## Label Engine Depth

Given forge3d's offline-rendering goal, it should not chase Mapbox/Cesium label
parity as a product goal. A better target is a deterministic cartographic label
compiler:

```text
terrain + vector/building/point inputs + camera + output size
    -> stable label plan
    -> sharp snapshot/SVG/PDF/review-bundle output
```

This aligns with offline rendering because the camera, viewport, output
resolution, terrain transform, and cartographic furniture are known before
rendering. forge3d can therefore optimize for reproducibility and output
quality instead of tiled, live, web-map collision behavior.

### Current Label Reality

forge3d already has a useful label substrate:

- `LabelManager`
- atlas-based glyph rendering
- R-tree collision
- priorities
- zoom ranges
- horizon fade
- leader lines
- viewer IPC helpers

Relevant anchors:

- `src/labels/mod.rs:51`
- `src/labels/mod.rs:119`
- `src/labels/mod.rs:139`
- `python/forge3d/viewer_ipc.py:206`
- `python/forge3d/viewer_ipc.py:342`
- `python/forge3d/viewer_ipc.py:430`

Several pieces are not mature yet:

- `add_label()` is the real path today.
- `add_line_label()` and `add_curved_label()` compute or route line-label data,
  but line glyphs are not emitted into render instances yet.
- The code notes that line labels track rotation but glyph rendering does not
  apply it yet at `src/labels/mod.rs:436`.
- `set_label_typography()` and `set_declutter_algorithm()` are currently IPC
  acknowledgements, not real state changes. See
  `src/viewer/cmd/labels_command.rs:247` and
  `src/viewer/cmd/labels_command.rs:256`.
- Python docs say label responses may include IDs, but generic IPC success
  responses do not currently carry created IDs. See
  `src/viewer/ipc/protocol/response.rs:7` and
  `src/viewer/ipc/protocol/response.rs:30`.
- The default atlas is limited; examples explicitly filter to ASCII labels.
  See `examples/fuji_labels_demo.py:184` and `examples/picking_demo.py:224`.

### Layer 1: Make Existing Label Commands Truthful And Complete

The first milestone should be conservative and high value:

- Add `created_id` or `id` to IPC responses for labels, line labels, callouts,
  and overlays where relevant.
- Wire `set_label_typography()` into `LabelManager` instead of printing.
- Wire `set_declutter_algorithm()` into actual placement selection.
- Fully implement `add_line_label()` and `add_curved_label()` rendering, or mark
  them experimental until they render correctly.
- Add high-level `ViewerHandle.add_label()`, `ViewerHandle.add_labels()`,
  `ViewerHandle.load_label_atlas()`, `ViewerHandle.clear_labels()`, and
  `ViewerHandle.set_labels_enabled()` methods so users do not need raw IPC for
  common label workflows.

This would turn labels from demo-accessible into API-stable.

### Layer 2: Implement Real Static Placement

For offline rendering, static placement matters more than live pan/zoom
behavior. Add a `LabelPlan` stage:

```text
features + style + camera + output_size + terrain sampling
    -> candidate anchors
    -> score/collision solve
    -> accepted labels
    -> render/export payload
```

The placement engine should support:

- point labels: center, above, below, left, right, and radial candidates
- line labels: along-path, center-on-line, repeat distance, and upside-down
  avoidance
- polygon labels: centroid plus visual-center/polylabel fallback
- terrain labels: automatic elevation sampling from DEM/terrain transform
- priority classes: capitals, cities, rivers, peaks, roads, and annotations
- keepout regions: title boxes, legends, scale bars, and manually reserved areas
- deterministic seeds for reproducible exports

This is more valuable for forge3d than tile-pyramid collision.

### Layer 3: Upgrade Typography And Font Handling

The current atlas path is enough for demos, not production cartography.

Priorities:

- bundled default Latin atlas with documented coverage
- runtime or build-time atlas generation from TTF/OTF
- font fallback ranges
- Unicode text coverage reports before rendering
- kerning, tracking, and line-height actually applied
- multiline labels and callouts
- optional shaping through a HarfBuzz-compatible stack for non-Latin scripts

Do not start with full international shaping unless the product needs it
immediately. Add coverage diagnostics early, for example:

```text
37 glyphs are missing from the active atlas.
```

### Layer 4: Connect Labels To Style And Data

There is already Mapbox-style parsing in `python/forge3d/style.py:401`, but it
stops short of becoming a real label workflow.

Add:

- `LabelLayer.from_geodataframe(...)`
- `LabelLayer.from_features(...)`
- `LabelLayer.from_style_layer(style_layer, features, text_field=...)`
- expression support for `{name}`, simple `get`, `concat`, `coalesce`, and
  casing
- automatic CRS transform using `forge3d.crs`
- terrain-height sampling for world positions
- `compile_labels(camera, viewport, terrain)` returning a `LabelPlan`

That would make style import useful for offline map production without
pretending to be Mapbox GL.

## Integration Maturity

The bigger architectural gap is that many workflows live in examples instead of
a unified scene API.

forge3d already has useful parts:

- `ViewerHandle` at `python/forge3d/viewer.py:148`
- scene bundles and review layers at `python/forge3d/bundle.py:333`
- raster overlays through `ViewerHandle.load_overlay()`
- vector overlays through `viewer_ipc.add_vector_overlay()`
- labels through `viewer_ipc.add_label()`
- point clouds through `ViewerHandle.load_point_cloud()`
- buildings through `forge3d.buildings`
- SVG/PDF export through `forge3d.export`
- CRS helpers through `forge3d.crs`

The missing piece is a single typed scene recipe.

Illustrative API shape:

```python
scene = f3d.MapScene(
    terrain=TerrainSource(path=dem, crs="EPSG:32632"),
    camera=OrbitCamera(...),
    lighting=LightingPreset(...),
    layers=[
        RasterOverlay(...),
        VectorOverlay(...),
        LabelLayer(...),
        BuildingLayer(...),
        PointCloudLayer(...),
    ],
    output=OutputSpec(width=3840, height=2160),
)

scene.validate()
scene.render("map.png")
scene.save_bundle("map.forge3d")
```

The point is ownership, not exact syntax. Today the user has to know which
combination of `viewer`, `viewer_ipc`, `style`, `bundle`, `export`, `crs`, and
example code to stitch together. A `MapScene` or `SceneRecipe` layer would make
forge3d feel like one tool.

### Recommended Priority

1. Fix label IPC/API truth: IDs, real typography state, real declutter state,
   and high-level `ViewerHandle` methods.
2. Render line and curved labels properly.
3. Add `LabelPlan` as deterministic offline placement output.
4. Add typed `LabelLayer` ingestion from GeoDataFrame, feature dictionaries, and
   style layers.
5. Fold terrain, overlays, labels, buildings, point clouds, lighting, camera,
   export, and bundles into a `MapScene` or `SceneRecipe` API.
6. Add production diagnostics: CRS mismatch, missing glyphs, label count by
   class, rejected labels by reason, estimated GPU memory, and unsupported style
   fields.

## Final Bottom Line

forge3d should strengthen its offline rendering identity rather than chase
web-first delivery or hosted tile providers.

The highest-leverage next step is not "be Mapbox" or "be Cesium." It is to make
forge3d a coherent offline map-production engine:

- deterministic label planning
- typed scene recipes
- reproducible snapshots and bundles
- strong diagnostics
- export paths that preserve cartographic intent

That would directly improve forge3d where it is already best positioned:
high-quality, reproducible offline terrain and cartographic rendering.

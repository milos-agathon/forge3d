# Data And Scene Workflows

This guide covers how forge3d gets terrain and other scene assets into either
the interactive viewer or the lower-level rendering path.

## Terrain Inputs

forge3d primarily works with terrain height data in these forms:

- bundled sample arrays from `forge3d.mini_dem()` and `forge3d.mini_dem_path()`
- on-demand tutorial datasets from `forge3d.fetch_dem(...)`
- GeoTIFF assets passed directly to `open_viewer_async(...)`
- COG-backed datasets accessed through `forge3d.cog`

The common high-level path is:

```python
import forge3d as f3d

dem_path = f3d.fetch_dem("rainier")
with f3d.open_viewer_async(terrain_path=dem_path) as viewer:
    viewer.snapshot("rainier.png")
```

When you need direct COG access rather than viewer loading, use
`forge3d.cog.open_cog(...)` and read windows or tiles yourself. That is the
workflow shown by `cog_streaming_demo.py`.

## Datasets And CRS Helpers

`forge3d.datasets` provides the sample registry used throughout the docs,
tutorials, and tests:

- `available()`, `list_datasets()`, and `dataset_info()` enumerate datasets
- `fetch_dem()`, `fetch_cityjson()`, and `fetch_copc()` resolve concrete assets
- `fetch_dataset()` is the more general entry point for named artifacts

`forge3d.crs` handles coordinate-system glue for mixed GIS inputs:

- `transform_coords()` and `reproject_geom()` for geometry transforms
- `crs_to_epsg()`, `get_crs_from_rasterio()`, and `get_crs_from_geopandas()` for CRS discovery

Use those helpers when you are feeding overlays, labels, buildings, or point
clouds that do not already match the DEM coordinate system.

## Raster Overlays

Raster overlays are part of the high-level viewer surface:

```python
viewer.load_overlay(
    name="land-cover",
    path=overlay_path,
    opacity=0.82,
    z_order=10,
    preserve_colors=True,
)
```

That is the main pattern in:

- `swiss_terrain_landcover_viewer.py`
- `bosnia_terrain_landcover_viewer.py`
- `belgium_bivariate_climate_map.py`
- `poland_population_spikes_height_shade.py`
- `pnoa_river_showcase_video.py`

Those examples also show the common real-world pattern of taking a raw viewer
snapshot and then compositing final labels, shadows, or layout outside the
viewer with PIL, Matplotlib, or NumPy.

## Vector Overlays, Labels, And Styles

There are two vector paths:

### 1. Viewer overlays and labels

Use the raw IPC helpers when the content belongs inside the 3D scene:

- `forge3d.viewer_ipc.add_vector_overlay`
- `forge3d.viewer_ipc.add_label`
- `forge3d.viewer_ipc.set_labels_enabled`
- `ViewerHandle.send_ipc(...)`

Examples:

- `luxembourg_rail_overlay.py`
- `fuji_labels_demo.py`
- `picking_demo.py`

### 2. Style translation and 2D vector scenes

Use `forge3d.style` when you need Mapbox-style parsing and translation into
forge3d vector or label settings for local/provided feature styling. This is
not full Mapbox GL parity and does not stream MVT tiles. The documented P0
style subset is `fill, line, and circle`; unsupported style fields and layer
types must surface as `unsupported_style_field` and
`unsupported_style_layer_type` diagnostics before render.

For feature `005-map-assets-bundles-p1`, typed data-driven labels live in
`forge3d.map_scene.LabelLayer` through `from_features`, `from_geodataframe`,
`from_style_layer`, and `compile_labels`. These APIs feed the product
`MapScene`/`LabelPlan` path and are `underdeveloped` until story-specific P1
tests complete.

P1 typography support is explicit about what is available. Use
`FontAtlas.default_latin` for the bundled Basic Latin atlas,
`FontAtlas.from_font` for TTF/OTF atlas setup or typed asset diagnostics,
`FontFallbackRange` for deterministic fallback declarations, and
`TypographySettings` for kerning, tracking, line-height, multiline, and callout
layout metadata. Complex-script shaping uses the native LITTERA font-chain,
bidi, positioned-outline, and analytic-coverage pipeline; unsupported Unicode coverage must be reported before
render through typed diagnostics such as `unicode_coverage_gap` or
`missing_glyphs`.

Example:

- `style_viewer_interactive.py`

## Point Clouds

Point-cloud workflows split between the live viewer and utility helpers.

### Viewer path

```python
with f3d.open_viewer_async() as viewer:
    viewer.load_point_cloud("MtStHelens.laz", point_size=2.0, max_points=250_000)
    viewer.set_point_cloud_params(color_mode="rgb", visible=True)
```

### Module path

`forge3d.pointcloud` covers data-side helpers such as `PointBuffer`,
`copc_laz_enabled()`, and `read_laz_points_info()`.

Example:

- `pointcloud_viewer_interactive.py`

## Buildings And 3D Tiles

Building-oriented workflows live in `forge3d.buildings`:

- `add_buildings()` for footprint-based sources
- `add_buildings_cityjson()` for CityJSON
- `add_buildings_3dtiles()` for 3D Tiles-backed metadata flows
- material helpers such as `material_from_name()` and `material_from_tags()`

`buildings_viewer_interactive.py` shows how those building meshes can be turned
into viewer geometry. The GIS tutorial and gallery entries use the same basic
pipeline.

`forge3d.tiles3d` is the lower-level module for 3D Tiles parsing and traversal.
Use it when you need direct control rather than the building convenience layer.
The typed product-scene path is `forge3d.map_scene.Tiles3DLayer`, including
`Tiles3DLayer.from_tileset_json` and `Tiles3DLayer.from_b3dm`. This is not full Cesium runtime parity; unsupported formats, unsupported B3DM/GLB features, and
incomplete public render paths must report `unsupported_tile_format`,
`unsupported_tile_feature`, or `python_public_3dtiles_incomplete` diagnostics.

The product-scene building adapter is `forge3d.map_scene.BuildingLayer`, also
available as the top-level `MapSceneBuildingLayer` alias. The legacy
`forge3d.BuildingLayer` name remains the `forge3d.buildings` class for
backward compatibility. Product-scene building paths stay `underdeveloped`,
`Pro-gated`, `placeholder/fallback`, or `unsupported` unless the selected path
has specific test evidence.
scalar PBR material metadata is supported as review and summary data on the
typed building layer. textured PBR is not implemented end to end in the public
MapScene path; requests for textured PBR must remain diagnostic-bearing with
`unsupported_feature` rather than silently falling back to untextured geometry.

## Scene Bundles

Bundles are the packaging format for portable scene state:

- `save_bundle()` writes a `.forge3d` directory
- `load_bundle()` reads it back
- `ViewerHandle.load_bundle()` installs it into a live viewer
- `MapScene.save_bundle()` writes deterministic recipe, review, layer-source,
  label-source, and validation-report payloads for typed map scenes
- `MapScene.load_bundle()` reloads the saved recipe and diagnostics where the
  assets are available

This is the path for repeatable scene variants, saved bookmarks, and shipping a
reviewable scene between machines. Missing external bundle assets remain
diagnostic-bearing through `missing_external_asset`; bundle save/load must not
turn missing, unsupported, `Pro-gated`, `placeholder/fallback`, or
`experimental` asset state into successful renderability. See `terrain_demo.py`
and Python tutorial 04.

## Example Map

- `terrain_single_tile.py`: minimal data-to-image sanity check
- `cog_streaming_demo.py`: direct COG tile/window access
- `swiss_terrain_landcover_viewer.py` and `bosnia_terrain_landcover_viewer.py`: terrain + raster overlay
- `luxembourg_rail_overlay.py`: vector overlays
- `fuji_labels_demo.py`: labels and decluttering
- `pointcloud_viewer_interactive.py`: point-cloud loading
- `buildings_viewer_interactive.py`: building geometry pipelines
- `style_viewer_interactive.py`: style translation
- `terrain_demo.py`: bundle-aware scene CLI

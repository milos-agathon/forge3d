# Feature Map

This page is the shortest complete map of the repo's user-facing surface. It is
organized by workflow rather than by Rust module layout.

## Open-Source Core

| Area | Main APIs | Covered by examples |
| --- | --- | --- |
| Interactive terrain viewing | `open_viewer_async`, `ViewerHandle`, `ViewerWidget` | `examples/terrain_viewer_interactive.py`, `examples/terrain_camera_rigs_demo.py` |
| Terrain inputs and datasets | `mini_dem`, `fetch_dem`, `datasets`, `cog.open_cog` | `examples/terrain_demo.py`, `examples/mapscene_bundled_datasets_showcase.py`; no dedicated tracked COG script |
| Raster overlays | `ViewerHandle.load_overlay` | `examples/swiss_terrain_landcover_viewer.py`, `examples/bosnia_terrain_landcover_viewer.py`, `examples/population_ghsl/iberia_builtup_cover_3d.py` |
| Vector overlays and labels | `ViewerHandle.add_vector_overlay`, `ViewerHandle.add_label`, `ViewerHandle.add_labels`, `ViewerHandle.add_line_label`, `ViewerHandle.add_callout` | `examples/label_api_truth_basic.py` |
| Typed MapScene labels | `MapScene`, `LabelLayer` | `examples/fuji_labels_demo.py`, `examples/mapscene_vector_labels.py` |
| Picking and selection | `viewer_ipc` picking helpers | No dedicated tracked runnable example |
| Point clouds | `ViewerHandle.load_point_cloud`, `forge3d.pointcloud` | `examples/pointcloud_viewer_interactive.py` |
| Camera automation | `forge3d.animation`, `forge3d.camera_rigs` | `examples/camera_animation_demo.py`, `examples/terrain_camera_rigs_demo.py` |
| Terrain quality controls | `terrain_params`, `presets`, `terrain_scatter` | `examples/colorado_rem_forge3d.py`, `examples/platte_rem_forge3d.py` |
| Native/offscreen rendering | `Session`, `TerrainRenderer`, `MapScene.render` | `examples/platte_rem_forge3d.py`, `examples/uk_ireland_lighthouse_map.py`, `examples/mapscene_terrain_raster.py` |
| Typed map scenes | `MapScene`, `LabelLayer`, `MapSceneBuildingLayer`, `Tiles3DLayer`, `MapScene.save_bundle`, `MapScene.load_bundle` | `examples/mapscene_terrain_raster.py`, `examples/mapscene_vector_labels.py`, `examples/mapscene_buildings_labels.py` |
| Geometry, mesh, vector, SDF, path tracing | `geometry`, `mesh`, `vector`, `sdf`, `path_tracing` | API-level usage; not every module has a dedicated showcase script |
| Device and memory diagnostics | `has_gpu`, `device_probe`, `mem` | diagnostics and tooling flows rather than gallery scripts |

`examples/luxembourg_rail_overlay.py` remains a lower-level raw-IPC
compatibility example for vector overlays.

## Pro Workflows

| Area | Main APIs | Covered by examples or tutorials |
| --- | --- | --- |
| Map plates and cartographic furniture | `MapPlate`, `Legend`, `ScaleBar`, `NorthArrow` | `examples/notebooks/map_plate.ipynb`, GIS tutorial 03 |
| Vector export | `export_svg`, `export_pdf`, `VectorScene` | gallery vector-export workflow |
| Buildings | `add_buildings`, `add_buildings_cityjson`, `add_buildings_3dtiles` | `examples/osm_city_demo.py`, `examples/mapscene_buildings_labels.py`, GIS tutorial 04 |
| Mapbox-style import | `style.load_style`, `style.apply_style` | `examples/sample_style.json` is an input asset; no dedicated tracked runnable example |
| Scene bundles | `save_bundle`, `load_bundle`, `ViewerHandle.load_bundle`, `MapScene.save_bundle` | `examples/terrain_demo.py`, `examples/mapscene_offline_quality.py`, Python tutorial 04 |

## Core Package Modules

The main package surface splits naturally into a few groups:

- Viewer and notebook control: `forge3d.viewer`, `forge3d.viewer_ipc`, `forge3d.widgets`, `forge3d.interactive`
- Terrain configuration: `forge3d.terrain_params`, `forge3d.presets`, `forge3d.terrain_scatter`
- Scene animation and rigs: `forge3d.animation`, `forge3d.camera_rigs`
- Terrain and asset loading: `forge3d.datasets`, `forge3d.crs`, `forge3d.cog`, `forge3d.pointcloud`, `forge3d.tiles3d`
- Production-oriented scene assets: `forge3d.buildings`, `forge3d.style`, `forge3d.bundle`, `forge3d.map_plate`, `forge3d.export`
- Lower-level rendering and geometry: `forge3d.geometry`, `forge3d.io`, `forge3d.mesh`, `forge3d.vector`, `forge3d.sdf`, `forge3d.path_tracing`, `forge3d.lighting`

`forge3d.map_scene` is the typed offline map-production surface. P1 asset
adapters such as `LabelLayer`, `MapSceneBuildingLayer`, and `Tiles3DLayer` are
`underdeveloped` until their feature `005` story tests complete, and unsupported
or incomplete paths must remain diagnostic-bearing rather than silently
renderable.

## What The Examples Directory Actually Covers

`examples/` is not just a gallery dump. It spans:

- minimal PNG and array round-trips
- baseline terrain viewing
- raster overlays and terrain composition
- labels, styles, vector overlays, and picking
- point clouds, buildings, and streaming terrain inputs
- camera animation, camera rigs, and cinematic rendering
- pure-Python post-processing on top of forge3d snapshots
- notebooks for quickstart, terrain exploration, and map plates

Use the [Examples Catalog](../examples/index.md) when you want the exact file
for one of those jobs.

## Choosing The Right Entry Point

- Start with `open_viewer_async()` if you want a live scene.
- Add `ViewerWidget` when the same workflow needs to run inside Jupyter.
- Use `Scene` or `TerrainRenderer` when you need an explicit offscreen pipeline.
- Use `ViewerHandle` methods for the public label workflow. Use `viewer_ipc`
  only for advanced compatibility commands that do not yet have a truthful
  high-level wrapper.
- Reach for Pro modules only after you already have a stable viewer or renderer workflow.

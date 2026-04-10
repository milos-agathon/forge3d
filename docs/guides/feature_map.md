# Feature Map

This page is the shortest complete map of the repo's user-facing surface. It is
organized by workflow rather than by Rust module layout.

## Open-Source Core

| Area | Main APIs | Covered by examples |
| --- | --- | --- |
| Interactive terrain viewing | `open_viewer_async`, `ViewerHandle`, `ViewerWidget` | `terrain_viewer_interactive.py`, `terrain_camera_rigs_demo.py` |
| Terrain inputs and datasets | `mini_dem`, `fetch_dem`, `datasets`, `cog.open_cog` | `terrain_single_tile.py`, `cog_streaming_demo.py` |
| Raster overlays | `ViewerHandle.load_overlay` | `swiss_terrain_landcover_viewer.py`, `bosnia_terrain_landcover_viewer.py`, `belgium_bivariate_climate_map.py` |
| Vector overlays and labels | `viewer.send_ipc`, `viewer_ipc.add_vector_overlay`, `viewer_ipc.add_label` | `luxembourg_rail_overlay.py`, `fuji_labels_demo.py` |
| Picking and selection | `viewer_ipc` picking helpers | `picking_demo.py`, `picking_test_interactive.py` |
| Point clouds | `ViewerHandle.load_point_cloud`, `forge3d.pointcloud` | `pointcloud_viewer_interactive.py` |
| Camera automation | `forge3d.animation`, `forge3d.camera_rigs` | `camera_animation_demo.py`, `terrain_camera_rigs_demo.py` |
| Terrain quality controls | `terrain_params`, `presets`, `terrain_scatter` | `terrain_atmosphere_path_demo.py`, `pnoa_river_showcase_video.py` |
| Native/offscreen rendering | `Scene`, `Session`, `TerrainRenderer`, `render_offline` | `terrain_atmosphere_path_demo.py`, `triangle_png.py` |
| Geometry, mesh, vector, SDF, path tracing | `geometry`, `mesh`, `vector`, `sdf`, `path_tracing` | API-level usage; not every module has a dedicated showcase script |
| Device and memory diagnostics | `has_gpu`, `device_probe`, `mem` | diagnostics and tooling flows rather than gallery scripts |

## Pro Workflows

| Area | Main APIs | Covered by examples or tutorials |
| --- | --- | --- |
| Map plates and cartographic furniture | `MapPlate`, `Legend`, `ScaleBar`, `NorthArrow` | `map_plate.ipynb`, GIS tutorial 03 |
| Vector export | `export_svg`, `export_pdf`, `VectorScene` | gallery vector-export workflow |
| Buildings | `add_buildings`, `add_buildings_cityjson`, `add_buildings_3dtiles` | `buildings_viewer_interactive.py`, GIS tutorial 04 |
| Mapbox-style import | `style.load_style`, `style.apply_style` | `style_viewer_interactive.py` |
| Scene bundles | `save_bundle`, `load_bundle`, `ViewerHandle.load_bundle` | `terrain_demo.py`, Python tutorial 04 |

## Core Package Modules

The main package surface splits naturally into a few groups:

- Viewer and notebook control: `forge3d.viewer`, `forge3d.viewer_ipc`, `forge3d.widgets`, `forge3d.interactive`
- Terrain configuration: `forge3d.terrain_params`, `forge3d.presets`, `forge3d.terrain_scatter`
- Scene animation and rigs: `forge3d.animation`, `forge3d.camera_rigs`
- Terrain and asset loading: `forge3d.datasets`, `forge3d.crs`, `forge3d.cog`, `forge3d.pointcloud`, `forge3d.tiles3d`
- Production-oriented scene assets: `forge3d.buildings`, `forge3d.style`, `forge3d.bundle`, `forge3d.map_plate`, `forge3d.export`
- Lower-level rendering and geometry: `forge3d.geometry`, `forge3d.io`, `forge3d.mesh`, `forge3d.vector`, `forge3d.sdf`, `forge3d.path_tracing`, `forge3d.lighting`

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
- Use `viewer_ipc` only when the higher-level handle does not expose the command you need.
- Reach for Pro modules only after you already have a stable viewer or renderer workflow.

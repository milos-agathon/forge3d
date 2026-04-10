# Output And Integration

forge3d does not stop at a viewer window. This guide covers how the repo turns
scene state into files, notebooks, and downstream products.

## Snapshots And Frame Sequences

The simplest output path is a viewer snapshot:

```python
with f3d.open_viewer_async(terrain_path=f3d.fetch_dem("rainier")) as viewer:
    viewer.snapshot("rainier.png", width=1920, height=1080)
```

For repeated capture, pair the viewer with animation or camera-rig helpers and
write frames to disk. That is the basic pattern in:

- `camera_animation_demo.py`
- `terrain_camera_rigs_demo.py`
- `pnoa_river_showcase_video.py`

For lower-level helper functions, see the top-level re-exports from
`forge3d.helpers.offscreen` and `forge3d.helpers.frame_dump`.

## Notebook Integration

`ViewerWidget` is the notebook-side integration layer. It keeps the viewer and
IPC model intact while giving you an embeddable front end inside Jupyter.

The notebooks in `examples/notebooks/` cover:

- `quickstart.ipynb`: basic terrain and viewer usage
- `terrain_explorer.ipynb`: notebook-driven exploration
- `map_plate.ipynb`: map-plate composition flow

## Pure-Python Composition On Top Of forge3d

Several examples use forge3d to render the base terrain image and then compose
the final deliverable outside the viewer:

- `belgium_bivariate_climate_map.py`
- `bosnia_terrain_landcover_viewer.py`
- `poland_population_spikes_height_shade.py`
- `pnoa_river_showcase.py`
- `pnoa_river_showcase_video.py`

That is a supported pattern. The viewer gives you the terrain image; PIL,
Matplotlib, NumPy, or other Python tools can handle annotations, poster layout,
or video assembly.

## Map Plates And Cartographic Furniture

The map-plate workflow is the main packaged-output layer on top of viewer
snapshots.

Core APIs:

- `MapPlate`
- `MapPlateConfig`
- `Legend` and `LegendConfig`
- `ScaleBar` and `ScaleBarConfig`
- `NorthArrow` and `NorthArrowConfig`

Use this workflow when you need titles, legends, scale bars, north arrows, and
panel layouts around a rendered map image.

See:

- `examples/notebooks/map_plate.ipynb`
- GIS tutorial 03
- gallery entry 10

## Vector Export

`forge3d.export` is the 2D vector-output path. It is separate from the raster
snapshot path and is intended for publication-style SVG/PDF generation.

Core APIs:

- `VectorScene`
- `generate_svg()`
- `export_svg()`
- `export_pdf()`
- `validate_svg()`

Use this path when the output needs to stay vector rather than being baked into
a PNG snapshot.

## Bundles And Repeatable Scenes

Bundles are the packaging and handoff format for saved scenes:

- `save_bundle()`
- `load_bundle()`
- `ViewerHandle.load_bundle()`

They matter when you need repeatable review scenes, saved presets, camera
bookmarks, or a stable asset package for another user or another machine.

See:

- `terrain_demo.py`
- Python tutorial 04

## CLI And Integration Entry Points

Beyond the viewer itself, the repo includes a few integration-oriented helpers:

- `terrain_demo.py` for the terrain-demo CLI wrapper
- `forge3d.terrain_pbr_pom` for scripted terrain rendering helpers
- `forge3d.viewer_ipc` for explicit command-based integration
- `forge3d.interactive` for example-side command loops

Use those when you are building your own wrapper or tool around forge3d rather
than following the shortest tutorial path.

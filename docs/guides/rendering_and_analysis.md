# Rendering And Analysis

This guide covers the rendering-side modules: what drives image generation, how
terrain quality is configured, and where the lower-level helpers fit.

## Viewer-First Rendering

For most users, forge3d rendering begins with the viewer:

- `open_viewer_async()` launches the scene runtime
- `ViewerHandle` applies terrain, camera, sun, overlay, and point-cloud changes
- `terrain_params` and `presets` define reproducible scene settings
- `animation` and `camera_rigs` automate camera motion
- `terrain_scatter` adds deterministic terrain population and related helpers

This is the path behind most of the terrain, overlay, point-cloud, and camera
examples in the repo.

## Terrain Configuration

`forge3d.terrain_params` is the central configuration module for advanced
terrain rendering. It groups settings into focused dataclasses such as:

- lighting, shadows, fog, reflections, bloom, height AO, and sun visibility
- material noise, POM, triplanar, and other terrain surface controls
- LOD, sampling, clamp, denoise, and offline quality settings
- terrain virtual-texture settings and overlay-related settings

Use it directly when you are building repeatable terrain render presets or
feeding `TerrainRenderer`.

## Native And Offscreen Rendering

forge3d exposes two lower-level rendering layers.

### `Scene`

Use `Scene` when you want a compact native offscreen surface. It is suitable for
image generation, quick experiments, and targeted feature testing.

### `Session` + `TerrainRenderer`

Use this stack when you need the full terrain-native pipeline:

- `Session` owns the native runtime context
- `MaterialSet` defines terrain materials
- `IBL` provides environment maps
- `TerrainRenderParams` configures the terrain renderer
- `TerrainRenderer` renders frames, AOVs, and offline accumulation passes

`terrain_atmosphere_path_demo.py` is the clearest example of this lower-level
terrain path in the repo today.

## Offline Accumulation And Denoise

`forge3d.offline.render_offline()` wraps the multi-batch accumulation flow
around `TerrainRenderer`. Use it when you want progressive offline quality
instead of a single interactive snapshot.

For the exact accumulation sequence, output contract, and denoise handoff, see
`terrain/offline-render-quality.md`.

`forge3d.denoise_oidn.oidn_denoise()` applies CPU denoise to beauty buffers,
optionally with albedo and normal guidance.

This pair is the main route for high-quality terrain output without building
your own accumulation loop from scratch.

## Geometry, Mesh, And Vector Helpers

These modules support scene construction before rendering:

- `forge3d.geometry`: mesh generation, extrusion, subdivision, displacement, tangents, transforms, and validation
- `forge3d.mesh`: TBN generation and mesh-side utilities
- `forge3d.io`: DEM and mesh IO helpers
- `forge3d.vector`: 2D vector scene primitives and rendering helpers

They are not just internal plumbing. They are the intended building blocks when
you need to generate or preprocess geometry in Python before passing it into a
viewer or export pipeline.

## Path Tracing, SDF, And Lighting

forge3d also exposes lower-level rendering utilities outside the terrain viewer:

- `forge3d.path_tracing`: path-tracing helpers and camera creation
- `forge3d.sdf`: SDF primitives, builders, and scene helpers
- `forge3d.lighting`: lower-level lighting and ReSTIR-oriented helpers
- `forge3d.materials` and `forge3d.textures`: material containers and supporting wrappers

These modules are more specialized than the viewer workflow, but they are part
of the public package surface and documented in the API reference.

## Diagnostics And Device Utilities

Use the top-level helpers and `forge3d.mem` when you need environment or memory
information:

- `has_gpu()`, `enumerate_adapters()`, `device_probe()`, `get_device()`
- `memory_metrics()`, `budget_remaining()`, `utilization_ratio()`

These are useful when a script needs to decide between a viewer path, a native
offscreen path, or a pure-Python fallback.

## Example Map

- `triangle_png.py`: minimal native renderer sanity check
- `terrain_atmosphere_path_demo.py`: full terrain-native rendering stack
- `camera_animation_demo.py`: animation plus viewer-driven frame export
- `terrain_camera_rigs_demo.py`: camera-rig automation on top of the viewer
- `pnoa_river_showcase_video.py`: terrain composition plus cinematic output

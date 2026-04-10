# Architecture

forge3d has two main public execution paths and one shared data model behind
them.

## The Two Rendering Paths

### 1. Viewer-first scene control

This is the default user-facing path:

1. resolve terrain or other scene assets
2. launch `interactive_viewer` with `forge3d.open_viewer_async()`
3. drive the scene through `ViewerHandle`
4. capture snapshots, frame sequences, or downstream products

This path covers most of the repo's examples, tutorials, and gallery entries.
It is the right default for terrain scenes, overlays, point clouds, labels,
camera automation, and notebook-driven inspection.

### 2. Native/offscreen rendering

This is the lower-level path for controlled rendering without the desktop viewer:

- `Scene` for compact offscreen scene rendering
- `Session` + `TerrainRenderer` + `TerrainRenderParams` for terrain PBR/POM
- `render_offline()` for accumulation-based offline output
- `oidn_denoise()` for CPU denoise after accumulation

Use this path when you need explicit renderer ownership, deterministic batch
rendering, AOVs, or advanced terrain-quality settings that belong in a pipeline
rather than an interactive session.

## Shared Runtime Layers

### Native runtime

The Rust layer owns GPU resources, the terrain renderer, point-cloud support,
screen-space effects, and the `interactive_viewer` executable.

### Python package

The Python layer is the user-facing control surface. The main modules are:

- `forge3d.viewer`, `forge3d.viewer_ipc`, and `forge3d.widgets` for live scenes
- `forge3d.datasets`, `forge3d.crs`, and `forge3d.cog` for terrain/data access
- `forge3d.terrain_params`, `forge3d.presets`, `forge3d.animation`, and `forge3d.camera_rigs` for scene configuration
- `forge3d.pointcloud`, `forge3d.buildings`, `forge3d.tiles3d`, and `forge3d.style` for non-raster scene assets
- `forge3d.map_plate`, `forge3d.export`, and `forge3d.bundle` for packaged outputs

### Examples, tutorials, and gallery

The `examples/` directory shows the supported workflows in executable form.
The tutorials explain the common onboarding paths. The gallery provides smaller
recipe-style references tied to concrete outputs.

## IPC Model

`ViewerHandle` exposes the high-frequency operations directly:

- `load_overlay()`
- `load_point_cloud()`
- `set_point_cloud_params()`
- `set_orbit_camera()`
- `set_camera_lookat()`
- `set_sun()`
- `snapshot()`

Everything else flows through explicit NDJSON commands over TCP. You can send
those with `ViewerHandle.send_ipc()` or by using helper functions from
`forge3d.viewer_ipc`.

That split keeps the common workflow ergonomic while leaving the full scene
command surface available.

## Data Flow

### Terrain

- small bundled samples come from `forge3d.datasets`
- larger samples resolve from local assets first, then download as needed
- `numpy` DEMs can still feed the viewer path through wrapper-side conversion

### Scene layers

- raster overlays load directly through `ViewerHandle.load_overlay()`
- vector overlays and labels use raw IPC or `forge3d.viewer_ipc`
- point clouds and scene assets use viewer or module-specific helpers

### Packaged outputs

- viewer snapshots feed map-plate and pure-Python composition examples
- scene state can be saved and loaded through bundle workflows
- vector scenes can be exported separately through `forge3d.export`

## Open-Source And Pro Boundaries

The open-source core covers the viewer, terrain rendering, datasets, point
clouds, raster overlays, labels, animations, widgets, geometry helpers, and the
native rendering path.

Pro-gated workflows are layered on top of that core rather than replacing it:
map plates, vector export, style import, bundles, and building-focused product
pipelines.

## Reading The Repo

If you are navigating the codebase:

- start with `python/forge3d/__init__.py` for the public package surface
- use `examples/` for concrete workflows
- use `docs/examples/index.md` to map example scripts to modules and features
- use `docs/api/api_reference.rst` when you need the module-by-module reference

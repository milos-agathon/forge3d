# Architecture

Phase 2 treats forge3d as a developer platform around one rendering core, not a
collection of disconnected demos.

## Layers

### 1. Rust runtime

The Rust side owns the renderer, terrain scene, point-cloud path, and the
`interactive_viewer` desktop binary.

### 2. Python control surface

The Python package provides:

- launch helpers in `forge3d.viewer`
- low-level IPC helpers in `forge3d.viewer_ipc`
- dataset resolution in `forge3d.datasets`
- notebook wrappers in `forge3d.widgets`
- packaging utilities in `forge3d.map_plate`, `forge3d.bundle`, and
  `forge3d.export`

### 3. Tutorials and samples

The docs, bundled data, gallery, and notebooks are the contract for how the
platform is supposed to feel to a new user.

## Primary rendering path

The intended runtime path is:

1. Resolve a DEM or scene asset.
2. Launch `interactive_viewer` with `forge3d.open_viewer_async()`.
3. Hold on to `ViewerHandle`.
4. Send explicit updates over TCP/NDJSON IPC.
5. Capture snapshots or export downstream assets.

That keeps Python thin and makes the viewer the single place where terrain,
overlays, point clouds, and camera state actually render.

## Notebook path

`ViewerWidget` is the notebook wrapper around the same viewer process. When the
full viewer cannot launch, the widget falls back to an inline pseudo-3D preview
so examples still run in constrained environments.

That fallback is intentionally narrower than the full viewer. It is useful for
documentation, CI, and quick previews, but it remains an implementation detail
of the widget; the authoritative interactive path is still the Rust viewer plus
IPC.

## IPC model

The high-level wrapper covers common commands directly:

- `load_terrain()`
- `load_overlay()`
- `load_point_cloud()`
- `set_point_cloud_params()`
- `set_orbit_camera()`
- `set_sun()`
- `snapshot()`

Anything more specialized can still be issued through `ViewerHandle.send_ipc()`
or the helpers in `forge3d.viewer_ipc`, including vector overlays, labels, and
bundle commands.

## Data flow

### DEMs

- Bundled samples ship as `.npy` and GeoJSON metadata.
- Larger tutorial datasets resolve from local repo assets or download through
  `pooch`.
- `.npy` terrain inputs are converted to temporary TIFFs before viewer launch so
  the current binary can consume them without changing the viewer CLI contract.

### Overlays

- Raster overlays are loaded directly with `load_overlay()`.
- Vector overlays and labels are sent as explicit IPC payloads.

### Buildings

- `forge3d.buildings` parses GeoJSON, CityJSON, or 3D Tiles metadata.
- When native geometry extraction is available, the resulting triangles can be
  forwarded to the viewer through `add_vector_overlay`.
- Without the native path, the Python fallback still exposes counts, materials,
  bounds, and attributes for inspection.

### Bundles and export

- `forge3d.bundle` writes portable `.forge3d` directories for terrain,
  overlays, presets, and bookmarks.
- `forge3d.MapPlate` composes rendered terrain with legends, scale bars, north
  arrows, and inset images.
- `forge3d.export` generates SVG or PDF from 2D vector scenes.

## Design constraints

- One live rendering path: the viewer binary.
- Thin Python wrappers over explicit commands.
- Small bundled examples and larger on-demand datasets.
- Notebook support without creating a second full renderer stack.
- Production-output features layered on top of the same scene assets.

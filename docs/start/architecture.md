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

Point-cloud LOD traversal consumes only `root_node()` and `children(key)` spatial
facts. COPC and EPT adapters share key-to-bounds subdivision and child enumeration
in the private `_pointcloud_octree` module, with children ordered by octant. Each
adapter retains its own format parsing, hierarchy loading, point counts and
spacing rules: COPC uses its declared spacing divided by `2 ** depth`, while EPT uses
node width divided by span. Decoding capabilities remain format-specific; sharing
hierarchy traversal does not give EPT a point decoder. The existing selection,
depth and point-budget policies remain in `PointCloudRenderer`.

Native capability dispatch belongs to `forge3d._native`. Runtime callers resolve
the current module with `get_native_module()` or an optional symbol with
`get_native_symbol()`; required-module guards use `require_native_module()`.
These lookups do not retry imports or create a GPU context. An explicit
`refresh_native_module()` invalidates Python's import caches and retries the
extension import, updating the module, availability flag, import-error cause,
and legacy `python.forge3d._forge3d` alias together. It does not reload an already
imported binary; replacing a loaded extension binary requires a new process.

Imports of availability flags by value are snapshots. Exported classes and
exception types in the package facade, animation, GIS, text, and smoke modules
retain their import-time identities. Smoke's backend selection stays coupled to
its exported classes and Python-version safety guard. Existing native objects
retain their owning module; newly created wrappers resolve current capabilities.
Missing native-only facade exports resolve against the current module without
rebinding exports that are already present. Extension absence and missing
symbols continue to use each caller's existing fallback or error behavior.

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

Both entry points delegate command encoding, NDJSON framing, receive buffering,
and connection failures to the private `forge3d._viewer_transport` module. Each
call sends one command and consumes one reply, retaining extra bytes for the
next call on that socket. Calls on a socket must be sequential. Socket timeout
settings stay unchanged: `ViewerHandle` keeps its exceptions, while the raw
helpers keep their error dictionaries and partial-reply parsing at EOF/timeout.

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

# Native layers and meshes: what the engine actually offers (measured 2026-11-16)

The plan lists three increments that need engine capability Studio cannot
reach yet: **meshes in Rendered** (roadmap §7.2), **COG tile-laddered
streaming** (`phase-17-plan.md` §2.4) and **MapScene native layers** (§8 —
buildings, 3D Tiles, point clouds). This page records what each one actually
runs into, measured on this machine, so the next person does not have to
rediscover it. Every number here is re-derivable from the commands quoted.

Short version: **all three are engine-facing.** The shell can wire them, but
the engine's current contract cannot deliver them, and shipping a Studio
control that renders nothing (or that the engine refuses) is exactly what
this repo forbids. The measurements below say what each one needs.

---

## 1. Meshes in Rendered (roadmap §7.2)

The plan's design: "`io.import_gltf` + `geometry` +
`instance_mesh_gpu_render`, composited over the terrain frame with the same
camera."

**The composited-with-the-same-camera part is not available.**
`forge3d.geometry.instance_mesh_gpu_render(mesh, transforms, width, height,
pbr=…)` (`python/forge3d/geometry.py:300`) takes **no camera** — no view
matrix, no target, no azimuth. Its Rust entry point
(`src/render/instancing.rs:126`) builds the mesh's vertices, a GPU context and
a `MeshInstancedRenderer`, and then hard-codes the camera
(`src/render/instancing.rs:204`):

```rust
let view = glam::Mat4::look_at_rh(
    glam::Vec3::new(3.0, 2.0, 3.0),   // fixed eye
    glam::Vec3::ZERO,                  // fixed target
    glam::Vec3::Y,                     // fixed up
);
let proj = crate::camera::perspective_wgpu(45.0f32.to_radians(), aspect, 0.1, 100.0);
```

The renderer itself *can* take a view/projection
(`src/render/mesh_instanced.rs:776 set_view_proj`) — the Python entry point
just never exposes it. So a mesh rendered this way is always seen from
(3, 2, 3) looking at the origin, while the terrain behind it was rendered
from the recipe's orbit camera: the two cannot be composited into one frame.

The entry point itself works, and is in this build: `enable-gpu-instancing`
is part of `default` (`Cargo.toml:22`), so the symbol is present. Measured
on the engine test's own glb: `instance_mesh_gpu_render` returns a 256×256
RGBA8 frame with the model drawn from that fixed camera (the call succeeds;
the camera is the problem, not the feature).

**The engine's in-recipe route for a glTF is a texture composite, not 3D.**
`MapScene`'s building layer can carry a glTF
(`map_scene.py:4039 _building_gltf_path`), but what the engine composites is
the model's **albedo texture**, drawn into a screen rectangle
(`map_scene.py:4075 _composite_textured_landmark_layers`,
`map_scene.py:4054 _screen_rect_pixels`) — a landmark card, not a model in
the scene. That route also wants the texture as a **separate PNG**
(`metadata.textured_materials[].albedo_texture`), while the engine's glTF
importer exposes texture *slots*
(`python/forge3d/io.py:496 GltfTextureSlot`), not image bytes — so a caller
would have to re-implement glTF image extraction to use it.

**What it needs:** either a camera-aware mesh pass in the engine (the
instanced renderer would have to accept the recipe's camera and share the
terrain's depth), or CONCORDIA participation for world-space geometry (the
same participation the native path already demands of buildings, see §3).

---

## 2. COG tile-laddered streaming (`phase-17-plan.md` §2.4)

The engine's reader is **tile-based**: `forge3d.open_cog(url)` returns a
`CogDataset` with `read_tile(x, y, lod)`, `ifd_info(level)` and
`overview_count()` (`python/forge3d/cog.py:131–223`); tiles are 256 px, and
LOD is a power-of-two step. The probe and the whole-file materialization
already work (Phase 17.2.4).

The ladder itself is shell work in principle — read the overview for the far
view, read finer tiles as the camera closes — but Studio's DEM pipeline
expects a **local file**: the asset registry stores a path, the renderer
opens that path, and the viewport's heightfield is a full numpy grid. A
streamed DEM therefore needs a new asset kind ("remote DEM, live"), a tile
cache feeding the heightfield, and a rule for when the whole file is
materialized for a final still. That is a day of work, not an afternoon, and
it touches the asset model, the viewport mesh and the render path.

**What it needs:** nothing from the engine beyond what exists — the reader
is already there. The work is Studio's asset model and the preview's
heightfield.

---

## 3. MapScene native layers: buildings, point clouds, 3D Tiles

The engine's recipe owns `PointCloudLayer` (`map_scene.py:5080`),
`BuildingLayer` (`map_scene.py:5101`) and `Tiles3DLayer`
(`map_scene.py:5212`) — exactly the layers Studio's map stack cannot author.
Two measured facts decide how each one lands.

### 3.1 The buildings layer renders — but only at city scale

The engine's own suite passes here (7/7, `tests/test_mapscene_buildings.py`),
including `test_mapscene_render_composites_building_layer`, which asserts
more than 100 changed pixels. So the route works.

Over Studio's terrain it changes **0 samples**. The arithmetic:

* The engine normalizes a building layer's own bounding box onto the
  terrain's normalized space (`map_scene.py:3628 _building_point_to_scene`),
  then maps that space onto the heightmap's pixels
  (`map_scene.py:3793 _terrain_scatter_building_batches_for_recipe`).
* A building's height is a **fraction of a ~45 m terrain extent**:
  `wall_height = height_m / 45`, clamped to `0.08 … 1.4` contract units
  (`map_scene.py:3745`). A 12 m building is therefore `0.27` units in a
  scene that is `1.7` units wide — sensible for a city model, sub-pixel for
  a 24 km DEM.

Measured on Mount Fuji (24 km, 2 594²): the layer validates as
`buildings.mapscene_render: supported` and the engine renders it, and the
frame changes **0 of 262 144 samples** (|Δ| > 0: 0). The buildings are
there, at the size of a pixel in a country.

The anchoring has the same root cause: the layer's own bbox is stretched
onto the whole terrain, so a 5th-station footprint would cover the entire
Fuji DEM rather than sit on the flank.

### 3.2 A GeoJSON building layer is refused when the still's TAA is on

With screen-space effects enabled (the still's TAA path,
`studio_render.py` sets `screen_space: {enabled: true, taa_enabled: true}`),
the native path raises:

```
MapSceneNativeUnavailable: MapScene native rendering unavailable:
layer=mount_fuji_buildings: world-space textured geometry has no shared
UV/material/velocity outputs (required_native=CONCORDIA layer participation)
```

The condition is `map_scene.py:4315`: `isinstance(layer, BuildingLayer) and
_building_gltf_path(layer) and not is_cartographic_landmark(layer)`. For a
GeoJSON layer, `BuildingLayer.from_geojson` sets
`source={"path": <the .geojson>, "source_format": "geojson"}`
(`map_scene.py:5124`), and `_building_gltf_path` (`map_scene.py:4039`)
reads `source["path"]` — so **a GeoJSON path is mistaken for a glTF path**
and the layer is refused. Building the layer with `features=` and no
`source=` sidesteps it (measured: no refusal), which is how Studio would
have to pass it.

This is an engine-side inconsistency worth its own fix: `validate()` reports
`buildings.mapscene_render: supported` for the same layer that `render()`
then refuses. The fix is small — `_building_gltf_path` should only treat the
source path as a glTF when the source format says so — and it should come
with a test that renders a GeoJSON building layer with screen-space effects
on. It is **not** done here: it changes no pixels by itself (the scale
problem above still stands), and the goal's rule is that the engine stays
untouched unless a failing test makes a fix unavoidable.

### 3.3 The point cloud needs a matching DEM

`PointCloudLayer` carries a path and a CRS, and the engine projects the
points through the recipe's camera (`map_scene.py:3068
_composite_native_point_cloud_layers`). The projection is anchored to the
terrain, so the cloud and the DEM must be the same place: the repository's
only COPC is `assets/lidar/MtStHelens.laz`, and there is no St Helens DEM
locally (the registry's `mt-st-helens` entry is the cloud, not a DEM). The
layer cannot be demonstrated against the terrain that ships here.

3D Tiles (`Tiles3DLayer`) is projected the same way, with the same
requirement.

**What it needs, concretely:**

| Layer | Engine-side | Studio-side |
|---|---|---|
| buildings | real-world height anchoring (not "a fraction of 45 m"), and the `_building_gltf_path` fix | a "buildings" cartography layer kind; the service hands the engine its own `BuildingLayer` |
| point clouds | nothing measured as missing — the projection exists | a matching DEM in the cloud's CRS; a "point cloud" layer kind |
| 3D Tiles | nothing measured as missing | a Tiles3D asset kind and layer kind |

---

## 4. What this means for the plan

Roadmap §12 keeps these three as increments, and they stay there — but the
*shape* changes: they are engine increments with a Studio half, not Studio
increments with an engine dependency. The honest next moves, in order:

1. **Engine: make the building layer's heights real-world** (or accept a
   metres-relative convention) and anchor it to the terrain's bounds rather
   than its own bbox. Without this, a buildings layer over any country-scale
   DEM is a no-op — measured 0 samples.
2. **Engine: fix `_building_gltf_path`** so a GeoJSON layer is not refused
   as textured geometry, with the test from §3.2.
3. **Engine: give the instanced mesh renderer a camera** (or let world-space
   geometry participate in CONCORDIA) — the only way "meshes in Rendered"
   means a model in the scene rather than a texture card.
4. **Studio (no engine change): the COG ladder** — a remote DEM asset kind, a
   tile cache, and a heightfield that refines with the camera.
5. **Studio (after 1): a "buildings" layer kind**, the service handing the
   engine its own `BuildingLayer`.

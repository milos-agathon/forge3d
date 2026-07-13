# Full Geospatial Viewer — MENSURA M-06, Option 2

Status: implementation specification; no source implementation is included here.

Evidence baseline: `C:/tmp/mensura`, branch `mensura`, commit `5e625c0a91f8db6967b2e8b4705e4153a7f21e6a`, inspected 2026-07-13. The worktree already contained the unrelated modified file `python/forge3d/forge3d.pdb`; it must remain untouched. All file and line references below refer to that commit unless a path is explicitly identified as the primary planning checkout.

## Goal

Make the `interactive_viewer` path render terrain, vector overlays, labels, object translations, and point clouds at projected/Earth-scale coordinate magnitudes without narrowing absolute positions to `f32`. World positions must remain `f64` until they are translated by one shared `camera::Anchor`, then narrow only through `Anchor::to_render_*` for GPU/CPU render-space work.

The invariant is:

```text
world position (f64) - one frame-frozen anchor origin (f64)
    -> render position (f32)
    -> view/light-view-projection matrices and GPU buffers (f32)
```

This removes the roughly 0.5 m quantization step near Earth-radius magnitudes while retaining the existing `f32` renderer.

## Scope

### In scope

- GeoTIFF affine-transform, CRS-authority, and height-system ingestion for interactive-viewer terrain.
- A single `Anchor` owned by `Viewer`, with one rebase decision at the start of a frame.
- `f64` camera/target/object/vector/label/point-cloud source positions and anchor-relative render data.
- Terrain simple, PBR, and shadow WGSL; terrain screen and snapshot paths; CSM, volumetrics, picking, GI/TAA history, sky/fog, and point-cloud matrix paths.
- Rust IPC request/command/scene-review schemas, Python high- and low-level helpers, and `.pyi` coordinate semantics.
- Backward-compatible JSON decoding for existing clients.
- Contract, unit, pipeline-construction, IPC, and live GPU differential tests.

### Out of scope

- A planet, globe camera, ellipsoid intersection, horizon/curvature model, quadtree/clipmap LOD, browser target, or CRS reprojection. The primary planning document assigns these to ORBIS and says MENSURA “does not render a planet” (`C:/Users/milos/forge3d/docs/prompts/fable5-moonshots/13-mensura.md:101-107`).
- A general WKT2/EPSG database or automatic transformation between overlay CRSs.
- Widening WGSL matrices, vertex formats, or rasterization to `f64`.
- Reinterpreting raster-overlay `extent` as geospatial bounds; it is an existing normalized UV contract.
- Changing terrain-scatter mesh vertices or instance matrices into absolute-world inputs; they are terrain-local contracts.
- Refactoring unrelated renderer architecture or repairing every pre-existing raw GPU allocation in this older branch.

Option 2 intentionally overrides the planning document's “do not touch the label engine” non-goal only where label source positions and projection must participate in the shared anchor (`13-mensura.md:106`). It does not override the globe/ORBIS boundary.

## Non-negotiable invariants

1. There is one viewer anchor: `Viewer.camera_anchor: Anchor`. Terrain, labels, vectors, objects, point clouds, shadows, picking, screen rendering, and snapshot rendering never own or rebase independent anchors.
2. `Viewer::render` makes at most one rebase decision before creating any matrix or projecting any label. The resulting copied `Anchor` is immutable for the entire screen + snapshot + motion-blur frame.
3. Persistent absolute positions are `f64` (`DVec3`, `[f64; 3]`, or a typed source structure). Persistent GPU/render positions remain `f32`.
4. The only world-coordinate narrowing implementation remains `Anchor::narrow` (`src/camera/anchor.rs:88-103`). New code calls `to_render_vec3`, `to_render_direction`, `to_render_f32`, `view_look_at`, or `model_offset`; it does not cast a world component.
5. Missing CRS is preserved as missing and is never guessed. Missing transform uses the explicitly documented legacy local fallback. Rotated/sheared GeoTIFFs fail with a typed/clear viewer diagnostic until a full affine terrain shader is designed.
6. Local non-georeferenced terrain has origin `[0, 0]`, span `[width, height]`, and the same height/camera conventions as today. Its simple and PBR pixels are regression-locked.
7. A rebase is a rigid translation. It must not change relative positions, culling, picking, shadow cascades, label placement, or raster-overlay UVs.
8. The enforced host-visible budget is 512 MiB. No task may add a raw `create_buffer`, `create_buffer_init`, `create_texture`, or `create_texture_with_data` call outside `src/core/resource_tracker.rs`.

## Verified current state

Every orientation hypothesis from the task is classified below. “Confirmed” never means “complete”; it means the stated current behavior was found in the requested worktree.

| Hypothesis | Finding | Repository evidence and correction |
|---|---|---|
| Viewer terrain ignores GeoTIFF transform and CRS. | **Confirmed.** | `ViewerTerrainScene::load_terrain` reads TIFF dimensions and image samples only (`src/viewer/terrain/scene/terrain_load.rs:8-27`). `create_grid_mesh` emits duplicated `(u,v)` attributes in `[0,1]` (`terrain_load.rs:259-269`). |
| Terrain positions are pixel-index based. | **Confirmed, with a height-formula correction.** | PBR computes `world_x = uv.x * terrain_width`, `world_z = uv.y * terrain_depth`, and `(h-min)*z_scale` (`src/viewer/terrain/shader_pbr/terrain_pbr.wgsl:91-119`). Simple terrain uses a different normalized height formula (`src/viewer/terrain/shader.rs:25-35`) but the same X/Z formula (`shader.rs:66-85`). Both, plus the shadow shader, must change. |
| `terrain_width` is TIFF width. | **Confirmed.** | Load sets the width in initial terrain parameters (`src/viewer/terrain/scene/terrain_load.rs:107-115`); `ViewerTerrainData::terrain_width` returns the image width as `f32` (`src/viewer/terrain/scene.rs:93-100`). |
| Current terrain uniform sizes are known. | **Confirmed.** | `TerrainUniforms` is mat4 + five vec4 = 144 bytes (`src/viewer/terrain/render.rs:34-44`); `TerrainPbrUniforms` is 240 bytes (`render.rs:57-73`); `ShadowPassUniforms` is documented and shaped as 112 bytes (`render.rs:46-55`, `src/shaders/terrain_shadow_depth.wgsl:5-17`). There are no compile-time size/offset assertions. |
| Raster overlays are terrain-local/normalized. | **Confirmed.** | `OverlayLayer.extent` is explicitly terrain UV `[u_min,v_min,u_max,v_max]` (`src/viewer/terrain/overlay/config.rs:20-29`), and the compositor compares normalized pixel-center UVs against it (`overlay/stack/composite.rs:98-114`). It is not a world-coordinate field and must remain `f32`. |
| Vector overlays are local `f32`. | **Confirmed.** | `VectorVertex.position` is `[f32;3]` (`src/viewer/terrain/vector_overlay.rs:34-42`). Draping subtracts a `f32 terrain_origin` (`vector_overlay.rs:263-327`), and the caller always supplies `(0,0)` (`src/viewer/terrain/scene/overlays.rs:140-157`). |
| Labels are local `f32`. | **Confirmed.** | `LabelData.world_pos` is `Vec3` (`src/labels/types.rs:160-180`) and is directly multiplied by the supplied matrix (`src/labels/mod.rs:351-408`). Line projection also consumes `Vec3` (`src/labels/line_label.rs:27-40,192-196`). |
| Nothing in the interactive viewer receives absolute geospatial data safely. | **Confirmed for the named camera/vector/label/object paths; raster overlays are a deliberate normalized exception.** | The Rust IPC types are `f32` at `src/viewer/ipc/protocol/request.rs:23-24,34-49,65,74-79,96-120`. The interactive LAS loader also narrows before centering (`src/viewer/pointcloud/load.rs:52-59`). |
| Option 1 is shipped and rejects `abs(coord) > 1e6`. | **Confirmed.** | `VIEWER_LOCAL_FRAME_MAX_COORD` and finite/absolute checks are at `src/viewer/camera_controller.rs:9-69`. `set_orbit_pose_target` and `set_look_at` return `Result` and validate before mutation (`camera_controller.rs:291-356`). Terrain commands sanitize targets (`src/viewer/cmd/terrain_command.rs:1-22,64-77`) and IPC logs camera rejection (`src/viewer/cmd/ipc_command.rs:89-103`). Rust tests are at `camera_controller.rs:365-481`; source contract locks are at `tests/test_m06_anchoring_boundary.py:65-88`. The contract must invert under this spec. |
| A reusable `f64` anchor already exists. | **Confirmed.** | `Anchor` stores `origin: DVec3` and epsilon (`src/camera/anchor.rs:17-22`), defaults to a 1 km epsilon (`anchor.rs:24-40`), rebases deterministically (`anchor.rs:71-86`), and exposes `to_render_vec3`, `to_render_direction`, typed conversion, anchored view, and model offset (`anchor.rs:95-133`). Reuse it unchanged unless a test proves a missing minimal API. |
| Offscreen Scene, PNTS, point clouds, and CityJSON are all already anchored. | **Partly confirmed, partly refuted.** | Offscreen Scene rebases and narrows its f64 camera (`src/scene/py_api/base.rs:17-54`); generic `PointBuffer` retains f64 and uses `Anchor` (`src/pointcloud/renderer.rs:11-20,47-74`); PNTS uses f64 plus `render_positions(anchor)` (`src/tiles3d/pnts.rs:75-97`). **Interactive-viewer point cloud is not anchored** (`src/viewer/pointcloud/load.rs:52-59`, `state.rs:177-223`). **CityJSON is not origin-relative at tessellation:** `SurfaceProjection::project` casts absolute point components (`src/import/cityjson/geometry.rs:253-258`) and `unproject` casts reconstructed absolute values (`geometry.rs:261-291`); the origin subtraction at `geometry.rs:214-224` only chooses a normal. |
| `test_world_coord_f32_gate.py` proves renderer-wide anchoring. | **Refuted.** | Its own header says f32 storage/constructors are invisible (`tests/test_world_coord_f32_gate.py:10-20`). It filters casts by a same-line world-word regex (`test_world_coord_f32_gate.py:28-38,92-102`), so it misses both `point.x as f32` and `point[n] as f32`. It does correctly lock the one sanctioned textual site (`test_world_coord_f32_gate.py:105-121`) and anchor implementation uniqueness (`:141-173`). |
| Existing M-06 docs accurately describe production closure. | **Refuted.** | The plan claims CityJSON and “point clouds” are already anchored (`docs/carto-engine/rust-gis-implementation-plan.md:200-211`), while the anchoring note includes normalized `LoadOverlay.extent` in its world-field residual (`docs/carto-engine/mensura-m06-world-coord-anchoring.md:74-86`). Both must be corrected. |
| The GIS layer already reads the required metadata. | **Confirmed, with a CRS limitation.** | `read_raster_info` reads custom height tag 65001 (`src/gis/raster_info.rs:52-63`), transform/bounds/resolution (`raster_info.rs:78-95`), and CRS authority/WKT (`raster_info.rs:97-117`). It reads ModelTransformation or ModelPixelScale + ModelTiepoint (`raster_info.rs:894-944`). Writer symmetry is at `src/gis/raster_write.rs:714-761,792-803`. However it rejects all EPSG codes except 4326, 3857, and 32631 (`raster_info.rs:1017-1028`); metadata ingestion must preserve other valid authority codes without pretending to implement their projection. |
| The 512 MiB budget and raw-allocation gate are currently enforced in this branch. | **Budget confirmed; gate premise refuted for this worktree.** | Registry default is enforced 512 MiB (`src/core/memory_tracker/registry.rs:5-7,28-46`). Tracked buffer helpers enforce/account (`src/core/resource_tracker.rs:338-398`), and tracked textures account (`resource_tracker.rs:400-423`). But `tests/test_allocation_gate.py` is absent and `src/viewer` contains 104 raw allocation-call matches, including `terrain_load.rs:52` and `viewer/pointcloud/state.rs:249`. This spec therefore requires no new raw calls and migration of any allocation site touched by the implementation; integration onto the later allocation-gated baseline is a rollout prerequisite, not something to falsely report as present here. |
| The authoritative MENSURA scope excludes a planet and f64 rasterizer. | **Confirmed.** | `C:/Users/milos/forge3d/docs/prompts/fable5-moonshots/13-mensura.md:101-107` assigns globe/clipmap/browser work to ORBIS and explicitly calls the deliverable anchoring rather than an f64 rasterizer. |

## Shared design

### Coordinate conventions

The interactive terrain viewer remains a right-handed, Y-up Cartesian renderer:

```text
viewer world X = source projected/map X (east for ordinary projected CRSs)
viewer world Y = existing display height above DEM minimum
viewer world Z = -source projected/map Y (north is -Z)
```

The `-map Y` basis conversion is deliberate: north-up GeoTIFFs have a negative row-to-map-Y coefficient, so they retain the current positive terrain Z span and winding. Python/Rust world positions supplied alongside a projected terrain must already use this viewer basis: `(map_x, display_height, -map_y)`. No hidden CRS conversion is performed.

Vertical behavior remains the existing display contract: simple and PBR terrain resolve heights relative to the DEM minimum, and `z_scale` remains a presentation scale. Height-system tag 65001 is retained as metadata but does not silently turn display Y into an orthometric/ellipsoidal absolute altitude. This is necessary for local pixel compatibility and honest because vertical-datum conversion is not implemented.

Consequences:

- Projected UTM/Mercator terrain is horizontally absolute and camera-relative.
- Generic object/vector/label/point-cloud scenes may use any Cartesian Earth-scale triple and rebase all three axes.
- “ECEF-scale” tests prove magnitude/precision behavior. Rendering an ECEF ellipsoid or tangent frame is still ORBIS work.

### Raster affine and CRS policy

Reuse `gis::read_raster_info`; do not add a second GeoTIFF tag parser. `AffineTransform` is `f64` and follows:

```text
map_x = a * col + b * row + c
map_y = d * col + e * row + f
```

as implemented at `src/gis/types.rs:94-128,175-179`.

For the two-uniform terrain representation, accept only `b == 0 && d == 0`. For an image `(width,height)`:

```text
world_origin_xz = (c, -f)
world_span_xz   = (a * width, -e * height)
```

Use the same width/height edge convention as `AffineTransform::bounds`, which evaluates `(0,0)`, `(width,0)`, `(0,height)`, `(width,height)` (`src/gis/types.rs:138-168`). Reject non-finite, zero-span, rotated, or sheared transforms with the path and six coefficients in the diagnostic; `is_rotated_or_sheared` already detects `b != 0 || d != 0` (`types.rs:171-173`). Do not flatten them to a bounding box.

If transform metadata is missing, use the explicit legacy viewer fallback:

```text
world_origin_xz = (0, 0)
world_span_xz   = (width, height)
crs             = None if absent
height_system   = value from tag 65001 or "unspecified"
```

This “identity fallback” names the viewer outcome, not an invented CRS. A present transform with missing CRS may still render as an unlabelled Cartesian frame; diagnostics and state report `crs=None`.

CRS authority ingestion must preserve any finite GeoKey EPSG code other than the GeoTIFF user-defined sentinel. It must not claim projection support, synthesize WKT, or reproject. The three-code rejection at `raster_info.rs:1017-1028` is inappropriate for metadata-only viewer placement and must be relaxed with focused GIS regression tests.

### Anchor ownership and frame lifecycle

Add `camera_anchor: Anchor` beside the camera in `Viewer` (`src/viewer/viewer_struct.rs:33-49`) and initialize it with `Anchor::new()` in the `Viewer` literal (`src/viewer/init/viewer_new.rs:119-137`). Do not put an anchor in `ViewerTerrainScene`, `LabelManager`, a vector layer, or `PointCloudState`.

At the first line of `Viewer::render` (`src/viewer/render/main_loop.rs:34-40`):

1. Select the active world camera: terrain camera when terrain is present; otherwise point-cloud orbit camera when point cloud is the active scene; otherwise the general viewer camera.
2. Validate its f64 eye and target are finite.
3. In terrain-orbit mode, rebase toward `(target.x, 0, target.z)` because the orbit target is the stable scene focus and terrain Y is intentionally DEM-min-relative. A pure orbit around a stationary target must not rebase. In a non-terrain Cartesian/FPS scene, rebase toward the full eye.
4. Call `camera_anchor.rebase_if_needed` once.
5. If it rebased, recompute persistent render-relative vector/point/object data and invalidate temporal history.
6. Copy `let frame_anchor = self.camera_anchor;` and pass that value through every render/snapshot path. No callee may call `rebase_if_needed`.

Command validation must not mutate the shared anchor between frames. To validate a new camera, copy the current anchor, perform the deterministic prospective rebase on the copy using the same target-vs-eye policy as frame start, validate eye and target residuals, then commit only the f64 camera state. The real anchor rebases at the next frame boundary.

### Render-space definition and narrowing chokepoints

| Source | Persistent representation | Chokepoint | Rebase behavior |
|---|---|---|---|
| Terrain footprint | `DVec2 origin/span` plus raw raster dimensions | `Anchor::to_render_vec3(origin)` and `to_render_direction(span)` while filling uniforms | No vertex-buffer rewrite; uniforms are rewritten for every screen/shadow/snapshot pass. |
| General camera | f64 eye/target; f32 up/distance/angles | `Anchor::view_look_at` and `to_render_vec3(eye)` | Rebuild frame matrices; no persistent GPU position. |
| Terrain camera | f64 target; f32 orbit angles/radius | Derive eye in f64, then same `Anchor` methods | Same. |
| Object transform | local mesh vertices f32; translation/origin `DVec3`; rotation/scale f32 | `Anchor::model_offset` when composing model matrix | Recompose matrix, do not rewrite mesh. |
| Vector overlay | typed f64 source XYZ; f32 RGBA; u32 feature ID | CPU drape in f64 terrain coordinates, then `to_render_vec3` | Rebuild/rewrite render vertices and render-space BVH in existing buffers. |
| Point/line/curved/callout labels | `DVec3` source points | Convert to transient render `Vec3` before projection/layout | Reproject every frame; no duplicate viewer-owned label mirror. |
| Pick results | Render-space hit plus the frame anchor, or the retained f64 label source | Restore `RichPickResult.world_pos` to absolute viewer-world `[f64;3]` before returning it | Geometry/vector and label producers return one frame; no render-relative coordinate escapes to Python. |
| Viewer LAS/LAZ | f64 source points plus attributes | `to_render_vec3` when packing `PointInstance3D` | Rewrite the existing instance buffer with `Queue::write_buffer`; do not allocate per rebase. |
| Terrain scatter | existing f32 terrain-contract matrices/mesh | Compose terrain origin/span-to-render transform once per frame | Remains local; no IPC widening. |
| Density volumes | existing f32 terrain-contract center/size | Evaluate through the same terrain-contract transform | Remains local; no IPC widening. |
| Raster overlays | normalized f32 UV extent | None; texture is sampled by anchored terrain | No change on rebase. |

Any persistent render-space cache must be regenerated synchronously before encoding the first pass after a rebase. If a buffer lacks `COPY_DST`, add it at initial creation; never replace buffers every 1 km.

### Terrain uniform and WGSL contract

Append the two fields, adjacent and in this order, to all three Rust structs and matching WGSL structs:

```rust
render_origin_xz: [f32; 2],
render_span_xz: [f32; 2],
```

| Uniform | Existing size | New offsets | Required size |
|---|---:|---|---:|
| `TerrainUniforms` | 144 | origin 144, span 152 | 160 bytes |
| `ShadowPassUniforms` | 112 | origin 112, span 120 | 128 bytes |
| `TerrainPbrUniforms` | 240 | origin 240, span 248 | 256 bytes |

Two `vec2<f32>` values have 8-byte alignment; the pair preserves the required 16-byte struct multiple. Add Rust `size_of` and `offset_of` assertions plus WGSL source-contract/pipeline-construction tests. Do not add a bind group or binding: the existing uniform binding grows with the struct.

In `src/viewer/terrain/shader.rs:25-85`, `src/viewer/terrain/shader_pbr/terrain_pbr.wgsl:91-156`, and `src/shaders/terrain_shadow_depth.wgsl:55-123`:

```wgsl
let world_xz = u.render_origin_xz + uv * u.render_span_xz;
let world_pos = vec3<f32>(world_xz.x, existing_world_y, world_xz.y);
```

Replace normal spacing derived from pixel width/depth with:

```wgsl
let step_x = u.render_span_xz.x / max(f32(dims.x) - 1.0, 1.0);
let step_z = u.render_span_xz.y / max(f32(dims.y) - 1.0, 1.0);
```

Use signed tangents so mirrored transforms are mathematically correct, then orient the final normal toward +Y if its Y component is negative. Both on-screen terrain raster pipelines (`src/viewer/terrain/scene/core.rs:90` and `scene/pbr_compute.rs:228`) currently use `FrontFace::Ccw` plus back-face culling; set `cull_mode: None` on both so a negative signed span cannot reverse the fixed index winding and disappear. The shadow terrain pipeline already has no culling. Do not add per-sign pipeline variants.

Replace the PBR atmospheric span calculation at `terrain_pbr.wgsl:743-747` with the maximum absolute component of `render_span_xz`. For the simple shader, rename/document `terrain_params.z` as `simple_height_extent`: set it to the raster width for the legacy local fallback, and to `abs(world_span_xz.x)` for georeferenced terrain. `height_to_world_y` uses this value, never the pixel width directly, so two rasters covering the same physical X span produce the same simple-shader relief regardless of resolution. PBR keeps `(h-min)*z_scale` unchanged.

Screen uniform fill sites are `src/viewer/terrain/render/screen/setup.rs:105-138,163-221`; snapshot sites are `render/offscreen/setup.rs:160-188,212-277`; shadow fill is `render/shadow.rs:43-87`. All must derive identical origin/span from the same `frame_anchor`.

## Complete interactive-viewer matrix inventory

This inventory covers every producer/consumer found under the reachable interactive-viewer terrain, object, labels, point-cloud, shadow, and screen-space-effect paths. It includes paths that keep `view` and `projection` separate rather than spelling the combined value `view_proj`. The already-anchored offscreen `Scene` API is intentionally excluded; the viewer's terrain `render_to_texture` snapshot path is included.

| Path | Current producer | Current consumers | Required Option-2 behavior |
|---|---|---|---|
| General OBJ/glTF geometry | `proj * (camera.view_matrix() * object_transform)` at `src/viewer/render/main_loop/geometry/pass.rs:19-48` | Geometry camera buffer and screen-space `CameraParams` at `pass.rs:41-60`; the geometry shader consumes the separately packed model-view/projection. | Active camera view must be anchor-relative; object translation must be `Anchor::model_offset`; current and previous VP must be in the same anchor frame. |
| Mouse picking | Duplicates `proj * (view * object_transform)` and inverts it at `src/viewer/input/viewer_input.rs:61-83` | `unproject_cursor` ray and unified picking. | Call the same anchored camera/model helper used by rendering; never independently rebuild an unanchored matrix. |
| Per-input GI update | Duplicates camera/model matrices at `viewer_input.rs:291-328` | `ScreenSpaceEffectsManager::update_camera`, motion vectors, `prev_view_proj`. | Remove the duplicate producer or route it through the one anchored frame-camera result. |
| One-shot geometry/GI helper | Duplicates projection/view/model and previous VP at `src/viewer/state/viewer_helpers/gi/geometry.rs:24-55` | One-shot G-buffer and GI camera upload. | Use the same anchor snapshot; do not rebase inside the helper. |
| Generic fog and fog shadows | Separate view/projection/VP/eye at `src/viewer/render/main_loop/geometry/fog.rs:19-48`; it also updates a fog CSM at `fog.rs:24-30`. | `FogCameraUniforms` (`src/viewer/viewer_types.rs:96-108`), ray reconstruction, and the fog CSM depth pass at `fog.rs:169-216`. Its embedded shader multiplies raw geometry position by `light_view_proj` (`src/viewer/init/viewer_new.rs:460-476`) and currently omits the object transform. | Upload render-space eye/view; derive both fog and light matrices in the frame anchor; apply the same anchor-relative object transform in the depth pass. This is distinct from terrain CSM. |
| Sky | Separate view/projection/inverses and f32 eye at `src/viewer/render/main_loop/frame_setup.rs:128-161`. | Sky compute camera buffer and sun-direction reconstruction. | Use the active anchored camera (terrain when terrain is active), not the always-generic camera currently used. Directions remain untranslated. |
| Automatic labels | Chooses terrain or generic camera and constructs a duplicate VP at `frame_setup.rs:79-125`. | `LabelManager::update_with_camera`; point/occlusion projection at `src/labels/projection.rs:25-91` and `src/labels/mod.rs:351-408`; line/curved projection at `src/labels/line_label.rs:27-40,192-196` and `src/labels/curved.rs:234-253`. | Pass the frame anchor and render-space camera. Convert f64 label sources before all point, line, curved-glyph, horizon, and callout projection. |
| Manual label update | Builds another generic VP at `src/viewer/state/labels.rs:73-87`. | `LabelManager::update`. | Delegate to the same anchored update path or remove the duplicate method body. |
| Terrain interactive screen | Terrain f32 eye/view and `proj * view` at `src/viewer/terrain/render/screen/setup.rs:12-63`. | Simple uniform upload `:105-138`; PBR + camera position `:163-221`; screen state `:226-239`; simple WGSL multiply `src/viewer/terrain/shader.rs:80-85`; PBR multiply `shader_pbr/terrain_pbr.wgsl:114-119`. | Terrain camera target/eye become f64; view uses `frame_anchor`; origin/span and camera position are render-space. |
| Terrain CSM producer | Screen calls `render_shadow_passes(view,proj)` at `screen/setup.rs:74-80`; snapshot does the same at `render/offscreen/setup.rs:147-153`. CSM inverts the camera VP and derives f32 light VP at `src/shadows/csm_renderer.rs:88-103,119-176`; inverse VP becomes frustum corners at `src/shadows/cascade_math.rs:7-31`. | Shadow uniform upload at `src/viewer/terrain/render/shadow.rs:32-87`; depth WGSL at `src/shaders/terrain_shadow_depth.wgsl:114-123`; PBR CSM lookups at `terrain_pbr.wgsl:436-445,603-619`. | Camera frusta, light cascades, depth terrain, and shaded terrain must all use the same render-space origin/span. CSM remains f32 because it operates after anchoring. |
| Raster overlays | No independent matrix. | CPU normalized compositor at `src/viewer/terrain/overlay/stack/composite.rs:98-114`, then terrain PBR sampling. | No coordinate conversion; inherits terrain VP and origin/span. |
| Vector overlays | Copies terrain VP from `ScreenRenderState` (`screen/setup.rs:226-230`) and snapshot state (`offscreen/setup.rs:282-295`). | Screen opaque/OIT calls at `render/screen/scene.rs:137-150,198-213`; snapshot calls at `render/offscreen/scene.rs:176-189,298-313`; upload at `vector_overlay/pipelines.rs:392-405,492-539`; WGSL multiply at `vector_overlay.rs:343-380`. | Render vertices are re-narrowed from f64 source with `frame_anchor`; keep VP identical to terrain. |
| Terrain scatter | Receives separate `view`, `proj`, and render eye at `src/viewer/terrain/scene/scatter.rs:24-48`. | Instanced mesh renderer and terrain-blend sampling. | Compose the existing terrain-local matrices through the terrain origin/span-to-render transform; do not widen the mesh-local IPC fields. |
| Terrain volumetrics | Interactive passes inverse terrain VP and eye at `src/viewer/terrain/render/screen/effects.rs:80-121`; snapshot does so at `render/offscreen/effects.rs:35-62`. | `VolumetricsUniforms.inv_view_proj` at `src/viewer/terrain/volumetrics.rs:15-33,352-413`. | Inverse must be of the anchored terrain VP; camera position is render-space. |
| Terrain DoF | **No view-projection producer or consumer.** | Depth plus near/far/focus parameters at `src/viewer/terrain/dof/types.rs:1-21` and `dof/pass/execute.rs:5-70`. | No matrix edit. Prove it receives depth produced by anchored geometry and keep this correction in tests/docs. |
| Terrain snapshot `render_to_texture` | Independently rebuilds terrain view/projection/VP at `src/viewer/terrain/render/offscreen/setup.rs:86-136`. | Simple/PBR uniforms `:160-188,212-277`; vector state `:282-295`; volumetrics above. Entry is `render/offscreen/mod.rs:22-76`, called by `render/main_loop/secondary.rs:125-169`. | Accept the already-frozen `frame_anchor`; screen and snapshot may differ in aspect/projection but never in origin. |
| Terrain motion blur | Calls `render_to_texture` for fallback and every shutter sample at `src/viewer/terrain/render/motion_blur.rs:14-24,83-122`. | Each sample's terrain, shadows, vectors, and effects. | Freeze one anchor across all samples; camera angle/radius may vary but no sample may rebase. |
| Viewer point cloud | Creates its own origin-centered eye/VP at `src/viewer/render/main_loop/secondary.rs:250-275` and a second snapshot projection at `:307-315`. | Uniform upload at `src/viewer/pointcloud/state.rs:260-284`; WGSL multiply at `src/viewer/pointcloud/pointcloud.wgsl:51-80`. | Retain f64 LAS positions, derive the active camera from f64 bounds, use the shared anchor, and stop clearing/replacing a terrain frame if terrain and points coexist. Snapshot reuses the same view and anchor. |
| Previous-VP/temporal consumers | `prev_view_proj` lives at `src/viewer/viewer_struct.rs:248-251` and is written by geometry/GI producers above. Terrain and general TAA jitter are advanced separately (`geometry/pass.rs:54-62`, `terrain/render/screen/setup.rs:52-63`). | Motion vectors, TAA, SSAO/SSGI/SSR histories, and fog history. | On rebase set previous VP to the new current VP, reset jitter/history validity, call the existing SSGI reset (`src/core/screen_space_effects/manager/accessors.rs:287-290`), and add minimal reset hooks for other enabled temporal histories. A rebase must not create a one-frame motion/fog flash. |

The PBR fragment's `u.view_proj * world_pos` at `terrain_pbr.wgsl:722-725` is also a consumer even though the result is only used for cascade/view-depth work; it must receive the same anchored world position and matrix. Matrix enforcement gates on producer operations, not one method name: normalize comments, UFCS spelling, whitespace, and multiline calls, then reject direct `Mat4::look_at_*`, `Mat4::perspective_*`, and direct projection/view composition outside an explicit producer-file/callsite allowlist. Each allowlisted producer must positively assert use of the shared frame anchor/helper. Every inventory row marked no-matrix must have a positive test proving it constructs no view/projection matrix.

## IPC and public API coordinate inventory

JSON numbers have no f32/f64 wire encoding, so ordinary three-number arrays remain byte-shape compatible. The loss occurs in Rust deserialization/storage. Python `float` is already binary64; Python changes preserve values, validate shape/finite-ness, name the frame, and prevent accidental coercion.

| Surface | Current Rust fields/sinks | Current Python / `.pyi` | Current frame and lane semantics | Target contract |
|---|---|---|---|---|
| `cam_lookat.eye`, `target`, `up` | `[f32;3]` at `src/viewer/ipc/protocol/request.rs:23-25` and `viewer_enums/commands.rs:37-39`; sink is f32 `CameraController`. | `ViewerHandle.set_camera_lookat` at `python/forge3d/viewer.py:999-1011`; stub `viewer.pyi:168-173`. | Eye/target are world positions; up is a direction. | Eye/target `[f64;3]` -> `DVec3`; up remains `[f32;3]`/`Vec3`. JSON array unchanged. |
| `set_transform.translation`, `rotation_quat`, `scale` | Translation `[f32;3]` alongside f32 rotation/scale at `request.rs:23`, `commands.rs:65`; object transform state is f32 (`src/viewer/viewer_struct.rs:238-243`). | `viewer.py:983-997`; `viewer.pyi:162-167`. | Translation is an absolute object origin; quaternion and scale are not positions. | Translation `[f64;3]`/`DVec3`; rotation and scale remain f32. Compose anchor-relative model offset. |
| `set_terrain_camera.target` | `Option<[f32;3]>` at `request.rs:34-40`, `commands.rs:67-73`; stored as `[f32;3]` at `src/viewer/terrain/scene.rs:74-80`. | `set_orbit_camera` at `viewer.py:1082-1111`; stub `viewer.pyi:185-192`. | Viewer-world terrain target; angles/radius/FOV are local scalar quantities. | Target `[f64;3]`/`DVec3`; scalars remain f32. Radius is metres/viewer units after georeferencing, no longer “terrain-width pixels.” |
| `set_terrain.target` | `Option<[f32;3]>` at `request.rs:42-49`, `commands.rs:75-80`. | Accessible through raw `send_ipc`; no dedicated typed high-level method. | Same as terrain camera target. | `[f64;3]`, same validation and state transaction. |
| `load_overlay.extent` | `Option<[f32;4]>` at `request.rs:65`, `commands.rs:98`, scene-review payload `src/viewer/ipc/protocol/payloads.rs:323-333`. | `viewer.py:914-940`; `viewer.pyi:140-148`. | **Not world coordinates:** `[u_min,v_min,u_max,v_max]` terrain UV. | Remains f32 and shape-compatible. Improve docs to say normalized UV. A future `world_bounds` is a separate API, not part of M-06. |
| `add_vector_overlay.vertices` | `Vec<[f32;8]>` at `request.rs:74-79`, `commands.rs:103-106`; scene-review copy at `payloads.rs:335-357`. Conversion is `src/viewer/cmd/vector_overlay_command.rs:25-31`. | High-level `viewer.py:721-752`, stub `viewer.pyi:97-110`; low-level docs incorrectly describe seven lanes at `python/forge3d/viewer_ipc.py:618-638`. | Exactly eight lanes: **0=X, 1=Y, 2=Z world position; 3=R, 4=G, 5=B, 6=A f32 color; 7=u32 feature ID encoded as a JSON number. There are no UV lanes.** Internal UV/normal defaults are synthesized at `src/viewer/terrain/vector_overlay.rs:45-63`. | Introduce a typed Rust source vertex `{ position:[f64;3], color:[f32;4], feature_id:u32 }` with a custom eight-number-array deserializer. Validate finite XYZ/color and integral in-range ID. Preserve the legacy array wire shape; only XYZ remain f64. Python validates length 8 and ID semantics. |
| `add_label.world_pos` | `[f32;3]` at `request.rs:96-101`, `commands.rs:121-126`; direct Vec3 conversion at `src/viewer/state/labels.rs:8-30`. | `viewer.py:485-513`; `viewer.pyi:37-54`; low-level helper serializes at `viewer_ipc.py:250-279`. | Absolute viewer-world point. Styling/offsets are nonposition f32. | `[f64;3]` and persistent `DVec3`; render projection converts through frame anchor. |
| `add_line_label.polyline` | `Vec<[f32;3]>` at `request.rs:102-106`, `commands.rs:127-131`. | `viewer.py:594-630`; `viewer.pyi:56-70`; low-level helper `viewer_ipc.py:342-399`. | Absolute viewer-world points; repeat distance is screen pixels. | `Vec<[f64;3]>`/`Vec<DVec3>`, transient render polyline per frame. |
| `add_curved_label.polyline` | `Vec<[f32;3]>` at `request.rs:113-116`, `commands.rs:135-139`. | High-level method is currently diagnostic-only at `viewer.py:658-676`, but raw helper sends points at `viewer_ipc.py:430-481`; stub `viewer.pyi:71-83`. | Absolute viewer-world points; tracking is typography. | Widen the raw/scene-review path even though the high-level renderer is experimental, so it cannot remain a hidden truncation bypass. |
| `add_callout.anchor` | `[f32;3]` at `request.rs:117-120`, `commands.rs:140-143`. | `viewer.py:678-701`; `viewer.pyi:84-96`; low-level `viewer_ipc.py:484-539`. | Absolute viewer-world point. `offset` is screen-space pixels; colors/sizes remain f32. | `[f64;3]`/`DVec3`; render through label anchor conversion. |
| Scene-review labels | Untyped label maps at `payloads.rs:360-389`, later decoded into f32 point/line/curved/callout structs at `src/viewer/scene_review.rs:699-770,1256-1337`; `value_as_array3` returns f32 at `scene_review.rs:891-903`. | `ViewerHandle.load_bundle` forwards `scene_state.to_dict()` at `viewer.py:905-912`. | Same world fields as direct commands; currently a secondary truncation path. | Add a dedicated f64 world-array parser and use it only for positions. Keep color-array parser f32. Route all decoded state through the same typed commands. |
| Scene-review raster/vector layers | Raster extent and vector vertices are f32 at `src/viewer/scene_review.rs:23-45`; vector conversion/picking uses f32 at `scene_review.rs:493-520`. | Indirect through bundle/review APIs. | Raster extent normalized; vector XYZ world, remaining lanes nonposition. | Raster unchanged; vector uses the typed f64 source vertex and common conversion. |
| Picking result `world_pos` | `RichPickResult.world_pos` is `[f32;3]` at `src/picking/unified.rs:15-21`; geometry/vector ray hits fill it in render/BVH space, while the label-hit branch copies `LabelData.world_pos` at `src/viewer/input/viewer_input.rs:105-108`; PyO3 exposes f32 tuples at `src/py_types/picking.rs:107-171`. | Python receives `RichPickResult.world_pos`. | Public output is an absolute viewer-world position, but the two producers become frame-inconsistent once the BVH is anchor-relative and labels retain `DVec3`. | Widen the shared result/PyO3 tuple to f64. Restore render-space geometry/vector hits with the immutable frame anchor before constructing the result; return the retained f64 label source directly. Test both producers at Earth scale and after rebase. |
| Volumetric density `center` | `Option<[f32;3]>` at `payloads.rs:152-178`, mapped unchanged at `ipc/protocol/translate/terrain.rs:263-280`, stored at `viewer_enums/config.rs:118-149`. | Raw PBR payload only. | Terrain-volume contract coordinates, evaluated against the terrain footprint; `size` and `wind` are dimensions/directions. | Remains f32 terrain-local. Apply the terrain contract-to-render transform when used; do not misclassify it as external world IPC. |
| Terrain scatter `transforms` and LOD `positions` | Matrices `Vec<[f32;16]>` and mesh positions `Vec<[f32;3]>` at `payloads.rs:222-232,263-281`; translation lanes are row-major **3, 7, 11** (`src/terrain/scatter.rs:974-1005`). | `ViewerHandle.set_terrain_scatter` accepts dictionaries (`viewer.pyi:178`). | Explicit terrain-contract instance transforms and mesh-local vertices; `scene/scatter.rs:12-20` currently documents the local contract. | Remain f32. Replace the identity contract-to-render mapping with the anchored terrain mapping. Rotation/scale/mesh positions are not world fields. |
| `load_point_cloud` | IPC contains a path only (`request.rs:86`); LAS supplies f64 coordinates, currently narrowed at `src/viewer/pointcloud/load.rs:52-59`. | `viewer.py:949-965`; `viewer.pyi:149-155`. | LAS X/Y/Z are absolute source positions; current renderer remaps `(x,z,y)` and then centers in f32. | Retain remapped `DVec3(x,z,y)` source values and anchor before GPU packing. No wire change. |

Add `.pyi` aliases to make semantics reviewable without pretending Python exposes an f64 type:

```python
WorldPosition = tuple[float, float, float]
VectorOverlayVertex = tuple[float, float, float, float, float, float, float, int]
NormalizedExtent = tuple[float, float, float, float]
```

Use those aliases throughout `viewer.pyi`. `python/forge3d/viewer_ipc.py` has no separate stub, so its runtime annotations/docstrings must use the same names imported from a non-cyclic public typing location or repeat precise tuple types.

### Wire compatibility

- Three-number world arrays stay three-number JSON arrays. Existing clients gain precision without a protocol-version branch.
- Eight-number vector arrays stay eight-number arrays. The custom Rust visitor stores only XYZ as f64 and validates lane 7 as an integer u32. Do not add an ambiguous seven-lane form.
- Normalized raster extent, colors, UVs, normals, screen offsets, dimensions, angles, radii, directions, scale, and quaternions remain f32 after finite/range validation.
- Bundle/scene-review JSON remains readable. Re-saving may canonicalize numeric spelling, but the field shapes and meanings do not change.
- No new PyO3 registration is required: this viewer protocol is socket JSON, not a new native `_forge3d` symbol. Rust/Python/stub changes are still one atomic API task.

## Prioritized implementation tasks

The sequence is intentionally bottom-up. Tasks M06-FGV-02 through M06-FGV-08 must live behind one feature branch or temporary internal gate; do not merge an intermediate commit that widens storage while still advertising the final old absolute-coordinate rejection contract.

### M06-FGV-01 — Ingest and retain terrain georeferencing

**Priority:** P0 — every downstream anchor calculation is wrong without an authoritative f64 terrain footprint.

**Description:** Call the existing GIS metadata reader before/alongside height decoding, retain its `RasterInfo`, derive a minimal f64 viewer footprint, and preserve the legacy local fallback. Relax metadata-only EPSG authority preservation and reject transforms that cannot be expressed by origin/span.

**Rationale:** Anchoring arbitrary pixel indices does not make terrain geospatial. The absolute origin and physical span must exist in f64 before any camera, overlay, or shader work, or the f32 cliff merely moves.

**Dependencies:** None.

**Files and symbols to change:**

- `src/viewer/terrain/scene/terrain_load.rs:8-27` — call `crate::gis::read_raster_info(path)` and keep the existing image decode; do not duplicate tag parsing. At `:107-183`, initialize `ViewerTerrainData.raster_info`, `world_origin_xz: DVec2`, `world_span_xz: DVec2`, and local/georeferenced status.
- `src/viewer/terrain/scene.rs:61-112` — add only the retained `RasterInfo` and derived `DVec2` fields/accessors needed by rendering and state reporting. Keep raster dimensions and local fallback behavior.
- `src/gis/raster_info.rs:1017-1028` — preserve arbitrary valid EPSG authority codes as metadata rather than rejecting all but three. Do not add a CRS database or projection claim.
- `src/gis/types.rs:94-179` — reuse `AffineTransform`; expose no new public API unless a crate-private helper avoids duplicate formulae.
- `src/viewer/cmd/terrain_command.rs:79-164` and `GetTerrainParams` response path — report transform tuple, origin/span, optional authority/WKT, height system, and whether fallback was used, so the contract is observable.
- Allocation rule: because `terrain_load.rs` contains existing raw texture/buffer creation (`:52,91,99`) and will be edited, migrate the touched allocations to `tracked_create_texture` / `tracked_create_buffer_init` and store their tracked wrappers or integrate atop the allocation-gated baseline. Do not add another raw call.

**Definition of Done:**

- North-up axis-aligned GeoTIFF produces `(c,-f)` origin and `(a*width,-e*height)` span exactly in f64.
- Missing transform produces `(0,0)` / `(width,height)` and a warning/status, with no CRS invention.
- Missing CRS remains `None`; a present non-whitelisted EPSG code is retained as authority metadata.
- Rotated/sheared, non-finite, and zero-span transforms fail before GPU resources are committed and identify the source path and coefficients.
- Height-system tag 65001 round-trips into viewer state.

**Verification/tests:**

- Add Rust tests beside `terrain_load.rs:289` for local fallback, north-up transform, south-up signed span, and rotated/sheared rejection.
- Extend GIS metadata tests for a projected EPSG outside `{4326,3857,32631}` and verify authority preservation without WKT synthesis.
- Add an IPC state test confirming optional CRS/transform fields and fallback status.
- Run `cargo test --lib raster_info` and the focused viewer terrain-load tests before GPU work.

### M06-FGV-02 — Add the shared viewer anchor and f64 camera/object state

**Priority:** P0 — one owner and one frame boundary are the central correctness invariant.

**Description:** Add `Viewer.camera_anchor`, promote general/terrain/point-cloud camera positions and object translation to f64 source state, define active-camera selection, and make `Viewer::render` the sole rebase site. This task may keep the old absolute-bound validation as a temporary internal compatibility check; M06-FGV-08 replaces it only after all consumers are anchored.

**Rationale:** Independent subsystem anchors or mid-frame rebases create rigid-translation disagreements that look like missing geometry, incorrect shadows, or jumping labels. f64 storage after an f32 IPC decode is too late.

**Dependencies:** M06-FGV-01.

**Files and symbols to change:**

- `src/viewer/viewer_struct.rs:33-49,238-251` — add `camera_anchor: Anchor`, store object translation/origin as `DVec3`, and retain rotation/scale separately as f32. Do not store a second frame anchor.
- `src/viewer/init/viewer_new.rs:119-137,303-305` — initialize the anchor and f64 world state.
- `src/viewer/camera_controller.rs:77-126,270-356` — change orbit target/FPS position/eye/target source state to `DVec3`; angles, up, speed, and radius stay f32. View construction accepts an `&Anchor` and delegates to `Anchor::view_look_at`.
- `src/viewer/terrain/scene.rs:74-155` — change terrain target and derived eye to `DVec3`; compute the default target from `world_origin_xz + 0.5 * world_span_xz` while preserving current display Y.
- `src/viewer/pointcloud/state.rs:140-175` — retain f64 bounds center for point-cloud camera derivation.
- `src/viewer/render/main_loop.rs:34-40` — call a small `prepare_frame_anchor`/equivalent before `prepare_render_frame`; select terrain, point-cloud, or general camera; in terrain-orbit mode rebase from the stable target rather than the orbiting eye; rebase once; copy `frame_anchor`.
- `src/camera/anchor.rs:17-134` — reuse the existing API. Add an API only if the implementation otherwise duplicates subtraction/narrowing, and keep the one-cast invariant.

**Definition of Done:**

- Only one production viewer call to `rebase_if_needed` exists, at frame start.
- All camera/object world state remains f64 between commands and frames.
- Terrain-orbit mode rebases horizontal `(target.x,0,target.z)`; a stationary-target orbit performs zero rebases. Non-terrain Cartesian/FPS mode rebases full eye, as documented.
- Screen, snapshot, and every motion-blur sample receive the same copied anchor value.
- A rebase callback runs before encoding and has a deterministic list of caches/histories to refresh.

**Verification/tests:**

- Rust tests for active-camera precedence, exact epsilon behavior, terrain Y=0 anchor policy, one-rebase-per-frame, and a complete stationary-target orbit with zero rebases using a pure helper (no GPU required).
- Source contract test permits `rebase_if_needed` in the viewer only in the frame-boundary helper.
- Existing `camera::anchor::tests` remain green, including sub-millimetre Earth-radius coverage (`src/camera/anchor.rs:192-234`).

### M06-FGV-03 — Render terrain and CSM from anchored origin/span

**Priority:** P0 — the camera cannot view absolute terrain until every terrain vertex and shadow vertex shares its render frame.

**Description:** Append the exact origin/span uniform fields, fill them from the frame anchor, update simple/PBR/shadow WGSL positions and normal spacing, and keep local output unchanged.

**Rationale:** A relative camera matrix with absolute/pixel-index terrain is still a frame mismatch. Shadow depth and shaded geometry must generate byte-identical render positions or CSM comparisons fail.

**Dependencies:** M06-FGV-01, M06-FGV-02.

**Files and symbols to change:**

- `src/viewer/terrain/render.rs:34-73` — append the two `[f32;2]` fields with the exact 160/128/256-byte layouts specified above; add size/offset assertions.
- `src/viewer/terrain/shader.rs:5-12,25-85` — mirror fields; use origin/span for world XZ and span-per-texel for normals.
- `src/viewer/terrain/shader_pbr/terrain_pbr.wgsl:5-22,91-156,743-747` — same changes and render-span atmospheric scale.
- `src/shaders/terrain_shadow_depth.wgsl:5-17,55-123` — grow documented size to 128 and use identical XZ generation.
- `src/viewer/terrain/render/screen/setup.rs:12-80,105-221` — accept `frame_anchor`, build anchored terrain view, and fill all three uniform variants.
- `src/viewer/terrain/render/offscreen/setup.rs:86-153,160-295` — same for snapshot.
- `src/viewer/terrain/render/shadow.rs:32-87` — fill shadow origin/span from the already-passed anchor-derived values; never recompute an anchor.
- `src/viewer/terrain/scene/core.rs:90` and `scene/pbr_compute.rs:228` — set `cull_mode: None` on the two on-screen terrain raster pipelines. They currently use `FrontFace::Ccw` plus back-face culling, so a negative signed span reverses the fixed mesh winding. Keep the existing no-cull shadow pipeline and do not create per-sign variants.
- `src/viewer/terrain/shader.rs:8,25-35` plus its uniform fill sites — rename/document `terrain_params.z` as `simple_height_extent`; use raster width only for legacy local terrain and `abs(world_span_xz.x)` for georeferenced terrain. Do not couple georeferenced vertical relief to raster resolution.

**Definition of Done:**

- Rust and WGSL layouts match at the stated byte offsets and pipeline creation succeeds.
- Simple, PBR, and shadow vertex shaders use exactly the same `origin + uv*span` expression.
- Normal gradients use physical render span, not TIFF pixel width.
- North-up, south-up, and either-X-mirrored terrain remain visible with correct orientation; the on-screen simple/PBR pipelines do not back-face cull terrain.
- Simple-shader relief is identical for different raster resolutions covering the same georeferenced X span; the local fallback remains pixel-compatible.
- `camera_pos`, `world_pos`, CSM frustum corners, and cascade light matrices are all render-space f32.
- Local origin/span reproduces current simple and PBR output within the existing golden thresholds.

**Verification/tests:**

- Rust layout tests plus WGSL source-contract tests for field order and no residual `uv * terrain_width` world-position formula.
- Headless pipeline-construction smoke for simple, PBR, and shadow entries after `maturin develop`.
- A live south-up/mirrored fixture asserts non-zero coverage and asymmetric orientation in both simple and PBR modes, so winding and Z-sign failures cannot pass as two equally blank/doubly mirrored images.
- A simple-shader unit/render test compares two DEM resolutions with the same physical footprint and height samples.
- Existing terrain visual goldens plus a local-before/after differential with SSIM >= 0.999 and normalized mean absolute RGB error <= 0.5/255 (equivalently 0.5 in u8 units).
- CSM regression with a non-zero Earth-scale origin proves lit and shadow depth agree.

### M06-FGV-04 — Unify every camera/matrix producer on the frame anchor

**Priority:** P0 — one missed matrix is the highest-probability catastrophic defect in Option 2.

**Description:** Route all rows in the complete matrix inventory through active f64 camera state and the immutable frame anchor. Remove duplicate direct camera-matrix construction where practical; where different aspect/FOV is required, share pose and anchor while deriving only projection.

**Rationale:** Correct terrain shaders do not help if labels, picking, sky, fog, snapshot, or temporal history uses another origin.

**Dependencies:** M06-FGV-02, M06-FGV-03.

**Files and symbols to change:**

- General geometry: `src/viewer/render/main_loop/geometry/pass.rs:19-61`.
- Picking and duplicate GI update: `src/viewer/input/viewer_input.rs:61-83,291-328`.
- One-shot GI: `src/viewer/state/viewer_helpers/gi/geometry.rs:24-55`.
- Labels/sky: `src/viewer/render/main_loop/frame_setup.rs:79-161`; manual labels `src/viewer/state/labels.rs:73-87`.
- Fog/fog CSM: `src/viewer/render/main_loop/geometry/fog.rs:19-48,169-216`, `src/viewer/viewer_types.rs:96-108`, and the depth shader at `src/viewer/init/viewer_new.rs:460-476`; include anchor-relative object transform rather than multiplying raw local vertices by light VP.
- Terrain screen/snapshot/effects: `src/viewer/terrain/render/screen/setup.rs:12-80`, `screen/effects.rs:80-121`, `offscreen/setup.rs:86-153`, `offscreen/effects.rs:35-62`, `offscreen/mod.rs:22-76`.
- Motion blur: `src/viewer/terrain/render/motion_blur.rs:14-24,83-140`.
- Temporal state: `src/viewer/viewer_struct.rs:248-251`, `src/core/screen_space_effects/manager/accessors.rs:287-290`, SSAO validity at `src/core/screen_space_effects/ssao/constructor.rs:43,143`, SSR/SSGI history resources under `src/core/screen_space_effects/ssr.rs:33-34` and `ssgi/constructor/mod.rs:84-94`, fog history at `src/viewer/init/fog_init.rs:254-291`.

**Definition of Done:**

- Every matrix-inventory row either uses the shared helper/frame anchor or is documented as no-matrix normalized/depth-only work.
- Screen and snapshot share pose/anchor; only aspect/projection differ.
- Picking ray and rendered object agree in the shared frame after a rebase. Render-space BVH rebuild and pick-output restoration are M06-FGV-05 deliverables.
- Rebase invalidates previous VP, TAA jitter/history, SSAO/SSGI/SSR history, and fog history without reallocating routine buffers.
- DoF has no invented matrix dependency and renders anchored depth correctly.

**Verification/tests:**

- A source-contract test normalizes UFCS spelling, comments, whitespace, and multiline calls; it rejects direct `Mat4::look_at_*`, `Mat4::perspective_*`, and projection/view composition outside an explicit producer-file/callsite allowlist. Every allowlisted producer has a positive shared-anchor/helper assertion, and each no-matrix inventory row has a positive test proving it constructs no matrix.
- Unit test simulates a rebase and asserts previous VP is reset to current rather than crossing anchor frames.
- Live two-frame test crosses the 1 km epsilon and asserts no one-frame temporal/fog/shadow flash above the image-difference threshold.

### M06-FGV-05 — Retain and re-narrow every anchored renderable

**Priority:** P0 — f64 IPC is meaningless if vectors, labels, objects, or LAS data are cached as absolute f32 before rendering.

**Description:** Store absolute f64 source positions, generate render-space caches through `Anchor`, and refresh them on rebase. Keep normalized/terrain-local subsystems local and compose their existing contract into render space.

**Rationale:** Rebase correctness depends on source data surviving after the first upload. Subtracting a new anchor from already-truncated f32 cannot recover lost millimetres.

**Dependencies:** M06-FGV-02, M06-FGV-03, M06-FGV-04.

**Files and symbols to change:**

- Vector source/cache/drape: `src/viewer/terrain/vector_overlay.rs:34-122,263-331`; layer creation and BVH at `src/viewer/cmd/vector_overlay_command.rs:20-130`; terrain caller `src/viewer/terrain/scene/overlays.rs:140-157`; GPU uploads in `vector_overlay/pipelines.rs:392-405,492-539`. Add a typed f64 source vertex; keep `VectorVertex` as GPU f32. Drape UV computation uses f64 terrain origin/span before narrowing.
- Labels: `src/labels/types.rs:160-202`, `src/labels/mod.rs:351-408,500-530`, `src/labels/projection.rs:33-85`, `src/labels/line_label.rs:27-40,192-196`, `src/labels/curved.rs:234-253`, plus viewer command sinks `src/viewer/state/labels.rs:8-30` and `src/viewer/cmd/labels_command.rs:120-245`. Retain DVec3 source points and make anchored update the shared path; legacy local callers use `Anchor::new()` only at a compatibility boundary, not an owned persistent anchor.
- Object transform: `src/viewer/cmd/ipc_command.rs:110-192` and `viewer_struct.rs:238-243`; keep local vertices, rotation, and scale, but derive translation with `Anchor::model_offset`.
- Picking output: `src/picking/unified.rs:15-21,314-351`, `src/viewer/input/viewer_input.rs:61-108`, and `src/py_types/picking.rs:107-171` — make `RichPickResult.world_pos` f64 absolute viewer-world output. Restore render-space BVH hits with the immutable frame anchor; copy label `DVec3` sources directly. Do not expose two frames through one field.
- Interactive point cloud: `src/viewer/pointcloud/load.rs:9-84` retains `DVec3(point.x,point.z,point.y)`; `state.rs:177-257` keeps f64 source/bounds, packs render points, and rewrites an existing `COPY_DST` instance buffer; `render/main_loop/secondary.rs:250-343` uses active camera and load semantics rather than an independent clear pass.
- Scatter: replace identity `viewer_render_from_contract` at `src/viewer/terrain/scene/scatter.rs:12-20` with the frame's terrain contract-to-render transform; retain row-major translation lanes 3/7/11 in f32 local matrices.
- Density volumes: retain local config at `src/viewer/viewer_enums/config.rs:118-149`, but ensure `src/viewer/terrain/volume_density.rs:134-219,299-372` evaluates centers in the same terrain contract.
- Allocation rule: migrate the touched point-cloud instance allocation at `state.rs:249-257` to `tracked_create_buffer`; rebase uses `Queue::write_buffer` and allocates zero GPU resources.

**Definition of Done:**

- No named source path stores an absolute world position only as Vec3/[f32;3].
- Ten repeated rebases reproduce vector/label/point/object render positions within 0.25 mm of direct subtraction.
- Drape sampling maps absolute f64 XZ to terrain UV before narrowing and preserves extrusion/local Y offsets.
- Render-space vector picking BVH is rebuilt/rebased together with its GPU vertices.
- Picking restores render-space BVH hits to absolute `[f64;3]` with the immutable frame anchor; label hits return retained `DVec3` source positions, so every `RichPickResult.world_pos` producer uses the same viewer-world frame.
- Point-cloud precision is retained before centering; rebase does not allocate a new instance buffer.
- Raster overlay, scatter mesh, density size/wind, colors, normals, UVs, and other nonposition data remain f32.

**Verification/tests:**

- Pure Rust unit tests for vector drape UVs and point-cloud packing at UTM/Earth-radius origins with 1 mm separation.
- Label point/line/curved projection tests compare local and translated-world screen coordinates.
- Picking test before/after rebase returns the same feature ID.
- Geometry/vector and label picking tests assert the same f64 world-position contract through `RichPickResult`/PyO3 before and after rebase.
- Resource-ledger test records stable buffer count and bytes across at least ten rebases.

### M06-FGV-06 — Widen Rust IPC and hidden scene-review world fields

**Priority:** P0 — Rust deserialization is the current first irreversible f64-to-f32 cliff.

**Description:** Apply the target types from the IPC inventory to direct requests, commands, translators, scene-review payloads, and command sinks. Add the typed eight-lane vector visitor while preserving JSON shapes.

**Rationale:** Python already sends binary64 values, but `[f32;3]` request fields discard precision before any anchor exists. Hidden scene-review parsers would otherwise bypass the fixed direct commands.

**Dependencies:** M06-FGV-05.

**Files and symbols to change:**

- Direct schema: `src/viewer/ipc/protocol/request.rs:23-24,34-49,74-79,96-120`.
- Internal commands: `src/viewer/viewer_enums/commands.rs:37-39,65-80,103-106,121-143`.
- Translation modules: `src/viewer/ipc/protocol/translate/core.rs`, `terrain.rs`, `overlays.rs`, and `labels.rs`; copy world f64 without intermediate f32.
- Command sinks: `src/viewer/cmd/ipc_command.rs:89-192`, `terrain_command.rs:1-22,64-164`, `vector_overlay_command.rs:20-130`, `labels_command.rs:120-245`.
- Scene-review: `src/viewer/ipc/protocol/payloads.rs:323-389`; `src/viewer/scene_review.rs:23-45,493-520,699-770,891-905,1256-1337`.
- Protocol unit tests: `src/viewer/ipc/mod.rs:24-52` and adjacent command tests.

**Definition of Done:**

- Every table row classified as world position deserializes/stores f64 through the sink.
- Direction, color, extent, scale, quaternion, UV, size, offset, and terrain-local fields remain f32.
- Vector lane 7 rejects fractional, negative, non-finite, and `>u32::MAX` values; exactly eight lanes are required.
- Legacy request and scene-review JSON fixtures decode without field-shape changes.
- No conversion path creates a `Vec3` from absolute request components before calling `Anchor`.

**Verification/tests:**

- Serde round-trip tests use coordinates `6_378_137.000_25` and assert exact f64 preservation.
- Negative tests cover non-finite world values and vector lane/type errors with stable diagnostics.
- Scene-review state test proves point, line, curved, callout, and vector fields retain a 0.25 mm offset.

### M06-FGV-07 — Update Python helpers and stubs without coercion

**Priority:** P1 — the core can be correct without it, but the shipped Python package is the primary public viewer API.

**Description:** Introduce semantic coordinate aliases, validate world/vector payloads, fix documentation, and ensure high- and low-level helpers preserve Python floats.

**Rationale:** Python `float` is already f64; the risk is misleading annotations, malformed vector lanes, or helper-side f32/NumPy coercion. The package contract must state the viewer basis and normalized exceptions.

**Dependencies:** M06-FGV-06.

**Files and symbols to change:**

- `python/forge3d/viewer.py:485-513,594-630,658-701,721-752,914-940,983-1011,1082-1111` — use semantic aliases/validators, keep `float(...)` (binary64), validate finite values and eight vector lanes, and correct orbit-radius/target docs.
- `python/forge3d/viewer_ipc.py:250-279,342-399,430-539,618-661` — update raw helpers; fix the false seven-lane vector description by documenting XYZ/RGBA/feature ID.
- `python/forge3d/viewer.pyi:1-110,140-192` — add `WorldPosition`, `VectorOverlayVertex`, `NormalizedExtent` aliases and use them in every public signature.
- `python/forge3d/__init__.py` / `__init__.pyi` only if aliases are deliberately public; do not expand exports without a caller need.

**Definition of Done:**

- No Python helper converts world positions to NumPy float32 or packs them into a float32 array.
- All public world-position docs state `(X, display Y, Z)` and projected-terrain `Z=-map Y`.
- Raster extent is documented as normalized UV, not f64 geospatial bounds.
- Vector docs/types identify all eight lanes and feature ID as integer-valued.
- High-level and raw helpers emit compatible JSON accepted by the Rust tests.

**Verification/tests:**

- Extend `tests/test_viewer_ipc.py` with captured-payload assertions using Earth-radius + 0.25 mm values.
- Add stub/source contract tests for the aliases and vector tuple shape.
- Run `tests/test_api_contracts.py` and install-smoke tests to prove no public import regression.

### M06-FGV-08 — Invert Option-1 validation and harden the f32 gate

**Priority:** P0 — leaving the old absolute ceiling or a dishonest global gate would make the completed feature unreachable or falsely certified.

**Description:** Replace “reject large absolute value” with finite + anchor-residual validation, remove silent target sanitization, rewrite locks/docs, and close the two direct-cast holes discovered by the audit.

**Rationale:** Earth-scale input is now valid; only an object too far from the chosen render origin is unsafe. The contract must protect f32 render precision without rejecting the very coordinates anchoring was built to support.

**Dependencies:** M06-FGV-02 through M06-FGV-07.

**Files and symbols to change:**

- `src/viewer/camera_controller.rs:9-69,291-356` — rename the bound to `VIEWER_RENDER_FRAME_MAX_COORD: f64 = 1_000_000.0`; keep `CameraFrameError` if compatible but change variants/messages to non-finite world coordinate and out-of-render-frame residual. Validate `max(abs(world-anchor)) <= bound`, inclusive.
- `src/viewer/cmd/terrain_command.rs:1-22,64-77` — delete `sanitize_terrain_target`; use a `Result`-returning transactional validator. Never turn invalid input into `None` and continue.
- `src/viewer/cmd/ipc_command.rs:89-103` — return/log a stable rejection with role, residual, bound, and unchanged-state guarantee.
- `src/viewer/camera_controller.rs:365-481` and all of `tests/test_m06_anchoring_boundary.py:10-88` — invert the behavioral and source contracts, including the file header and the second test at lines 48-63; delete every assertion that still describes the viewer as categorically non-absolute-world.
- `tests/test_world_coord_f32_gate.py:10-20,28-38,92-173` — retain the single textual narrowing assertion, add explicit storage/schema checks, and positively require every DVec/f64 world-to-render conversion to call `Anchor`. Forbidden narrowing patterns must cover component/field/index casts such as `origin.x as f32`, `map_x as f32`, and array conversion helpers, not only same-line world-vocabulary matches or three-identical-component constructors. Do not claim regex alone proves renderer-wide anchoring.
- `src/import/cityjson/geometry.rs:214-291` — make tessellation projection origin-relative before narrowing and restore the origin in f64 on unprojection, or remove the false global production-path claim and track CityJSON separately. For full M-06 acceptance, the preferred outcome is to fix this narrow discovered defect through the sanctioned anchor conversion so the strengthened repository-wide gate can pass.
- `docs/carto-engine/mensura-m06-world-coord-anchoring.md:3-99` — remove the Option-1 final contract and the false CityJSON/interactive-point-cloud/overlay-extent claims.

**Precise validation transaction:**

1. Reject any non-finite eye/target component.
2. Copy the current anchor.
3. Derive the candidate eye for the complete requested pose; prospective-rebase the copy using the same terrain-target/non-terrain-eye policy as frame start.
4. Compute component-wise residuals for eye, target, object/label bounds as applicable.
5. Reject if any absolute residual is greater than 1,000,000 m. Equality is accepted.
6. Commit camera/content state only after all checks pass. The real shared anchor remains unchanged until `Viewer::render`.

**Definition of Done:**

- A camera near `(6_378_137, ..., ...)` is accepted when eye/target residuals fit the candidate anchor.
- The same target is rejected if its residual exceeds the bound; non-finite values always reject.
- Rejected commands mutate neither camera nor anchor nor target.
- No symbol or doc still says viewer absolute coordinates above 1e6 are categorically invalid.
- The strengthened gate finds the current interactive LAS, CityJSON, and a fixture containing `origin.x as f32` before their fixes and passes only after all are corrected through the positive Anchor chokepoint contract.
- Exactly one sanctioned world-coordinate `as f32` implementation remains.

**Verification/tests:**

- Rust cases: local accepted; UTM/ECEF-scale accepted; prospective rebase accepted; far residual rejected; non-finite rejected; inclusive boundary; no mutation; orbit target derives candidate eye correctly.
- Python/source cases lock new names/messages and forbid `VIEWER_LOCAL_FRAME_MAX_COORD`, `coord_within_local_frame`, and `sanitize_terrain_target`.
- Run `python -m pytest tests/test_world_coord_f32_gate.py tests/test_m06_anchoring_boundary.py -q`.

### M06-FGV-09 — Prove Earth-scale visual equivalence and millimetre separation

**Priority:** P0 — this is the measurable feature claim; unit/type tests alone cannot prove the live viewer paths agree.

**Description:** Add a deterministic interactive-viewer GPU test that renders equivalent local and Earth-translated scenes, crosses a rebase boundary, and renders two 1 mm-separated features at Earth scale.

**Rationale:** The defect is visible precision loss across a live Rust/WGSL/IPC path. Only a rendered differential exercises deserialization, anchoring, all matrices, shader layout, and snapshot code together.

**Dependencies:** M06-FGV-01 through M06-FGV-08.

**Files and symbols to change:**

- New `tests/test_m06_full_geospatial_viewer.py`, reusing the process/socket/snapshot harness at `tests/test_terrain_viewer_pbr.py:39-135` rather than inventing another launcher.
- Reuse `tests/_ssim.py:12-82` for SSIM and NumPy mean absolute RGB error, but call `ssim(..., data_range=1.0)` for normalized `[0,1]` RGB. The helper's `255.0` default is invalid for this comparison.
- Generate the GeoTIFF fixtures with the shipped native `forge3d.gis.write_raster`, which already accepts data, CRS, and affine transform; do not require optional `rasterio`. Install/require Pillow explicitly in the local acceptance environment for PNG overlay/snapshot I/O. `rasterio` and Pillow are optional extras at `pyproject.toml:45-63`, not base dependencies.
- Tag the live test `@pytest.mark.interactive_viewer` as registered at `tests/conftest.py:150`. The existing `test-interactive-viewer-macos` job is informational (`continue-on-error: true`), runs Metal, is absent from `ci-success.needs`, and can skip when the binary is missing; it is not P0 acceptance evidence.

**Required live test:**

1. Generate a deterministic 64x64 DEM and overlay with `forge3d.gis.write_raster`. Local transform spans 64 m; translated transform uses an EPSG:32631 UTM origin. Render identical relative camera, vector, label, and overlay placement.
2. Generate a second asymmetric south-up fixture (`e > 0`) and an X-mirrored case. Render simple and PBR terrain and assert non-zero coverage plus the expected asymmetric orientation, independently of local-vs-translated SSIM.
3. Disable TAA/motion blur and other stochastic effects; use the same backend, size, PBR settings, sun, and snapshot sequence.
4. Normalize RGB to `[0,1]`, then compare local and translated output with `ssim(..., data_range=1.0) >= 0.999` and mean absolute error `<= 0.5/255` (equivalently 0.5 in u8 units). Record adapter/backend and both metrics in failure output.
5. Pan the terrain target across the 1 km rebase threshold without changing its relative scene pose; compare before/after images with the same thresholds and check stable f64 picking output/label screen positions.
6. Orbit 360 degrees at the documented 5400-unit radius around a fixed terrain target. Assert zero rebases/history invalidations, stable relative picking/labels, and no temporal/fog/shadow flash across the complete sequence.
7. Generate a second centimetre-scale translated fixture/camera or tight vector-only view. Add two differently colored point/short-line features whose f64 positions differ by exactly `0.001` m around an Earth-radius-scale origin. Assert both color components exist and their centroids are separated by at least one pixel. Also assert the Rust-side packed render positions differ by approximately 0.001 m so a coincidental shader-size artifact cannot pass the test.

**Definition of Done:**

- Acceptance runs locally through a freshly built `interactive_viewer`, IPC, terrain screen/snapshot, vectors, labels, and raster overlay on the specified RTX 3070/Vulkan adapter. Record the command, adapter, backend, pass count, and **zero skipped tests**; a green or skipped informational macOS CI job does not satisfy this gate.
- Local and translated render meet the stated metrics; the two 1 mm features remain distinct.
- South-up/X-mirrored terrain has non-zero correctly oriented coverage in simple and PBR modes, and a stationary-target orbit causes no rebase/history churn.
- A deliberately unanchored test double/negative control fails the millimetre or differential assertion.
- Existing offscreen Scene/recipe goldens and determinism hash are unchanged.
- Failure artifacts include local, translated, difference, backend, SSIM, MAE, and detected feature centroids.

**Verification/tests:**

- `python -m pip install pillow` in the local acceptance environment; no rasterio install is required.
- `cargo build --release --bin interactive_viewer --features async_readback` before pytest; binary absence is a failure, not a skip.
- `python -m pytest tests/test_m06_full_geospatial_viewer.py -m interactive_viewer -vv -rs`; acceptance output must report zero skipped tests on RTX 3070/Vulkan.
- `python -m pytest tests/test_terrain_viewer_pbr.py tests/test_vector_overlay_rendering.py -m interactive_viewer -q`
- `python -m pytest tests/test_recipe_goldens.py tests/test_determinism_hash.py -q` for offscreen non-regression.

### M06-FGV-10 — Close documentation, enforcement, and rollout evidence

**Priority:** P1 — code can render correctly without prose, but contradictory M-06 status and unenforced build constraints will regress it.

**Description:** Update authoritative M-06 status, coordinate/API docs, validation commands, budget evidence, and branch-integration prerequisites. Remove obsolete Option-1 language only after live acceptance is green.

**Rationale:** The current plan already overclaims CityJSON and generic “point clouds.” M-06 completion must be evidence-based and reproducible by an engineer who did not participate in the implementation.

**Dependencies:** M06-FGV-09.

**Files and symbols to change:**

- `docs/carto-engine/rust-gis-implementation-plan.md:200-214` — mark full viewer complete only with measured live evidence; correct CityJSON/point-cloud claims and link this spec.
- `docs/carto-engine/mensura-m06-world-coord-anchoring.md:3-99` — replace Option-1 residual with final shared-anchor data flow, coordinate basis, and normalized exceptions.
- Viewer API docs around `python/forge3d/viewer.py:914-1011,1082-1111` and low-level vector/label docs.
- Picking API docs/stubs for `RichPickResult.world_pos` — state that the returned tuple is f64 absolute viewer-world `(X, display Y, Z)`, never anchor-relative render space.
- Live-acceptance docs/CI comments — state that P0 evidence is the recorded local RTX 3070/Vulkan run with zero skips unless a required real-GPU CI job is added to `ci-success.needs`; do not cite the informational macOS lane as a merge gate.
- If implementation is still based on this old `mensura` commit, integrate/rebase onto the repository revision that contains the canonical allocation gate before declaring closure. Re-run line/source audits after the integration because citations in this spec describe the evidence baseline, not future line numbers.

**Definition of Done:**

- No doc claims the viewer is a globe, a full ECEF terrain renderer, or an f64 rasterizer.
- No doc claims normalized raster extents are absolute world coordinates.
- M-06 status names the exact live metrics/backend and the one remaining narrowing site.
- Budget/allocation evidence reports stable resource counts through rebases and peak host-visible bytes below 512 MiB.
- Branch has only intended implementation/doc/test files plus the preserved pre-existing `forge3d.pdb` modification.

**Verification/tests:** Run the complete rollout checklist below and attach exact command output/metrics to the implementation handoff or PR.

## Backward-compatibility policy

- A non-georeferenced TIFF remains the legacy local scene: origin `(0,0)`, span `(width,height)`, DEM-min-relative display Y, existing default camera framing, and normalized raster overlay UVs.
- Existing JSON clients send the same command names and array shapes. JSON numeric values that previously fit f32 continue to behave identically; larger/more precise values cease being truncated.
- Existing eight-lane vector arrays remain valid. The implementation becomes stricter only for malformed lengths, non-finite values, and non-integral/out-of-range feature IDs that could never be interpreted reliably.
- Object mesh vertices, scatter meshes, colors, normals, UVs, sizes, angles, directions, and screen offsets remain f32. Only absolute position/origin fields widen.
- Projection matrices and all WGSL interfaces remain f32. No shader f64 feature is requested.
- Offscreen `Scene` keeps its existing anchor and API. This feature must not route it through the viewer anchor or change its goldens/certificates.
- Bundle/scene-review readers accept old numeric JSON. If bundle output gains optional CRS/transform metadata, absence means the documented legacy local fallback.
- The Python API continues to accept ordinary tuples/sequences. Semantic aliases improve static review without changing runtime object types.

## Risks and open questions

| Risk / question | Impact | Required mitigation or decision |
|---|---|---|
| Anchor changes after one pass has encoded. | Terrain, labels, shadows, or snapshot jump into different frames. | Sole rebase at `Viewer::render` entry; pass copied `Anchor` by value; source gate forbids callee rebase. Motion-blur samples freeze it. |
| Active-camera ambiguity when terrain, object geometry, and point cloud coexist. | Subsystems render with different views despite one anchor. | Use explicit precedence (terrain, then point cloud, then general) for the whole frame. Document it in state/stats. Point-cloud pass must load rather than clear when terrain is present. |
| Terrain CSM is derived in absolute space or shadow geometry omits origin/span. | Large false-shadow regions or detached shadows. | CSM frustum and light matrices operate entirely in anchor-relative f32; shadow WGSL uses the exact main-shader position expression; live non-zero-origin CSM test. |
| Rebase crosses temporal history. | One-frame TAA/GI/SSR/fog ghost or flash. | Invalidate every enabled history and previous VP atomically on rebase; test the 1 km crossing with two consecutive snapshots. |
| Terrain orbit rebases from the moving eye. | A fixed-target orbit triggers roughly 20-34 synchronous cache rewrites/history resets per revolution and causes periodic hitch/shimmer. | Terrain orbit anchors on `(target.x,0,target.z)`; add a full 360-degree orbit test asserting zero rebases/history invalidations. FPS/non-terrain motion remains eye-anchored. |
| `render_origin_xz`/`render_span_xz` cannot encode rotation/shear. | Silent spatial distortion if a rotated raster is flattened. | Reject before resource commit. A future full affine uniform is a separate task; do not substitute bounds. |
| A negative signed span reverses fixed mesh winding. | South-up or mirrored terrain is back-face culled to blank despite correct normal orientation. | Disable culling on both on-screen simple/PBR terrain raster pipelines and render asymmetric south-up/X-mirrored GPU fixtures. |
| Source CRS and viewer basis are confused. | Labels/vectors mirror northing or appear off-terrain. | Publicly state `X=map X`, `Z=-map Y`; expose transform/CRS in terrain stats; tests use asymmetric features that reveal a sign flip. |
| Vertical “absolute” meaning is assumed. | Labels using orthometric elevation disagree with DEM-min-relative terrain. | State that Y remains display height above DEM minimum. Absolute vertical datums/reprojection are out of scope; tag 65001 is metadata only here. |
| Arbitrary EPSG preservation is mistaken for projection support. | Users assume automatic reprojection. | Preserve authority code only; no synthesized WKT or transform; missing/unsupported transformation remains explicit. |
| Current f32 gate passes despite direct casts/storage. | False certification. | Add schema/storage assertions, positive Anchor chokepoint checks for DVec/f64 operands, and component/field/index cast fixtures including `origin.x as f32`; fix interactive LAS and CityJSON. Keep one sanctioned implementation. |
| Matrix gate searches only one camera method token. | A new `Mat4::look_at_*`, perspective call, or direct projection/view multiply silently creates an unanchored producer. | Normalize Rust source spelling/layout, forbid producer operations outside an explicit file/callsite allowlist, positively assert the shared helper at every allowed producer, and positively test no-matrix rows. |
| Pick producers return different frames/types. | Geometry/vector hits escape as anchor-relative f32 while label hits are absolute DVec3, or the label branch stops compiling after widening. | Make `RichPickResult.world_pos` absolute f64 for every producer; restore render hits with the frame anchor and return retained label sources directly. |
| Legacy vector lane 7 is a float-shaped JSON number. | Fractional/large IDs silently truncate. | Custom deserializer validates integer u32; Python typing uses `int`; retain eight-lane wire shape. |
| Re-narrowing allocates GPU resources every kilometre. | Stutter, budget growth, ledger leaks. | Retain CPU f64 source, create COPY_DST buffers once with tracked helpers, use queue writes, and assert stable ledger counts across ten rebases. |
| f64 point-cloud source retention increases CPU memory. | Higher resident memory for max-point loads. | Measure exact CPU/GPU bytes at 500k points; avoid redundant f64 copies; remain under host-visible GPU budget and report CPU cost separately. |
| This branch lacks the canonical allocation gate and has 104 viewer raw calls. | A local “pass” cannot prove the requested repository invariant. | Do not claim the gate is present. Add no new raw calls, migrate touched sites, and integrate onto the later gated baseline before closure; then run the actual gate. |
| Local render parity is assumed rather than measured. | Existing users see a camera/orientation/normal regression. | Identity-fallback differential plus existing simple/PBR/shadow goldens are mandatory. |
| Normalized RGB uses the SSIM helper's `data_range=255.0` default. | Stabilizer constants swamp the signal and make the 0.999 threshold nearly unfalsifiable. | Call `ssim(..., data_range=1.0)` explicitly for `[0,1]` images and retain the MAE gate. |
| Informational interactive-viewer CI is treated as P0 evidence. | Metal, `continue-on-error`, or a missing-binary skip appears green without exercising the RTX/Vulkan contract. | Require a freshly built binary and zero-skip local RTX 3070/Vulkan evidence, or later add a required real-GPU job to `ci-success.needs`. |
| Test fixture generation assumes optional raster packages. | Base/CI environments fail or skip before rendering. | Write DEMs with native `forge3d.gis.write_raster`; explicitly install Pillow for local PNG I/O; do not describe rasterio/Pillow as base dependencies. |
| Simple-shader height remains tied to raster pixel width. | The same geospatial footprint changes relief when DEM resolution changes. | Use raster width only for legacy local fallback and physical `abs(world_span_xz.x)` as `simple_height_extent` for georeferenced terrain; regression-test both contracts. |
| A visual 1 mm point test passes because oversized markers overlap. | False precision win. | Pair the visual centroid assertion with a Rust packed-render-position delta assertion and a deliberately unanchored negative control. |

No unresolved question above permits a silent fallback. If the active-camera coexistence or vertical contract must change, update this spec/API documentation first and add a compatibility test before implementation proceeds.

## Build, test, budget, and enforcement gates

### Verified feature-list truth for this worktree

`CLAUDE.md` is absent from `C:/tmp/mensura`. The current branch's CI commands at `.github/workflows/ci.yml:104-116` use exactly:

```text
default,async_readback,copc_laz,weighted-oit,enable-pbr,enable-ibl,enable-csm,enable-tbn,enable-normal-mapping,enable-hdr-offscreen,enable-render-bundles,enable-renderer-config,enable-memory-pools,enable-staging-rings,geos-topology
```

Do not silently substitute a remembered list. The canonical lint alias in this same worktree intentionally uses a different, newer matrix at `.cargo/config.toml:5-14`:

```text
extension-module,default,async_readback,copc_laz,cog_streaming,gis-remote,geos-topology,weighted-oit,wsI_bigbuf,wsI_double_buf,enable-pbr,enable-tbn,enable-normal-mapping,enable-hdr-offscreen,enable-renderer-config,enable-staging-rings
```

`cargo forge3d-clippy` is the lint command; plain `cargo clippy` is not an acceptable substitute. If the implementation branch first integrates a newer CI baseline, re-read and quote that branch's `ci.yml` rather than carrying the list above forward.

### Mandatory commands

After any `src/*.rs` or `*.wgsl` change:

```text
maturin develop
cargo build --bin interactive_viewer --features async_readback
cargo fmt --check
cargo forge3d-clippy
```

On the evidence baseline, the curated Rust commands are:

```text
cargo check --workspace --features default,async_readback,copc_laz,weighted-oit,enable-pbr,enable-ibl,enable-csm,enable-tbn,enable-normal-mapping,enable-hdr-offscreen,enable-render-bundles,enable-renderer-config,enable-memory-pools,enable-staging-rings,geos-topology

cargo test --workspace --features default,async_readback,copc_laz,weighted-oit,enable-pbr,enable-ibl,enable-csm,enable-tbn,enable-normal-mapping,enable-hdr-offscreen,enable-render-bundles,enable-renderer-config,enable-memory-pools,enable-staging-rings,geos-topology -- --test-threads=1 --skip gpu_extrusion --skip brdf_tile

cargo test --doc --workspace --features default,async_readback,copc_laz,weighted-oit,enable-pbr,enable-ibl,enable-csm,enable-tbn,enable-normal-mapping,enable-hdr-offscreen,enable-render-bundles,enable-renderer-config,enable-memory-pools,enable-staging-rings,geos-topology
```

Focused contract/live commands:

```text
python -m pytest tests/test_world_coord_f32_gate.py tests/test_m06_anchoring_boundary.py -q
python -m pytest tests/test_m06_full_geospatial_viewer.py -m interactive_viewer -vv
python -m pytest tests/test_viewer_ipc.py tests/test_api_contracts.py -q
python -m pytest tests/test_terrain_viewer_pbr.py tests/test_vector_overlay_rendering.py -m interactive_viewer -q
python -m pytest tests/test_recipe_goldens.py tests/test_determinism_hash.py -q
```

The P0 live command is a local evidence gate on RTX 3070/Vulkan unless CI gains a required equivalent runner. Build the release viewer first, install Pillow explicitly for snapshot/overlay PNG I/O, and reject any run reporting a skipped M-06 test. The existing informational macOS lane is supplementary only.

### Allocation and budget gates

The implementation must satisfy all of the following:

- Default `global_tracker` policy remains enforce and limit remains 512 MiB (`src/core/memory_tracker/registry.rs:5-7,28-46`).
- Every new/touched GPU buffer/texture allocation uses a tracked constructor from `src/core/resource_tracker.rs:338-423` and propagates `RenderError::Budget` rather than warning/falling back.
- A no-new-raw-call diff check produces no added line matching `create_buffer`, `create_buffer_init`, `create_texture`, or `create_texture_with_data` outside `resource_tracker.rs`.
- Rebase tests show unchanged GPU buffer/texture counts, total bytes, and host-visible bytes after initial upload.
- The full 500k-point + terrain + vector/label fixture reports peak host-visible GPU bytes below 536,870,912.
- Once integrated onto the allocation-gated baseline, `python -m pytest tests/test_allocation_gate.py -q` is mandatory. On commit `5e625c0a`, that file does not exist; absence is a blocking rollout fact, not a passing result.

## Rollout and verification checklist

1. **Preflight truth**
   - [ ] Confirm branch/HEAD and record `git status --short`; preserve `python/forge3d/forge3d.pdb`.
   - [ ] Re-read `ci.yml`, `.cargo/config.toml`, `pyproject.toml`, budget policy, and allocation gate after any baseline integration.
   - [ ] Re-run the world-field and matrix `rg` inventories; update a table/test if new producers exist.
   - [ ] Confirm the live environment has NumPy and Pillow; generate DEMs with native `forge3d.gis.write_raster` and do not require optional rasterio.
2. **Land bottom-up with compile checkpoints**
   - [ ] M06-FGV-01 metadata/fallback tests green.
   - [ ] M06-FGV-02 one-owner/one-rebase tests green, including zero rebases for a stationary-target 360-degree orbit.
   - [ ] M06-FGV-03 uniform layout and WGSL pipeline construction green; local visual parity, south-up/X-mirrored coverage, and resolution-independent georeferenced simple relief measured.
   - [ ] M06-FGV-04 operation-based matrix source contract covers every row above, positive no-matrix tests pass, and temporal reset test is green.
   - [ ] M06-FGV-05 re-narrow/rebase resource tests and unified f64 `RichPickResult.world_pos` producer tests are green.
   - [ ] M06-FGV-06 Rust serde and scene-review precision tests green.
   - [ ] M06-FGV-07 Python payload/stub/API tests green.
   - [ ] M06-FGV-08 old Option-1 symbols/header/assertions removed from the complete boundary file, residual contract tests green, and the positive/component-aware f32 gate catches `origin.x as f32`.
3. **Build and static enforcement**
   - [ ] `maturin develop` succeeds after final Rust/WGSL edit.
   - [ ] Interactive viewer binary build succeeds.
   - [ ] `cargo fmt --check` and `cargo forge3d-clippy` succeed.
   - [ ] Curated cargo check/test/doctest commands succeed with the feature list read from the implementation branch.
   - [ ] Allocation gate exists and passes after baseline integration; no new raw allocation appears in the diff.
   - [ ] World-coordinate gate reports exactly one sanctioned narrowing implementation, positively requires Anchor conversion for DVec/f64 operands, and explicit viewer storage contracts pass.
   - [ ] Matrix gate rejects look-at/perspective/projection-view producer operations outside its explicit file/callsite allowlist; every allowlisted and no-matrix inventory row has a positive assertion.
4. **Live measurable acceptance**
   - [ ] Fresh release `interactive_viewer` binary built; local RTX 3070/Vulkan M-06 run reports zero skipped tests. Informational macOS CI status is not substituted.
   - [ ] Local vs UTM-translated render: `ssim(..., data_range=1.0) >= 0.999`, normalized mean absolute RGB error <= 0.5/255; adapter/backend recorded.
   - [ ] South-up and X-mirrored simple/PBR renders have non-zero coverage and correct asymmetric orientation.
   - [ ] Pre/post target-driven 1 km rebase render meets the same thresholds with stable pick ID, absolute f64 pick world position, and label screen position.
   - [ ] A complete stationary-target orbit produces zero rebases/history invalidations and no temporal/fog/shadow flash.
   - [ ] Two Earth-scale features 1 mm apart have distinct packed render positions and visible separated centroids; negative control fails.
   - [ ] Non-zero-origin PBR/CSM, volumetrics, DoF, raster overlay, vector, label, point-cloud, and snapshot paths all execute without validation errors.
   - [ ] Offscreen Scene goldens/determinism hash remain unchanged.
5. **Resource and compatibility acceptance**
   - [ ] Ten rebases allocate no new GPU buffers/textures and remain below 512 MiB host-visible usage.
   - [ ] Old local JSON/bundle/vector fixtures load and render within existing thresholds.
   - [ ] Malformed vector IDs/non-finite world values reject transactionally with stable diagnostics.
6. **Documentation and handoff**
   - [ ] M-06 plan and anchoring note state the final contract without CityJSON/point-cloud/overlay overclaims.
   - [ ] Python docs/stubs state coordinate basis, f64 source semantics, and normalized exceptions.
   - [ ] Final `git diff --check` and `git status --short` show only intended changes plus the preserved pre-existing dirty artifact.
   - [ ] Handoff records exact commands, pass counts, SSIM/MAE, millimetre centroid delta, backend, peak memory, allocation counts, and sole narrowing-site count.

M-06 Option 2 is complete only when the live viewer demonstrates rigid-translation equivalence and millimetre retention at Earth scale while the local fallback, offscreen Scene, budget, allocation, and single-narrowing contracts remain green. Type widening without that evidence is not completion.

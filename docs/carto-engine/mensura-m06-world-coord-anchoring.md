# MENSURA M-06 — World-Coordinate f32 Audit & Anchoring Roadmap

Status: the single **narrowing** invariant is closed; renderer-wide **storage**
of absolute world coordinates in `f32` is the remaining, larger work. This file
is the source-level audit the plan requires ("record each audited binding") and
the precise roadmap for the rest.

## What is closed

- **One narrowing site.** The only textual `f64 → f32` narrowing of a world
  coordinate is `Anchor::narrow` (`src/camera/anchor.rs`), grep-gated by
  `tests/test_world_coord_f32_gate.py` (which now also asserts exactly one
  `as f32` in the anchor module and forbids any `Vec3::new(x as f32, y as f32,
  z as f32)` reconstruction anywhere).
- **`Anchor` primitive.** f64 origin, validated 1 km rebase threshold
  (`try_with_epsilon`), non-finite-eye guard, `to_render_f32`, and rebase
  integration tests at ECEF/UTM magnitudes.
- **Subsystems already f64 → single anchor cliff** (verified): `camera`
  (`camera_look_at`/`camera_view_proj` take `(f64,f64,f64)`), offscreen `scene`
  (`set_camera_look_at` f64; `Text3DInstance.origin: DVec3`), `tiles3d`
  (`[f64;…]`/`DVec3`/`DMat4` end to end; `pnts` positions `Vec<f64>`),
  `pointcloud` (`PointBuffer.positions: Vec<f64>`), and `gis/geometry`
  (`Coord{x:f64,y:f64}`).

## GPU-uniform / WGSL conclusion

No GPU uniform or WGSL input carries an **absolute** world position. Shader-facing
`f32` positions (e.g. `FogCameraUniforms.eye_position`, terrain-local vertex
transforms, path-tracer `cam_origin`, shadow/light-space bounds) are
**render-space or terrain-local** by construction, which is exactly where `f32`
is correct. The world→render narrowing happens on the CPU, before the uniform is
built. So the GPU boundary is clean; the remaining work is CPU-side storage.

## Remaining: absolute world coordinates stored as f32 (renderer-wide, pending)

These hold projected/ECEF world positions in `f32` (glam `Vec2/Vec3`, `[f32;N]`,
or PyO3 `f32` params) — no `as f32` token, so invisible to the textual gate.
Magnitude of the resulting planimetric error: UTM easting ≈ 0.03–0.06 m, UTM
northing / ECEF ≈ 0.25–1 m, Web Mercator ≈ 1–2 m. Ranked by how load-bearing
each is for "a feature in the wrong place":

1. **Viewer IPC scene placement — HIGHEST.** `src/viewer/viewer_enums/commands.rs`:
   `SetCamLookAt.{eye,target,up}: [f32;3]`, `SetTerrain(Camera).target: [f32;3]`,
   `LoadOverlay.extent: [f32;4]`, `AddVectorOverlay.vertices: Vec<[f32;8]>`,
   `AddLabel.world_pos`, `AddLineLabel/AddCurvedLabel.polyline`,
   `AddCallout.anchor`, `SetTransform.translation`. The interactive viewer's
   camera itself is not anchored.
2. **Vector API + overlay — HIGH.** `src/vector/api/py.rs` narrows f64 numpy →
   `Vec2` at ingest (`parse_polygon_from_numpy`, `add_lines/points/graph`);
   `src/vector/{data,extrusion}.rs`, `src/vector/api/core.rs` store `[f32;2]`/
   `Vec2` "world coordinates"; `src/viewer/terrain/vector_overlay.rs`
   `VectorVertex.position: [f32;3]`, `terrain_origin: (f32,f32)`.
3. **Labels — HIGH.** `src/labels/{types,layer,projection}.rs` `world_pos: Vec3`,
   `polyline: Vec<Vec3>` (unanchored world anchors).
4. **CityJSON import — HIGH.** `src/import/cityjson/{types,geometry}.rs` decode
   correctly to `[f64;3]` (`x*scale + translate`) then store `Vec<f32>` — the CRS
   `translate` (absolute easting/northing) lands in `f32`.
5. **Export — MEDIUM.** `src/export/{svg_labels,mod,projection}.rs` project label
   `world_pos: Vec3` for SVG/PDF output.
6. **Offscreen Scene lights / instances — LOW.** `src/scene/py_api/*lights*`,
   `src/lighting/*` `position: [f32;3]` — world-space in principle, but the
   offscreen `Scene` authors near the origin so magnitudes stay small.

### Anchoring plan (per subsystem)

Give each geospatial renderable an `f64` object origin, subtract the current
`Anchor` origin, and recompute the offset on rebase — the pattern `tiles3d`/
`pointcloud` already use. Concretely: widen the viewer IPC command world fields
to `f64` and rebase them in the viewer camera (1); carry `Vec<[f64;2/3]>` in the
vector ingest/overlay path and narrow only through `Anchor` at upload (2); store
label/callout anchors as `DVec3` and project post-anchor (3, 5); keep the
CityJSON `translate` as the object origin in `f64` (4). Each step needs a
GPU/offscreen visual check at Earth-scale magnitudes before it is called done.

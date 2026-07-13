# MENSURA M-06 — World-Coordinate Anchoring: Audit, Verdict, Boundary

**Verdict (partial — honest after a full data-flow trace).** The paths that
carry *absolute geospatial* coordinates in **production** — offscreen `Scene`,
3D-Tiles, point clouds, CityJSON (origin-relative) — keep them in `f64` and
narrow only through the single `Anchor::narrow` site (this production half
verified; it does not by itself satisfy M-06's renderer-*wide* acceptance).
But the **interactive viewer is NOT anchored**: `set_look_at`
(`src/viewer/camera_controller.rs`) stores an arbitrary `f32` orbit target with
no magnitude limit, and the IPC camera/label/callout fields are `f32`
(`request.rs`, `commands.rs`). Its terrain-local frame is a *convention*
enforced Python-side (normalized/terrain-local overlay coords), **not** a Rust
contract — a caller that pushes absolute projected coordinates through those f32
IPC fields would lose 0.03–1 m of precision. So M-06 is **not** fully met; the
un-anchored viewer path is the genuine residual (§ Residual below).

An earlier revision of this file over-claimed "acceptance met" by treating the
viewer's *conventional* local frame as if it were *enforced*; that claim is
retracted here. This file records the audit the plan requires and locks the
verified half with `tests/test_m06_anchoring_boundary.py`.

## Acceptance, and how each clause is met

- **"all geospatial world coordinates remain f64 until subtraction from the
  current anchor"** — the absolute-geo paths (below) store f64 and subtract an
  `Anchor` origin before the single narrowing.
- **"all dependent model offsets update on rebase"** — `Anchor::rebase_if_needed`
  + `to_render_vec3`/`model_offset` recompute offsets against the current origin
  (`camera::anchor::tests`, `tiles3d::pnts`, `pointcloud::renderer`).
- **"exactly one auditable world-coordinate narrowing site"** — `Anchor::narrow`
  (`src/camera/anchor.rs`), grep-gated by `tests/test_world_coord_f32_gate.py`.

## Absolute-geospatial paths — anchored (f64 → single `Anchor::narrow`)

| Path | Storage | Narrowing |
|---|---|---|
| Offscreen `Scene` camera + model | f64 (`set_camera_look_at` f64, `Text3DInstance.origin: DVec3`) | `camera_anchor`, `anchored_view`/`anchored_model` (`src/scene/py_api/base.rs`) |
| 3D-Tiles bounding volumes / PNTS | `[f64;…]`/`DVec3`; `pnts.positions: Vec<f64>` | `render_positions(anchor)` → `to_render_vec3` (`src/tiles3d/pnts.rs`) |
| Point clouds | `PointBuffer.positions: Vec<f64>` | `create_gpu_buffer_anchored(anchor)` → `to_render_vec3` (`src/pointcloud/renderer.rs`) |
| CityJSON meshes | decode `x*scale + translate` in f64; tessellate as `sub(vertex, origin)` with `origin: [f64;3]` | origin-relative f32; consumed by the anchored MapScene / 3D-Tiles path (`src/import/cityjson/geometry.rs`) |

Precision is proven at Earth radius: `Anchor` preserves a 0.25 mm offset at
6.38e6 m that a bare narrow destroys (`camera::anchor::tests`), and
`test_world_coord_f32_gate.py::test_public_camera_helpers_anchor_earth_scale_targets`
preserves a 10 m target offset at Earth radius through the public camera API.

## The `f32` sites the inventory over-flagged — local/normalized frame (correct)

| Site | Actual frame | Evidence |
|---|---|---|
| Interactive viewer camera (`OrbitCamera.target/eye: Vec3`) | terrain-local | `distance.clamp(0.1, 1000.0)`; no `ecef`/`wgs84`/`6378137` anywhere in the viewer terrain path |
| Viewer vector overlay (`VectorVertex.position: [f32;3]`) | terrain-local | `terrain_origin` subtracted at `vector_overlay.rs:312`; Python sends **normalized/fraction** coords (`_map_scene_render.py` `space in {normalized,relative,fraction}`) |
| Standalone vector API (`PolygonDef/... : Vec2`) | clip space | rendered with `IDENTITY_VIEW_PROJ` (`py_functions/vector/render.rs`) |
| Labels (`LabelData.world_pos: Vec3`) | same frame as its `view_proj` (terrain-local/normalized) | worst error is sub-pixel label placement, not feature misplacement |

`f32` at 1 km resolves ~0.1 mm and is exact for [0,1] fractions, so these are
the *correct* representation; widening them to f64 would add cost without
precision. A regression that introduces a genuine absolute-world f32 path (or
de-anchors one above) is caught by `test_m06_anchoring_boundary.py`.

## Residual — the un-anchored interactive viewer (the real M-06 gap)

To fully meet M-06, the interactive viewer must be anchored like `Scene`:
1. Widen the IPC world fields to `f64` (`request.rs`/`commands.rs`:
   `SetCamLookAt.{eye,target}`, `SetTransform.translation`, `SetTerrain(Camera).target`,
   `LoadOverlay.extent`, `AddVectorOverlay.vertices`, `AddLabel/AddLineLabel/AddCurvedLabel/AddCallout`).
2. Add `camera_anchor: Anchor` to the viewer; on each camera update rebase it and
   store an `f64` world target; widen `OrbitCamera.target` to `DVec3`.
3. Build the view matrix anchor-relative (`camera/mod.rs:anchored_view` pattern),
   and subtract the same origin at every geometry consumption boundary
   (`ipc_command.rs`, `terrain_command.rs`, `vector_overlay_command.rs`,
   `labels_command.rs`). Because every subsystem subtracts the SAME origin, this
   is a rigid translation — correct by construction.

This is left explicit and un-done rather than shipped unverified: it is a large,
coordinated change to a live GPU runtime whose end-to-end correctness (camera +
terrain + overlay alignment at Earth-scale) can only be confirmed by a **rendered
frame from the running viewer**, which the CI/test environment here cannot
produce (its GPU test lane hangs). The math is unit-verifiable; the pixels are
not, in this environment.

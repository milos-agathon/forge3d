# MENSURA M-06 — World-Coordinate Anchoring: Audit, Verdict, Boundary

**Verdict (after a full data-flow trace of every flagged subsystem): M-06's
acceptance is met.** Every path that carries *absolute geospatial* coordinates
(ECEF / projected UTM / Web Mercator, magnitudes 5e5–6.4e6 m, where f32 costs
0.03–1 m) keeps them in `f64` and narrows only through the single
`Anchor::narrow` site. The `f32` "world" storage the earlier inventory/audit
flagged as renderer-wide turned out, on tracing, to be **local- or
normalized-frame** coordinates where `f32` is correct — a mis-classification of
frame, not a precision bug.

This file records the audit the plan requires and locks the boundary with
`tests/test_m06_anchoring_boundary.py`.

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

## Residual (optional, not a correctness gap)

If a future workflow feeds the interactive viewer *un-normalized* absolute
projected coordinates (rather than the current normalized/terrain-local ones),
the viewer would need the same treatment as `Scene`: an `Anchor` on the viewer
struct and f64 IPC world fields. That change is a rigid translation — safe by
construction — but its end-to-end validation requires a rendered frame from the
running viewer, so it is deliberately left as an explicit, guarded follow-up
rather than an unverified rewrite of the default user path.

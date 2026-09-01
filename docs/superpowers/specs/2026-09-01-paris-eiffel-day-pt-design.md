# Paris Eiffel daylight still — path-traced look-match

Date: 2026-09-01
Target: `examples/paris_eiffel_day.py` (+ engine changes in `src/`)
Reference: `C:\Users\milos\Downloads\ZR8duX05LH6ajR-X.mp4`, daylight frame at t = 3.3 s

## Goal

Reproduce the daylight look of the reference video as a single high-resolution
still, rendered through forge3d's GPU path tracer with the scene carried as real
triangle geometry.

The reference is a 6.1 s, 960x720 iPad screen recording of Apple Maps 3D over the
Eiffel Tower, cycling night to day to night and ending with the Control Centre
pulled down. The usable daylight frame is at t = 3.3 s. Extract with:

```bash
ffmpeg -y -ss 3.3 -i ZR8duX05LH6ajR-X.mp4 -frames:v 1 reference_day.png
```

Scope decisions taken during design:

- **Full look-match**, not a grade-only pass: framing, coverage, palette, light,
  and the surface treatment (trees, cars, footpaths, pitch).
- **A still**, not the night-to-day sequence.
- **Path tracing**, chosen by the user over the CPU painter that the example
  currently drives.

## Measured reference

All values sampled from the t = 3.3 s frame. These supersede the current
`REFERENCE_*` constants in the example, every one of which is wrong.

| Sample | Reference | Current constant |
|---|---|---|
| Frame mean RGB | (140, 149, 164) | (223, 234, 227) |
| Frame median RGB | (144, 155, 175) | (255, 255, 247) |
| Mean HSV saturation | 65 | 40 |
| Street / road | (203, 191, 186) | (244, 232, 207) |
| Ground base | (202, 196, 185) | (238, 240, 228) |
| Lawn | (153, 164, 153) | (170, 220, 119) |
| Roof, lit | (207, 200, 198) | (180, 188, 214) |
| Wall, shadow face | (150, 142, 179) | no wall tint |
| Water | (39, 56, 175) | (34, 112, 224) |
| Tower, lit | (181, 97, 90) | (153, 89, 35) |
| Pitch | (130, 156, 64) | (92, 176, 78) |

Two properties define the look and neither is currently reproduced:

1. **A cool violet ambient cast over the whole frame.** Blue exceeds green
   exceeds red in the global mean. The violet shadow face at (150, 142, 179) is
   the single most identifying feature.
2. **Desaturated greens against saturated accents.** Overall saturation is
   *higher* than the current render (65 vs 40) because of the deep cobalt Seine
   and the salmon tower, while the lawns are far *less* saturated. The existing
   `apply_reference_day_grade()` applies a uniform +10% saturation and +3.5%
   exposure, which is the wrong direction on both counts.

The current render's median of (255, 255, 247) reflects that roughly half the
frame is empty background, addressed by workstream C.

## Renderer: what the engine can and cannot do

Established by reading the source, not assumed:

- `forge3d.path_tracing.PathTracer` **is a stub.** `add_triangle` discards
  geometry ("accepted for API parity but ignored by the deterministic
  fallback") and `render_rgba` synthesizes a noise gradient behind a
  `synthetic_ok=True` gate. Unusable.
- The real entry point is
  `forge3d.path_tracing.hybrid_render_terrain_reference()`, GPU-backed via
  Vulkan. Confirmed working adapter: NVIDIA RTX 3070, driver 595.95.
- It accepts `mesh_vertices` (N,3 float32) and `mesh_indices` (M,3 uint32),
  mixed into the scene through the shared hybrid traversal. This contradicts
  `osm_city_pt_3d.py`'s docstring claim that the tracer handles "heightfields,
  not triangle meshes".
- It returns `rgba`, plus `albedo`, `normal` and `depth` AOVs, and converges on
  a per-pixel luminance variance threshold.
- **It takes a single global `albedo`.** There is no per-mesh, per-triangle or
  per-vertex material. This constraint drives workstream B.

No example and no test in the repository passes `mesh_vertices`. This work is
the first caller of that path.

## Architecture

```
OSM + IGN LoD2.2 + Eiffel STL
    |
    +-- merged triangle mesh ------> PT (neutral albedo) --> light field
    |                                                            |
    +-- class-ID raster, same camera --> colour map -------------+
                                                                 |
                                                          multiply, grade
                                                                 |
                                                            daylight still
```

**PT supplies the light; a matched-camera class-ID raster supplies the hue.**
This is forced by the single-global-albedo constraint, and it is the pattern
already used by `osm_city_pt_3d.py`. Because we trace real meshes at the
oblique camera, we avoid the nadir-render-plus-ground-homography contortion
that the heightfield route requires: both passes share one camera directly.

What path tracing buys over the CPU painter is real light — lattice
self-shadowing, the tower's shadow thrown across the Champ de Mars, and true
ambient occlusion in the street canyons.

## Workstream A — engine: BVH traversal for the hybrid mesh path

The Rust side already does nearly all of this. `render_terrain.rs:583` builds an
SAH BVH over the supplied mesh, uploads it, sets `mesh_bvh_node_count`, and sets
`traversal_mode`. The buffer is bound at `@group(1) @binding(4)`.

The gap is in the shader. `intersect_mesh` in `src/shaders/hybrid_traversal.wgsl`
ignores `mesh_bvh_nodes` and sweeps every triangle linearly, under a comment
that reads "BVH traversal for mesh intersection":

```wgsl
// Brute-force triangle sweep. This keeps the shader simple and guarantees
// we shade meshes even when no GPU BVH data is available.
for (var tri = 0u; tri + 2u < index_count; tri = tri + 3u) {
```

At 186k triangles this is roughly 10^13 ray-triangle tests for one frame: it
will hit a TDR, not finish. With SAH BVH traversal it becomes about 50 node
tests per ray, which is minutes on the target card.

### Changes

1. **Write stack-based BVH traversal** in `intersect_mesh`, mirroring the
   working traversal in `src/shaders/pt_intersect.wgsl`. `MAX_BVH_STACK_SIZE`
   is already declared. Leaf nodes carry `left` = first triangle and
   `right` = triangle count, with `flags` bit 0 marking a leaf.
2. **Permute the index stream at upload.** `build_bvh` reorders triangles and
   leaves reference `BvhHandle.tri_indices`, but `sdf/hybrid.rs:353` uploads
   `self.indices` raw and unpermuted, and `tri_indices` is never uploaded.
   Reorder at upload so leaves index directly, mirroring
   `path_tracing/mesh/upload.rs:65`. Chosen over uploading `tri_indices` as a
   fourth buffer because it needs no binding or layout change and matches the
   convention the atlas path already set.
3. **Pad the BVH upload to a 48-byte stride.** Rust's `BvhNode` is 40 bytes
   (`static_assert` in `accel/cpu_bvh/types.rs`). The WGSL struct lays out at 48,
   because `vec3f` forces 16-byte alignment. The atlas path has `GpuBvhNode48`
   for exactly this; the hybrid path writes raw 40-byte records.

Each of these three fails **silently**: wrong triangles, garbage nodes, or an
invisible mesh. None produces an error.

### Known tripwire

`upload_mesh_to_gpu` early-returns on `BvhBackend::Gpu` before creating any mesh
buffers, leaving `mesh_buffers: None`, which makes the bind group fall back to
dummy buffers and the mesh silently invisible. `render_terrain.rs` passes
`GpuContext::NotAvailable` so the CPU backend is selected and this path is not
taken. Do not change that call without addressing this.

### Test

A unit test asserting that BVH traversal and brute-force sweep return identical
hits (t, normal, triangle) for a fixed ray set against a non-trivial mesh. This
must exist and pass before the Paris scene is pointed at the mesh path.

## Workstream B — colour

1. **Neutral-albedo PT pass** over the merged mesh, producing the light field.
2. **Class-ID raster** at an identical camera, reusing the existing CPU
   projector from `osm_city_daycycle.prepare_scene`. Classes: base, landuse,
   park, water, road, road_hi, building roof, building wall, tower, tower deck,
   tree, pitch, car.
3. **Composite**: colour map multiplied by the light field.
4. **Grade** to the measured targets, including the violet ambient cast.

The camera match is a direct correspondence — PT takes a pinhole dict of
`origin` / `look_at` / `up` / `fov_y`, mapping one-to-one onto the projector's
`eye` / `target` / `up` / `fov_deg`. If analytic matching drifts, fall back to
fitting a DLT against a marker grid.

## Workstream C — coverage and framing

- Raise `DEFAULT_RADIUS_M` from 400 to about 900 so the AOI circle circumscribes
  the view footprint and the ground reaches every frame corner. Costs roughly 40
  IGN tiles instead of 10, downloaded once.
- Drive framing from the camera. Delete `zoom_reference_frame()` and
  `REFERENCE_CONTENT_ZOOM`; a top-anchored post-crop is not framing.
- Re-solve `REFERENCE_ROTATION_DEG`. The current render runs the Seine from
  top-left to bottom-right; the reference runs it from mid-left to top-right, so
  the scene is out by roughly a quadrant. Verify by landmark alignment against
  Pont d'Iena, the Trocadero basin, and the track at Stade Emile Anthoine —
  not by eye.

## Workstream D — palette and grade

Replace every `REFERENCE_*_RGB` constant with the measured values above. Rewrite
`apply_reference_day_grade()` to move toward the reference: desaturate the
greens, hold the blues and the salmon, apply the violet ambient cast.

## Workstream E — surface treatment

- **Trees**: OSM `natural=tree` points plus row-scatter along avenues at about
  12 m spacing, replacing the current 27 m. The scene reports 211 trees today
  and none are visible in the render.
- **Cars**: small coloured boxes on road centrelines.
- **Footpaths**: thin cream surfaces through the Champ de Mars and Trocadero.
- **Pitch**: olive green with its track.

This workstream is new geometry rather than retuning, and is the most likely to
sprawl. If cars need more scaffolding than they are worth, say so explicitly
rather than dropping them silently.

## Verification

In order, each gating the next:

1. BVH-vs-brute-force equivalence test passes.
2. Low-resolution PT smoke render confirms the mesh path produces hits at all.
3. 960x720 render compared region-by-region against the reference frame.
4. High-resolution final pass.

**Acceptance gate**: frame mean within 8 per channel of (140, 149, 164), and
each of the sampled surface classes within 12 of its measured target. Numeric,
not asserted.

## Risks

- The engine change requires a Rust rebuild and may surface further stride or
  binding mismatches beyond the three identified here.
- PT convergence at full resolution with a lattice in frame may need tiling to
  avoid a TDR. Prior work on this card required capping `grid_max` at 1536 and
  isolating tiles in subprocesses.
- Exact camera match with Apple's proprietary view is unlikely; expect a few
  degrees of drift in orientation.
- The reference is a screen recording, so it carries compression artefacts and
  the iPad's display transform. Sampled colours are the target as *displayed*,
  which is the right target for a look-match but is not a colorimetric ground
  truth.

# Paris Eiffel daylight still — path-traced look-match

Date: 2026-09-01
Revision: 2 (rev 1 failed contract review; see "Corrections from review" at the end)
Target: `examples/paris_eiffel_day.py` + engine changes in `src/`
Reference: `ZR8duX05LH6ajR-X.mp4`, daylight frame at t = 3.3 s

## Goal

Reproduce the daylight look of the reference video as a single high-resolution
still, rendered through forge3d's GPU path tracer with the scene carried as real
triangle geometry.

Scope decisions taken during design: **full look-match** (framing, coverage,
palette, light, surface treatment), a **still** rather than the night-to-day
sequence, and **path tracing**, chosen by the user over the CPU painter that the
example currently drives.

This revision splits the work into two parts, because review established that
the engine work is a project in its own right rather than a preamble:

- **Part 1 — engine.** Make the hybrid path tracer capable of tracing a
  186k-triangle mesh at all, and of emitting linear light. Independently
  testable, no Paris content involved.
- **Part 2 — the Paris still.** The look-match, which depends on Part 1.

Part 2 must not begin until Part 1's gates pass.

## 0. Reproducibility prerequisites

Rev 1 was not reproducibly bound. Binding it now.

### 0.1 Tracked-source problem (must be resolved first)

`.gitignore:395` ignores `examples/**/*.py`. Both `examples/paris_eiffel_day.py`
and the cited precedent `examples/osm_city_pt_3d.py` are therefore **untracked**.
`examples/osm_city_daycycle.py` *is* tracked, so the ignore is already being
overridden case by case.

Before any implementation: force-add the target and its direct dependencies
(`git add -f`), or add negation rules to `.gitignore`. A spec cannot gate work on
a file whose state is not recorded. **This is a blocking prerequisite**, not a
cleanup task.

### 0.2 Reference frame

| Property | Value |
|---|---|
| Source | `ZR8duX05LH6ajR-X.mp4` |
| SHA256 | `7CEC3ACB0F7C4294699DECCFA07E124EC5C9920190C70BC8CBC43A38CD05CD4B` |
| Dimensions | 960x720 |
| Duration | 6.106848 s |
| Frame timestamp | t = 3.3 s |

Extraction is pinned by tool version and by the hash of the extracted frame:

```bash
ffmpeg -y -ss 3.3 -i ZR8duX05LH6ajR-X.mp4 -frames:v 1 reference_day.png
```

ffmpeg 8.0.1-full_build (Gyan). The extracted `reference_day.png` must be
committed as the fixture and its SHA256 recorded, so the comparison target is
immutable regardless of ffmpeg behaviour on any other machine.

**Colour-space caveat.** The reference is a screen recording of an iPad display.
Sampled values are the target *as displayed*, carrying H.264 chroma subsampling
and the device transform. This is the correct target for a look-match but is not
colorimetric ground truth, and no claim of colour accuracy beyond "matches this
recording" is supported.

### 0.3 Dataset and environment pinning

Recorded in the run manifest, and required for any acceptance claim: OSM Overpass
query text and retrieval date; IGN tile URLs with per-tile SHA256 (the existing
manifest already records URLs and must add hashes); Eiffel STL SHA256 (already
pinned: `bff2ce1d…55adf`); env-map source; wgpu backend actually selected;
adapter name and driver version; seed; spp; frame counts; variance threshold; and
the SHA256 of every output image.

## 1. Measured reference

Sampled from the pinned frame. These supersede the current `REFERENCE_*`
constants, every one of which is wrong.

Global measurements, independently reproduced during review:

| Metric | Value |
|---|---|
| Mean RGB | (140.116, 149.472, 163.736) |
| Median RGB | (144, 155, 175) |
| Mean HSV saturation | 65.327 |

Per-class samples:

| Class | Reference | Current constant |
|---|---|---|
| Street / road | (203, 191, 186) | (244, 232, 207) |
| Ground base | (202, 196, 185) | (238, 240, 228) |
| Lawn | (153, 164, 153) | (170, 220, 119) |
| Roof, lit | (207, 200, 198) | (180, 188, 214) |
| Wall, shadow face | (150, 142, 179) | no wall tint |
| Water | (39, 56, 175) | (34, 112, 224) |
| Tower, lit | (181, 97, 90) | (153, 89, 35) |
| Pitch | (130, 156, 64) | (92, 176, 78) |

The per-class figures above are single-patch samples taken during design and are
**not yet acceptance-grade**. Section 8.2 defines how they must be re-derived as
masked region medians before they can gate anything.

Two properties define the look:

1. **A cool violet ambient cast.** Blue exceeds green exceeds red in the global
   mean. The violet shadow face at (150, 142, 179) is the most identifying
   feature.
2. **Desaturated greens against saturated accents.** Overall saturation is
   *higher* than the current render (65 vs 40) because of the deep cobalt Seine
   and salmon tower, while lawns are far *less* saturated. The existing
   `apply_reference_day_grade()` applies uniform +10% saturation and +3.5%
   exposure — wrong on both counts.

# Part 1 — Engine

## 2. What the engine actually provides

- `forge3d.path_tracing.PathTracer` **is a stub**: `add_triangle` discards
  geometry, `render_rgba` synthesizes noise behind `synthetic_ok=True`. Unusable.
- The real entry point is `hybrid_render_terrain_reference()`
  (`python/forge3d/path_tracing.py:893`), GPU-backed **via wgpu**. `gpu.rs:159`
  selects all backends when no backend env var is set, so the backend in use is
  whatever wgpu picks and must be *recorded per run*, not asserted. Local
  hardware observed: NVIDIA RTX 3070, driver 595.95.
- It requires a **heightmap** as its first positional argument and always
  constructs a `TerrainPtScene` (`render_terrain.rs:548`). Mesh geometry is
  *mixed into* a terrain scene; there is no mesh-only mode. Section 5 defines
  the terrain/mesh split this forces.
- Mesh hits use a hardcoded albedo `vec3f(0.7, 0.7, 0.8)`
  (`hybrid_traversal.wgsl:240`); the `albedo` parameter applies to terrain only.
  Two flat classes, against the thirteen this scene needs.
- The mesh path is exercised by exactly one test,
  `test_mixed_scene_mesh_and_terrain` (`tests/test_hybrid_terrain_pt.py:718`),
  which mixes a **two-triangle quad** into a heightfield. The path works; two
  triangles is also why the defects below have gone unnoticed.

## 3. Engine defect 1 — BVH build is O(n²)

`sah_cpu.rs:250` `find_best_split` has **no binning**. For each of 3 axes it
sorts all centroids, then for every unique centroid calls `evaluate_split`,
which scans all n primitives.

At the root with 186k triangles: 3 × 186,000 × 186,000 ≈ **1.04 × 10¹¹** AABB
operations, for the root split alone. This is decided by arithmetic; it does not
need benchmarking to rule out.

**Fix: binned SAH.** Replace the exhaustive centroid scan with fixed bucket
counts (16 bins per axis), giving O(n) per node and O(n log n) overall. This is
the standard construction and keeps the work on the CPU builder, self-contained
and testable without new GPU surface.

Rejected alternatives: the GPU LBVH builder (`accel/lbvh_gpu`) would be faster
but selecting it trips the `BvhBackend::Gpu` early-return in section 4.3, and
carries its own packing contract; offline caching leaves the O(n²) one-off cost
unaddressed and helps no other caller.

**Gate:** build of a 186k-triangle mesh completes in bounded time, recorded.
Tree quality regression: SAH cost of the binned tree within 5% of the exhaustive
tree on a mesh small enough to build both ways.

## 4. Engine defect 2 — GPU packing contract is undefined

Rev 1 documented the wrong type. There are two unrelated `BvhNode` definitions:

- `accel::cpu_bvh::BvhNode` — 40 bytes, fields `flags`/`left`/`right`, with
  reordered indices on `BvhCPU.tri_indices`. **Not used by the render path.**
- `accel::types::BvhNode` — 48 bytes, used via `CpuBvhData`. **This is the one
  that reaches the GPU.**

### 4.1 Field-offset mismatch

`accel::types::BvhNode` is `{ aabb: Aabb, kind, left_idx, right_idx, parent_idx }`
where `Aabb` is `{ min: [f32;3], _pad0: f32, max: [f32;3], _pad1: f32 }`. Against
the WGSL struct:

| WGSL field | bytes | reads Rust | result |
|---|---|---|---|
| `aabb_min` | 0–12 | `aabb.min` | correct |
| `left` | 12–16 | `aabb._pad0` | **padding** |
| `aabb_max` | 16–28 | `aabb.max` | correct |
| `right` | 28–32 | `aabb._pad1` | **padding** |
| `flags` | 32–36 | `kind` | correct by coincidence |
| `_pad` | 36–40 | `left_idx` | discarded |

Both types are 48 bytes, so the **stride is already correct** and rev 1's
"40-vs-48 stride" diagnosis was wrong. The defect is field offsets: every child
and primitive index reads as padding.

**Fix:** define an explicit GPU packing conversion — a `#[repr(C)] Pod` struct
matching the WGSL layout exactly, converted from `accel::types::BvhNode` at
upload, mirroring how `GpuBvhNode48` serves the atlas path. Do not reinterpret
the Rust type directly.

### 4.2 Root index and index permutation

- **Root is the last node.** `sah_cpu.rs:69` does `nodes.push(root_node)` after
  recursion completes. Traversal must start at `node_count - 1`, not 0. The root
  index must be passed explicitly in the uniforms rather than assumed.
- **Reordered primitive indices live in `CpuBvhData.indices`**, not
  `BvhHandle.tri_indices` (which does not exist — `BvhHandle` is `backend`,
  `triangle_count`, `node_count`, `world_aabb`, `build_stats`). Leaf
  `left_idx`/`right_idx` are a range into that permutation.
- `sdf/hybrid.rs:353` uploads `self.indices` raw and unpermuted, and
  `CpuBvhData.indices` is never uploaded. **Fix:** permute the triangle index
  stream at upload so leaves index it directly, mirroring
  `path_tracing/mesh/upload.rs:65`.
- **Stack depth:** `MAX_BVH_STACK_SIZE` is 32. A binned-SAH tree over 186k
  triangles is ~18 levels deep balanced, but degenerate splits can exceed 32.
  Traversal must either bound tree depth at build time or handle stack overflow
  explicitly — silently dropping nodes is not acceptable.

### 4.3 Known tripwire

`upload_mesh_to_gpu` early-returns on `BvhBackend::Gpu` **before creating mesh
buffers**, leaving `mesh_buffers: None`, so the bind group falls back to dummies
and the mesh is silently invisible. `render_terrain.rs` passes
`GpuContext::NotAvailable`, selecting the CPU backend, so this path is not taken
today. Do not change that call without fixing this first.

## 5. Engine defect 3 — traversal

`intersect_mesh` (`hybrid_traversal.wgsl:136`) ignores `mesh_bvh_nodes` and
sweeps every triangle linearly, under a comment claiming BVH traversal:

```wgsl
// Brute-force triangle sweep. This keeps the shader simple and guarantees
// we shade meshes even when no GPU BVH data is available.
for (var tri = 0u; tri + 2u < index_count; tri = tri + 3u) {
```

**Fix:** stack-based traversal reading the packed nodes from section 4.1,
rooted at the index from section 4.2, mirroring `pt_intersect.wgsl`.

**Gate — equivalence test.** BVH traversal and brute-force sweep must return
identical hits for a fixed ray set against a mesh large enough that leaf
ordering and node stride matter (order 10⁴ triangles, not two).

`HybridHitResult` exposes **no triangle ID** (`t`, `point`, `normal`,
`material_id`, `hit_type`, `hit`), so triangle identity cannot be asserted
directly as rev 1 proposed. Compare on `t` and `normal` at fixed tolerance, and
add a test-only triangle-index field behind a debug flag if identity proves
necessary to localize failures.

## 6. Engine defect 4 — no linear light output

`hybrid_terrain_traversal.wgsl:518` applies `reinhard_tonemap(mean_rgb,
cam_exposure)` before `textureStore`, so the returned `rgba` is **tonemapped
uint8 LDR**. Multiplying class colours by that is not a defined light-field
operation, as rev 1 assumed.

`AovKind` declares `Direct`, `Indirect`, `Emission` and `Visibility` in addition
to `Albedo`/`Normal`/`Depth`, but only `aov_visibility` is written by the terrain
shader, and only Albedo/Normal/Depth are returned to Python
(`render_terrain.rs:1249–1265`). So the linear signal is not currently reachable.

**Fix:** write the pre-tonemap `mean_rgb` to a linear radiance target and expose
it. Reinhard here is `exposed/(1+exposed)` with `exposed = color * exposure`,
which is analytically invertible as `color = ldr / (exposure * (1 - ldr))` — but
inversion is rejected as the primary path because uint8 quantization makes the
top stop unrecoverable and precision collapses as `ldr → 1`, exactly in the lit
regions that matter. Inversion may serve as a cross-check on the new output.

**Gate:** linear output round-trips to the existing LDR output through the
forward Reinhard within tolerance.

# Part 2 — The Paris still

## 7. Architecture

```
OSM surfaces ------> terrain heightfield ---+
IGN LoD2.2 + STL --> merged triangle mesh --+--> PT (linear radiance) --> light
                                                                            |
class-ID raster, same camera --> colour map --------------------------------+
                                                                            |
                                                              multiply in linear
                                                                            |
                                                                grade, encode
```

PT supplies light; a matched-camera class-ID raster supplies hue. This is forced
by the two-flat-albedo constraint of section 2. What PT buys over the CPU painter
is real light transport — lattice self-shadowing, the tower's shadow across the
Champ de Mars, occlusion in street canyons.

**Lighting-model caveat.** `hybrid_terrain_traversal.wgsl:454` implements one
directional sun term, one cosine-weighted environment sample, and binary
visibility. That is direct lighting with environment occlusion. It is **not**
multi-bounce indirect and **not** a separate AO model, and must not be described
as either.

## 8. Terrain / mesh ownership

Forced by section 2: the API requires a heightmap, so the scene must be split.

- **Terrain**: a flat heightfield at ground reference (the existing manifest
  records `ground_reference_m` = 30.369). Concretely: a 2x2 grid — the API
  minimum, and sufficient for a plane — of constant height 0 in the local ENU
  frame, `exaggeration` 1.0, extent 1800 m square (2 × the 900 m radius of
  section 10), giving `spacing` = (1800.0, 1800.0). A minimal grid also keeps
  terrain out of the section 11 memory budget. The plane must sit in the same
  local ENU frame as the meshes, with ground reference subtracted identically.
- **Mesh**: IGN LoD2.2 buildings, the Eiffel STL landmark, tower deck, and trees.
- **Ownership rule**: ground, roads, water, landuse, parks and pitch are
  currently CPU surface layers (`paris_eiffel_day.py:1369`) and stay
  **flat** — they contribute to the class-ID raster and to the terrain plane, not
  to the mesh. Any surface that must occlude or cast shadow becomes mesh.
  No geometry may appear in both, and the spec must name the priority when a
  source provides both.
- Winding, vertex units, and the merge offset applied to each source must be
  stated so that mesh and terrain register.

Undeclared geometry: `build_eiffel_deck_mesh()` exists
(`paris_eiffel_day.py:401`) but is **never added to the scene**
(`paris_eiffel_day.py:1359`), so the declared `REFERENCE_TOWER_DECK_RGB` class is
unreachable. Either wire it in or drop the class.

## 9. Class-ID raster and camera registration

### 9.1 Class provenance

`PreparedTriangle` carries `rgba` and `is_wall`, not a class ID
(`osm_city_daycycle.py:56`). Colour is not identity: two classes may share a
colour, and the grade must move classes independently. **Add an explicit class
field** carried from source through to the raster.

### 9.2 Visibility

The CPU renderer sorts by mean triangle depth and draws with Pillow
(`osm_city_daycycle.py:634`). Painter sorting on mean depth produces wrong
occlusion for interpenetrating and large triangles — the Eiffel lattice is the
worst case. **The class raster must not rely on painter order.** Reconcile
against the PT `depth` AOV: a class pixel is valid only where its rasterized
depth agrees with PT depth within tolerance; disagreement is a defect to report,
not to blend away.

### 9.3 Camera

Rev 1 claimed a one-to-one camera correspondence. That is contradicted by the
projector: `prepare_scene` applies a **content-dependent fit scale and offsets**
(`osm_city_daycycle.py:372`) and then transforms projected points
(`osm_city_daycycle.py:407`), while PT uses raw centered-NDC projection
(`hybrid_terrain_traversal.wgsl:421`). Passing the same eye/target/up/fov does
**not** register the two.

**Fix:** define one canonical projection. Either drive the class raster from the
PT camera parameters directly with the fit disabled, or compute the fit first and
feed the resulting camera to both. **Gate:** landmark reprojection residual
**≤ 2.0 px RMS at 960x720**, measured with a marker-grid DLT as the instrument.
The bound is derived from the class mask: a residual above ~2 px would shift
class boundaries by more than the thinnest features the mask must resolve
(footpaths and road edges, 2–3 px at this resolution), which would corrupt the
section 12.2 per-class statistics rather than merely soften edges.

## 10. Coverage, framing, palette, surface treatment

- **Coverage:** raise `DEFAULT_RADIUS_M` from 400 to ~900 so the AOI circumscribes
  the view footprint and reaches every frame corner. Costs ~40 IGN tiles rather
  than 10, once.
- **Framing:** drive from the camera. Delete `zoom_reference_frame()` and
  `REFERENCE_CONTENT_ZOOM`; a top-anchored post-crop is not framing.
- **Rotation:** re-solve `REFERENCE_ROTATION_DEG`. The current render runs the
  Seine top-left to bottom-right; the reference runs mid-left to top-right, so
  the scene is out by roughly a quadrant. Verify by landmark residual (section
  9.3) against Pont d'Iena, the Trocadero basin, and the Stade Emile Anthoine
  track.
- **Palette:** replace every `REFERENCE_*_RGB` with section 1 values.
- **Grade:** rewrite `apply_reference_day_grade()` to move toward the reference —
  desaturate greens, hold blues and salmon, apply the violet cast — operating in
  linear space before encode.
- **Surface treatment:** trees from OSM `natural=tree` points (the scene reports
  211 trees today and none are visible); cars as small coloured boxes on road
  centrelines; footpaths as thin cream surfaces; olive pitch with track. Avenue
  row-scatter is **deferred**: start with the OSM points and add scatter only if
  the section 11 comparison shows a density gap. This workstream is new geometry
  rather than retuning and is the most likely to sprawl; if cars need more
  scaffolding than they are worth, say so rather than dropping them silently.

## 11. Memory budget and output dimensions

The renderer registers per-pixel allocations (accumulation 16 B, Welford 8 B,
three 80-byte ReSTIR reservoirs, G-buffer, output, AOVs) against a **hard 512 MiB
gate** (`render_terrain.rs:827`), at approximately **352 bytes/pixel**.

The example's current default of 1600x1200 (`paris_eiffel_day.py:1508`) requires
1,920,000 × 352 = 675,840,000 B = **644.53 MiB**, before any terrain or mesh
data. **The default configuration cannot render.** Consequences:

- Maximum single-pass resolution under the gate is ~1,525,000 px
  (e.g. 1424x1068), before scene data. Reference-comparison renders at 960x720
  (675,840 px, ~227 MiB) fit comfortably.
- **Final output dimensions must be stated explicitly** and the memory
  arithmetic shown, not left to a default.
- Any resolution above the single-pass ceiling requires a **defined tile-and-
  stitch protocol**: tile size, overlap, camera sub-frustum derivation, seam
  handling, and per-tile process isolation. "May need tiling" is not a protocol.
  Prior work on this card also required capping grid dimensions at 1536 and
  isolating tiles in subprocesses.

## 12. Acceptance

Rev 1's criteria were unenforceable. Replacing them.

### 12.1 Gate order

Each gates the next; none may be skipped.

| # | Gate | Pass condition |
|---|---|---|
| 1 | Binned SAH build | 186k-triangle build completes; SAH cost within 5% of exhaustive on a small mesh |
| 2 | BVH equivalence | BVH and brute-force hits identical on ~10⁴-triangle mesh, fixed ray set |
| 3 | Existing regression | `test_mixed_scene_mesh_and_terrain` still passes |
| 4 | Linear output | Round-trips to LDR through forward Reinhard within tolerance |
| 5 | PT smoke | Low-res render of the Paris mesh produces hits; hit fraction recorded |
| 6 | Camera registration | Landmark reprojection residual under stated pixel maximum |
| 7 | Class coverage | Every declared class reachable in the raster; unreachable = defect |
| 8 | Colour match | Section 12.2 |
| 9 | Final render | Stated dimensions, within memory gate, output SHA256 recorded |

### 12.2 Colour metric

Deliverable **before** tuning begins, as an executable fixture:

- `reference_day.png`, committed, with SHA256.
- A **mask PNG** assigning each pixel of the reference to one of the thirteen
  classes or to "unassigned", hand-authored once and committed.
- A metric script taking (render, reference, mask) and emitting per-class
  statistics.

Targets are **masked region medians**, not the single-patch samples of section 1,
which were design-time estimates. Tolerances must be **derived**, not asserted:
compute per-class within-region spread on the reference itself and set the
tolerance from it, so a class with genuinely variable colour is not held to a
tolerance tighter than its own variance.

Additional gates so colour alone cannot pass a wrong image: **coverage** —
fraction of frame that is background must be under a stated bound, catching the
current island-on-white failure; **landmark alignment** — reprojection residual
from gate 6; **shadow presence** — the tower's cast shadow must be measurably
present, since it is the point of using PT at all.

Global mean and median are reported for continuity with section 1 but are
**not** sufficient: a uniformly wrong image can match a global mean.

## 13. Risks

- Part 1 is engine work on a shared shader. `intersect_mesh` is reached by any
  caller of the hybrid path, not only this example. The equivalence test is what
  keeps that honest, but the blast radius extends past `paris_eiffel_day.py`.
- Binned SAH changes tree topology for every existing BVH consumer. Gate 1's
  quality regression bounds the change but does not make it invisible.
- The linear-output change adds a render target, raising per-pixel bytes and
  lowering the section 11 resolution ceiling. Recompute the ceiling after it
  lands.
- Exact camera match with Apple's proprietary view is unlikely; a few degrees of
  drift is expected and gate 6 measures rather than assumes it.

## Corrections from review

Rev 1 failed contract review. Substantive corrections, all verified against
source before acceptance:

1. **BVH types conflated.** Rev 1 documented `cpu_bvh::BvhNode` (40 B,
   `tri_indices`); the render path uses `accel::types::BvhNode` (48 B,
   `CpuBvhData.indices`). The "40-vs-48 stride" trap was a misdiagnosis — stride
   is correct, field offsets are not. `BvhHandle.tri_indices` does not exist.
2. **Root index.** Root is the last node, not node 0.
3. **Build feasibility.** The CPU SAH builder is O(n²) with no binning; ~1.04 ×
   10¹¹ operations for the root split alone. Rev 1's premise that the engine
   "already does nearly all the work" was wrong.
4. **Linear light.** Output is Reinhard-tonemapped uint8; rev 1's multiply was
   undefined. Direct/Indirect AOVs are declared but never written.
5. **Camera.** The projector applies a content-dependent fit; rev 1's claimed
   one-to-one correspondence was contradicted.
6. **Class raster.** No class ID exists on `PreparedTriangle`; painter sorting is
   not visibility-correct; the tower deck class is unreachable.
7. **Memory.** The default 1600x1200 exceeds the hard 512 MiB gate.
8. **Terrain.** The API requires a heightmap; rev 1 never defined the
   terrain/mesh split.
9. **Reproducibility.** Target file is gitignored; datasets and environment were
   unpinned.
10. **Backend wording.** "Vulkan confirmed" downgraded to "GPU-backed via wgpu",
    recorded per run.
11. **Lighting wording.** "True ambient occlusion" and multi-bounce claims
    removed; the shader does direct lighting with environment occlusion.

Applied from the over-engineering pass: DLT calibration demoted from a fallback
mechanism to the measurement instrument for gate 6; avenue row-scatter deferred
behind measured need. **Not** applied: deletion of the AOV inventory, because
section 9.2 consumes the `depth` AOV for visibility reconciliation — the
`albedo`/`normal` mention was dropped instead.

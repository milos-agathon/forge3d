# Paris Eiffel daylight still — path-traced look-match

Date: 2026-09-01
Revision: 3 (rev 1 and rev 2 failed contract review; see git history for the diff)
Target: `examples/paris_eiffel_day.py` + engine changes in `src/`
Reference: `ZR8duX05LH6ajR-X.mp4`, daylight frame at t = 3.3 s

## Goal

Reproduce the daylight look of the reference video as a single high-resolution
still, rendered through forge3d's GPU path tracer with the scene carried as real
triangle geometry.

Scope: **full look-match** (framing, coverage, palette, light, surface
treatment), a **still** rather than the night-to-day sequence, and **path
tracing**, chosen by the user over the CPU painter the example currently drives.

Two parts, because the engine work is a project in its own right:

- **Part 1 — engine.** Make the hybrid tracer capable of tracing this scene at
  all, and of emitting light that can be composited. Independently testable.
- **Part 2 — the Paris still.** Depends on Part 1. Must not begin until Part 1's
  gates pass.

## 0. Reproducibility

### 0.1 Tracked source — RESOLVED

`.gitignore:395` ignores `examples/**/*.py`. 43 of 94 example scripts are
nevertheless tracked, so per-file exception is established practice.
`examples/paris_eiffel_day.py` and `examples/osm_city_pt_3d.py` have been
force-added to this branch. `examples/osm_city_daycycle.py` was already tracked.

### 0.2 Reference frame — RESOLVED

Committed at `tests/fixtures/paris_eiffel_day/reference_day.png`. `.gitignore:277`
ignores `*.png`, so the fixture is force-added; any further fixture in this
directory (the §12.2 class mask) needs the same treatment.

| Property | Value |
|---|---|
| Source video SHA256 | `7CEC3ACB0F7C4294699DECCFA07E124EC5C9920190C70BC8CBC43A38CD05CD4B` |
| Video dimensions | 960x720, 6.106848 s |
| Frame timestamp | t = 3.3 s |
| Extractor | ffmpeg 8.0.1-full_build (Gyan) |
| Extracted PNG SHA256 | `CA13E48F90F8C3ACDA40BC086C76E3F1C0CF0D7B456228D7F3D2EA92ED22A067` |
| Extracted PNG size | 972,931 bytes |

```bash
ffmpeg -y -ss 3.3 -i ZR8duX05LH6ajR-X.mp4 -frames:v 1 reference_day.png
```

The committed PNG, not the command, is the acceptance target — ffmpeg behaviour
on another machine cannot change it.

**Colour-space caveat.** The reference is a screen recording of an iPad display,
carrying H.264 chroma subsampling and the device transform. Sampled values are
the target *as displayed*. No claim of colour accuracy beyond "matches this
recording" is supported.

### 0.3 Run manifest

The example already writes `paris_eiffel_day_manifest.json` (schema_version 1).
Bump to schema_version 2, adding: per-tile IGN SHA256 (URLs are already
recorded); Overpass query text and retrieval timestamp; env-map source and hash;
**wgpu backend and adapter name/driver as reported by the adapter that actually
served the run**; seed; spp; min/max frames; variance threshold; achieved frame
count and variance; render dimensions and tiling; and the SHA256 of every output
image. `gpu.rs:159` selects all backends when no backend env var is set, so the
backend is an observation, not a setting — record it, do not assert it.

Eiffel STL SHA256, already pinned in the example:
`bff2ce1d08609b2d761a225cb7c8c1da782369714011567ede7a5008a1e55adf`.

## 1. Measured reference

Global measurements, independently reproduced during review:

| Metric | Value |
|---|---|
| Mean RGB | (140.116, 149.472, 163.736) |
| Median RGB | (144, 155, 175) |
| Mean HSV saturation | 65.327 |

Design-time single-patch samples, superseding the current constants:

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

These eight are **design-time estimates, not acceptance targets**. Section 12.2
defines how all fifteen classes are re-derived as masked region medians. The
seven classes without a measured value (landuse, park, road_hi, footpath,
tower_deck, tree, car) are measured when the mask is authored.

Two properties define the look: a **cool violet ambient cast** (blue > green >
red in the global mean; the violet shadow face at (150, 142, 179) is the most
identifying feature), and **desaturated greens against saturated accents**
(overall saturation 65 vs the current render's 40, driven by the cobalt Seine
and salmon tower, while lawns are far less saturated). The existing
`apply_reference_day_grade()` applies uniform +10% saturation and +3.5% exposure
— wrong on both counts.

# Part 1 — Engine

## 2. What the engine provides

- `forge3d.path_tracing.PathTracer` **is a stub**: `add_triangle` discards
  geometry, `render_rgba` synthesizes noise behind `synthetic_ok=True`.
- The real entry point is `hybrid_render_terrain_reference()`
  (`python/forge3d/path_tracing.py:893`), GPU-backed **via wgpu**.
- It requires a **heightmap** and always constructs a `TerrainPtScene`
  (`render_terrain.rs:548`). Mesh geometry is *mixed into* a terrain scene;
  there is no mesh-only mode. Section 8 defines the split this forces.
- Mesh hits use a hardcoded albedo `vec3f(0.7, 0.7, 0.8)`
  (`hybrid_traversal.wgsl:240`); the `albedo` parameter applies to terrain only.
- The mesh path is exercised by exactly one test,
  `test_mixed_scene_mesh_and_terrain` (`tests/test_hybrid_terrain_pt.py:718`),
  with a **two-triangle quad**. The path works; two triangles is why the defects
  below have gone unnoticed.

## 3. Scene scale — pin before gating

Rev 2 used 186k triangles throughout. That figure is the **radius-400** scene
(`bugfix_preview_960.log:28`: `prepared triangles=185,960`). Recorded runs at
other radii:

| Radius | IGN triangles | Prepared total | Notes |
|---|---|---|---|
| 400 | 41,095 | 185,960 | with 147,604-triangle STL landmark |
| 800 | 261,843 | 277,373 | `preview_800_v2.log:34`, procedural tower, **no STL** |

At radius 800 *with* the STL landmark the merged count is
261,843 − 1,592 + 147,604 ≈ **407,855**; radius 900 is higher again. So the true
Part 1 workload is roughly 400–500k triangles, not 186k.

**Requirement:** run the scene-preparation step at the final radius, record the
exact tile inventory and merged triangle count in the manifest, and run gates 1,
2 and 5 **at that count**. No gate may be run at 186k and claimed for the final
scene. Triangle count also drives BVH node count, stack depth (§4.3) and scene
memory (§11).

## 4. Engine defect 1 — BVH build and root convention

### 4.1 Build is O(n²)

`sah_cpu.rs:250` `find_best_split` has **no binning**: for each of 3 axes it
sorts all centroids, then for every unique centroid calls `evaluate_split`,
which scans all n primitives. At 186k triangles the root split alone is
3 × 186,000² ≈ **1.04 × 10¹¹** AABB operations; at the real ~450k it is ~6× worse.
Decided by arithmetic, not benchmarking.

**Fix: binned SAH**, 16 bins per axis, O(n) per node, O(n log n) overall.
Rejected: the GPU LBVH builder trips §4.4 and carries its own packing contract;
offline caching leaves the one-off cost unaddressed and helps no other caller.

### 4.2 Root is the last node — and every CPU consumer is already wrong

`sah_cpu.rs:69` does `nodes.push(root_node)` **after** recursion, so the root is
at `node_count - 1`. But every existing consumer assumes index 0:

| Consumer | Assumption |
|---|---|
| `path_tracing/accel.rs:232` | `cpu_stack.push(0) // Start with root node` |
| `sah_cpu.rs:117` | `refit_recursive(..., 0)` |
| `sah_cpu.rs:121` | `world_aabb = nodes[0].aabb` |

This is a **pre-existing bug in the repository**, not merely an omission in this
spec — CPU traversal, refit, and world-AABB reporting all currently walk from a
non-root node.

**Fix: make the builder emit the root at index 0**, restructuring the push order
so the existing convention becomes true. This repairs all three consumers at
once and requires no new plumbing. The alternative — storing an explicit
`root_index` and threading it through every consumer — is rejected as strictly
more work that leaves a trap for the next caller.

**Gates:** CPU traversal regression (rays that must hit a known mesh do hit it);
refit regression; `world_aabb` equals the true scene bounds.

### 4.3 Index permutation, packing, stack depth

- **Permutation.** Reordered primitive indices live in `CpuBvhData.indices`
  (**not** `BvhHandle.tri_indices`, which does not exist). `sdf/hybrid.rs:353`
  uploads `self.indices` raw and unpermuted, and `CpuBvhData.indices` is never
  uploaded. Fix: permute the triangle index stream at upload so leaves index it
  directly, mirroring `path_tracing/mesh/upload.rs:65`.
- **Packing.** Two unrelated `BvhNode` types exist. The render path uses
  `accel::types::BvhNode` (48 B, `{aabb: Aabb, kind, left_idx, right_idx,
  parent_idx}` with `Aabb = {min:[f32;3], _pad0, max:[f32;3], _pad1}`), *not*
  `cpu_bvh::BvhNode` (40 B). Against the WGSL struct:

  | WGSL field | bytes | reads Rust | result |
  |---|---|---|---|
  | `aabb_min` | 0–12 | `aabb.min` | correct |
  | `left` | 12–16 | `aabb._pad0` | **padding** |
  | `aabb_max` | 16–28 | `aabb.max` | correct |
  | `right` | 28–32 | `aabb._pad1` | **padding** |
  | `flags` | 32–36 | `kind` | correct by coincidence |
  | `_pad` | 36–40 | `left_idx` | discarded |

  Stride is already 48 and correct; **field offsets are not**. Fix: an explicit
  `#[repr(C)] Pod` GPU struct matching the WGSL layout, converted at upload,
  mirroring `GpuBvhNode48`. Never reinterpret the Rust type directly.
- **Stack depth.** `MAX_BVH_STACK_SIZE` is 32. A binned-SAH tree over ~450k
  triangles is ~19 levels balanced, but degenerate splits can exceed 32.
  Bound tree depth at build time to 32 (forcing leaves at the limit) and assert
  it; silently dropping nodes is not acceptable.

### 4.4 Tripwire

`upload_mesh_to_gpu` early-returns on `BvhBackend::Gpu` **before creating mesh
buffers**, leaving `mesh_buffers: None`, so the bind group falls back to dummies
and the mesh is silently invisible. `render_terrain.rs` passes
`GpuContext::NotAvailable`, so this is not hit today. Do not change that call
without fixing this.

## 5. Engine defect 2 — traversal

`intersect_mesh` (`hybrid_traversal.wgsl:136`) ignores `mesh_bvh_nodes` and
sweeps every triangle linearly, under a comment claiming BVH traversal.

**Fix:** stack-based traversal over the packed nodes of §4.3, rooted at index 0
per §4.2, mirroring `pt_intersect.wgsl`.

**Gate — equivalence.** BVH and brute-force must return identical hits over a
fixed ray set on a mesh large enough that leaf ordering and stride matter.
Fixture: `tests/fixtures/bvh_equivalence_mesh.npz`, a ~10⁴-triangle mesh
committed alongside a seeded ray set (`numpy.random.default_rng(7)`, 4096 rays
spanning hits, misses and grazing edges). Pass: for every ray, both report the
same hit/miss, and where both hit, `|t_bvh − t_brute| ≤ 1e-4 × scene_diagonal`
and `dot(n_bvh, n_brute) ≥ 0.9999`. `HybridHitResult` exposes no triangle ID, so
identity is not asserted.

## 6. Engine defect 3 — compositable light

`hybrid_terrain_traversal.wgsl:518` applies `reinhard_tonemap(mean_rgb,
cam_exposure)` before `textureStore`, so returned `rgba` is tonemapped uint8.

**But a pre-tonemap target alone is insufficient**, and this was rev 2's error.
`hybrid_terrain_traversal.wgsl:433` computes `albedo = get_surface_properties(hit)`
and shades `albedo * light_color * ndotl`. The accumulated signal is therefore
**radiance, which already contains material albedo**. Multiplying a class colour
into it applies albedo twice.

**Fix — render at unit albedo.** Two changes make the linear target true
irradiance:

1. Pass `albedo = (1.0, 1.0, 1.0)` for terrain (already a parameter).
2. Promote the hardcoded mesh albedo to a parameter (default unchanged at
   `(0.7, 0.7, 0.8)` for existing callers), and pass `(1.0, 1.0, 1.0)`.

Then radiance = irradiance, and `class_colour × irradiance` is well defined.

3. Write pre-tonemap `mean_rgb` to a linear target exposed to Python.

**Miss/background.** On a miss the shader adds `terrain_env_radiance(rd)` — sky,
not surface irradiance. Composite must not multiply a class colour into sky.
Background pixels are identified by the PT `depth` AOV being non-finite and are
owned entirely by the class raster's background class.

**Gate — albedo invariance.** For a fixed scene and camera, rendering at albedo
`A` must equal `A × (render at unit albedo)` within tolerance, for at least two
non-grey values of `A`. This proves the separation is real rather than assumed.

Reinhard here is `exposed/(1+exposed)` with `exposed = color * exposure`.
Forward round-trip (linear → Reinhard → compare against the LDR output) is the
consistency gate. Analytical inversion is **not** used: uint8 quantization makes
the top stop unrecoverable exactly in the lit regions that matter.

# Part 2 — The Paris still

## 7. Architecture

```
OSM surfaces ------> terrain heightfield ---+
IGN LoD2.2 + STL --> merged triangle mesh --+--> PT @ unit albedo --> irradiance
                                                                          |
class-ID + depth raster, same camera --> colour map ----------------------+
                                                                          |
                                                        multiply in linear
                                                          grade, encode
```

PT supplies irradiance; a matched-camera class raster supplies hue. Forced by
the two-flat-albedo constraint of §2.

**Lighting-model caveat.** `hybrid_terrain_traversal.wgsl:454` implements one
directional sun term, one cosine-weighted environment sample, and binary
visibility. That is **direct lighting with environment occlusion** — not
multi-bounce indirect, and not a separate AO model.

## 8. Terrain / mesh ownership

- **Terrain**: flat heightfield, 2x2 grid (the API minimum, sufficient for a
  plane), constant height 0 in local ENU, `exaggeration` 1.0, extent 1800 m
  square (2 × the 900 m radius), `spacing` = (1800.0, 1800.0). Ground reference
  (`ground_reference_m` = 30.369 at r=400; re-recorded at final radius) is
  subtracted identically from mesh and terrain.
- **Mesh**: IGN LoD2.2 buildings, Eiffel STL landmark, tower deck, trees.
- **Ownership rule**: anything that must occlude or cast shadow is mesh;
  everything else is flat and contributes only to the class raster. No geometry
  appears in both. Where a source supplies both, mesh wins.
- Winding, vertex units, and the per-source merge offset are recorded in the
  manifest so mesh and terrain register.

## 9. Class table

Fifteen classes. This table is the single source of truth; §12.2's mask uses
these IDs.

| ID | Class | Geometry | Source | Colour |
|---|---|---|---|---|
| 0 | background | — | sky / no hit | n/a |
| 1 | base | flat | OSM fallback ground | measured (202,196,185) |
| 2 | landuse | flat | OSM landuse | **to measure** |
| 3 | park | flat | OSM leisure=park | **to measure** |
| 4 | water | flat | OSM natural=water | measured (39,56,175) |
| 5 | road | flat | OSM highway | measured (203,191,186) |
| 6 | road_hi | flat | OSM highway, major | **to measure** |
| 7 | footpath | flat | OSM highway=footway/path | **to measure** |
| 8 | building_roof | mesh | IGN LoD2.2 | measured (207,200,198) |
| 9 | building_wall | mesh | IGN LoD2.2 | measured (150,142,179) |
| 10 | tower | mesh | Eiffel STL | measured (181,97,90) |
| 11 | tower_deck | mesh | `build_eiffel_deck_mesh()` | **to measure** |
| 12 | tree | mesh | OSM `natural=tree` + park scatter | **to measure** |
| 13 | pitch | flat | OSM leisure=pitch | measured (130,156,64) |
| 14 | car | mesh | synthetic, seeded | **to measure** |

Three decisions previously left open, now resolved:

- **Tower deck.** `build_eiffel_deck_mesh()` exists (`paris_eiffel_day.py:401`)
  but is never added to the scene (`paris_eiffel_day.py:1359`). **Wire it in.**
  The reference clearly shows the teal second-floor band.
- **Trees.** `combined_osm_query()` (`paris_eiffel_day.py:920`) has **no
  `node["natural"="tree"]` selector**; the current 211 trees come entirely from
  polygon scatter. **Add the selector**, keep scatter as fill for parks with no
  mapped trees. Avenue row-scatter stays deferred until the §12.2 comparison
  shows a density gap.
- **Cars.** OSM carries no vehicle data. Cars are **synthetic props**, placed
  deterministically along road centrelines from a fixed seed recorded in the
  manifest, so renders remain reproducible.

**Gate:** every class 0–14 must be reachable in the raster on the final scene.
An unreachable declared class is a defect, not an acceptable absence.

## 10. Class raster, depth, and camera

### 10.1 Per-pixel raster

`PreparedTriangle` carries `rgba` and `is_wall`, not a class ID
(`osm_city_daycycle.py:56`); colour is not identity, since classes may share a
colour and must grade independently. **Add an explicit class field** carried
from source to raster.

The existing renderer sorts by **mean triangle depth** and draws with Pillow
(`osm_city_daycycle.py:634`), which is not visibility-correct for
interpenetrating or large triangles — the lattice is the worst case. **The class
raster must be a real per-pixel depth-buffered rasterization**, emitting both a
class ID and a per-pixel depth. Painter order is not acceptable.

### 10.2 Depth reconciliation — units

PT `depth` is **ray distance** (`chit.t`, NaN on miss). A camera-space
rasterizer produces **camera-forward z**. These are not comparable. Convert PT
depth to camera-forward z:

```
z_cam = t * dot(normalize(ray_dir), forward)
```

with `forward = normalize(look_at - origin)`. Compare in metres.

- **Agreement tolerance:** 0.5 m, ~1/3600 of the 1800 m scene extent, tight
  enough to catch a wrong surface and loose enough to absorb rasterizer
  sampling differences.
- **Miss handling:** PT depth non-finite ⇒ class 0 (background). Raster hit
  where PT missed, or vice versa, is a **defect to report**, not to blend.
- **Edges:** disagreement is expected on silhouettes. Pixels adjacent to a class
  boundary are excluded from the §12.2 statistics rather than reconciled, and
  the excluded fraction is reported.

### 10.3 Camera

`prepare_scene` applies a **content-dependent fit scale and offsets**
(`osm_city_daycycle.py:372`) then transforms projected points
(`osm_city_daycycle.py:407`), while PT uses raw centered-NDC projection
(`hybrid_terrain_traversal.wgsl:421`). Passing the same eye/target/up/fov does
**not** register them.

**Fix:** compute the fit first, then derive one canonical camera and feed it to
both, with the projector's post-fit transform disabled.

**Gate:** landmark reprojection residual **≤ 2.0 px RMS at 960x720**, measured
via marker-grid DLT. The bound is derived from the mask: above ~2 px, class
boundaries shift by more than the thinnest features the mask must resolve
(footpaths and road edges, 2–3 px at this resolution), corrupting §12.2's
per-class statistics.

## 11. Memory, dimensions, tiling

The renderer registers ~**352 B/pixel** (accumulation 16, Welford 8, three
80-byte ReSTIR reservoirs, G-buffer, output, AOVs) against a **hard 512 MiB
gate** (`render_terrain.rs:827`). The §6 linear target adds one `Rgba16Float`
render target at 8 B/px, giving **360 B/px**.

Measured against the gate:

| Size | Pixels | at 352 B/px | at 360 B/px |
|---|---|---|---|
| 960x720 | 691,200 | 232.0 MiB | 237.3 MiB |
| 1280x960 | 1,228,800 | 412.5 MiB | 421.9 MiB |
| 1408x1056 | 1,486,848 | 499.1 MiB | 510.5 MiB |
| 1424x1068 | 1,520,832 | 510.5 MiB | **522.1 MiB OVER** |
| 1600x1200 | 1,920,000 | 644.5 MiB | **659.2 MiB OVER** |

Scene data also counts against the gate: at ~450k triangles, BVH nodes alone are
~900k × 48 B ≈ 41 MiB, plus vertices and indices, so budget ~90 MiB for scene.

**Decisions:**

- **Single-pass ceiling: 1280x960** (421.9 MiB + ~90 MiB scene ≈ 512 MiB). The
  example's current default of 1600x1200 (`paris_eiffel_day.py:1508`) **cannot
  render** and must be changed.
- **Acceptance renders: 960x720**, matching the reference exactly so comparison
  needs no resampling.
- **Final deliverable: 1920x1440**, produced as **2x2 tiles of 960x720**, each
  well inside the gate. Tiling protocol: each tile uses a sub-frustum of the
  canonical camera derived by splitting NDC into quadrants; no overlap is needed
  because the projection is exact per tile and the camera is shared; tiles run in
  **separate subprocesses** for TDR isolation, as prior work on this card
  required. Seam check: adjacent tile edge columns/rows must agree within the
  §12.2 per-class tolerance, reported as a gate.

## 12. Acceptance

### 12.1 Gate order

| # | Gate | Pass condition |
|---|---|---|
| 1 | Binned SAH build | Build at the §3 final triangle count completes in ≤ 60 s wall-clock, recorded; SAH cost within 5% of exhaustive on a ≤5k-triangle mesh where both are computable |
| 2 | Root convention | CPU traversal, refit, and `world_aabb` regressions pass |
| 3 | BVH equivalence | §5 fixture and formula |
| 4 | Existing regression | `test_mixed_scene_mesh_and_terrain` passes |
| 5 | Albedo invariance | §6 formula, two non-grey albedos |
| 6 | Linear round-trip | Linear → forward Reinhard matches LDR output within 1/255 per channel on 99% of pixels |
| 7 | PT smoke | Paris mesh at §3 count renders; hit fraction ≥ 0.95 of non-sky pixels; recorded |
| 8 | Camera registration | ≤ 2.0 px RMS landmark residual |
| 9 | Class reachability | All 15 IDs present in the raster |
| 10 | Colour match | §12.2 |
| 11 | Seam check | §11 tile-edge agreement |
| 12 | Final render | 1920x1440, manifest complete, output SHA256 recorded |

### 12.2 Colour metric

Deliverable **before tuning begins**, as committed fixtures:

- `tests/fixtures/paris_eiffel_day/reference_day.png` — committed (§0.2).
- `tests/fixtures/paris_eiffel_day/class_mask.png` — hand-authored once,
  assigning every reference pixel one of the §9 IDs or "unassigned".
- `tools/paris_eiffel_day/colour_metric.py` — takes (render, reference, mask),
  emits per-class median RGB, per-class delta, and the global statistics.

**Targets** are masked region medians on the reference, replacing §1's
single-patch estimates.

**Tolerances are derived, not asserted:** for each class, compute the
within-region interquartile range on the reference itself and set that class's
tolerance to `max(6, IQR)`. A class with genuinely variable colour is not held
to a tolerance tighter than its own variance; the floor of 6 prevents a flat
class from demanding impossible precision. Both the derived tolerance and the
achieved delta are reported per class.

**Colour space:** all statistics computed in sRGB 8-bit, matching the reference's
native encoding. The composite's linear→sRGB encode happens before comparison.

**Colour alone cannot pass a wrong image**, so three structural gates run
alongside:

- **Coverage** — background (class 0) fraction ≤ 2% of frame, catching the
  current island-on-white failure, where background is ~50%.
- **Landmark alignment** — gate 8's residual.
- **Shadow presence** — the tower's cast shadow must be measurably present:
  within the Champ de Mars mask region, the darkest decile must be at least 15%
  below the region median. Without this, PT earns nothing over flat shading.

Global mean and median are reported for continuity but are **not sufficient**: a
uniformly wrong image can match a global mean.

## 13. Risks

- Part 1 touches a shader every hybrid-path caller reaches. The §5 equivalence
  gate bounds this, but the blast radius exceeds `paris_eiffel_day.py`.
- The §4.2 root fix changes behaviour for existing CPU consumers that are
  currently walking from a non-root node. They are already wrong; the fix makes
  them right, but any code depending on present behaviour will change.
- Binned SAH changes tree topology for every BVH consumer. Gate 1's quality
  regression bounds this without making it invisible.
- `tests/test_paris_eiffel_day.py:73` asserts `zoom_reference_frame`, which §10
  deletes. Update the test as part of the change.
- Exact camera match with Apple's proprietary view is unlikely; gate 8 measures
  drift rather than assuming it away.

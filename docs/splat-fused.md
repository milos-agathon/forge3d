# SPLAT-FUSED: path-traced splat + LiDAR + terrain fusion

`forge3d.splat.render_fused` renders 3D Gaussian splats, a COPC LiDAR point
cloud and DEM terrain in one ReSTIR path-traced pass. All three representations
live in one acceleration structure and occlude each other through one function,
so a splat-captured tree shadows the terrain and a ridge shadows the splats and
the LiDAR. Nothing is rasterized and there is no second integrator: the fused
kernel is the hybrid ReSTIR terrain kernel with its traversal replaced.

The feature is behind the Cargo feature `splat-fusion` (part of the wheel).

```python
import forge3d as f3d
from forge3d import splat

cloud = f3d.load_gaussian_splats("scene.ply")          # 3DGS .ply
cloud = cloud.transformed(scale, (w, x, y, z), (tx, ty, tz))  # place a capture in the scene frame
image = f3d.render_fused(
    splats=cloud,
    pointcloud="swath.copc.laz",
    terrain=dem,                                          # 2-D float32 heights
    camera=splat.FusedCamera(origin=(6, 52, 46), look_at=(0, 1, 1)),
    samples=64,
)                                                         # (H, W, 4) uint8
```

`return_aovs=True` returns a `FusedRenderResult` with radiance, albedo, normal,
position, depth, the per-pixel transmittance split
`(T_total, T_splat, T_lidar, T_terrain)`, the hit kind, the self-shadow bias
and paging/memory statistics.

LiDAR points enter either from a file (`pointcloud=` a COPC path, a point page
store path or a `FusedPointCloud`) or straight from memory: pass
`point_positions` — an `(N, 3)` float32 array of world-space y-up positions —
optionally with `point_colors`, an `(N, 3)` uint8 sRGB array. In-memory points
are k-d paginated exactly like a file-backed cloud (same residency pool,
traversal and unified transmittance), so no file is written and a point set
derived in the calling script renders with the same pipeline as a streamed
COPC scan.

## Map-sheet example

`examples/splat_fused_map.py` renders all three representations as one lit
"sensor + terrain" map sheet: title, legend with per-layer hit counts,
scene/camera/render info blocks, a scale bar in real metres, a north arrow,
optional AOV insets (hit kind, depth, sun transmittance) and a sidecar
`report.json`.

```bash
python examples/splat_fused_map.py                # offline fixture: ridge + 3,500 splats + COPC swath
python examples/splat_fused_map.py --dem fuji     # real DEM fetched from the web (cached), full extent:
                                                  # hypsometric tint + a splat scan block + a derived
                                                  # LiDAR swath, all digitized from the same DEM
python examples/splat_fused_map.py --dem fuji --view top --aovs   # north-up hypsometric map
```

The camera is a rig, not a hard-coded pose: `--view
hero|canonical|oblique|high|top|low` presets (`hero` — the `--dem` default —
is a low cinematic oblique at az 35 / el 10 / fov 55 with sky above the
summit), `--azimuth/--elevation/--distance/--target/--fov` orbit controls
(or raw `--origin/--look-at`), and the chosen pose is printed and echoed in
the sheet's CAMERA block. `--dem` scenes convert elevation into the same
world units as the grid, so terrain slopes keep their real steepness and
path-traced relief stays crisp; the full DEM extent is used by default and
`--crop-size N` zooms to a central square.

Lighting follows the view: the hero shot is daytime (sun az 125 / el 42,
intensity 3, near-neutral color) with a lit sky. The sky is a real HDRI by
default on `--dem` runs — a CC0 Poly Haven "pure sky" equirect (midday,
partly cloudy, fetched once into the dataset cache) passed through
`splat.render_fused(env_map=...)`, which replaces the Hosek-Wilkie bake,
normalises the HDRI's exposure on load and clamps its sun disc so it reads
as a bright blob instead of an IBL firefly; `--hdr PATH` supplies any
`.hdr/.rgbe/.exr`, `--hdr none` forces the analytic Hosek-Wilkie sky
(turbidity 5, ground albedo 0.35), and `--hdr-rotation DEG` places the
HDRI's sun. Map views keep the cartographic NW light (az 215 / el 45).
Distance is rendered as aerial perspective: the same exponential height fog
that dims sun rays also blends distant surfaces toward the sky colour
(`--fog` density, default 4e-4, and `--fog-falloff`, default 1.5e-3, both
0 for the fixture), so far ridges recede into haze instead of ending at a
hard silhouette. Every other knob is overridable with
`--sun-azimuth/--sun-elevation/--sun-color/--sky-intensity/
--sky-turbidity/--sky-ground-albedo/--exposure`. Reruns with the same seed
reproduce the beauty and the sheet byte for byte.

`--dem` runs digitize all three representations from the DEM itself, so the
sheet shows the full fusion story with no external dataset: the hypsometric
heightfield, the violet splat scan block, and — since no public LiDAR archive
matches the DEM's coordinates — an airborne scan swath derived from the same
grid (`_derive_lidar_swath`): flight lines with realistic pulse and line
spacing plus deterministic jitter, returns sitting on the terrain, colour
driven by local slope as an intensity proxy, and a pulse-sized sphelet radius
so neighbouring returns overlap into a scan strip. The strip underlaps the
splat block, so `T_splat x T_lidar x T_terrain` acts in a shared region. The
SCENE block states the provenance ("derived scan swath") and `--no-lidar`
drops the layer; passing `--pointcloud FILE` replaces it with a real cloud.

Cartography and motion: plan views (`--view top|high`) get an
elevation-contour overlay — marching squares projected through the actual
camera, an interval chosen for the relief, index lines every fifth contour
(`--no-contours` disables it). Annotation is drawn on the sheet's map copy
only; the beauty PNG stays the raw tracer output. `--animate N` additionally
renders an N-frame orbit fly-through GIF (`--anim-size WxH`): every frame is
its own deterministic fused render with the still's lighting, so the GIF
reruns byte-for-byte too. Large stills render in seamless tiles
(`--tile W H`) so e.g. 1280 x 896 at 128 spp fits the 512 MiB budget.

## Options

* `tile=(w, h)`: the frame is rendered in seamless tiles. Every fused render
  is seamless (global-pixel camera rays and seeds, self-only spatial ReSTIR
  reuse), so a tiled render equals the single-tile render bit for bit. The
  default is the whole frame up to one megapixel, otherwise 1024 x 1024 tiles.
  Per-pixel GPU memory is bounded by one tile, so 1080p and 4K fit the
  512 MiB budget. Without `return_aovs` only the beauty is read back and the
  AOV targets are not allocated (48 B per pixel less).
* `FusedTerrain(albedo_map=..., albedo_sampling="bilinear" | "nearest")`:
  a `(rows, cols, 3|4)` uint8 sRGB colour map on the DEM grid, stored as
  RGBA8 sRGB (4 B per cell). With four channels, alpha 255 marks a mapped
  cell and any other alpha falls back to the uniform `albedo`.
* `point_positions=(N, 3) float32, point_colors=(N, 3) uint8`: an in-memory
  LiDAR point set in world space (y-up), an alternative to `pointcloud=` that
  never touches disk. Positions must be finite; colours are sRGB bytes
  (absent = white). The arrays are k-d paginated with the file-backed sources
  and share one residency pool, so their pages appear in the same paging
  stats and behave identically at render time. The only provenance
  difference is the Rust-side source list (`FusedScene::describe()`, which
  labels them `point-array[N points, memory]`); the Python stats dict
  reports paging totals, not per-source labels.
* `terrain_smooth_normals=True`: terrain is shaded with central-difference
  vertex normals interpolated across each cell, continuous across cell
  edges. A normal that would face away from the cell's bilinear patch falls
  back to the patch normal. The intersection is the exact patch either way.
  `False` reproduces the per-cell patch normal.
* `lidar_surfels=True`: a LiDAR return whose page neighbourhood (within three
  sphelet radii, at least six neighbours) is planar becomes an oriented disc
  (surfel) instead of a sphere. Grazing shadow rays then no longer clip
  neighbouring returns of the same surface. Other returns stay sphelets.
  Normals are estimated per page, so a return at a page boundary sees only
  part of its neighbourhood, and the surfel classification can change with
  `page_capacity` (a large COPC node is split into runs of that size).
* `splat_self_bias_sigmas`, `lidar_self_bias_radii`: secondary rays leaving a
  splat (point) skip the stretch in which they rise this many standard
  deviations (sphelet radii) along the hit normal.
* `GaussianSplatCloud.transformed(scale, (w, x, y, z), (tx, ty, tz))`: a
  copy of the cloud under a similarity transform, used to georeference a
  capture.
* `render_fused_sequence(views=[FusedView(camera, sun..., exposure, seed),
  ...])`: renders several views of one scene with one kernel, one residency
  pool and one terrain upload (heights, min-max pyramid and colour map are
  built once; each view only swaps in its sky). Pages stay resident across
  views and are evicted LRU; each result equals `render_fused` of that view.
* Terrain is stored with a conservative half-float min-max pyramid
  (`stats["terrain_minmax_f16"]`), which halves terrain memory without
  changing any output bit. COPC nodes smaller than half a page share pages.

## Occlusion model

For a ray `x(t) = o + t d` and a Gaussian with mean `mu` and covariance
`Sigma` (`Delta = o - mu`):

```
t*  = -(Delta^T Sigma^-1 d) / (d^T Sigma^-1 d)
g*  = Mahalanobis distance^2 at t*
rho = alpha * exp(-g*/2) * (fraction of the 1-D Gaussian inside [tmin, tmax])
T   = exp(-kappa * rho)
```

`src/shaders/splat/gaussian_intersect.wgsl` implements this as a closest-hit
and an any-hit form; `src/splat/kernel.rs` is the CPU mirror the unit tests
check against the closed form. A splat contributes inside its 3-sigma ellipsoid
(`g <= 9`), which is also its BLAS proxy box.

`src/shaders/fusion/unified_occlusion.wgsl` exposes

```
shadow_transmittance(ray) = T_splat * T_lidar * T_terrain
```

* `T_splat` is the product of the per-splat transmittances above.
* `T_lidar` is the product of `1 - c` over LiDAR returns with coverage
  `c = opacity * (1 - b^2 / r^2)`. For a sphelet, `b` is the ray's distance
  to the point; for a surfel (`lidar_surfels`), `b` is the distance from the
  disc centre at which the ray crosses the disc plane.
* `T_terrain` is the exact height-field any-hit (0 or 1).

Primary and bounce rays accept a splat or sphelet stochastically with
probability `1 - T`; G-buffer rays use the deterministic threshold `1/2`.
Surfaces are shaded through `eval_brdf`, lit by the Hosek-Wilkie sky
(`SkyParams`), and attenuated by height fog laid out as `VolumetricParams`.

The ReSTIR target function is `delta + (1 - delta) * shadow_transmittance`
over directions in the solar disc, so reservoirs are importance-resampled by
the fused visibility.

## Out-of-core paging

Splats and LiDAR points are stored in pages (4096 primitives by default). The
TLAS holds one leaf per page, per terrain tile, and nothing else; page payloads
are resident only in a fixed-size GPU pool. A ray that reaches a non-resident
page records a request in the page table and continues; the host loads the
requested pages asynchronously and evicts least-recently-used pages.

* `policy="exact"` (default): a frame that missed a page is discarded and
  accumulation restarts once the pages are resident, so the image never
  contains a missing occluder. If one view needs more pages than the pool
  holds, the render fails with a budget error instead of dropping occluders.
* `policy="progressive"`: frames are kept while pages stream in. A pixel whose
  shadow ray missed a page reuses its reservoir's last known visibility.
  `stats["stale_frames"]` reports how many frames were affected. Pending page
  decodes reserve pool slots; requests beyond the unpinned capacity are
  deferred in traversal-priority order and counted in `paging.deferred_pages`,
  even before background decoding finishes.

Every buffer is created through the memory tracker. The render peak is bounded
by the 512 MiB budget and reported in `stats` (`peak_total_bytes`,
`peak_host_visible_bytes`, `paging.pool_bytes`).

`render_fused` and `render_fused_reference` follow the render-entrypoint
contract: `certificate=True` (or a path) emits the signed CENSOR certificate
with the timed `hybrid_pt.fused*` / `hybrid_pt.restir_*` passes, the shader
module hashes and the occlusion model; `cache=` is accepted and ignored,
because the fused scene is streamed per view and has no render-graph cache.

`splat.build_splat_page_store` converts a `.ply` of any size into a page store
with bounded memory; `splat.write_synthetic_point_field` writes the
billion-point index used by the acceptance test.

## Acceptance

`tests/test_splat_fusion_occlusion.rs` and
`tests/test_splat_fusion_occlusion.py` render the committed fixture in
`tests/fixtures/splat_fusion/` (65 x 65 DEM with a ridge, 3,500 splats, 5,500
COPC points) embedded in a 1.0 x 10^9-point page index, and compare the fused
shadow mask with a reference path trace:

* shadow IoU > 0.9 for splat shadow on terrain, LiDAR shadow on terrain,
  terrain shadow on splats, terrain shadow on LiDAR;
* logical primitives > 10^9 with the tracked peak under 512 MiB;
* the GPU transmittance agrees with a brute-force CPU evaluation of the same
  model on every shadow ray;
* a golden image (`tests/golden/splat_fusion/fused_fixture.png`).

The reference is the AEQUITAS wavefront path tracer
(`forge3d.splat.render_fused_reference`). It traces triangles, so each splat is
replaced by the ellipsoid on which the fused model's transmittance is 1/2, and
each LiDAR sphelet by an icosahedron of the radius at which its coverage is
1/2. The two renders therefore agree on where a shadow is, not on the soft
falloff at its edge: many overlapping faint splats can add up to a shadow in
the fused model where no single hard proxy exists.

The GPU assertions need a physical adapter with at least 13 storage buffers per
shader stage. They run on the NVIDIA Vulkan lane
(`FORGE3D_SPLAT_FUSION_REQUIRED_GPU=1` forbids skipping) and skip on software
and virtualized adapters.

## Measured

Local NVIDIA RTX 3070, Vulkan (`docs/superpowers/plans/2026-10-03-splat-fused-limits-evidence.md`):

| | |
|---|---|
| 1920x1080 fixture render, default tiles | 4 tiles, peak 471.0 MiB |
| 1080p and 4K, 960x540 tiles | peak equal to the single 960x540 render (272.5 MiB; 248.8 MiB without AOVs) |
| Mount St. Helens, 1920x1080, 64 spp | 9.7 s, peak 473.3 MiB |
| Lauterbrunnen, 1920x1080, 64 spp, 6 m DEM + colour map | 39.6 s, peak 467.1 MiB, terrain 67.0 MiB |
| Min-max pyramid, 1250 x 2500 DEM | 89.5 MB -> 44.7 MB, output bit-identical |
| Lauterbrunnen COPC at 3.3 m, capacity 2700 | 12,959 -> 1,228 pages |
| Smooth terrain normals vs analytic surface | mean error 0.48 deg (patch normal 1.42 deg) |
| Dense LiDAR plane, sun 5-40 deg | mean T_lidar 1.000 with surfels (0.78 as unbiased spheres at 5 deg) |
| Mount St. Helens crater LiDAR (ground returns), default settings, sun 22 deg | T_lidar 0.705-0.788 -> 0.753-0.822 with surfels, LiDAR radiance +8 to +25 % |

## Limits

* Splat colour on the GPU uses spherical-harmonic bands 0 and 1. Higher bands
  are loaded, stored and evaluated on the CPU (`GaussianSplatCloud.color`).
* The terrain min-max pyramid is padded to powers of two per axis (the shared
  traversal assumes power-of-two node coordinates).
* Heightfields cannot represent overhangs: cliffs stay 2.5D even with smooth
  shading normals.
* Splats are treated as emission-free scattering surfaces lit by the scene sun
  and sky; capture lighting baked into the SH colour is used as albedo.

The fused path needs `max_storage_buffers_per_shader_stage >= 13`; on a
smaller adapter `render_fused` raises a capability diagnostic.

**LiDAR at low sun.** Surfels fix the self-shadowing of ground and roof
returns. On the Mount St. Helens crater (ground returns, default
self-shadow bias, sun 22 degrees) mean T_lidar rises from 0.705-0.788 to
0.753-0.822 and LiDAR radiance by 8-25 % across the cameras and sun
azimuths measured. Tree canopy stays dark: on the Lauterbrunnen cloud
(trees and buildings) mean T_lidar is 0.533 without surfels and 0.532 with
them, because the canopy is a volume whose returns occlude each other,
which is largely physical shadowing rather than a model artefact.

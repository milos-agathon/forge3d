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
`(T_total, T_splat, T_lidar, T_terrain)`, the hit kind and paging/memory
statistics.

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
* `T_lidar` is the product of `1 - c` over LiDAR sphelets with coverage
  `c = opacity * (1 - b^2 / r^2)` at impact parameter `b`.
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
  `stats["stale_frames"]` reports how many frames were affected.

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

## Limits

* The fused path needs `max_storage_buffers_per_shader_stage >= 13`; on a
  smaller adapter `render_fused` raises a capability diagnostic.
* Splat colour on the GPU uses spherical-harmonic bands 0 and 1. Higher bands
  are loaded, stored and evaluated on the CPU (`GaussianSplatCloud.color`).
* Splats are treated as emission-free scattering surfaces lit by the scene sun
  and sky; baked-in capture lighting in the SH colour is used as albedo.
* LRU eviction only happens across views (`render_fused_sequence` in Rust): in
  a single static view every resident page is in use.

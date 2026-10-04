# Participating media

`forge3d.media.Medium` is the canonical validated participating-medium object.
It carries absorption, scattering, derived extinction, phase, density payload,
explicit version, and a deterministic full identity shared by resource caches
and temporal history.

```python
from forge3d.media import Medium, render_volumetric_reference
import numpy as np

medium = Medium.homogeneous(
    sigma_a=(0.01, 0.02, 0.03),
    sigma_s=(0.10, 0.12, 0.14),
    density=0.8,
    version=1,
)

heightmap = np.zeros((64, 64), dtype=np.float32)
camera = {
    "fov_y": 45.0,
    "terrain_camera": {
        "target": (0.0, 15.0, 0.0),
        "radius": 50.990195,
        "phi_deg": 90.0,
        "theta_deg": 78.690068,
        "mode": "mesh:yup",
    },
}
result = render_volumetric_reference(
    medium,
    heightmap,
    width=128,
    height=96,
    camera=camera,
    samples_per_pixel=16,
    homogeneous_medium_reach=250.0,
    seed=7,
    certificate=True,
)
beauty = result["beauty"]
```

The reference renderer combines CPU transport with production GPU terrain
queries and follows the repository certificate contract. Set `certificate=True`
to retain the execution certificate in the diagnostics surface, or pass a path
to write it. The
certificate records the negotiated adapter and capabilities, tracked
allocations, the executed terrain-trace WGSL, the media pass, and any
degradations.

Attach the same object to an offscreen configuration with the public terrain
configuration helpers:

```python
from forge3d import TerrainRenderParams
from forge3d.media import Medium
from forge3d.terrain_params import AovSettings, make_terrain_params_config

medium = Medium.homogeneous(
    sigma_a=(0.01, 0.02, 0.03),
    sigma_s=(0.10, 0.12, 0.14),
    density=0.8,
    version=1,
)
config = make_terrain_params_config(
    size_px=(128, 96),
    render_scale=1.0,
    terrain_span=64.0,
    msaa_samples=1,
    z_scale=1.0,
    exposure=1.0,
    domain=(0.0, 1.0),
    media=medium,
    aov=AovSettings(
        enabled=True,
        albedo=False,
        normal=False,
        depth=False,
        transmittance=True,
        in_scatter=True,
        cloud_shadow=True,
        optical_depth=True,
    ),
)
params = TerrainRenderParams(config)
```

## Physical terrain surface

`TerrainRenderParams.terrain_shading_model` and
`make_terrain_params_config(terrain_shading_model=...)` accept `"stylized"`
(the default) or `"lambert_physical"`. The default preserves the established
terrain look. The physical model is available with or without an attached
medium and evaluates the pre-exposure terrain radiance as

\[
L_o = \frac{\rho}{\pi}(L_{sun}\max(n\mathbin{\cdot}l,0)T_{medium}V_{terrain})
    + \frac{\rho}{\pi}E_{env}.
\]

Here `rho` is the resolved terrain albedo, `n` is the analytic geometric
normal of the rendered bilinear DEM cell, and `L_sun` is the uploaded sun
color multiplied by intensity. `V_terrain` is the raw product of cascade and
heightfield sun visibility; the physical path adds no shadow floor or clamp.
For untextured `MaterialSet.custom` layers, `base_color` is linear reflectance.
A single constant physical material retains its authored floating-point value
through a shading uniform, avoiding sRGB8 quantization; general untextured
layers use the physical material cache's sRGB encoding. File-backed albedo
texture bytes retain their normal sRGB interpretation. The default stylized
material cache is unchanged.
`T_medium` affects direct sunlight only and is one when no medium is attached.
The terrain IBL cube stores normalized irradiance `E_env / pi`: its
cosine-weighted convolution returns the mean under a cosine-weighted sampling
distribution. The physical path converts that sample back to `E_env`, applies
the configured IBL intensity, and divides by pi exactly once in the Lambertian
equation. Thus constant radiance 0.25 corresponds to irradiance pi/4 in every
channel, while its normalized cube value is 0.25.

The physical path does not add the stylized ambient floor, edge enhancement,
ambient occlusion, hue rotation, subsurface term, or terrain specular term.
It also does not estimate terrain-to-terrain interreflection. Reference
comparisons must report the remaining positive indirect-energy residual rather
than adding an unmeasured bounce term. Compare unoccluded surface regions to
separate that residual from geometric-normal or PCF visibility differences,
which concentrate at slopes, silhouettes, and shadow boundaries.

When real-time media diagnostics are captured, `terrain_shading_model` records
the active value so physical-reference evidence cannot silently use the
default stylized path.

Pass ``params`` to ``TerrainRenderer.render_with_aov`` together with the
material set, environment maps, and heightmap used by your existing terrain
rendering setup. Assign ``None`` to ``config.media`` before constructing new
render parameters to remove the medium. For an interactive terrain already
loaded through ``open_viewer_async``, call ``viewer.set_media(medium)`` to
attach it and ``viewer.set_media(None)`` to remove it.

A homogeneous medium is unbounded unless you give it a reach, and an
unbounded medium blocks all direct sunlight. Pass
``viewer.set_media(medium, homogeneous_reach=distance)`` to fill only that
distance along every camera and sun ray, matching
``render_volumetric_reference(..., homogeneous_medium_reach=distance)``.
Grid and noise media are bounded by their own ``bounds`` and reject a reach.

Each froxel's single-scatter estimate uses one stochastic continuation
direction. Before multiple scattering and integration, the real-time path
averages it over a 3x3 tent of neighbouring froxel columns in the same depth
slice, using only froxels that contain medium.

The reference result contains `beauty`, `transmittance`, `in_scatter`,
`cloud_shadow`, and `optical_depth` arrays with shape `(height, width, 3)`, plus
the additive `terrain_slice` acceptance array with shape `(height, width)`.
`terrain_slice` maps the production terrain-trace hit distance into the same
64-slice logarithmic froxel coordinate used by real-time termination capture;
the mapping uses the public `clip=(near, far)` argument.
`result["diagnostics"]` uses the same schema as offscreen AOV capture and viewer
statistics: majorant proof and validity; sample and step counts; temporal
history decision and reason; separate host, froxel, density, majorant, and
staging byte counts; adapter, backend, driver, and source revision; and whether
multiple-scattering transport executed. Invalid spectra, density, phase,
majorant, camera, or transport input raises `forge3d.media.MediaError`; no
medium is silently clamped or disabled.

The real-time froxel path stores sampled density and extinction as f16. A
positive sample below the smallest f16 subnormal (2^-24, about 6e-8), which
occurs where an interpolated density field fades to zero, is stored as zero
and counted: real-time diagnostics report `f16_flushed_sample_count` and
`f16_flushed_max_value`. Values too large for f16 still raise.

The returned AOV frame exposes the same `has_*`, readback, and `save_*` methods
for all four media AOVs, plus `media_diagnostics`.
Requesting one without `TerrainRenderParams.media` fails instead of returning a
placeholder texture.

`TerrainRenderParams.media = None` preserves the existing non-media renderer.
While a canonical medium is attached to the interactive viewer, its independent
legacy height-fog volumetric pass is bypassed. Its froxel depth range spans the
nearest to farthest point of the terrain box joined with the medium bounds
(the terrain alone for homogeneous media), so thin layers keep depth resolution. Analytic sky configuration stays
independent and remains valid. `viewer.get_stats()` reports
`media_diagnostics` after a successful media frame and `media_render_error`
after a failed media frame. Passing `None` to `viewer.set_media` removes the
canonical pass and restores the legacy behavior.

## Owner decisions — 2026-09-30

Milos approved the CPU reference kernel in `crate::media::reference` with GPU
participation limited to production `terrain_trace` visibility (OD-8 option a).
This is an explicit exemption from reference WGSL tracking; it does not waive
any G1–G6 estimator, visual, memory, determinism, or physical-lane requirement.

The terrain reference uses shared RGB paths with spectral real/null-event
likelihood weights and environment/surface MIS. Real collisions are proposed
from mean RGB extinction; real weights are sigma_t,c / mean(sigma_t), and
null weights are (majorant - sigma_t,c) / (majorant - mean(sigma_t)). This
preserves each channel's expectation while reducing hero-channel color noise.
The existing single-channel Rust reference API and production
`delta_track_counted` sampler remain available and keep their contracts.

Gate 4 uses `per_pixel_pass_fraction`: at least 95% of reference-cloud-shadowed
terrain pixels must have DeltaE2000 strictly below 2.0. The owner record is
`tests/nephele/gate4-policy.json`; the same criterion governs convergence.
Gate 2 uses three independent ensembles of 10,000,000 paths each, routed through
production `delta_track_counted` and `Phase::sample` (OD-7 a + 7b). Its uncertainty
and sample-count derivation are reported alongside the measured energy terms.
For independent outcome estimates with variance sum V, the residual standard
error is sqrt(V/N). With the fixed threshold tau = 1e-3, requiring k standard
errors below tau gives N >= k^2 V / tau^2. The threshold alone does not select
a unique N without a confidence choice: the owner selected N = 10,000,000, and
the report gives the resulting measured k = tau sqrt(N/V) and approximate
false-failure probability. These are uncertainty reports, not new gates.
The registry `wgpu-hal` is used without a vendored patch (OD-3 a).

Physical evidence retains reference wheels under
`reference-wheel/<wheel-sha256>/<original-filename>`. Candidate wheels remain
at the archive root, so two builds of the same package version cannot overwrite
each other. Both retain their original filenames and verified native bytes.

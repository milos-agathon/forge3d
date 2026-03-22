# TV5: Terrain Local Probe Lighting

Epic TV5 from `docs/plans/2026-03-16-terrain-viz-epics.md` is implemented on `main` as a terrain-local diffuse + reflection probe workflow. TV5.1 and TV5.2 ship through diffuse irradiance probes, and TV5.3 adds local reflection probes that reuse the same placement discipline for terrain-local specular.

## What shipped

- Public terrain control surface: `ProbeSettings` in `python/forge3d/terrain_params.py`
- Heightfield analytical bake path: `HeightfieldAnalyticalBaker` in `src/terrain/probes/heightfield_baker.rs`
- Heightfield analytical reflection bake path: `HeightfieldReflectionBaker` in `src/terrain/probes/reflection_baker.rs`
- Probe placement and rebake path: `src/terrain/renderer/probes.rs`
- GPU packing and upload path: `src/terrain/probes/gpu.rs`
- Terrain shader integration: `src/shaders/terrain_probes.wgsl` included by `src/shaders/terrain_pbr_pom.wgsl`
- Probe memory reporting: `TerrainRenderer.get_probe_memory_report()`
- Example/demo coverage: `examples/terrain_tv5_probe_lighting_demo.py`
- Regression coverage: `tests/test_terrain_probes.py`

## Public API

```python
import forge3d as f3d
from forge3d.terrain_params import (
    PomSettings,
    ProbeSettings,
    ReflectionProbeSettings,
    make_terrain_params_config,
)

config = make_terrain_params_config(
    size_px=(1280, 720),
    terrain_span=2400.0,
    z_scale=1.6,
    cam_radius=3600.0,
    cam_phi_deg=138.0,
    cam_theta_deg=58.0,
    ibl_enabled=True,
    ibl_intensity=3.0,
    probes=ProbeSettings(
        enabled=True,
        grid_dims=(6, 6),
        height_offset=5.0,
        ray_count=48,
        sky_color=(0.6, 0.75, 1.0),
        sky_intensity=1.0,
    ),
    reflection_probes=ReflectionProbeSettings(
        enabled=True,
        grid_dims=(4, 4),
        height_offset=5.0,
        ray_count=16,
        ground_color=(0.22, 0.18, 0.14),
        strength=1.0,
    ),
    pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
)

params = f3d.TerrainRenderParams(config)
frame = renderer.render_terrain_pbr_pom(
    material_set=material_set,
    env_maps=ibl,
    params=params,
    heightmap=heightmap,
)
memory = renderer.get_probe_memory_report()
```

## Workflow notes

- `ProbeSettings(enabled=False)` and `probes=None` preserve the pre-TV5 IBL-only baseline.
- `ReflectionProbeSettings(enabled=False)` and `reflection_probes=None` preserve the pre-TV5 specular baseline.
- Default probe config is disabled, with `grid_dims=(8, 8)`, `height_offset=5.0`, `ray_count=64`, `sky_color=(0.6, 0.75, 1.0)`, and `sky_intensity=1.0`.
- Default reflection probe config is disabled, with `grid_dims=(4, 4)`, `height_offset=5.0`, `ray_count=16`, `ground_color=(0.22, 0.18, 0.14)`, and `strength=1.0`.
- When `origin` and `spacing` are omitted, placement auto-spans the terrain bounds. A `1x1` grid collapses to a centered single probe.
- Enabled probes validate `grid_dims >= (1, 1)`, total probe count `<= 4096`, `ray_count >= 1`, positive explicit spacing, non-negative blend distance, and non-negative sky intensity.
- Enabled reflection probes validate the same placement rules with a tighter probe-count limit of `<= 256` and an explicit `strength` range of `[0, 1]`.
- Probe positions are resolved from the terrain heightfield with bilinear height sampling plus `height_offset`.
- Probe payloads are baked as SH L2 coefficients, packed into `GpuProbeData`, uploaded once per cache key, and sampled with bilinear grid interpolation in the terrain shader.
- Reflection payloads are baked as directional local-environment face colors, packed into `GpuReflectionProbeData`, uploaded once per cache key, and sampled in the terrain shader to replace the local specular term where probes are present.
- The diffuse probe term is blended against the existing IBL diffuse term. Outside probe coverage, the shader falls back smoothly to the old IBL behavior.
- The reflection probe term is blended against the existing global specular IBL term. Outside probe coverage, terrain specular falls back smoothly to the old IBL behavior.
- `fallback_blend_distance` defaults to `2 * min(spacing_x, spacing_y)` when omitted.
- Rebake invalidation keys include terrain span, `z_scale`, full heightfield contents and dimensions, `grid_dims`, `origin`, `spacing`, `height_offset`, `ray_count`, `sky_color`, and `sky_intensity`.
- `renderer.get_probe_memory_report()` returns diffuse and reflection probe counts, byte totals per subsystem, and a combined `total_bytes`.
- Debug mode `50` shows raw probe irradiance. Debug mode `51` shows probe blend weight.
- Debug mode `52` shows raw reflection probe color. Debug mode `53` shows reflection probe blend weight.

## Current boundary

- TV5 now changes both diffuse indirect terrain lighting and the local specular fallback path.
- Terrain specular still falls back to the global IBL cubemap outside reflection probe coverage.
- The current baker is heightfield-analytical and terrain-only; it does not perform scene cubemap capture.

## Example and tests

- Example: `python examples/terrain_tv5_probe_lighting_demo.py`
- API and behavior tests: `tests/test_terrain_probes.py`
- Detailed design/spec: `docs/superpowers/specs/2026-03-21-tv5-local-probe-lighting-design.md`

The bundled demo writes six outputs:

- probes disabled
- probes enabled
- probe irradiance debug
- probe weight debug
- reflection probe color debug
- side-by-side comparison

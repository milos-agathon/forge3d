# TV5: Terrain Local Probe Lighting

Epic TV5 from `docs/plans/2026-03-16-terrain-viz-epics.md` is partially implemented on `main` as a terrain-local diffuse GI workflow. TV5.1 and TV5.2 are shipped through diffuse irradiance probes, while TV5.3 reflection probes are still future work.

## What shipped

- Public terrain control surface: `ProbeSettings` in `python/forge3d/terrain_params.py`
- Heightfield analytical bake path: `HeightfieldAnalyticalBaker` in `src/terrain/probes/heightfield_baker.rs`
- Probe placement and rebake path: `src/terrain/renderer/probes.rs`
- GPU packing and upload path: `src/terrain/probes/gpu.rs`
- Terrain shader integration: `src/shaders/terrain_probes.wgsl` included by `src/shaders/terrain_pbr_pom.wgsl`
- Probe memory reporting: `TerrainRenderer.get_probe_memory_report()`
- Example/demo coverage: `examples/terrain_tv5_probe_lighting_demo.py`
- Regression coverage: `tests/test_terrain_probes.py`

## Public API

```python
import forge3d as f3d
from forge3d.terrain_params import PomSettings, ProbeSettings, make_terrain_params_config

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
- Default probe config is disabled, with `grid_dims=(8, 8)`, `height_offset=5.0`, `ray_count=64`, `sky_color=(0.6, 0.75, 1.0)`, and `sky_intensity=1.0`.
- When `origin` and `spacing` are omitted, placement auto-spans the terrain bounds. A `1x1` grid collapses to a centered single probe.
- Enabled probes validate `grid_dims >= (1, 1)`, total probe count `<= 4096`, `ray_count >= 1`, positive explicit spacing, non-negative blend distance, and non-negative sky intensity.
- Probe positions are resolved from the terrain heightfield with bilinear height sampling plus `height_offset`.
- Probe payloads are baked as SH L2 coefficients, packed into `GpuProbeData`, uploaded once per cache key, and sampled with bilinear grid interpolation in the terrain shader.
- The diffuse probe term is blended against the existing IBL diffuse term. Outside probe coverage, the shader falls back smoothly to the old IBL behavior.
- `fallback_blend_distance` defaults to `2 * min(spacing_x, spacing_y)` when omitted.
- Rebake invalidation keys include terrain span, `z_scale`, full heightfield contents and dimensions, `grid_dims`, `origin`, `spacing`, `height_offset`, `ray_count`, `sky_color`, and `sky_intensity`.
- `renderer.get_probe_memory_report()` returns `probe_count`, `grid_uniform_bytes`, `probe_ssbo_bytes`, and `total_bytes`.
- Debug mode `50` shows raw probe irradiance. Debug mode `51` shows probe blend weight.

## Current boundary

- TV5 currently changes diffuse indirect terrain lighting only.
- Terrain specular still comes from the global IBL cubemap.
- No local reflection probes, cubemap capture path, or per-probe specular prefiltering ship in the current implementation.
- The current baker is heightfield-analytical and terrain-only; TV5.3 remains the planned reflection-probe upgrade path.

## Example and tests

- Example: `python examples/terrain_tv5_probe_lighting_demo.py`
- API and behavior tests: `tests/test_terrain_probes.py`
- Detailed design/spec: `docs/superpowers/specs/2026-03-21-tv5-local-probe-lighting-design.md`

The bundled demo writes five outputs:

- probes disabled
- probes enabled
- probe irradiance debug
- probe weight debug
- side-by-side comparison

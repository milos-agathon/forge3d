# TV10: Terrain Subsurface Materials

Epic TV10 from `docs/plans/2026-03-16-terrain-viz-epics.md` is implemented as a terrain-specific subsurface response for snow, ice, wet soil, and similar broad natural layers. The shipped path is a bounded in-shader approximation built around per-layer scatter strength/tint controls, wrap lighting, and a curvature-weighted diffusion estimate. It is intentionally terrain-first, not a full screen-space diffusion pipeline.

Shipped in `forge3d 1.17.0`.

## What shipped

- Public terrain controls:
  - `MaterialLayerSettings.snow_subsurface_strength`
  - `MaterialLayerSettings.snow_subsurface_color`
  - `MaterialLayerSettings.rock_subsurface_strength`
  - `MaterialLayerSettings.rock_subsurface_color`
  - `MaterialLayerSettings.wetness_subsurface_strength`
  - `MaterialLayerSettings.wetness_subsurface_color`
- Native terrain config plumbing:
  - `src/terrain/render_params/decode_materials.rs`
  - `src/terrain/render_params/native_material.rs`
- GPU uniform wiring:
  - `src/terrain/renderer/uniforms.rs`
  - `src/terrain/renderer/bind_groups/terrain_pass.rs`
- Terrain shader integration:
  - `src/shaders/terrain_pbr_pom.wgsl`
  - per-layer weight evaluation now feeds a terrain subsurface state
  - the beauty path adds a bounded subsurface contribution based on wrap lighting, view/light alignment, and derivative-based diffusion
- Real-DEM example coverage:
  - `examples/terrain_tv10_subsurface_demo.py`
  - renders Mount Rainier and Gore Range from repo DEM assets
- Regression coverage:
  - `tests/test_terrain_materials.py`
  - `tests/test_terrain_tv10_subsurface.py`
  - `tests/test_terrain_tv10_goldens.py`
  - `tests/test_terrain_tv10_demo.py`

## Public API

```python
import forge3d as f3d
from forge3d.terrain_params import MaterialLayerSettings, make_terrain_params_config

config = make_terrain_params_config(
    size_px=(1280, 720),
    terrain_span=2400.0,
    z_scale=1.4,
    cam_radius=3200.0,
    cam_phi_deg=144.0,
    cam_theta_deg=48.0,
    materials=MaterialLayerSettings(
        snow_enabled=True,
        snow_altitude_min=1800.0,
        snow_altitude_blend=650.0,
        snow_slope_max=58.0,
        snow_slope_blend=18.0,
        rock_enabled=True,
        rock_slope_min=36.0,
        rock_slope_blend=10.0,
        wetness_enabled=True,
        wetness_strength=0.18,
        wetness_slope_influence=0.45,
        snow_subsurface_strength=0.58,
        snow_subsurface_color=(0.72, 0.85, 0.98),
        rock_subsurface_strength=0.04,
        rock_subsurface_color=(0.45, 0.38, 0.30),
        wetness_subsurface_strength=0.16,
        wetness_subsurface_color=(0.38, 0.27, 0.18),
    ),
)

params = f3d.TerrainRenderParams(config)
```

## Workflow notes

- Defaults stay backward-compatible at the scene level because all terrain material layers remain disabled by default.
- Snow defaults now carry a non-zero terrain subsurface response:
  - `snow_subsurface_strength = 0.35`
  - `snow_subsurface_color = (0.78, 0.88, 0.98)`
- Rock defaults stay neutral:
  - `rock_subsurface_strength = 0.0`
- Wetness defaults carry a mild warm fill:
  - `wetness_subsurface_strength = 0.12`
  - `wetness_subsurface_color = (0.40, 0.28, 0.18)`
- Setting all `*_subsurface_strength` values to `0.0` restores the pre-TV10 terrain response.
- `*_subsurface_color` is a scatter tint, not an albedo override.
- The terrain SSS path is additive and bounded. It does not add a new full-screen post-process or a generic character-material pipeline.
- The approximation is most visible on low-angle lighting over snow/ice or damp terrain where the layer weights already dominate.

## Example output

`examples/terrain_tv10_subsurface_demo.py` writes:

- `mount_rainier_baseline.png`
- `mount_rainier_subsurface.png`
- `mount_rainier_comparison.png`
- `gore_range_baseline.png`
- `gore_range_subsurface.png`
- `gore_range_comparison.png`
- `terrain_tv10_summary.png`

The bundled demo uses two real repo DEM assets so the effect is validated on representative terrain rather than only synthetic fixtures.

## Regression contract

- Explicit zero-strength SSS matches the pre-TV10 baseline image contract.
- Two dedicated SSS scenes are locked with goldens:
  - `tests/golden/terrain/terrain_tv10_scene_a_sss.png`
  - `tests/golden/terrain/terrain_tv10_scene_b_sss.png`
- The zero-strength baseline is locked separately:
  - `tests/golden/terrain/terrain_tv10_zero_sss.png`

## Example and tests

- Example: `examples/terrain_tv10_subsurface_demo.py`
- Config/API tests: `tests/test_terrain_materials.py`
- Runtime render tests: `tests/test_terrain_tv10_subsurface.py`
- Golden-image tests: `tests/test_terrain_tv10_goldens.py`
- Real-DEM example test: `tests/test_terrain_tv10_demo.py`

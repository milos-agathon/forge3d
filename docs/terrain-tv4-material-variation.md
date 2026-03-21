# TV4: Terrain Material Variation Upgrade

Epic TV4 from `docs/plans/2026-03-16-terrain-viz-epics.md` is implemented as a bounded terrain-material variation workflow built on a shared WGSL noise module, explicit Python controls, and renderer/example regression coverage.

## What shipped

- Shared terrain noise module: `src/shaders/terrain_noise.wgsl`
- Terrain shader refactor:
  - `terrain_pbr_pom.wgsl` now includes the shared noise unit instead of inlining duplicate value-noise helpers.
  - Existing procedural detail normals and albedo variation reuse the shared `terrain_value_noise(...)` path.
- Higher-quality bounded variation set:
  - FBM for broad snow breakup
  - ridged FBM for rock exposure breakup
  - cellular-distance variation for wetness breakup
- Public terrain controls:
  - `MaterialNoiseSettings`
  - `MaterialLayerSettings.variation`
- Zero-regression defaults:
  - all per-layer amplitudes default to `0.0`
  - setting amplitudes to zero preserves the pre-TV4 material output

## Public API

```python
import forge3d as f3d
from forge3d.terrain_params import (
    MaterialLayerSettings,
    MaterialNoiseSettings,
    make_terrain_params_config,
)

config = make_terrain_params_config(
    size_px=(1280, 720),
    terrain_span=2400.0,
    z_scale=1.25,
    cam_radius=3200.0,
    cam_phi_deg=144.0,
    cam_theta_deg=52.0,
    materials=MaterialLayerSettings(
        snow_enabled=True,
        snow_altitude_min=1800.0,
        snow_altitude_blend=700.0,
        rock_enabled=True,
        rock_slope_min=28.0,
        rock_slope_blend=18.0,
        wetness_enabled=True,
        wetness_strength=0.45,
        wetness_slope_influence=0.85,
        variation=MaterialNoiseSettings(
            macro_scale=4.0,
            detail_scale=20.0,
            octaves=5,
            snow_macro_amplitude=0.22,
            snow_detail_amplitude=0.10,
            rock_macro_amplitude=0.18,
            rock_detail_amplitude=0.14,
            wetness_macro_amplitude=0.16,
            wetness_detail_amplitude=0.08,
        ),
    ),
)

params = f3d.TerrainRenderParams(config)
```

## Output and perf contract

- `macro_scale` and `detail_scale` must be `> 0`
- `octaves` is intentionally bounded to `[1, 8]`
- amplitudes are bounded to `[0, 1]`
- zero amplitudes preserve the previous terrain-material baseline
- the richer path is covered by a bounded render-time budget test

## Example and tests

- Example: `python examples/terrain_tv4_material_variation_demo.py`
- Material config tests: `tests/test_terrain_materials.py`
- TV4 regression tests: `tests/test_terrain_tv4_material_variation.py`
- TV4 demo test: `tests/test_terrain_tv4_demo.py`
- CI example lane: `.github/workflows/ci.yml`

Use the bundled demo when you want a side-by-side comparison between:

- baseline terrain material layering
- TV4 terrain material variation with explicit snow, rock, and wetness breakup controls

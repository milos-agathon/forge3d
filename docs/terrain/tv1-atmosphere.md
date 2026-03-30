# TV1: Terrain Atmosphere Path Parity

Epic TV1 from `docs/plans/2026-03-16-terrain-viz-epics.md` is implemented and exposed through the terrain renderer, terrain params surface, and a bundled example/test set.

## What shipped

- `TerrainRenderParams.sky` now drives the terrain render path instead of acting as dead plumbing.
- The terrain path honors `SkySettings.enabled`, `turbidity`, `ground_albedo`, `sun_intensity`, `sun_size`, `aerial_density`, and `sky_exposure`.
- Terrain haze semantics are split cleanly:
  - `FogSettings` owns height fog density, falloff, base height, and inscatter tint.
  - `SkySettings` owns aerial perspective and atmosphere tinting for the terrain path.
- The regression surface includes clear-sky, hazy, and low-sun validation scenes.

## Public API

```python
import forge3d as f3d
from forge3d.terrain_params import PomSettings, SkySettings, make_terrain_params_config

config = make_terrain_params_config(
    size_px=(1280, 720),
    terrain_span=2500.0,
    z_scale=1.2,
    cam_radius=3600.0,
    cam_phi_deg=135.0,
    cam_theta_deg=52.0,
    sky=SkySettings(
        enabled=True,
        turbidity=3.0,
        ground_albedo=0.25,
        sun_intensity=1.6,
        sun_size=1.2,
        aerial_density=1.4,
        sky_exposure=0.9,
    ),
    pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
)
params = f3d.TerrainRenderParams(config)
```

Runtime expectations:

- `enabled=False` preserves the pre-TV1 terrain baseline.
- Increasing `turbidity` or `aerial_density` pushes the terrain toward a hazier, lower-contrast result.
- Lower sun elevation plus larger `sun_size` warms the image and broadens the sun contribution.

## Example and tests

- Example: `python examples/terrain_atmosphere_path_demo.py`
- Behavior tests: `tests/test_terrain_sky_parity.py`
- Semantics note: `docs/notes/terrain_atmosphere_semantics.md`

The example renders the three regression looks used during implementation:

- clear alpine air
- warm haze / aerial perspective
- low sun / evening relief

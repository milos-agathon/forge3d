# TV2: Terrain Output and Compositing Foundation

Epic TV2 is implemented for terrain beauty plus AOV extraction, directory saves, and single-file multi-channel EXR export.

## What shipped

- `TerrainRenderer.render_with_aov(...)` returns a `(Frame, AovFrame)` tuple.
- The terrain AOV path supports beauty, albedo, normal, and depth outputs.
- `AovFrame.save_all(...)` still writes per-pass files for quick inspection.
- `AovFrame.save_exr(path, beauty_frame)` writes a single multi-channel EXR containing terrain beauty and the enabled AOVs.
- CI includes a terrain TV2 example lane so this path is exercised in automation.

## Public API

```python
import forge3d as f3d
from forge3d.terrain_params import AovSettings, make_terrain_params_config

config = make_terrain_params_config(
    size_px=(1280, 720),
    terrain_span=4000.0,
    z_scale=1.0,
    cam_radius=5200.0,
    cam_phi_deg=142.0,
    cam_theta_deg=64.0,
    aov=AovSettings(enabled=True, albedo=True, normal=True, depth=True),
)

params = f3d.TerrainRenderParams(config)
beauty_frame, aov_frame = renderer.render_with_aov(
    material_set=material_set,
    env_maps=ibl,
    params=params,
    heightmap=heightmap,
)

beauty_frame.save("terrain_beauty.png")
aov_frame.save_all("out", "terrain")
aov_frame.save_exr("out/terrain_multichannel.exr", beauty_frame)
```

## Output contract

- Beauty dimensions match the AOV dimensions exactly.
- Albedo and normal are color outputs.
- Depth is exported as a scalar channel.
- EXR channel naming follows the native writer contract:
  - `beauty.R/G/B/A`
  - `albedo.R/G/B`
  - `normal.X/Y/Z`
  - `depth.Z`

## Example and tests

- Example: `python examples/terrain_tv2_aov_exr_demo.py`
- API tests: `tests/test_aov.py`
- EXR channel test: `tests/test_exr_output.py`
- CI example lane: `.github/workflows/ci.yml`

Use the example when you want one command that writes:

- a beauty PNG
- individual albedo/normal/depth files
- one compositing-friendly EXR with the same data packed into named channels

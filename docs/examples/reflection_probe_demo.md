# TV24 Reflection Probe Demo

`examples/terrain_tv24_reflection_probe_demo.py` demonstrates the delivered TV24 / TV5.3 terrain reflection-probe feature set on a real repo DEM: `assets/tif/dem_rainier.tif`.

## What TV24 Ships

- Optional local reflection probes for terrain scenes via `ReflectionProbeSettings`.
- Automatic or explicit probe-grid placement with `grid_dims`, `origin`, `spacing`, `height_offset`, `resolution`, `ray_count`, and `fallback_blend_distance`.
- CPU probe baking against the terrain heightfield plus mip-aware texture storage for roughness-aware sampling.
- Runtime invalidation when terrain heights, terrain span, `z_scale`, reflection-probe settings, or HDR environment inputs change.
- GPU memory tracking through `TerrainRenderer.get_reflection_probe_memory_report()`.
- Terrain shader debug views:
  - `debug_mode=52`: raw reflection-probe color
  - `debug_mode=53`: reflection-probe blend weight
- Water-path integration so local probes can change reflective water pixels instead of only existing as a debug-only path.

## Python Surface

The public Python entry points added for TV24 are:

- `forge3d.ReflectionProbeSettings`
- `TerrainRenderParams(..., reflection_probes=...)`
- `make_terrain_params_config(..., reflection_probes=...)`
- `TerrainRenderer.get_reflection_probe_memory_report()`

`ReflectionProbeSettings(enabled=False)` is the baseline-preserving default. When disabled or omitted, terrain rendering falls back to the existing global IBL-only behavior.

## Example Behavior

The demo renders five outputs:

- `terrain_tv24_diffuse_only.png`
- `terrain_tv24_reflection_on.png`
- `terrain_tv24_reflection_debug.png`
- `terrain_tv24_reflection_weight.png`
- `terrain_tv24_comparison.png`

The example fits the reflection-probe grid to the generated water mask before rendering. That keeps the local specular budget concentrated over the lake footprint instead of spreading probes across the full DEM extent.

The returned metrics include:

- `mean_abs_diff`: full-frame beauty difference
- `water_mean_abs_diff`: beauty difference only on rendered water pixels
- `probe_memory`
- `reflection_probe_memory`
- `reflection_probe_origin`
- `reflection_probe_spacing`
- `rendered_water_pixels`

## Running It

```bash
python examples/terrain_tv24_reflection_probe_demo.py --dem assets/tif/dem_rainier.tif
```

Optional controls:

- `--output-dir`: choose a custom output directory
- `--width` / `--height`: render resolution
- `--max-dem-size`: downsample the DEM before rendering

## Verification Coverage

TV24 is covered by:

- `tests/test_terrain_probes.py`
  - shader/debug-mode presence
  - fallback identity when reflection probes are disabled
  - memory reporting
  - local reflection color debug output
  - weight blending and edge falloff
  - bake invalidation
- `tests/test_terrain_reflection_probe_exports.py`
  - top-level Python re-export and public API listing
- `tests/test_terrain_tv24_demo.py`
  - real-DEM example execution
  - output image creation
  - non-trivial debug/weight output
  - non-trivial reflected-water beauty difference

## Practical Outcome

TV24 gives terrain scenes a bounded local-specular path that can represent nearby terrain and water-surface context better than a single global IBL cubemap. The result is most visible on reflective water, glossy built elements, and other places where global environment lighting alone is too spatially coarse.

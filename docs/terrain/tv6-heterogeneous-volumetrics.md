# Terrain TV6: Heterogeneous Volumetrics

Epic TV6 extends the terrain viewer's existing volumetric fog pass from global height/uniform fog into bounded, localized 3D density volumes. The result is a single terrain-volumetrics pipeline that can render:

- Valley fog banks that stay attached to low terrain.
- Rising plumes for smoke, ash, steam, or industrial emissions.
- Localized haze pockets that affect only part of a scene.

Legacy volumetric modes still work unchanged. When no density volumes are configured, the viewer falls back to the pre-TV6 global fog behavior.

## What TV6 Adds

### 1. Real 3D density-volume support

The terrain viewer volumetrics pass now accepts a bounded 3D density atlas. Each active volume contributes world-space density to the same raymarch path already used for terrain fog and light shafts.

The current implementation:

- Uses one shared 3D atlas for all active localized volumes.
- Rebuilds and uploads the atlas only when the terrain or volume configuration changes.
- Exposes a runtime report so Python can inspect what the viewer actually allocated.

### 2. Procedural terrain-volume presets

TV6 ships three Python-side preset constructors, all backed by the same 3D density path:

- `valley_fog_volume(...)`
- `plume_volume(...)`
- `localized_haze_volume(...)`

These presets generate CPU-side voxel density fields, upload them to the viewer, and render them through the same WGSL shader and volumetric pass. There are no preset-specific shader forks.

### 3. Budgeted and inspectable runtime behavior

The feature is deliberately bounded:

- Maximum active localized volumes: `4`
- Maximum per-axis volume resolution: `96`
- Density-atlas memory budget: `16 MiB`

The viewer reports:

- `active_volume_count`
- `atlas_dimensions`
- `total_voxels`
- `texture_bytes`
- `memory_budget_bytes`
- `raymarch_steps`
- `half_res`
- Per-volume preset, placement, resolution, atlas offset, and voxel count

Python can query this with `ViewerHandle.get_terrain_volumetrics_report()`.

## World-Space Placement

Density-volume placement uses the terrain viewer's terrain-space coordinates:

- `center.x` and `center.z` live in terrain viewer XZ space, where the terrain spans `[0, terrain_width]`.
- `center.y` uses the same exaggerated terrain height space as the PBR terrain path:
  - `world_y = (raw_height - min_height) * z_scale`

That means a volume can be positioned either:

- Relative to the DEM's scaled relief range, or
- Relative to a sampled terrain feature such as the highest peak

The TV6 demo does both: it finds the Mount Fuji summit from the DEM and places a plume there, while valley fog and haze use broader scene-relative placement.

## Python Usage

### Typed config

```python
from forge3d.terrain_params import VolumetricsSettings, valley_fog_volume

fog = valley_fog_volume(
    center=(420.0, 28.0, 460.0),
    size=(900.0, 120.0, 900.0),
    resolution=(72, 32, 72),
    density_scale=1.3,
    seed=11,
)

vol = VolumetricsSettings(
    enabled=True,
    mode="height",
    density=0.0025,
    height_falloff=0.18,
    scattering=0.76,
    absorption=0.08,
    light_shafts=True,
    shaft_intensity=1.3,
    shaft_samples=48,
    density_volumes=(fog,),
)
```

### Viewer IPC payload

Use `to_viewer_dict()` when sending the config to the interactive terrain viewer:

```python
viewer.send_ipc(
    {
        "cmd": "set_terrain_pbr",
        "enabled": True,
        "volumetrics": vol.to_viewer_dict(),
    }
)

viewer.snapshot("tv6_valley_fog.png", width=1400, height=900)
report = viewer.get_terrain_volumetrics_report()
```

`to_viewer_dict()` exists because the viewer IPC surface names the raymarch sample count `steps`, while the typed dataclass keeps the existing `shaft_samples` name.

## Demo

The reference demo is:

- `examples/terrain_tv6_heterogeneous_volumetrics_demo.py`

It uses the real repo DEM:

- `assets/tif/Mount_Fuji_30m.tif`

The demo writes:

- `terrain_tv6_baseline.png`
- `terrain_tv6_valley_fog.png`
- `terrain_tv6_plume.png`
- `terrain_tv6_localized_haze.png`
- `terrain_tv6_contact_sheet.png`
- `terrain_tv6_manifest.json`

The manifest records image-difference metrics, per-scene viewer reports, DEM metadata, and render timings.

## What the Feature Achieves

TV6 materially improves terrain-scene storytelling and review:

- Volcanic and industrial scenes can now show localized emissions instead of faking them with uniform scene fog.
- Valleys and basins can hold fog without washing out the entire terrain.
- Localized atmosphere can be inspected with concrete memory and raymarch-cost data instead of being an unbounded shader effect.
- Existing terrain-viewer volumetrics workflows remain backward compatible.

## Validation

TV6 coverage now includes:

- Python config tests for `DensityVolumeSettings`, preset constructors, volume-count limits, and viewer payload export.
- Rust unit tests for deterministic atlas generation, budget truncation, and valley-fog vertical behavior.
- A real-DEM viewer integration test that renders the TV6 example, checks image deltas, and verifies memory/report bounds.

Primary validation files:

- `tests/test_volumetrics_sky.py`
- `tests/test_viewer_ipc.py`
- `tests/test_terrain_tv6_heterogeneous_volumetrics.py`
- `src/viewer/terrain/volume_density.rs`

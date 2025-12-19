# Blender-Quality Rendering Features

This document describes the Blender-like offline rendering features implemented in forge3d Milestones 1-6.

## Overview

forge3d now supports high-quality offline rendering features comparable to Blender's Cycles/EEVEE:

| Milestone | Feature | Status |
|-----------|---------|--------|
| M1 | Accumulation Anti-Aliasing | ✅ Complete |
| M2 | Bloom Post-Processing | ✅ Complete |
| M4 | Material Layering | ✅ Complete |
| M5 | Depth-Correct Vector Overlays | ✅ Complete |
| M6 | Tonemap Enhancements | ✅ Complete |
| M7 | Integration & Validation | ✅ Complete |

## Feature Details

### M1: Accumulation Anti-Aliasing

Multi-sample jittered anti-aliasing for smooth edges without excessive MSAA overhead.

```python
from forge3d.terrain_params import make_terrain_params_config

params = make_terrain_params_config(
    size_px=(1920, 1080),
    terrain_span=10000.0,
    domain=(0.0, 4000.0),
    aa_samples=64,      # 1=off, 16/64/256 for quality
    aa_seed=42,         # Optional: deterministic jitter
)
```

**Parameters:**
- `aa_samples`: Number of jittered samples (1=disabled, 16/64/256 typical)
- `aa_seed`: Optional seed for reproducible results

### M2: Bloom Post-Processing

HDR glow effects for bright areas, simulating camera lens bloom.

```python
from forge3d.terrain_params import BloomSettings

bloom = BloomSettings(
    enabled=True,
    intensity=0.3,      # Bloom strength (0.0-1.0)
    threshold=0.8,      # HDR threshold for bloom
    radius=0.5,         # Blur radius
    soft_threshold=0.5, # Soft knee threshold
)

params = make_terrain_params_config(
    ...,
    bloom=bloom,
)
```

### M4: Material Layering

Automatic terrain material assignment based on altitude, slope, and aspect.

```python
from forge3d.terrain_params import MaterialLayerSettings

materials = MaterialLayerSettings(
    # Snow layer
    snow_enabled=True,
    snow_altitude_min=2500.0,    # Meters
    snow_altitude_blend=200.0,   # Transition range
    snow_slope_max=45.0,         # Degrees (no snow on steep slopes)
    
    # Rock layer
    rock_enabled=True,
    rock_slope_min=45.0,         # Exposed rock on steep slopes
    
    # Wetness layer
    wetness_enabled=True,
    wetness_strength=0.3,        # Darkening in low areas
)

params = make_terrain_params_config(
    ...,
    materials=materials,
)
```

### M5: Depth-Correct Vector Overlays

Vector overlays (contours, paths) with proper depth testing and halo outlines.

```python
from forge3d.terrain_params import VectorOverlaySettings

overlay = VectorOverlaySettings(
    # Depth testing
    depth_test=True,            # Occlude behind terrain
    depth_bias=0.001,           # Z-fighting prevention
    
    # Halo rendering
    halo_enabled=True,
    halo_width=2.0,             # Pixels
    halo_color=(0.0, 0.0, 0.0, 0.5),  # RGBA
    halo_blur=1.0,
)

params = make_terrain_params_config(
    ...,
    vector_overlay=overlay,
)
```

### M6: Tonemap Enhancements

HDR tonemapping with operator selection and white balance control.

```python
from forge3d.terrain_params import TonemapSettings

tonemap = TonemapSettings(
    # Operator selection
    operator="aces",            # reinhard, reinhard_extended, aces, uncharted2, exposure
    white_point=4.0,            # For extended operators
    
    # White balance
    white_balance_enabled=True,
    temperature=5500.0,         # Kelvin (2000-12000)
    tint=0.1,                   # Green-magenta (-1 to 1)
)

params = make_terrain_params_config(
    ...,
    tonemap=tonemap,
)
```

**Tonemap Operators:**
- `reinhard`: Simple Reinhard (fast, neutral)
- `reinhard_extended`: Extended Reinhard with white point
- `aces`: ACES filmic (cinematic, default)
- `uncharted2`: Uncharted 2 filmic (warm)
- `exposure`: Simple exposure mapping

## Interactive Viewer CLI

All features are accessible via the interactive viewer:

```bash
# Basic PBR with ACES tonemap
python examples/terrain_viewer_interactive.py \
    --dem assets/dem_rainier.tif --pbr

# Alpine scene with snow and rock
python examples/terrain_viewer_interactive.py \
    --dem assets/dem_rainier.tif --pbr \
    --snow --snow-altitude 2500 \
    --rock --rock-slope 45 \
    --tonemap aces

# Cinematic warm render
python examples/terrain_viewer_interactive.py \
    --dem assets/dem_rainier.tif --pbr \
    --tonemap uncharted2 \
    --white-balance --temperature 5500 \
    --exposure 1.2

# All features enabled
python examples/terrain_viewer_interactive.py \
    --dem assets/dem_rainier.tif --pbr \
    --height-ao --sun-vis \
    --snow --rock --wetness \
    --overlay-depth --overlay-halo \
    --tonemap aces --white-balance
```

## CLI Flag Reference

### Material Layering (M4)

| Flag | Default | Description |
|------|---------|-------------|
| `--snow` | off | Enable snow layer |
| `--snow-altitude` | 2500.0 | Snow altitude threshold (m) |
| `--snow-blend` | 200.0 | Snow blend range (m) |
| `--snow-slope` | 45.0 | Max slope for snow (°) |
| `--rock` | off | Enable rock layer |
| `--rock-slope` | 45.0 | Min slope for rock (°) |
| `--wetness` | off | Enable wetness |
| `--wetness-strength` | 0.3 | Wetness darkening |

### Vector Overlays (M5)

| Flag | Default | Description |
|------|---------|-------------|
| `--overlay-depth` | off | Enable depth testing |
| `--overlay-depth-bias` | 0.001 | Depth bias |
| `--overlay-halo` | off | Enable halo |
| `--overlay-halo-width` | 2.0 | Halo width (px) |
| `--overlay-halo-color` | 0,0,0,0.5 | Halo RGBA |

### Tonemap (M6)

| Flag | Default | Description |
|------|---------|-------------|
| `--tonemap` | aces | Operator |
| `--tonemap-white-point` | 4.0 | White point |
| `--white-balance` | off | Enable white balance |
| `--temperature` | 6500.0 | Color temp (K) |
| `--tint` | 0.0 | Green-magenta tint |

## Acceptance Criteria

All features meet these acceptance criteria:

1. **Backward Compatibility**: All features default to OFF
2. **No Test Regressions**: Existing tests pass
3. **Memory Budget**: ≤512 MiB host-visible heap
4. **Portable**: Vulkan 1.2 / WebGPU compatible

## Example Renders

See `examples/blender_quality_demo.py` for preset configurations:

```bash
# List available presets
python examples/blender_quality_demo.py --list-presets

# Alpine preset
python examples/blender_quality_demo.py --preset alpine --dry-run

# Cinematic preset
python examples/blender_quality_demo.py --preset cinematic --dry-run
```

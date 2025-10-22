# Terrain Ray Tracing Mode

## Overview

The terrain draping pipeline now supports two rendering modes:

1. **Raster mode** (default): Fast GPU rasterization for interactive previews
2. **Raytrace mode**: High-quality GPU path tracing for production renders

Both modes use the same camera parameters and produce consistent viewpoints.

## Usage

### Basic Example

```python
from forge3d.terrain import drape_landcover
import numpy as np

# Your terrain data
heightmap = np.load("dem.npy")  # (H, W) float32
landcover = np.load("landcover.npy")  # (H, W, 4) uint8 RGBA

# Raster mode (fast)
img_raster = drape_landcover(
    heightmap, landcover,
    render_mode="raster",
    width=1920, height=1080,
    camera_theta=45.0, camera_phi=30.0,
)

# Raytrace mode (high quality)
img_raytrace = drape_landcover(
    heightmap, landcover,
    render_mode="raytrace",
    rt_spp=96,  # Samples per pixel
    rt_seed=0,
    width=1920, height=1080,
    camera_theta=45.0, camera_phi=30.0,
)
```

### Command Line (Switzerland Example)

```bash
# Raster mode (fast)
python examples/switzerland_landcover_drape.py --render-mode raster

# Raytrace mode (high quality)
python examples/switzerland_landcover_drape.py \
    --render-mode raytrace \
    --rt-spp 128 \
    --denoiser oidn
```

## Parameters

### Render Mode Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `render_mode` | str | `"raster"` | Rendering mode: `"raster"` or `"raytrace"` |

### Ray Tracing Parameters (raytrace mode only)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rt_spp` | int | `64` | Samples per pixel. Higher = less noise, slower. Range: 4-512 recommended |
| `rt_max_bounces` | int | `2` | Maximum light bounces (currently not used, reserved for future) |
| `rt_seed` | int | `0` | Random seed for reproducible renders |

### Camera Parameters (both modes)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera_theta` | float | `45.0` | Azimuth angle in degrees (0=North, 90=East) |
| `camera_phi` | float | `25.0` | Elevation angle in degrees (0=horizon, 90=zenith) |
| `camera_gamma` | float | `0.0` | Roll angle in degrees |
| `camera_fov` | float | `35.0` | Field of view in degrees |
| `camera_distance` | float | `None` | Camera distance (auto-calculated if None) |

### Lighting & Effects (both modes)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `light_type` | str | `"directional"` | `"none"`, `"directional"`, or `"hemisphere"` |
| `light_azimuth` | float | `315.0` | Light direction azimuth |
| `light_elevation` | float | `45.0` | Light direction elevation |
| `enable_shadows` | bool | `True` | Enable shadow rendering |
| `shadow_intensity` | float | `0.6` | Shadow darkness (0=none, 1=full) |
| `hdri` | str | `None` | Path to HDRI environment map (.hdr, .exr) |
| `hdri_intensity` | float | `1.0` | HDRI brightness multiplier |
| `denoiser` | str | `"oidn"` | `"off"`, `"oidn"`, or `"bilateral"` |

## Mode Comparison

### Raster Mode
- ✅ **Fast**: Real-time rendering (milliseconds to seconds)
- ✅ **Interactive**: Suitable for previews and exploration
- ✅ **Consistent**: GPU rasterization pipeline
- ⚠️ **Lighting**: Simplified shading model

### Raytrace Mode
- ✅ **High Quality**: Physically-based lighting
- ✅ **Realistic**: Accurate shadows and ambient occlusion
- ✅ **Flexible**: HDRI environment support
- ⚠️ **Slower**: Minutes for high-quality renders
- ⚠️ **Noise**: Requires higher `rt_spp` for clean output

## Performance Tips

### Raster Mode
- Default settings are optimized for speed
- Adjust `shadow_map_res` (512-4096) for shadow quality vs speed
- Use `max_dim` to downsample large datasets

### Raytrace Mode
- **Quick preview**: `rt_spp=16`, `denoiser="off"`
- **Balanced**: `rt_spp=64`, `denoiser="oidn"`
- **Production**: `rt_spp=128-256`, `denoiser="oidn"`
- **Memory**: Ray tracing converts terrain to mesh (vertex colors), typically 2-5x raster memory usage

## Quality Settings Guide

| Use Case | rt_spp | Denoiser | Typical Time (1920x1080) |
|----------|--------|----------|--------------------------|
| Quick test | 4-16 | off | 5-15 seconds |
| Preview | 32-64 | oidn | 30-60 seconds |
| Production | 128-256 | oidn | 2-5 minutes |
| High-end | 512+ | oidn | 10+ minutes |

## Camera Consistency

Both modes use identical camera math, so switching between modes preserves the exact same viewpoint:

```python
camera_params = {
    "camera_theta": 45.0,
    "camera_phi": 30.0,
    "camera_fov": 35.0,
}

# Same viewpoint in both modes
img1 = drape_landcover(dem, lc, render_mode="raster", **camera_params)
img2 = drape_landcover(dem, lc, render_mode="raytrace", rt_spp=64, **camera_params)
```

## Implementation Details

### Architecture
- **Raster mode**: Direct GPU terrain draping shader (existing pipeline)
- **Raytrace mode**: Converts heightmap to triangle mesh with vertex colors, then calls `render_raytrace_mesh()`

### Coordinate System
- Both modes use Y-up coordinate system
- Triangle winding is counter-clockwise (front-facing upward)
- Camera framing uses horizontal extent for terrain (from memory 846a665e)

### Vertex Colors
In raytrace mode, landcover colors are mapped to mesh vertices:
- Each heightmap cell becomes one vertex
- Landcover RGBA is converted to vertex RGB colors (0-1 range)
- GPU path tracer interpolates colors across triangles

## Known Limitations

1. **Memory**: Raytrace mode requires full mesh in memory (no tiling yet)
2. **Max bounces**: `rt_max_bounces` parameter reserved for future (currently fixed at 2)
3. **AOVs**: Raytrace mode doesn't export AOVs (depth, normal, etc.) like `render_raytrace_mesh` can

## Examples

See:
- `examples/terrain_raytrace_demo.py` - Quick comparison demo
- `examples/switzerland_landcover_drape.py` - Full-featured CLI with both modes
- `tests/test_terrain_raytrace_mode.py` - Comprehensive test suite

## Testing

Run the test suite:

```bash
# Python tests
pytest tests/test_terrain_raytrace_mode.py -v

# Rust tests
cargo test test_raytrace_toggle
```

All tests validate:
- ✅ Both modes produce valid RGBA output
- ✅ Camera parameters behave identically
- ✅ Ray tracing variance decreases with higher rt_spp
- ✅ Mode switching works correctly
- ✅ Invalid render_mode raises ValueError

# High-Quality Terrain Rendering

This document describes the high-quality rendering features available in forge3d for production-grade terrain visualization.

## Overview

The HQ rendering system provides advanced lighting, material properties, and post-processing effects to create publication-quality terrain visualizations that rival commercial software like rayshader's `render_highquality()`.

## Key Features

### 1. Advanced Lighting Models

Three lighting models are available via the `lighting_model` parameter:

#### Lambert (Simple Diffuse)
- **Use case**: Fast preview, simple visualization
- **Characteristics**: Flat, matte appearance, no view-dependent effects
- **Performance**: Fastest
```python
img = terrain.drape_landcover(dem, landcover, lighting_model="lambert")
```

#### Phong Reflection
- **Use case**: View-dependent specular highlights
- **Characteristics**: Sharp, focused highlights using reflected ray
- **Performance**: Medium
```python
img = terrain.drape_landcover(
    dem, landcover,
    lighting_model="phong",
    shininess=32.0,
    specular_strength=0.3
)
```

#### Blinn-Phong (Default)
- **Use case**: High-quality rendering with smooth highlights
- **Characteristics**: Wider, smoother highlights using half-vector (HQ default)
- **Performance**: Medium (similar to Phong)
```python
img = terrain.drape_landcover(
    dem, landcover,
    lighting_model="blinn_phong",
    shininess=48.0,
    specular_strength=0.4
)
```

### 2. Material Properties

Control surface appearance with these parameters:

- **`shininess`** (1-256): Controls specular highlight tightness
  - Low (1-16): Very wide, soft highlights (matte/satin)
  - Medium (16-64): Natural highlights (default: 32)
  - High (64-256): Tight, sharp highlights (glossy)

- **`specular_strength`** (0-1): Controls reflectivity
  - 0.0: Completely matte (no specular)
  - 0.2-0.4: Natural terrain (rocks, soil)
  - 0.6-1.0: Glossy/wet surfaces

### 3. Background Control

Specify background color as hex string:

```python
# White background (publication quality)
img = terrain.drape_landcover(dem, landcover, background="#FFFFFF")

# Transparent background (for compositing)
img = terrain.drape_landcover(dem, landcover, background="#00000000")

# Custom color background
img = terrain.drape_landcover(dem, landcover, background="#E6F2FF")
```

### 4. Lighting Configuration

Fine-tune lighting for dramatic or subtle effects:

```python
img = terrain.drape_landcover(
    dem, landcover,
    light_type="directional",      # or "hemisphere", "none"
    light_elevation=55.0,           # degrees above horizon
    light_azimuth=325.0,            # degrees from +X (clockwise)
    light_intensity=1.2,            # brightness multiplier
    ambient=0.18,                   # minimum brightness (0-1)
)
```

**Lighting Tips:**
- **Morning/Evening**: `light_elevation=20-40`, warm colors
- **Midday**: `light_elevation=60-80`, high intensity
- **Dramatic shadows**: Lower ambient (0.1-0.2)
- **Soft lighting**: Higher ambient (0.3-0.5)

## Complete HQ Example

```python
import numpy as np
from forge3d import terrain

# Load data
dem = np.load("elevation.npy")
landcover = np.load("landcover.npy")

# Convert to relative heights (important!)
dem_relative = dem - dem.min()

# Render with HQ settings
img = terrain.drape_landcover(
    dem_relative,
    landcover,
    # Output
    width=3000,
    height=2000,
    
    # Terrain
    zscale=1.2,
    max_dim=3000,
    
    # Camera
    camera_theta=48.0,
    camera_phi=32.0,
    camera_fov=35.0,
    
    # HQ Lighting
    lighting_model="blinn_phong",
    shininess=48.0,
    specular_strength=0.4,
    
    # Light source
    light_elevation=50.0,
    light_azimuth=320.0,
    light_intensity=1.1,
    ambient=0.2,
    
    # Background
    background="#FFFFFF",
)

# Save
from PIL import Image
Image.fromarray(img).save("terrain_hq.png")
```

## Quality Presets

The `switzerland_landcover_hq.py` example provides four quality presets:

### Standard Preset
Fast rendering with good quality. Suitable for previews and iterative work.
```bash
python examples/switzerland_landcover_hq.py --preset standard --output standard.png
```

### High Preset (Recommended)
Balanced quality and performance. Best for most production use cases.
```bash
python examples/switzerland_landcover_hq.py --preset high --width 2400 --height 1600
```

### Ultra Preset
Maximum quality settings with strong specular highlights.
```bash
python examples/switzerland_landcover_hq.py --preset ultra --width 3840 --height 2160
```

### Rayshader Preset
Mimics the appearance of R's rayshader package `render_highquality()`.
```bash
python examples/switzerland_landcover_hq.py --preset rayshader
```

## Performance Considerations

| Resolution | Typical Render Time* | Memory Usage |
|------------|---------------------|--------------|
| 1920×1080  | 1-2 seconds         | ~50 MB       |
| 2400×1600  | 2-3 seconds         | ~80 MB       |
| 3840×2160  | 4-6 seconds         | ~150 MB      |

*On Apple M1/M2 GPU. Times may vary by hardware.

## Comparison with Other Tools

### forge3d vs rayshader

| Feature | forge3d HQ | rayshader::render_highquality |
|---------|------------|------------------------------|
| White background | ✅ | ✅ |
| Blinn-Phong lighting | ✅ | ✅ (via path tracing) |
| Specular highlights | ✅ | ✅ |
| Render time (2k)* | 2-3 sec | 30-60 sec |
| Real shadows | ⚠️ Planned | ✅ |
| HDRI environment | ⚠️ Planned | ✅ |
| Ray tracing | ⚠️ Planned | ✅ |
| Denoising | ⚠️ Planned | ✅ |

*Approximate times for 2400×1600 on comparable hardware

**forge3d advantages:**
- Much faster raster rendering (GPU accelerated)
- Real-time preview capability
- Python-native (no R dependency)
- Cross-platform (Windows, Mac, Linux)

**rayshader advantages:**
- True path-traced shadows and ambient occlusion
- HDRI environment lighting
- More mature post-processing pipeline

## Future Enhancements

The following features are planned for future releases:

### Shadow Mapping (Planned)
Real-time shadow mapping with PCF filtering:
```python
img = terrain.drape_landcover(
    dem, landcover,
    shadow_intensity=0.7,      # 0=off, 1=full shadows
    shadow_softness=2.0,       # PCF kernel radius
    shadow_map_res=2048,       # Shadow map resolution
)
```

### HDRI Environment Lighting (Planned)
High-dynamic-range environment maps for realistic ambient lighting:
```python
img = terrain.drape_landcover(
    dem, landcover,
    hdri="studio_small.hdr",
    hdri_intensity=0.8,
)
```

### Tone Mapping (Planned)
HDR to LDR conversion for proper color mapping:
```python
img = terrain.drape_landcover(
    dem, landcover,
    tonemap="aces",            # or "reinhard", "none"
    gamma_correction=2.2,
)
```

### Optional Ray Tracing (Planned)
Path-traced rendering with denoising:
```python
img = terrain.drape_landcover(
    dem, landcover,
    render_mode="raytrace",
    rt_spp=64,                 # Samples per pixel
    rt_max_bounces=2,
    denoiser="oidn",           # or "bilateral", "none"
)
```

## Troubleshooting

### Issue: Terrain appears too dark or too bright
**Solution**: Adjust `ambient` and `light_intensity`:
```python
# Too dark
img = terrain.drape_landcover(dem, lc, ambient=0.3, light_intensity=1.2)

# Too bright
img = terrain.drape_landcover(dem, lc, ambient=0.15, light_intensity=0.9)
```

### Issue: No specular highlights visible
**Solution**: Increase `specular_strength` and check camera angle:
```python
img = terrain.drape_landcover(
    dem, lc,
    specular_strength=0.6,     # Increase from default 0.3
    camera_phi=35.0,           # Adjust viewing angle
    light_elevation=60.0,      # Higher light angle
)
```

### Issue: Specular highlights too harsh
**Solution**: Lower `shininess` for softer highlights:
```python
img = terrain.drape_landcover(dem, lc, shininess=16.0)  # Default is 32
```

### Issue: Background not white
**Solution**: Ensure background parameter is set correctly:
```python
img = terrain.drape_landcover(dem, lc, background="#FFFFFF")  # Not "#FFF"
```

## API Reference

See `python/forge3d/terrain.py` for complete API documentation:
```python
help(terrain.drape_landcover)
```

## Examples

See the `examples/` directory for complete working examples:
- `switzerland_landcover_drape.py` - Basic example
- `switzerland_landcover_hq.py` - HQ rendering with presets
- `test_hq_rendering.py` - Test suite demonstrating all features

# Color Management & Tonemap Pipeline

This document describes the color management workflow in forge3d, including the linear lighting pipeline, tone mapping, and output format choices.

## Overview

The forge3d renderer implements a physically-based linear color workflow with the following pipeline:

```
Linear Scene → Lighting → Tone Mapping → Gamma Correction → sRGB Output
```

## Pipeline Details

### 1. Linear Scene Lighting

All lighting calculations are performed in linear color space:
- Height-based color mapping from LUT textures
- Lambert diffuse lighting: `N·L`
- Ambient term mixed in to avoid flat regions

### 2. Tone Mapping (Reinhard)

The Reinhard tone mapping operator is applied to compress HDR values to [0,1] range:

```glsl
fn reinhard(x: vec3<f32>) -> vec3<f32> {
    return x / (1.0 + x);
}
```

**Characteristics:**
- Simple and fast
- Preserves local contrast
- Asymptotically approaches 1.0 for bright inputs
- Good for landscapes and terrain visualization

### 3. Gamma Correction

Linear tone-mapped values are converted to gamma space for display:

```glsl
fn gamma_correct(x: vec3<f32>) -> vec3<f32> {
    return pow(x, vec3<f32>(1.0 / 2.2));
}
```

**Gamma 2.2** is used as a standard approximation of sRGB gamma curve.

### 4. Output Format

The final output uses **`Rgba8UnormSrgb`** texture format:
- Hardware handles the final linear→sRGB conversion automatically
- Ensures correct display on standard monitors
- Compatible with PNG, JPEG, and other standard image formats

## Format Choices & Rationale

### Why Rgba8UnormSrgb?

1. **Automatic sRGB handling**: Hardware performs the linear→sRGB conversion
2. **Standard compatibility**: Works correctly with standard image viewers  
3. **Memory efficient**: 8 bits per channel is sufficient for final display
4. **Cross-platform**: Universally supported format

### When to Use Linear vs sRGB Targets

**Use `Rgba8UnormSrgb` (sRGB) when:**
- Final output for display/saving to files
- Standard visualization workflows
- Compatibility with image processing tools

**Use `Rgba8Unorm` (linear) when:**
- Intermediate render targets for further processing
- HDR workflows requiring precision
- Custom tone mapping or post-processing effects

## Implementation Notes

### Shader Integration

The tone mapping functions are implemented in `src/shaders/terrain.wgsl`:

```glsl
// Apply explicit tonemap pipeline: reinhard -> gamma correction  
let lit_color = lut_color.rgb * exposure * shade;
let tonemapped = reinhard(lit_color);
let gamma_corrected = gamma_correct(tonemapped);

return vec4<f32>(gamma_corrected, 1.0);
```

### CPU Reference Implementation

CPU reference functions for validation are available in `tests/test_tonemap.py`:

```python
def reinhard(x):
    return x / (1.0 + x)

def gamma_correct(x, gamma=2.2):
    return x ** (1.0 / gamma)
```

## Best Practices

1. **Keep lighting linear**: Perform all lighting calculations before tone mapping
2. **Tone map once**: Apply tone mapping as the final step before gamma correction
3. **Validate with tests**: Use CPU reference implementations to verify shader behavior
4. **Monitor gamma settings**: Ensure consistent gamma handling across the pipeline

## Future Extensions

The current pipeline can be extended with:
- **ACES tone mapping**: Industry standard for film/video
- **Exposure control**: Dynamic range adjustment
- **Color grading**: Artistic color adjustments  
- **HDR output**: High dynamic range display support
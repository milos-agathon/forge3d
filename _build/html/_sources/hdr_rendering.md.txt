# HDR Off-Screen Rendering and Tone Mapping

Forge3D provides comprehensive HDR (High Dynamic Range) off-screen rendering capabilities with advanced tone mapping operators for converting HDR content to LDR (Low Dynamic Range) display output. The HDR pipeline supports multiple tone mapping algorithms and exposure controls for high-quality image processing.

## Overview

HDR rendering works by:

1. **Rendering to floating-point textures** with extended dynamic range beyond [0,1]
2. **Capturing high dynamic range lighting** including bright highlights and deep shadows
3. **Applying tone mapping operators** to convert HDR to display-appropriate LDR
4. **Providing exposure controls** for artistic and technical adjustments

This enables:
- **Realistic lighting simulation** with physically accurate light intensities
- **Post-processing flexibility** with exposure adjustment and tone mapping
- **High-quality image output** with proper highlight and shadow detail
- **Multiple output formats** including HDR preservation and LDR conversion

## API Reference

### Python API

```python
import forge3d.hdr as hdr

# Configure HDR rendering
config = hdr.HdrConfig(
    width=1920,
    height=1080,
    hdr_format="rgba16float",
    tone_mapping=hdr.ToneMappingOperator.ACES,
    exposure=1.5,
    white_point=4.0,
    gamma=2.2
)

# Create HDR renderer
renderer = hdr.HdrRenderer(config)

# Render HDR scene
scene_data = hdr.create_hdr_test_scene(sun_intensity=100.0, sky_intensity=5.0)
hdr_image = renderer.render_hdr_scene(scene_data)

# Apply tone mapping
ldr_image = renderer.apply_tone_mapping()

# Get HDR statistics
stats = renderer.get_hdr_statistics()
print(f"Dynamic range: {stats['dynamic_range']:.2f}")
print(f"Peak luminance: {stats['max_luminance']:.2f} cd/m²")

# Save results
renderer.save_hdr_data("scene.hdr")
renderer.save_ldr_data("scene_tone_mapped.png")
```

### Rust API

```rust
use forge3d::core::hdr::{HdrRenderTarget, HdrConfig, ToneMappingOperator};

// Create HDR render target
let config = HdrConfig {
    width: 1920,
    height: 1080,
    hdr_format: TextureFormat::Rgba16Float,
    tone_mapping: ToneMappingOperator::Aces,
    exposure: 1.5,
    white_point: 4.0,
    gamma: 2.2,
};

let hdr_target = HdrRenderTarget::new(&device, config)?;

// Render to HDR buffer
let mut encoder = device.create_command_encoder(&Default::default());
{
    let mut hdr_pass = hdr_target.begin_hdr_pass(&mut encoder);
    // ... render HDR scene
}

// Apply tone mapping
hdr_target.update_tone_mapping(&queue, 1.5, 4.0);
hdr_target.apply_tone_mapping(&mut encoder);

// Read results
let hdr_data = hdr_target.read_hdr_data(&device, &queue)?;
let ldr_data = hdr_target.read_ldr_data(&device, &queue)?;
```

## Usage Guide

### 1. HDR Configuration

Configure HDR rendering parameters:

```python
# Basic configuration
config = hdr.HdrConfig(
    width=1920,
    height=1080,
    hdr_format="rgba16float",  # or "rgba32float" for higher precision
    tone_mapping=hdr.ToneMappingOperator.REINHARD,
    exposure=1.0,
    white_point=4.0,
    gamma=2.2
)

# Validate configuration
try:
    renderer = hdr.HdrRenderer(config)
    print("HDR configuration valid")
except ValueError as e:
    print(f"Configuration error: {e}")
```

### 2. Tone Mapping Operators

Choose appropriate tone mapping based on content:

```python
# Available operators
operators = [
    hdr.ToneMappingOperator.REINHARD,          # Classic: color / (color + 1)
    hdr.ToneMappingOperator.REINHARD_EXTENDED, # Extended with white point
    hdr.ToneMappingOperator.ACES,              # Film-industry standard
    hdr.ToneMappingOperator.UNCHARTED2,        # Game-industry filmic
    hdr.ToneMappingOperator.EXPOSURE,          # Simple exponential
    hdr.ToneMappingOperator.GAMMA,             # Gamma correction only
    hdr.ToneMappingOperator.CLAMP,             # Simple clamping
]

# Compare operators on same HDR content
results = hdr.compare_tone_mapping_operators(hdr_image, operators, exposure=1.0)

for op_name, result in results.items():
    ldr_mean = result['ldr_mean']
    contrast = result['contrast_ratio']
    print(f"{op_name}: mean={ldr_mean:.3f}, contrast={contrast:.2f}")
```

### 3. HDR Scene Creation

Create test scenes for development:

```python
# Create synthetic HDR scene
scene_data = hdr.create_hdr_test_scene(
    width=512,
    height=512,
    sun_intensity=50.0,    # Direct sunlight: ~50,000 cd/m²
    sky_intensity=2.0      # Clear sky: ~2,000 cd/m²
)

# Render HDR scene
config = hdr.HdrConfig(width=512, height=512)
renderer = hdr.HdrRenderer(config)
hdr_image = renderer.render_hdr_scene(scene_data)

# Analyze HDR content
stats = renderer.get_hdr_statistics()
print(f"HDR Statistics:")
print(f"  Dynamic range: {stats['dynamic_range']:.1f}:1")
print(f"  Luminance range: {stats['min_luminance']:.6f} - {stats['max_luminance']:.2f} cd/m²")
print(f"  Bright pixels (>1): {stats['pixels_above_1']}")
print(f"  Very bright pixels (>10): {stats['pixels_above_10']}")
print(f"  Extreme pixels (>100): {stats['pixels_above_100']}")
```

### 4. Exposure and White Point Control

Adjust exposure and white point for optimal results:

```python
# Test exposure effects
exposure_values = [0.25, 0.5, 1.0, 2.0, 4.0]

for exposure in exposure_values:
    config.exposure = exposure
    renderer = hdr.HdrRenderer(config)
    renderer._hdr_data = hdr_image  # Reuse HDR data
    
    ldr_result = renderer.apply_tone_mapping()
    ldr_mean = np.mean(ldr_result[:, :, :3])
    
    print(f"Exposure {exposure:4.2f}: mean LDR = {ldr_mean:6.2f}")

# Adjust white point for extended Reinhard
config.tone_mapping = hdr.ToneMappingOperator.REINHARD_EXTENDED
config.white_point = 8.0  # Higher white point preserves more highlights

renderer = hdr.HdrRenderer(config)
ldr_extended = renderer.apply_tone_mapping(hdr_image)
```

## Tone Mapping Operators

### Reinhard Tone Mapping

Classic tone mapping with smooth rolloff:

```python
# Basic Reinhard: L_out = L_in / (1 + L_in)
config = hdr.HdrConfig(tone_mapping=hdr.ToneMappingOperator.REINHARD)

# Extended Reinhard with white point control
config = hdr.HdrConfig(
    tone_mapping=hdr.ToneMappingOperator.REINHARD_EXTENDED,
    white_point=4.0  # Luminance level mapped to white
)
```

**Characteristics:**
- Smooth tone curve with no hard clipping
- Preserves local contrast well
- Simple and fast computation
- May produce flat-looking images with low contrast

### ACES Filmic

Film industry standard with cinematic look:

```python
config = hdr.HdrConfig(tone_mapping=hdr.ToneMappingOperator.ACES)
```

**Characteristics:**
- Cinematic color response
- Good highlight rolloff
- Industry-standard color science
- Slightly desaturated look

### Uncharted 2 Filmic

Game industry filmic tone mapping:

```python
config = hdr.HdrConfig(
    tone_mapping=hdr.ToneMappingOperator.UNCHARTED2,
    white_point=11.2  # Typical white point for this operator
)
```

**Characteristics:**
- Developed for game rendering
- High contrast with punchy look
- Good shoulder and toe response
- Popular in real-time applications

### Exposure-Based

Simple exponential tone mapping:

```python
config = hdr.HdrConfig(tone_mapping=hdr.ToneMappingOperator.EXPOSURE)
```

**Characteristics:**
- Mimics camera exposure response
- Simple exponential curve: 1 - exp(-exposure * L_in)
- Good for natural lighting simulation
- May clip bright highlights

## Shader Integration

### WGSL Tone Mapping Shaders

```wgsl
struct ToneMappingUniforms {
    exposure: f32,
    white_point: f32,
    gamma: f32,
    operator_index: u32,
}

@group(0) @binding(0) var<uniform> uniforms: ToneMappingUniforms;
@group(0) @binding(1) var hdr_texture: texture_2d<f32>;
@group(0) @binding(2) var hdr_sampler: sampler;

// Reinhard tone mapping
fn reinhard_tonemap(color: vec3<f32>) -> vec3<f32> {
    return color / (color + vec3<f32>(1.0));
}

// ACES filmic tone mapping
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    
    return saturate((color * (color * a + b)) / (color * (color * c + d) + e));
}

// Fragment shader for tone mapping
@fragment
fn fs_tonemap(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample HDR texture
    let hdr_color = textureSample(hdr_texture, hdr_sampler, input.uv);
    
    // Apply exposure
    let exposed_color = hdr_color.rgb * uniforms.exposure;
    
    // Apply tone mapping based on operator
    var tone_mapped: vec3<f32>;
    if (uniforms.operator_index == 0u) {
        tone_mapped = reinhard_tonemap(exposed_color);
    } else if (uniforms.operator_index == 2u) {
        tone_mapped = aces_tonemap(exposed_color);
    }
    // ... other operators
    
    // Apply gamma correction
    let gamma_corrected = pow(tone_mapped, vec3<f32>(1.0 / uniforms.gamma));
    
    return vec4<f32>(gamma_corrected, hdr_color.a);
}
```

## HDR Statistics and Analysis

### Dynamic Range Analysis

```python
def analyze_hdr_content(hdr_image):
    """Analyze HDR content for optimal tone mapping settings."""
    stats = renderer.get_hdr_statistics(hdr_image)
    
    print("HDR Content Analysis:")
    print(f"  Dynamic Range: {stats['dynamic_range']:.1f}:1")
    
    if stats['dynamic_range'] < 10:
        print("  → Low dynamic range, consider gamma or clamp tone mapping")
    elif stats['dynamic_range'] < 100:
        print("  → Medium dynamic range, Reinhard or ACES recommended")
    else:
        print("  → High dynamic range, filmic tone mapping recommended")
    
    # Analyze brightness distribution
    bright_ratio = stats['pixels_above_1'] / (stats['width'] * stats['height'])
    if bright_ratio > 0.1:
        print(f"  → {bright_ratio*100:.1f}% bright pixels, consider lower exposure")
    
    # Recommend white point
    if stats['max_luminance'] > 10:
        recommended_white = min(stats['max_luminance'] / 2, 20)
        print(f"  → Recommended white point: {recommended_white:.1f}")
    
    return stats
```

### Luminance Histogram

```python
def create_luminance_histogram(hdr_image):
    """Create luminance histogram for HDR analysis."""
    import matplotlib.pyplot as plt
    
    # Compute luminance
    luminance = 0.299 * hdr_image[:, :, 0] + 0.587 * hdr_image[:, :, 1] + 0.114 * hdr_image[:, :, 2]
    
    # Create log-scale histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Linear histogram
    ax1.hist(luminance.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Luminance (cd/m²)')
    ax1.set_ylabel('Pixel Count')
    ax1.set_title('Luminance Distribution (Linear)')
    
    # Log histogram
    log_luminance = np.log10(np.maximum(luminance, 1e-6))
    ax2.hist(log_luminance.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Log₁₀ Luminance')
    ax2.set_ylabel('Pixel Count')
    ax2.set_title('Luminance Distribution (Log)')
    
    plt.tight_layout()
    plt.savefig('luminance_histogram.png', dpi=150, bbox_inches='tight')
    print("Saved luminance histogram: luminance_histogram.png")
```

## Validation and Testing

### Tone Mapping Validation

```python
def validate_tone_mapping_quality(hdr_image, operator):
    """Validate tone mapping quality and effectiveness."""
    
    # Create renderer with specific operator
    config = hdr.HdrConfig(
        width=hdr_image.shape[1],
        height=hdr_image.shape[0], 
        tone_mapping=operator
    )
    
    renderer = hdr.HdrRenderer(config)
    renderer._hdr_data = hdr_image
    
    # Apply tone mapping
    ldr_result = renderer.apply_tone_mapping()
    
    # Compute quality metrics
    ldr_luminance = 0.299 * ldr_result[:, :, 0] + 0.587 * ldr_result[:, :, 1] + 0.114 * ldr_result[:, :, 2]
    
    metrics = {
        'mean_luminance': float(np.mean(ldr_luminance)),
        'std_luminance': float(np.std(ldr_luminance)),
        'contrast_ratio': float(np.max(ldr_luminance) / max(np.min(ldr_luminance), 1)),
        'clipped_pixels': int(np.sum((ldr_result == 0) | (ldr_result == 255))),
        'utilization': float(np.mean(ldr_luminance) / 255.0),
    }
    
    # Quality assessment
    print(f"Tone Mapping Quality - {operator.value}:")
    print(f"  Mean luminance: {metrics['mean_luminance']:.1f} (target: 100-150)")
    print(f"  Contrast ratio: {metrics['contrast_ratio']:.1f} (higher is better)")
    print(f"  Clipped pixels: {metrics['clipped_pixels']} (lower is better)")
    print(f"  Utilization: {metrics['utilization']*100:.1f}% (target: 40-70%)")
    
    # Pass/fail criteria
    quality_score = 0
    if 80 <= metrics['mean_luminance'] <= 180:
        quality_score += 1
    if metrics['contrast_ratio'] > 50:
        quality_score += 1
    if metrics['clipped_pixels'] < hdr_image.size * 0.01:  # <1% clipped
        quality_score += 1
    if 0.3 <= metrics['utilization'] <= 0.8:
        quality_score += 1
    
    passed = quality_score >= 3
    print(f"  Quality Score: {quality_score}/4 ({'PASS' if passed else 'FAIL'})")
    
    return metrics, passed
```

### HDR Pipeline Test

```python
def test_hdr_pipeline():
    """Complete HDR pipeline test."""
    print("=== HDR Pipeline Test ===")
    
    # Create HDR test scene
    scene_data = hdr.create_hdr_test_scene(
        width=256,
        height=256,
        sun_intensity=100.0,
        sky_intensity=5.0
    )
    
    config = hdr.HdrConfig(width=256, height=256)
    renderer = hdr.HdrRenderer(config)
    
    # Render HDR
    hdr_image = renderer.render_hdr_scene(scene_data)
    print(f"✓ HDR rendering: {hdr_image.shape}, range=[{np.min(hdr_image):.3f}, {np.max(hdr_image):.3f}]")
    
    # Test all tone mapping operators
    operators = [
        hdr.ToneMappingOperator.REINHARD,
        hdr.ToneMappingOperator.ACES,
        hdr.ToneMappingOperator.UNCHARTED2,
    ]
    
    passed_operators = 0
    for operator in operators:
        config.tone_mapping = operator
        renderer.config = config
        
        try:
            ldr_result = renderer.apply_tone_mapping()
            print(f"✓ {operator.value} tone mapping: {ldr_result.shape}, dtype={ldr_result.dtype}")
            passed_operators += 1
        except Exception as e:
            print(f"✗ {operator.value} tone mapping failed: {e}")
    
    # HDR statistics
    stats = renderer.get_hdr_statistics()
    print(f"✓ HDR statistics: DR={stats['dynamic_range']:.1f}, bright_pixels={stats['pixels_above_1']}")
    
    # Overall result
    total_tests = len(operators) + 2  # operators + HDR render + stats
    passed_tests = passed_operators + 2
    
    print(f"\nHDR Pipeline Test Result: {passed_tests}/{total_tests} tests passed")
    return passed_tests == total_tests
```

## Performance Considerations

### Memory Usage

- **HDR textures**: RGBA16Float = 8 bytes/pixel, RGBA32Float = 16 bytes/pixel
- **1080p HDR buffer**: ~16 MB (16-bit) or ~32 MB (32-bit)
- **Multiple render targets**: HDR + LDR + depth ≈ 24 MB for 1080p

### Rendering Performance

- **HDR rendering**: Minimal overhead vs LDR rendering
- **Tone mapping**: Fullscreen pass, typically <1ms
- **Memory bandwidth**: Higher due to larger pixel size
- **GPU compatibility**: Requires floating-point render target support

### Optimization Tips

1. **Use 16-bit HDR** format when sufficient for quality requirements
2. **Combine tone mapping** with other post-processing passes
3. **Pre-compute tone curves** for complex operators when possible
4. **Use compute shaders** for tone mapping when available
5. **Profile memory bandwidth** usage with HDR buffers

## Examples

### Basic HDR Tone Mapping

```bash
python examples/hdr_tone_mapping.py --headless --out out/hdr_comparison.png
```

This example:
- Creates synthetic HDR scenes with high dynamic range
- Tests all available tone mapping operators
- Compares results with statistical analysis
- Saves individual and comparison images
- Validates tone mapping effectiveness

### Advanced HDR Workflow

```python
import forge3d.hdr as hdr
import numpy as np

# Create HDR scene
scene_data = hdr.create_hdr_test_scene(sun_intensity=200.0, sky_intensity=10.0)
config = hdr.HdrConfig(width=512, height=512, tone_mapping=hdr.ToneMappingOperator.ACES)
renderer = hdr.HdrRenderer(config)

# Render and analyze
hdr_image = renderer.render_hdr_scene(scene_data)
stats = renderer.get_hdr_statistics()

# Optimize exposure based on content
optimal_exposure = 1.0 / max(stats['mean_luminance'], 1.0)
renderer.config.exposure = optimal_exposure

# Apply tone mapping
ldr_result = renderer.apply_tone_mapping()

# Save results
renderer.save_hdr_data("scene.npy")  # Preserve HDR data
renderer.save_ldr_data("scene_ldr.png")  # Display-ready image

print(f"Optimal exposure: {optimal_exposure:.2f}")
print(f"Final dynamic range: {stats['dynamic_range']:.1f}:1")
```

## Troubleshooting

### Common Issues

1. **Flat tone mapped images**: Increase exposure or use filmic tone mapping
2. **Clipped highlights**: Reduce exposure or increase white point
3. **Dark images**: Increase exposure or adjust gamma
4. **No HDR effect**: Verify HDR content has values > 1.0
5. **Performance issues**: Check GPU memory usage and bandwidth

### Debug Techniques

```python
# Enable HDR debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check HDR content range
hdr_stats = renderer.get_hdr_statistics()
if hdr_stats['max_luminance'] <= 1.0:
    print("WARNING: No HDR content detected")

# Validate tone mapping differences
operators = [hdr.ToneMappingOperator.REINHARD, hdr.ToneMappingOperator.ACES]
results = hdr.compare_tone_mapping_operators(hdr_image, operators)

for op1, op2 in [(operators[0], operators[1])]:
    diff = np.mean(np.abs(results[op1.value]['ldr_data'] - results[op2.value]['ldr_data']))
    print(f"Difference {op1.value} vs {op2.value}: {diff:.2f}")

# False color HDR visualization
luminance = 0.299 * hdr_image[:, :, 0] + 0.587 * hdr_image[:, :, 1] + 0.114 * hdr_image[:, :, 2]
false_color = np.zeros_like(hdr_image[:, :, :3])
false_color[luminance > 10] = [1, 0, 0]  # Red for very bright
false_color[(luminance > 1) & (luminance <= 10)] = [1, 1, 0]  # Yellow for bright
false_color[luminance <= 1] = [0, 1, 0]  # Green for normal
```

## Feature Flags

HDR rendering requires appropriate feature flags:

```toml
[dependencies]
forge3d = { features = ["enable-hdr", "enable-tone-mapping"] }
```

## Cross-Platform Support

HDR rendering works consistently across:

- **Windows**: DirectX 12 with floating-point render targets
- **Linux**: Vulkan with HDR format support
- **macOS**: Metal with extended range pixel formats

All platforms use identical WGSL tone mapping shaders ensuring consistent image quality across different graphics backends.
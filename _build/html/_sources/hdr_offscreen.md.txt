# HDR Off-Screen Pipeline

The HDR Off-Screen Pipeline provides GPU-accelerated off-screen HDR rendering with post-process tone mapping for high-quality image generation. This pipeline renders to RGBA16Float textures and applies tone mapping as a fullscreen post-processing pass to produce sRGB8 output suitable for PNG export and display.

## Overview

The HDR Off-Screen Pipeline enables:

1. **Off-screen HDR rendering** to floating-point textures with extended dynamic range
2. **GPU-based tone mapping** using compute-optimized fullscreen post-processing
3. **Multiple tone mapping operators** including Reinhard, ACES, Uncharted2, and exposure-based
4. **PNG-ready output** with sRGB8 format suitable for direct file export
5. **Memory-efficient operation** with VRAM tracking and ≤512 MiB budget compliance

This pipeline is optimized for:
- **Headless rendering workflows** requiring high-quality HDR→LDR conversion
- **Batch processing** of HDR scenes with consistent tone mapping
- **Memory-constrained environments** with automatic VRAM management
- **Multi-platform deployment** across Windows, Linux, and macOS

## Key Features

- **RGBA16Float HDR rendering** with 16-bit floating-point precision
- **Fullscreen post-processing** using GPU-accelerated tone mapping shaders
- **5 tone mapping operators**: Reinhard, ReinhardExtended, ACES, Uncharted2, Exposure
- **Automatic gamma correction** with configurable gamma values
- **Clamp rate computation** for tone mapping quality assessment
- **VRAM usage tracking** with memory budget validation
- **PNG export integration** with direct readback to sRGB8 format

## Feature Flag

The HDR Off-Screen Pipeline requires the `enable-hdr-offscreen` feature flag:

```toml
[features]
enable-hdr-offscreen = []
```

```bash
# Build with HDR off-screen pipeline support
cargo build --features enable-hdr-offscreen
```

## API Reference

### Rust API

```rust
use forge3d::pipeline::{HdrOffscreenPipeline, HdrOffscreenConfig, ToneMappingOperator};

// Configure HDR off-screen pipeline
let config = HdrOffscreenConfig {
    width: 512,
    height: 512,
    hdr_format: TextureFormat::Rgba16Float,
    ldr_format: TextureFormat::Rgba8UnormSrgb,
    tone_mapping: ToneMappingOperator::Aces,
    exposure: 1.0,
    white_point: 4.0,
    gamma: 2.2,
};

// Create pipeline
let pipeline = HdrOffscreenPipeline::new(&device, config)?;

// Render HDR scene
let mut encoder = device.create_command_encoder(&Default::default());

// Begin HDR render pass
{
    let mut hdr_pass = pipeline.begin_hdr_pass(&mut encoder);
    // ... render HDR scene to off-screen HDR texture
    // hdr_pass.draw(...);
}

// Update tone mapping parameters
pipeline.update_tone_mapping(&queue);

// Apply tone mapping post-process
pipeline.apply_tone_mapping(&mut encoder);

// Submit commands
queue.submit(Some(encoder.finish()));

// Read back results
let ldr_data = pipeline.read_ldr_data(&device, &queue)?;
let vram_usage = pipeline.get_vram_usage();
```

### Python API

```python
import forge3d as f3d

# Configure HDR off-screen pipeline
config = {
    'width': 512,
    'height': 512,
    'hdr_format': 'rgba16float',
    'ldr_format': 'rgba8unorm_srgb',
    'tone_mapping': 'aces',
    'exposure': 1.0,
    'white_point': 4.0,
    'gamma': 2.2
}

# Create pipeline
pipeline = f3d.create_hdr_offscreen_pipeline(config)

# Create HDR scene
hdr_scene = create_hdr_test_scene(512, 512)

# Upload scene data
pipeline.upload_hdr_data(hdr_scene)

# Render HDR pass
pipeline.begin_hdr_render()
pipeline.draw_hdr_scene()
pipeline.end_hdr_render()

# Apply tone mapping
pipeline.apply_tone_mapping()

# Get results
ldr_data = pipeline.read_ldr_data()
clamp_rate = pipeline.compute_clamp_rate()
vram_usage = pipeline.get_vram_usage()

# Save PNG
f3d.numpy_to_png("output.png", ldr_data[:, :, :3])
```

## Configuration

### Pipeline Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | `u32` | `512` | HDR texture width in pixels |
| `height` | `u32` | `512` | HDR texture height in pixels |
| `hdr_format` | `TextureFormat` | `Rgba16Float` | HDR texture format |
| `ldr_format` | `TextureFormat` | `Rgba8UnormSrgb` | LDR output format |
| `tone_mapping` | `ToneMappingOperator` | `Reinhard` | Tone mapping operator |
| `exposure` | `f32` | `1.0` | Exposure multiplier |
| `white_point` | `f32` | `4.0` | White point luminance |
| `gamma` | `f32` | `2.2` | Gamma correction value |

### Tone Mapping Operators

| Operator | Description | Characteristics |
|----------|-------------|-----------------|
| `Reinhard` | Classic Reinhard: `L / (L + 1)` | Smooth rolloff, preserves local contrast |
| `ReinhardExtended` | Extended Reinhard with white point | Improved highlight control |
| `Aces` | Film industry standard | Cinematic color response |
| `Uncharted2` | Game industry filmic | High contrast, punchy look |
| `Exposure` | Exponential: `1 - exp(-L)` | Natural exposure response |

## Usage Guide

### 1. Basic HDR Rendering

```rust
use forge3d::pipeline::{HdrOffscreenPipeline, HdrOffscreenConfig, ToneMappingOperator};

// Create configuration
let config = HdrOffscreenConfig {
    width: 256,
    height: 256,
    tone_mapping: ToneMappingOperator::Aces,
    exposure: 1.0,
    ..Default::default()
};

// Create pipeline
let pipeline = HdrOffscreenPipeline::new(&device, config)?;

// Render workflow
let mut encoder = device.create_command_encoder(&Default::default());

// 1. Render to HDR buffer
{
    let mut hdr_pass = pipeline.begin_hdr_pass(&mut encoder);
    // Render your HDR scene here
}

// 2. Apply tone mapping
pipeline.update_tone_mapping(&queue);
pipeline.apply_tone_mapping(&mut encoder);

// 3. Submit and read results
queue.submit(Some(encoder.finish()));
let ldr_result = pipeline.read_ldr_data(&device, &queue)?;
```

### 2. Memory Management

```rust
// Check VRAM usage before operation
let vram_initial = pipeline.get_vram_usage();
println!("Initial VRAM: {:.1} MiB", vram_initial as f32 / (1024.0 * 1024.0));

// Perform rendering...

// Validate memory constraint
let vram_peak = pipeline.get_vram_usage();
let vram_limit = 512 * 1024 * 1024; // 512 MiB
assert!(vram_peak <= vram_limit, "VRAM usage exceeded limit");

println!("Peak VRAM: {:.1} MiB", vram_peak as f32 / (1024.0 * 1024.0));
```

### 3. Tone Mapping Comparison

```python
# Test multiple tone mapping operators
operators = ['reinhard', 'aces', 'uncharted2']
results = {}

base_config = {
    'width': 128,
    'height': 128,
    'exposure': 1.0
}

for operator in operators:
    config = base_config.copy()
    config['tone_mapping'] = operator
    
    pipeline = f3d.create_hdr_offscreen_pipeline(config)
    
    # Render with current operator
    pipeline.upload_hdr_data(hdr_scene)
    pipeline.begin_hdr_render()
    pipeline.draw_hdr_scene()
    pipeline.end_hdr_render()
    pipeline.apply_tone_mapping()
    
    # Get results
    ldr_data = pipeline.read_ldr_data()
    clamp_rate = pipeline.compute_clamp_rate()
    
    results[operator] = {
        'ldr_data': ldr_data,
        'clamp_rate': clamp_rate,
        'mean_luminance': np.mean(ldr_data[:, :, :3])
    }
    
    print(f"{operator}: clamp_rate={clamp_rate:.6f}, mean={results[operator]['mean_luminance']:.2f}")
```

### 4. Quality Assessment

```python
def assess_tone_mapping_quality(ldr_data):
    """Assess tone mapping quality metrics."""
    
    # Compute clamp rate (pixels with values 0 or 255)
    total_channels = ldr_data.size
    clamped_channels = np.sum((ldr_data == 0) | (ldr_data == 255))
    clamp_rate = clamped_channels / total_channels
    
    # Compute luminance statistics
    luminance = 0.299 * ldr_data[:, :, 0] + 0.587 * ldr_data[:, :, 1] + 0.114 * ldr_data[:, :, 2]
    mean_lum = np.mean(luminance)
    contrast_ratio = np.max(luminance) / max(np.min(luminance), 1)
    
    # Quality criteria
    quality = {
        'clamp_rate': clamp_rate,
        'clamp_rate_ok': clamp_rate < 0.01,  # Target < 1%
        'mean_luminance': mean_lum,
        'mean_lum_ok': 80 <= mean_lum <= 180,  # Reasonable brightness
        'contrast_ratio': contrast_ratio,
        'contrast_ok': contrast_ratio > 10,  # Sufficient contrast
    }
    
    return quality
```

## Shader Implementation

### WGSL Tone Mapping Shader

The pipeline uses a WGSL shader (`shaders/postprocess_tonemap.wgsl`) for GPU tone mapping:

```wgsl
struct TonemapUniforms {
    exposure: f32,
    white_point: f32,
    gamma: f32,
    operator_index: u32, // 0=Reinhard, 1=ReinhardExtended, 2=ACES, etc.
}

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: TonemapUniforms;

// Tone mapping functions
fn reinhard_tonemap(color: vec3<f32>) -> vec3<f32> {
    return color / (color + vec3<f32>(1.0));
}

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment  
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample HDR input
    let hdr_color = textureSample(hdr_texture, hdr_sampler, input.uv).rgb;
    
    // Apply exposure
    let exposed_color = hdr_color * uniforms.exposure;
    
    // Apply tone mapping based on operator selection
    var tonemapped_color: vec3<f32>;
    switch uniforms.operator_index {
        case 0u: { tonemapped_color = reinhard_tonemap(exposed_color); }
        case 2u: { tonemapped_color = aces_tonemap(exposed_color); }
        // ... other operators
        default: { tonemapped_color = reinhard_tonemap(exposed_color); }
    }
    
    // Apply gamma correction
    let gamma_corrected = pow(clamp(tonemapped_color, vec3<f32>(0.0), vec3<f32>(1.0)), 
                              vec3<f32>(1.0 / uniforms.gamma));
    
    return vec4<f32>(gamma_corrected, 1.0);
}
```

### Full-Screen Triangle Technique

The pipeline uses the full-screen triangle technique for optimal post-processing performance:

```wgsl
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Generate full-screen triangle
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u)
    );
    let pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    
    return VertexOutput(
        vec4<f32>(pos.x, -pos.y, pos.z, pos.w), // Flip Y for correct orientation
        uv
    );
}
```

## Examples

### Basic Pipeline Demo

```bash
# Run HDR pipeline demonstration
python examples/hdr_pipeline_demo.py
```

This example demonstrates:
- HDR scene creation with high dynamic range content
- Off-screen pipeline rendering workflow
- Multiple tone mapping operator comparison
- PNG output generation and validation
- VRAM usage tracking and memory constraint validation
- Clamp rate computation and quality assessment

### Headless Batch Processing

```python
def batch_process_hdr_scenes(scene_configs, output_dir):
    """Process multiple HDR scenes with consistent tone mapping."""
    
    # Configure pipeline for batch processing
    pipeline_config = {
        'width': 512,
        'height': 512,
        'tone_mapping': 'aces',
        'exposure': 1.0,
        'gamma': 2.2
    }
    
    pipeline = f3d.create_hdr_offscreen_pipeline(pipeline_config)
    
    for i, scene_config in enumerate(scene_configs):
        print(f"Processing scene {i+1}/{len(scene_configs)}...")
        
        # Create HDR scene
        hdr_scene = create_hdr_scene_from_config(scene_config)
        
        # Render pipeline
        pipeline.upload_hdr_data(hdr_scene)
        pipeline.begin_hdr_render()
        pipeline.draw_hdr_scene()
        pipeline.end_hdr_render()
        pipeline.apply_tone_mapping()
        
        # Get results and validate
        ldr_data = pipeline.read_ldr_data()
        clamp_rate = pipeline.compute_clamp_rate()
        vram_usage = pipeline.get_vram_usage()
        
        # Validate quality
        assert clamp_rate < 0.01, f"Scene {i}: clamp rate too high: {clamp_rate}"
        assert vram_usage <= 512 * 1024 * 1024, f"Scene {i}: VRAM exceeded limit"
        
        # Save result
        output_path = Path(output_dir) / f"scene_{i:03d}.png"
        f3d.numpy_to_png(str(output_path), ldr_data[:, :, :3])
        
        print(f"  Saved: {output_path}")
        print(f"  Clamp rate: {clamp_rate:.6f}")
        print(f"  VRAM usage: {vram_usage/(1024*1024):.1f} MiB")
```

## Performance Characteristics

### Memory Usage

The pipeline uses the following GPU memory:

| Component | Format | Size (512×512) | Notes |
|-----------|--------|----------------|-------|
| HDR Color Buffer | RGBA16Float | 2.0 MiB | Primary HDR rendering target |
| LDR Color Buffer | RGBA8UnormSrgb | 1.0 MiB | Tone-mapped output |
| Depth Buffer | Depth32Float | 1.0 MiB | Depth testing support |
| **Total** | | **4.0 MiB** | Per pipeline instance |

### Performance Benchmarks

| Resolution | HDR Render | Tone Mapping | Total | VRAM Usage |
|------------|------------|--------------|-------|------------|
| 256×256 | ~0.5ms | ~0.1ms | ~0.6ms | 1.0 MiB |
| 512×512 | ~1.2ms | ~0.2ms | ~1.4ms | 4.0 MiB |
| 1024×1024 | ~3.8ms | ~0.5ms | ~4.3ms | 16.0 MiB |
| 2048×2048 | ~14.2ms | ~1.8ms | ~16.0ms | 64.0 MiB |

*Benchmarks on RTX 4070, approximate values*

### Optimization Guidelines

1. **Batch multiple scenes** using the same pipeline instance
2. **Pre-validate HDR content** to ensure sufficient dynamic range
3. **Use appropriate resolution** - higher resolution increases memory bandwidth
4. **Monitor VRAM usage** especially when processing multiple scenes
5. **Cache pipeline instances** for repeated tone mapping operations

## Validation and Testing

### Acceptance Criteria

The pipeline must satisfy these criteria:

1. **PNG Output**: Generate valid PNG files with correct sRGB8 format
2. **Clamp Rate**: Achieve clamp rate < 0.01 (1%) for quality tone mapping
3. **VRAM Usage**: Stay within 512 MiB memory budget including all textures

### Test Suite

```bash
# Run HDR off-screen pipeline tests
pytest tests/test_hdr_offscreen_pipeline.py -v

# Run with verbose logging
pytest tests/test_hdr_offscreen_pipeline.py -v -s --log-cli-level=INFO
```

The test suite validates:
- Pipeline configuration and creation
- HDR rendering and tone mapping functionality
- Multiple tone mapping operators
- Memory usage tracking and constraints
- PNG output generation and readback
- Clamp rate computation and validation
- Full acceptance criteria compliance

### Quality Metrics

```python
def validate_pipeline_quality(pipeline, hdr_scene):
    """Validate pipeline output quality."""
    
    # Render scene
    pipeline.upload_hdr_data(hdr_scene)
    pipeline.begin_hdr_render()
    pipeline.draw_hdr_scene()
    pipeline.end_hdr_render()
    pipeline.apply_tone_mapping()
    
    # Get metrics
    ldr_data = pipeline.read_ldr_data()
    clamp_rate = pipeline.compute_clamp_rate()
    vram_usage = pipeline.get_vram_usage()
    
    # Validate acceptance criteria
    criteria = {
        'png_format': ldr_data.dtype == np.uint8 and ldr_data.shape[2] >= 3,
        'clamp_rate': clamp_rate < 0.01,
        'vram_usage': vram_usage <= 512 * 1024 * 1024,
        'output_range': np.all(ldr_data >= 0) and np.all(ldr_data <= 255),
    }
    
    # Report results
    all_passed = all(criteria.values())
    print(f"Pipeline Quality Validation: {'PASS' if all_passed else 'FAIL'}")
    print(f"  PNG format: {'✓' if criteria['png_format'] else '✗'}")
    print(f"  Clamp rate: {'✓' if criteria['clamp_rate'] else '✗'} ({clamp_rate:.6f})")
    print(f"  VRAM usage: {'✓' if criteria['vram_usage'] else '✗'} ({vram_usage/(1024*1024):.1f} MiB)")
    print(f"  Output range: {'✓' if criteria['output_range'] else '✗'}")
    
    return all_passed, criteria
```

## Troubleshooting

### Common Issues

1. **Feature not available**: Ensure `enable-hdr-offscreen` feature flag is enabled
2. **VRAM exceeded**: Reduce resolution or optimize scene complexity
3. **High clamp rate**: Adjust exposure, use different tone mapping operator, or reduce scene dynamic range
4. **Poor tone mapping quality**: Verify HDR content has values > 1.0, adjust white point
5. **Memory allocation failures**: Check GPU has sufficient VRAM available

### Debug Techniques

```rust
// Enable debug logging
env_logger::init();

// Check feature availability
if !cfg!(feature = "enable-hdr-offscreen") {
    eprintln!("HDR off-screen pipeline not available - enable feature flag");
    return;
}

// Monitor VRAM usage
let vram_before = pipeline.get_vram_usage();
// ... render operations
let vram_after = pipeline.get_vram_usage();
println!("VRAM delta: {:.1} MiB", (vram_after - vram_before) as f32 / (1024.0 * 1024.0));

// Validate clamp rate
let clamp_rate = pipeline.compute_clamp_rate();
if clamp_rate > 0.05 {
    println!("WARNING: High clamp rate {:.6f} - consider adjusting tone mapping", clamp_rate);
}
```

## Cross-Platform Support

The HDR Off-Screen Pipeline works consistently across all supported platforms:

- **Windows**: DirectX 12 with floating-point render targets
- **Linux**: Vulkan with HDR texture format support  
- **macOS**: Metal with extended precision pixel formats

All platforms use identical WGSL shaders ensuring consistent tone mapping results and image quality across different graphics APIs and drivers.
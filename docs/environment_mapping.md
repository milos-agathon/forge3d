# Environment Mapping and Image-Based Lighting (IBL)

Environment mapping provides realistic lighting by using environment textures to simulate complex lighting conditions. Forge3D's IBL implementation uses CPU-side environment map processing with roughness-based mip sampling for physically-based rendering.

## Overview

Environment mapping works by:

1. **Loading HDR environment maps** from equirectangular images
2. **Preprocessing** irradiance and specular maps for different roughness levels
3. **Sampling** environment lighting based on surface normal and view direction
4. **Combining** diffuse and specular contributions for realistic material appearance

This technique enables:
- **Realistic lighting** from real-world captured environments
- **Efficient rendering** using precomputed lighting data
- **Material variation** through roughness-based reflections
- **Cross-platform compatibility** using WebGPU/WGSL shaders

## API Reference

### Python API

```python
import forge3d.envmap as envmap

# Create or load environment maps
env_map = envmap.EnvironmentMap.create_test_envmap(256)
env_map = envmap.load_environment_map("assets/studio.hdr")

# Validate environment data
validation = envmap.validate_environment_map(env_map)
assert validation['valid']

# Sample environment in different directions
color = env_map.sample_direction([0, 1, 0])  # Sample upward

# Test roughness monotonicity
roughness_values = [0.1, 0.5, 0.9]
luminances = envmap.compute_roughness_luminance_series(env_map, roughness_values)
# Verify: luminances[0] > luminances[1] > luminances[2]

# Save processed environment maps
envmap.save_environment_map(env_map, "output/environment.png")
```

### Rust API

```rust
use forge3d::core::envmap::{EnvironmentMap, EnvMapConfig};

// Create environment map from HDR data
let env_map = EnvironmentMap::new(width, height, hdr_data)?;

// Generate IBL textures
let irradiance = env_map.generate_irradiance_map(32, 1024);
let prefiltered = env_map.generate_prefiltered_map(128, 0.5, 256);

// Upload to GPU
let env_texture = env_map.upload_to_gpu(&device, &queue);

// Sample environment programmatically
let color = env_map.sample(direction);
```

## Usage Guide

### 1. Environment Map Creation

Create synthetic or load real environment maps:

```python
# Option 1: Create test environment
env = envmap.EnvironmentMap.create_test_envmap(256)

# Option 2: Load from HDR file (future)
# env = envmap.load_environment_map("studio.hdr")

# Validate the environment data
validation = envmap.validate_environment_map(env)
if not validation['valid']:
    print("Issues:", validation['errors'])
```

### 2. Roughness Testing

Test roughness monotonicity for IBL validation:

```python
# Test different roughness levels
roughness_values = [0.1, 0.3, 0.5, 0.7, 0.9]
luminances = envmap.compute_roughness_luminance_series(env, roughness_values)

# Verify monotonic decrease (higher roughness = lower luminance)
for r, l in zip(roughness_values, luminances):
    print(f"Roughness {r:.1f}: Luminance {l:.4f}")

# Check key values for acceptance criteria
l_01, l_05, l_09 = luminances[0], luminances[2], luminances[4]
assert l_01 > l_05 > l_09, "Roughness monotonicity failed"
```

### 3. Directional Sampling

Sample environment lighting in different directions:

```python
# Sample various directions
directions = {
    'up': [0, 1, 0],
    'down': [0, -1, 0],
    'forward': [0, 0, 1],
    'right': [1, 0, 0]
}

for name, direction in directions.items():
    color = env.sample_direction(np.array(direction, dtype=np.float32))
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    print(f"{name}: RGB{tuple(color)} L={luminance:.3f}")
```

## Shader Integration

### WGSL Environment Mapping

```wgsl
// Environment map uniforms and textures
@group(1) @binding(0) var env_texture: texture_2d<f32>;
@group(1) @binding(1) var env_sampler: sampler;
@group(1) @binding(2) var irradiance_texture: texture_2d<f32>;
@group(1) @binding(3) var irradiance_sampler: sampler;

// Convert 3D direction to equirectangular UV
fn direction_to_uv(direction: vec3<f32>) -> vec2<f32> {
    let phi = atan2(direction.z, direction.x);
    let theta = acos(direction.y);
    
    let u = (phi / (2.0 * 3.14159265359) + 0.5) % 1.0;
    let v = theta / 3.14159265359;
    
    return vec2<f32>(u, v);
}

// Sample environment with roughness-based mip level
fn sample_environment_lod(direction: vec3<f32>, roughness: f32) -> vec3<f32> {
    let uv = direction_to_uv(direction);
    let mip_level = roughness * 8.0; // Map roughness to mip levels
    return textureSampleLevel(env_texture, env_sampler, uv, mip_level).rgb;
}

// Sample irradiance for diffuse lighting
fn sample_irradiance(normal: vec3<f32>) -> vec3<f32> {
    let uv = direction_to_uv(normal);
    return textureSample(irradiance_texture, irradiance_sampler, uv).rgb;
}

// Fragment shader with IBL
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let world_normal = normalize(input.world_normal);
    let view_direction = normalize(camera_position.xyz - input.world_position);
    let reflection_direction = reflect(-view_direction, world_normal);
    
    // Sample environment lighting
    let irradiance = sample_irradiance(world_normal);
    let reflection = sample_environment_lod(reflection_direction, material.roughness);
    
    // Combine diffuse and specular IBL
    let diffuse_ibl = base_color * irradiance;
    let specular_ibl = reflection * fresnel_factor;
    
    return vec4<f32>(diffuse_ibl + specular_ibl, 1.0);
}
```

## Environment Map Formats

### Equirectangular Projection

Environment maps use equirectangular (latitude-longitude) projection:

- **Horizontal**: 360° longitude mapping to texture U coordinate [0,1]
- **Vertical**: 180° latitude mapping to texture V coordinate [0,1]  
- **Poles**: North pole at V=0, south pole at V=1
- **Prime meridian**: At U=0.5, wrapping at edges

### HDR Data Format

- **Channels**: RGB float values for high dynamic range
- **Range**: [0, ∞) for realistic lighting values
- **Precision**: float32 for GPU compatibility
- **Alpha**: Unused (can store material masks)

### Coordinate System

Forge3D uses right-handed coordinate system:

- **X-axis**: Right (+X = east)
- **Y-axis**: Up (+Y = north pole)
- **Z-axis**: Forward (+Z = forward from prime meridian)

## Validation and Testing

### Roughness Monotonicity

Validate that roughness correctly affects lighting:

```python
def test_roughness_monotonicity(env_map):
    roughness_values = [0.1, 0.5, 0.9]
    luminances = envmap.compute_roughness_luminance_series(env_map, roughness_values)
    
    # Acceptance criteria: L(0.1) > L(0.5) > L(0.9)
    l_01, l_05, l_09 = luminances[0], luminances[1], luminances[2]
    
    assert l_01 > l_05, f"L(0.1)={l_01:.4f} not > L(0.5)={l_05:.4f}"
    assert l_05 > l_09, f"L(0.5)={l_05:.4f} not > L(0.9)={l_09:.4f}"
    
    print("PASS: Roughness monotonicity validated")
```

### Histogram Changes

Verify different roughness values produce different luminance distributions:

```python
import matplotlib.pyplot as plt

def test_histogram_changes(env_map):
    # Sample environment at different roughness levels
    directions = generate_sphere_samples(100)
    
    luminances_smooth = []
    luminances_rough = []
    
    for direction in directions:
        # Simulate different roughness sampling
        color_smooth = env_map.sample_direction(direction)  # Low roughness
        color_rough = sample_with_noise(env_map, direction, 0.1)  # High roughness
        
        luminances_smooth.append(compute_luminance(color_smooth))
        luminances_rough.append(compute_luminance(color_rough))
    
    # Compare histograms
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(luminances_smooth, bins=20, alpha=0.7, label='Roughness 0.1')
    plt.subplot(1, 2, 2) 
    plt.hist(luminances_rough, bins=20, alpha=0.7, label='Roughness 0.9')
    
    # Verify statistical difference
    from scipy.stats import ks_2samp
    statistic, p_value = ks_2samp(luminances_smooth, luminances_rough)
    assert p_value < 0.05, "Histograms should be statistically different"
```

### Environment Data Validation

```python
def validate_environment_comprehensive(env_map):
    validation = envmap.validate_environment_map(env_map)
    
    # Check basic validity
    assert validation['valid'], f"Validation errors: {validation['errors']}"
    
    # Check statistics
    stats = validation['statistics']
    assert stats['min_value'] >= 0.0, "Environment should not have negative values"
    assert stats['max_value'] < 1000.0, "Environment values seem unreasonably high"
    assert 0.001 < stats['mean_value'] < 10.0, "Environment mean in reasonable range"
    
    # Memory usage check
    assert stats['memory_mb'] < 100.0, f"Environment uses too much memory: {stats['memory_mb']:.1f} MB"
    
    print("PASS: Environment validation complete")
```

## Performance Considerations

### Memory Usage

- **Environment maps**: Typical size 512×256×3×4 = 1.5 MB for float32
- **Mip chain**: Additional ~33% memory for full mip pyramid
- **Irradiance maps**: Small (32×16) for diffuse lighting
- **Specular maps**: Medium (256×128) for reflection detail

### Rendering Performance

- **Preprocessing**: Generate irradiance/specular maps offline when possible
- **GPU upload**: Use compressed formats (BC6H) for production
- **Sampling**: Bilinear filtering sufficient for most cases
- **LOD**: Use roughness-based mip selection for specular reflections

### Optimization Tips

1. **Precompute** irradiance and specular maps on CPU
2. **Use mipmaps** for environment textures with linear filtering
3. **Compress textures** using BC6H for HDR environments
4. **Cache** frequently used environment maps
5. **Profile** shader performance with different environment sizes

## Examples

### Basic Environment Mapping

```bash
python examples/environment_mapping.py --headless --out out/environment_demo.png
```

This example:
- Creates synthetic HDR environment map
- Tests roughness monotonicity (L(0.1) > L(0.5) > L(0.9))
- Renders environment-lit spheres with different materials
- Saves demonstration image and environment texture

### Custom Environment Loading

```python
import forge3d.envmap as envmap

# Create custom environment
def create_sunset_environment():
    size = 512
    data = np.zeros((size, size//2, 3), dtype=np.float32)
    
    for y in range(size//2):
        for x in range(size):
            # Create sunset-like gradient
            u = x / size
            v = y / (size//2)
            
            # Sky gradient
            sun_dir = 0.8  # Sun position
            sun_intensity = max(0, 1 - abs(u - sun_dir) * 5)
            
            r = 1.0 + sun_intensity * 2  # Warm highlights
            g = 0.7 + sun_intensity * 1.5
            b = 0.3 + v * 0.4  # Blue sky gradient
            
            data[y, x] = [r, g, b]
    
    return envmap.EnvironmentMap(size, size//2, data)

# Use custom environment
sunset_env = create_sunset_environment()
validation = envmap.validate_environment_map(sunset_env)
print(f"Sunset environment: {validation}")
```

## Troubleshooting

### Common Issues

1. **Invalid HDR values**: Check for NaN, infinite, or negative values
2. **Memory issues**: Large environments may exceed GPU memory limits  
3. **Coordinate system**: Verify Y-up, right-handed coordinate convention
4. **Sampling artifacts**: Use sufficient sample counts for Monte Carlo integration
5. **Performance drops**: Profile GPU memory bandwidth and texture cache usage

### Debug Techniques

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check environment statistics
validation = envmap.validate_environment_map(env)
print(f"Environment stats: {validation['statistics']}")

# Sample test directions
test_dirs = [[0,1,0], [0,-1,0], [1,0,0], [-1,0,0], [0,0,1], [0,0,-1]]
for i, direction in enumerate(test_dirs):
    color = env.sample_direction(np.array(direction, dtype=np.float32))
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    print(f"Direction {i}: {color} -> L={luminance:.4f}")

# Validate roughness behavior  
rough_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
luminances = envmap.compute_roughness_luminance_series(env, rough_vals)
for r, l in zip(rough_vals, luminances):
    print(f"Roughness {r:.2f}: Luminance {l:.6f}")
```

## Feature Flags

Environment mapping requires appropriate feature flags:

```toml
[dependencies]
forge3d = { features = ["enable-ibl", "enable-environment-mapping"] }
```

## Cross-Platform Support

Environment mapping works consistently across:

- **Windows**: DirectX 12 + HLSL compilation
- **Linux**: Vulkan + SPIR-V compilation  
- **macOS**: Metal + MSL compilation

All platforms use identical WGSL shaders ensuring consistent lighting results across different graphics backends.
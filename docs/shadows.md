# Cascaded Shadow Maps (CSM) with PCF Filtering

## Overview

forge3d provides a comprehensive Cascaded Shadow Maps (CSM) implementation for high-quality directional light shadows. The system uses multiple shadow map cascades to provide detailed shadows across large view distances, with Percentage-Closer Filtering (PCF) for smooth shadow edges.

## Key Features

- **Multi-Cascade System**: Up to 4 cascade levels for optimal shadow quality across view distance
- **PCF Filtering**: 1x1, 3x3, 5x5, 7x7, and Poisson disk sampling for smooth shadow edges
- **Automatic Cascade Splitting**: Practical split scheme balancing uniform and logarithmic distribution
- **Bias Management**: Depth bias and slope-scaled bias to prevent shadow acne and light leaking
- **Debug Visualization**: Color-coded cascade visualization for debugging and optimization
- **Memory Efficient**: Texture array storage with configurable resolution per cascade

## Quick Start

```python
import forge3d.shadows as shadows

# Create shadow system with medium quality preset
config = shadows.get_preset_config('medium_quality')
renderer = shadows.ShadowRenderer(800, 600, config)

# Configure directional light
light = shadows.DirectionalLight(
    direction=(-0.4, -0.8, -0.5),  # Angled sunlight
    color=(1.0, 0.95, 0.8),        # Warm sunlight color
    intensity=3.0
)
renderer.set_light(light)

# Set camera position
renderer.set_camera(
    position=(10.0, 5.0, 10.0),
    target=(0.0, 0.0, 0.0),
    fov_y_degrees=60.0
)

# Create test scene and render with shadows
scene = shadows.create_test_scene()
shadow_image = renderer.render_with_shadows(scene)
```

## Core Concepts

### Cascaded Shadow Maps

CSM solves the common shadow mapping problem of limited resolution across large view distances by using multiple shadow maps (cascades) at different distances:

- **Cascade 0**: Nearest to camera, highest detail
- **Cascade 1**: Medium distance, moderate detail  
- **Cascade 2**: Far distance, lower detail
- **Cascade 3**: Furthest distance, lowest detail

Each cascade covers a specific depth range in the camera's view frustum, automatically selected during rendering.

### Cascade Splitting

The system uses a practical split scheme that balances uniform and logarithmic distribution:

```
distance[i] = λ × log_split[i] + (1-λ) × uniform_split[i]
```

Where:
- **λ = 0.0**: Uniform splits (equal distance intervals)
- **λ = 1.0**: Logarithmic splits (exponential distance intervals)
- **λ = 0.5**: Balanced approach (default)

### PCF Filtering

Percentage-Closer Filtering provides soft shadow edges by sampling multiple points around each pixel:

| Kernel Size | Samples | Quality | Performance |
|-------------|---------|---------|-------------|
| 1x1 | 1 | Hard edges | Fastest |
| 3x3 | 9 | Soft edges | Fast |
| 5x5 | 25 | Smooth edges | Medium |
| 7x7 | 49 | Very smooth | Slow |
| Poisson | 16 | High quality | Medium |

## API Reference

### DirectionalLight Class

```python
class DirectionalLight:
    def __init__(self,
                 direction=(0.0, -1.0, 0.3),
                 color=(1.0, 1.0, 1.0),
                 intensity=3.0,
                 cast_shadows=True):
        """
        Create directional light for shadow casting.
        
        Args:
            direction: Light direction (pointing towards light), normalized automatically
            color: RGB light color [0.0, 10.0]
            intensity: Light intensity multiplier [0.0, ∞)
            cast_shadows: Enable shadow casting
        """
```

### CsmConfig Class

```python
class CsmConfig:
    def __init__(self,
                 cascade_count=4,
                 shadow_map_size=2048,
                 camera_far=1000.0,
                 camera_near=0.1,
                 lambda_factor=0.5,
                 depth_bias=0.0001,
                 slope_bias=0.001,
                 pcf_kernel_size=3):
        """
        Configure cascaded shadow maps.
        
        Args:
            cascade_count: Number of cascade levels [1-4]
            shadow_map_size: Resolution per cascade (power of 2: 512-4096)
            camera_far: Camera far clipping plane
            camera_near: Camera near clipping plane  
            lambda_factor: Split scheme blend [0.0-1.0]
            depth_bias: Fixed depth bias to prevent shadow acne
            slope_bias: Slope-scaled bias for angled surfaces
            pcf_kernel_size: PCF filter size (1, 3, 5, 7)
        """
```

### ShadowRenderer Class

```python
class ShadowRenderer:
    def __init__(self, width, height, config=None):
        """Create shadow-aware renderer."""
    
    def set_camera(self, position, target, up=(0,1,0), fov_y_degrees=45.0):
        """Configure camera parameters."""
    
    def set_light(self, light):
        """Set directional light for shadow casting."""
    
    def enable_debug_visualization(self, enabled=True):
        """Enable cascade debug colors."""
    
    def render_with_shadows(self, scene_data):
        """Render scene with shadow mapping."""
    
    def get_shadow_stats(self):
        """Get shadow system statistics."""
```

### Preset Configurations

```python
# Quality presets for common scenarios
def get_preset_config(quality):
    """
    Get preset shadow configuration.
    
    Args:
        quality: 'low_quality', 'medium_quality', 'high_quality', 'ultra_quality'
    
    Returns:
        Pre-configured CsmConfig instance
    """

# Available presets:
PRESET_CONFIGS = {
    'low_quality': CsmConfig(
        cascade_count=2,
        shadow_map_size=1024,
        pcf_kernel_size=1
    ),
    'medium_quality': CsmConfig(
        cascade_count=3,
        shadow_map_size=2048,
        pcf_kernel_size=3
    ),
    'high_quality': CsmConfig(
        cascade_count=4,
        shadow_map_size=2048,
        pcf_kernel_size=5
    ),
    'ultra_quality': CsmConfig(
        cascade_count=4,
        shadow_map_size=3072,  # Maximum within 256 MiB constraint
        pcf_kernel_size=7
    )
}
```

### Validation and Debugging

```python
def validate_csm_setup(config, light, camera_near, camera_far):
    """
    Validate CSM configuration and provide recommendations.
    
    Returns:
        dict: {
            'valid': bool,                    # Overall validity
            'errors': [str],                  # Error messages
            'warnings': [str],                # Warning messages  
            'recommendations': [str],         # Optimization suggestions
            'memory_estimate_mb': float       # Memory usage estimate
        }
    """

def compare_shadow_techniques():
    """
    Compare performance of different shadow filtering techniques.
    
    Returns:
        dict: Technique name -> performance score [0.0-1.0]
    """
```

## Usage Patterns

### Basic Shadow Setup

```python
# Simple shadow setup for standard scenes
config = shadows.CsmConfig(
    cascade_count=4,
    shadow_map_size=2048,
    pcf_kernel_size=3
)

renderer = shadows.ShadowRenderer(1920, 1080, config)

# Typical outdoor lighting
light = shadows.DirectionalLight(
    direction=(-0.3, -0.8, -0.5),  # Sun angle
    color=(1.0, 0.95, 0.8),        # Warm sunlight
    intensity=3.0
)
renderer.set_light(light)
```

### Performance Optimization

```python
# For real-time applications, balance quality vs performance
if target_fps >= 60:
    config = shadows.get_preset_config('medium_quality')
elif target_fps >= 30:
    config = shadows.get_preset_config('high_quality')
else:
    config = shadows.get_preset_config('ultra_quality')

# Validate configuration
validation = shadows.validate_csm_setup(config, light, 0.1, 1000.0)
if validation['memory_estimate_mb'] > available_memory_mb:
    # Reduce shadow map size
    config.shadow_map_size = min(config.shadow_map_size, 1024)
```

### Debug Visualization

```python
# Enable cascade debug colors to visualize coverage
renderer.enable_debug_visualization(True)
debug_image = renderer.render_with_shadows(scene)

# Colors indicate cascade levels:
# Red = Cascade 0 (nearest, highest detail)
# Green = Cascade 1 (medium distance)
# Blue = Cascade 2 (far distance)
# Yellow = Cascade 3 (furthest distance)

# Check shadow statistics
stats = renderer.get_shadow_stats()
print(f"Memory usage: {stats.memory_usage / (1024*1024):.1f}MB")
print(f"Split distances: {stats.split_distances}")
print(f"Texel sizes: {stats.texel_sizes}")
```

### Custom Scene Creation

```python
# Create scene optimized for shadow demonstration
def create_shadow_test_scene():
    scene = {
        'ground': create_ground_plane(size=50.0),
        'objects': []
    }
    
    # Add various shadow casters
    for i in range(8):
        angle = 2.0 * np.pi * i / 8
        x = 15.0 * np.cos(angle)
        z = 15.0 * np.sin(angle)
        height = 2.0 + np.sin(angle * 3)
        
        scene['objects'].append({
            'type': 'box',
            'position': (x, height/2, z),
            'size': (1.0, height, 1.0),
            'geometry': create_box_geometry(x, height/2, z, 1.0, height, 1.0)
        })
    
    return scene
```

## Best Practices

### Shadow Quality Guidelines

1. **Cascade Count**:
   - 2 cascades: Basic scenes, performance critical
   - 3 cascades: Standard quality for most applications  
   - 4 cascades: High quality for cinematic rendering

2. **Shadow Map Resolution**:
   - 1024x1024: Mobile/low-end hardware
   - 2048x2048: Standard desktop quality
   - 4096x4096: High-end/cinematic quality

3. **PCF Kernel Size**:
   - 1x1: Hard shadows, maximum performance
   - 3x3: Soft shadows, good balance
   - 5x5: Smooth shadows, moderate performance impact
   - 7x7: Very smooth shadows, performance cost

### Bias Tuning

Shadow artifacts require careful bias tuning:

```python
# For thin objects or detailed surfaces
config = shadows.CsmConfig(
    depth_bias=0.0001,      # Small fixed bias
    slope_bias=0.0005       # Moderate slope scaling
)

# For chunky geometry or distance scenes  
config = shadows.CsmConfig(
    depth_bias=0.0005,      # Larger fixed bias
    slope_bias=0.002        # More aggressive slope scaling
)
```

### Memory Management

```python
# Calculate memory usage before creating system
def estimate_shadow_memory(config):
    bytes_per_texel = 4  # 32-bit depth
    texels_per_cascade = config.shadow_map_size ** 2
    total_bytes = bytes_per_texel * texels_per_cascade * config.cascade_count
    return total_bytes / (1024 * 1024)  # Convert to MB

memory_mb = estimate_shadow_memory(config)
if memory_mb > gpu_memory_limit:
    # Reduce resolution or cascade count
    config.shadow_map_size = min(config.shadow_map_size // 2, 1024)
```

### Light Setup

```python
# Outdoor scenes - sun lighting
sun_light = shadows.DirectionalLight(
    direction=(-0.3, -0.8, -0.5),  # Typical sun angle
    color=(1.0, 0.95, 0.8),        # Warm daylight
    intensity=3.0
)

# Indoor scenes - window lighting
window_light = shadows.DirectionalLight(
    direction=(-0.1, -0.9, 0.2),   # Steep angle from window
    color=(0.9, 0.95, 1.0),        # Cool daylight
    intensity=2.0
)

# Dramatic lighting - low sun
dramatic_light = shadows.DirectionalLight(
    direction=(-0.7, -0.3, -0.2),  # Low, angled sun
    color=(1.0, 0.7, 0.3),         # Orange/red sunset
    intensity=4.0
)
```

## Performance Considerations

### GPU Memory Usage

CSM memory usage scales with cascade count and resolution:

```
Memory = cascade_count × (shadow_map_size²) × 4 bytes
```

Examples:
- 4 cascades × 2048² × 4 bytes = 64 MB
- 3 cascades × 1024² × 4 bytes = 12 MB  
- 2 cascades × 4096² × 4 bytes = 128 MB

### Rendering Performance

Shadow rendering involves multiple passes:

1. **Shadow Pass**: Render scene geometry to each cascade
2. **Main Pass**: Render scene with shadow sampling
3. **PCF Filtering**: Multiple shadow map samples per pixel

Performance factors:
- **Scene complexity**: More geometry = slower shadow passes
- **PCF kernel size**: Larger kernels = more samples = slower
- **Cascade count**: More cascades = more shadow passes
- **Shadow map resolution**: Higher resolution = more memory bandwidth

### Optimization Strategies

```python
# Dynamic quality adjustment based on performance
class AdaptiveShadowQuality:
    def __init__(self):
        self.target_frame_time = 1.0 / 60.0  # 60 FPS
        self.current_config = shadows.get_preset_config('high_quality')
    
    def adjust_quality(self, measured_frame_time):
        if measured_frame_time > self.target_frame_time * 1.2:
            # Reduce quality
            if self.current_config.pcf_kernel_size > 1:
                self.current_config.pcf_kernel_size = max(1, 
                    self.current_config.pcf_kernel_size - 2)
            elif self.current_config.shadow_map_size > 1024:
                self.current_config.shadow_map_size //= 2
            elif self.current_config.cascade_count > 2:
                self.current_config.cascade_count -= 1
        
        elif measured_frame_time < self.target_frame_time * 0.8:
            # Increase quality
            if self.current_config.cascade_count < 4:
                self.current_config.cascade_count += 1
            elif self.current_config.shadow_map_size < 4096:
                self.current_config.shadow_map_size *= 2
            elif self.current_config.pcf_kernel_size < 7:
                self.current_config.pcf_kernel_size += 2
```

## Integration with Other Systems

### PBR Materials

CSM shadows integrate seamlessly with PBR materials:

```python
# PBR material with shadow support  
import forge3d.pbr as pbr

material = pbr.PbrMaterial(
    base_color=(0.7, 0.4, 0.2, 1.0),
    metallic=1.0,
    roughness=0.2
)

# Shadow calculations automatically account for material properties
# No additional configuration needed
```

### Environment Mapping

Combine CSM with IBL for complete lighting:

```python
import forge3d.envmap as envmap

# Environment lighting for ambient/specular
env_map = envmap.EnvironmentMap.load_hdr("environment.hdr")

# CSM for direct light shadows
csm_config = shadows.get_preset_config('high_quality')
shadow_renderer = shadows.ShadowRenderer(1920, 1080, csm_config)

# Render with both systems active
combined_image = render_with_env_and_shadows(scene, env_map, shadow_renderer)
```

### HDR Rendering

CSM works with HDR rendering for realistic lighting:

```python
import forge3d.hdr as hdr

# HDR off-screen rendering
hdr_renderer = hdr.HdrRenderer(1920, 1080, format='rgba16float')

# Shadow system provides linear lighting values
shadow_contribution = shadow_renderer.render_with_shadows(scene)

# Combine and tone map
final_image = hdr_renderer.tone_map(shadow_contribution, 'aces')
```

## Troubleshooting

### Common Issues

**1. Shadow Acne (surface self-shadowing)**
```python
# Increase depth bias
config.depth_bias = 0.001
config.slope_bias = 0.005
```

**2. Light Leaking (shadows too thick)**
```python
# Reduce depth bias
config.depth_bias = 0.0001
config.slope_bias = 0.0005
```

**3. Shadow Aliasing (jagged edges)**
```python
# Increase shadow map resolution
config.shadow_map_size = 4096
# Or enable PCF filtering
config.pcf_kernel_size = 5
```

**4. Performance Issues**
```python
# Reduce cascade count or resolution
config = shadows.get_preset_config('low_quality')
# Or disable PCF
config.pcf_kernel_size = 1
```

**5. Incorrect Shadow Coverage**
```python
# Enable debug visualization to check cascades
renderer.enable_debug_visualization(True)
debug_image = renderer.render_with_shadows(scene)
# Adjust lambda factor for better split distribution
config.lambda_factor = 0.7  # More logarithmic
```

### Validation Warnings

The validation system provides specific guidance:

```python
validation = shadows.validate_csm_setup(config, light, 0.1, 1000.0)

for warning in validation['warnings']:
    print(f"Warning: {warning}")

for recommendation in validation['recommendations']:
    print(f"Recommendation: {recommendation}")
```

Common warnings:
- "Low shadow map resolution may cause aliasing"
- "Large PCF kernel may impact performance"  
- "High memory usage: 128.0MB"
- "Large far/near ratio may cause precision issues"
- "Horizontal lights may cause long shadows"

## Examples

See `examples/shadow_demo.py` for comprehensive demonstrations including:

- Multi-cascade visualization with debug colors
- PCF filtering quality comparison
- Light angle effects on shadow quality
- Performance benchmarking across quality settings
- Memory usage analysis and optimization

The example generates multiple output images showing different aspects of the CSM system and provides detailed performance metrics.

## Implementation Notes

### GPU Shader Integration

CSM is implemented across `src/shaders/shadow_map.wgsl` and `src/shaders/shadow_sample.wgsl` with:

- **Shadow vertex shader**: Transforms geometry to light space per cascade
- **Shadow fragment shader**: Writes depth values to shadow maps
- **PCF sampling functions**: Multiple filtering algorithms (basic, grid, Poisson)
- **Cascade selection**: Automatic selection based on view depth
- **Debug visualization**: Color-coded cascade regions

### Memory Layout

Shadow data is efficiently packed for GPU consumption:

```rust
#[repr(C)]
struct CsmUniforms {
    light_direction: [f32; 4],          // 16 bytes
    light_view: [[f32; 4]; 4],          // 64 bytes
    cascades: [ShadowCascade; 4],       // 80 bytes × 4
    cascade_count: u32,                 // 4 bytes
    pcf_kernel_size: u32,               // 4 bytes
    depth_bias: f32,                    // 4 bytes
    slope_bias: f32,                    // 4 bytes
    shadow_map_size: f32,               // 4 bytes
    debug_mode: u32,                    // 4 bytes
    _padding: [f32; 2],                 // 8 bytes
}
```

### Feature Integration

CSM integrates with other forge3d systems:

- **Material System**: Shadow attenuation affects PBR lighting calculations
- **Scene Management**: Automatic frustum culling for shadow casters
- **Camera System**: Cascade splits based on camera parameters
- **Memory Manager**: Efficient texture array allocation and reuse

This provides a solid foundation for high-quality real-time shadows in forge3d applications.
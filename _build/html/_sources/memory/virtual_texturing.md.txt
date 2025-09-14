# Virtual Texture Streaming

The virtual texture streaming system allows efficient handling of very large textures (up to 16K×16K) that don't fit in GPU memory by streaming tiles on demand based on camera position and view frustum.

## Overview

Virtual texturing solves the problem of working with textures that exceed available GPU memory by:

- **Dividing large textures into tiles** (typically 256×256 pixels each)
- **Loading only visible tiles** into a resident cache based on camera position
- **Streaming new tiles** as the camera moves through the scene
- **Managing memory automatically** with LRU eviction of unused tiles

This enables applications to work with massive textures while maintaining predictable memory usage and good performance.

## Key Components

### VirtualTextureSystem

The main interface for managing virtual texture streaming:

```python
import forge3d.streaming as streaming

# Initialize with 256MB memory budget
vt_system = streaming.VirtualTextureSystem(device, max_memory_mb=256)

# Load a large texture for streaming
texture = vt_system.load_texture("large_terrain.ktx2")
```

### Tile Management

The system automatically manages tiles through:

- **Page table**: Maps virtual texture coordinates to physical atlas locations
- **Tile cache**: LRU cache of resident tiles with reference counting
- **Atlas texture**: Physical GPU texture storing cached tile data
- **Feedback buffer**: GPU→CPU communication for tile visibility analysis

### Camera-Based Streaming

Tile loading is driven by camera position and view parameters:

```python
# Update streaming based on camera movement
camera_pos = (1000, 2000, 500)  # World coordinates
result = vt_system.update_streaming(camera_pos, view_matrix, proj_matrix)

print(f"Loaded {result['tiles_loaded']} new tiles")
print(f"Evicted {result['tiles_evicted']} old tiles")
```

## Configuration

### Memory Budget

Set the memory budget based on your application's requirements:

```python
# Conservative budget for mobile/integrated GPUs
vt_system = streaming.VirtualTextureSystem(device, max_memory_mb=128)

# Higher budget for dedicated GPUs
vt_system = streaming.VirtualTextureSystem(device, max_memory_mb=512)
```

### Tile Size

Choose tile size based on texture characteristics and performance:

```python
# Smaller tiles (more granular streaming, higher overhead)
vt_system = streaming.VirtualTextureSystem(device, tile_size=128)

# Larger tiles (less overhead, coarser streaming)
vt_system = streaming.VirtualTextureSystem(device, tile_size=512)
```

**Recommendations:**
- **256×256**: Good default for most use cases
- **128×128**: Better for very detailed textures with fine viewing patterns
- **512×512**: Better for textures viewed from far distances

### Quality Settings

Tune streaming quality vs performance:

```python
# High quality (more aggressive loading)
vt_system.set_quality_settings(
    max_mip_bias=0.0,
    lod_scale=1.0,
    cache_priority_boost=2.0
)

# Performance focused (more conservative loading)
vt_system.set_quality_settings(
    max_mip_bias=1.0,
    lod_scale=0.8,
    cache_priority_boost=1.0
)
```

## Usage Patterns

### Basic Streaming Setup

```python
import forge3d.streaming as streaming
import numpy as np

# Initialize system
device = forge3d.get_device()
vt_system = streaming.VirtualTextureSystem(device, max_memory_mb=256)

# Load virtual texture
texture = vt_system.load_texture("assets/world_texture_8192x8192.ktx2")
print(f"Virtual texture: {texture.size[0]}×{texture.size[1]} pixels")
print(f"Tile grid: {texture.tile_count[0]}×{texture.tile_count[1]} tiles")

# Main loop
while running:
    # Update camera position
    camera_pos = get_camera_position()
    view_matrix = get_view_matrix()
    proj_matrix = get_projection_matrix()
    
    # Stream tiles based on camera
    result = vt_system.update_streaming(camera_pos, view_matrix, proj_matrix)
    
    # Render using virtual texture
    render_scene_with_virtual_texture(texture)
    
    # Monitor performance
    if frame % 60 == 0:  # Once per second
        stats = vt_system.get_statistics()
        print(f"Cache hit rate: {stats.cache_hit_rate:.1f}%")
        print(f"Memory usage: {stats.memory_utilization:.1f}%")
```

### Prefetching

For predictable camera movement, prefetch tiles ahead of time:

```python
# Prefetch a region around the camera
camera_x, camera_y = get_camera_world_position_2d()
region_size = 1024  # pixels

success = vt_system.prefetch_region(
    texture,
    camera_x - region_size // 2,
    camera_y - region_size // 2,
    region_size,
    region_size,
    mip_level=0
)
```

### Multiple Textures

Manage multiple virtual textures with shared memory budget:

```python
# Load multiple textures
terrain_diffuse = vt_system.load_texture("terrain_diffuse_8192.ktx2")
terrain_normal = vt_system.load_texture("terrain_normal_8192.ktx2") 
terrain_material = vt_system.load_texture("terrain_material_8192.ktx2")

# Update streaming for all textures
vt_system.update_streaming(camera_pos, view_matrix, proj_matrix)

# Check memory distribution
memory_info = vt_system.get_memory_info()
print(f"Total budget: {memory_info['total_budget'] // 1024 // 1024} MB")
print(f"Memory used: {memory_info['used_memory'] // 1024 // 1024} MB")
```

## Performance Monitoring

### Statistics Tracking

Monitor streaming performance with built-in statistics:

```python
stats = vt_system.get_statistics()

print(f"Performance Metrics:")
print(f"  Cache hit rate: {stats.cache_hit_rate:.1f}%")
print(f"  Memory utilization: {stats.memory_utilization:.1f}%")
print(f"  Atlas utilization: {stats.atlas_utilization * 100:.1f}%")
print(f"  Active tiles: {stats.active_tiles}")
print(f"  Tiles loaded this session: {stats.tiles_loaded}")
print(f"  Tiles evicted this session: {stats.tiles_evicted}")
```

### Memory Analysis

Analyze memory requirements before creating the system:

```python
# Calculate memory requirements for a texture
requirements = streaming.calculate_memory_requirements(
    texture_width=8192,
    texture_height=8192,
    tile_size=256,
    bytes_per_pixel=4  # RGBA8
)

print(f"Full texture size: {requirements['full_texture_size'] // 1024 // 1024} MB")
print(f"Total tiles: {requirements['tile_count']}")
print(f"Memory per tile: {requirements['tile_memory_size'] // 1024} KB")
print(f"Recommended cache: {requirements['recommended_cache_size'] // 1024 // 1024} MB")
```

### Performance Estimation

Estimate streaming performance characteristics:

```python
# Estimate performance for a given setup
perf = streaming.estimate_streaming_performance(
    texture_size=(8192, 8192),
    tile_size=256,
    cache_size_mb=256,
    target_fps=60
)

print(f"Performance Estimates:")
print(f"  Cache capacity: {perf['cache_capacity_tiles']} tiles")
print(f"  Tiles per frame budget: {perf['tiles_per_frame_budget']}")
print(f"  Memory pressure: {perf['memory_pressure_factor'] * 100:.1f}%")
print(f"  Recommended prefetch distance: {perf['recommended_prefetch_distance']} tiles")
```

## Integration with Other Systems

### Staging Rings Integration

Virtual texture streaming automatically integrates with staging buffer rings for efficient GPU uploads:

```python
# The system automatically uses staging rings when available
import forge3d.memory as memory

# Initialize memory management systems
memory.init_memory_system(device, staging_memory_mb=64, pool_memory_mb=128)

# Virtual texture system will use staging rings for tile uploads
vt_system = streaming.VirtualTextureSystem(device, max_memory_mb=256)
```

### Compressed Texture Support

Virtual textures work with compressed formats for memory efficiency:

```python
# Load compressed virtual texture
texture = vt_system.load_texture(
    "terrain_bc7_8192.ktx2",
    preferred_format="BC7",  # Will use compression if available
    quality="high"
)

# Compression reduces memory usage by ~50-75%
stats = vt_system.get_statistics()
print(f"Effective memory reduction from compression: ~{100 - (stats.memory_used / stats.memory_limit) * 100:.0f}%")
```

## Shader Integration

### WGSL Shader Code

Virtual textures can be sampled in shaders using the page table lookup system:

```wgsl
// Virtual texture sampling function
fn sample_virtual_texture(
    virtual_coords: vec2<f32>,
    mip_level: f32,
    page_table: texture_2d<u32>,
    atlas_texture: texture_2d<f32>,
    sampler: sampler
) -> vec4<f32> {
    // Convert virtual coordinates to tile coordinates
    let tile_coords = virtual_coords * virtual_texture_size / tile_size;
    let tile_id = vec2<u32>(floor(tile_coords));
    
    // Look up physical atlas location
    let page_entry = textureLoad(page_table, tile_id, 0);
    let atlas_coords = unpack_atlas_coords(page_entry);
    
    // Sample from atlas texture
    let local_coords = fract(tile_coords);
    let atlas_uv = (atlas_coords + local_coords) / atlas_size;
    
    return textureSample(atlas_texture, sampler, atlas_uv);
}
```

### C++ Integration

For applications with C++ render loops:

```cpp
// Update streaming from C++ (via Python embedding)
PyObject* update_result = PyObject_CallMethod(
    vt_system, "update_streaming", "(fff)(O)(O)", 
    camera_x, camera_y, camera_z,
    view_matrix_py, proj_matrix_py
);

// Extract results
int tiles_loaded = PyLong_AsLong(PyDict_GetItemString(update_result, "tiles_loaded"));
int tiles_evicted = PyLong_AsLong(PyDict_GetItemString(update_result, "tiles_evicted"));
```

## Best Practices

### Memory Budget Guidelines

- **Conservative**: 128-256 MB for mobile/integrated GPUs
- **Standard**: 256-512 MB for mid-range dedicated GPUs  
- **Aggressive**: 512+ MB for high-end GPUs with abundant VRAM

### Tile Size Selection

- **Small textures** (<2K): Use larger tiles (512×512) to reduce overhead
- **Medium textures** (2K-8K): Use standard tiles (256×256)
- **Large textures** (8K+): Use smaller tiles (128×128) for finer control

### Streaming Strategy

1. **Update frequency**: Call `update_streaming()` every 3-5 frames, not every frame
2. **Camera prediction**: Prefetch in the direction of camera movement
3. **LOD management**: Use appropriate mip levels based on distance to camera
4. **Memory pressure**: Monitor cache hit rates and adjust quality settings

### Performance Optimization

```python
# Optimize for your use case
if rendering_static_scenes:
    # Reduce update frequency, increase cache size
    vt_system.set_quality_settings(cache_priority_boost=3.0)
elif fast_camera_movement:
    # More aggressive prefetching
    vt_system.set_quality_settings(max_mip_bias=0.5, lod_scale=1.2)
elif memory_constrained:
    # More conservative loading
    vt_system.set_quality_settings(lod_scale=0.6)
```

## Troubleshooting

### Common Issues

**High cache miss rates:**
- Increase memory budget
- Reduce tile size for finer granularity
- Implement prefetching for predictable movement

**Memory pressure warnings:**
- Reduce texture resolution or use more compression
- Increase mip bias to use lower detail levels
- Reduce cache priority boost

**Poor streaming performance:**
- Check staging ring configuration
- Ensure textures use compressed formats
- Profile GPU memory bandwidth usage

### Debug Information

Enable detailed debugging:

```python
# Get detailed memory breakdown
memory_info = vt_system.get_memory_info()
for key, value in memory_info.items():
    if 'memory' in key:
        print(f"{key}: {value // 1024 // 1024} MB")
    else:
        print(f"{key}: {value}")

# Monitor tile loading patterns
stats = vt_system.get_statistics()
if stats.cache_hit_rate < 80.0:
    print("Warning: Low cache hit rate - consider increasing memory budget")

if stats.memory_utilization > 90.0:
    print("Warning: High memory pressure - consider reducing quality settings")
```

## API Reference

See the [streaming module documentation](../python/streaming.py) for complete API details including:

- `VirtualTextureSystem` class methods
- `VirtualTexture` properties and operations  
- `StreamingStats` performance metrics
- Utility functions for memory analysis
- Integration helpers for staging rings and compressed textures
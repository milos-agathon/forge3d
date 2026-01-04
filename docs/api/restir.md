# ReSTIR DI (Reservoir-based Spatio-Temporal Importance Resampling)

ReSTIR DI is an advanced lighting technique for efficiently handling scenes with many lights (thousands) while maintaining low variance in the lighting estimation. It uses reservoir sampling with temporal and spatial reuse to achieve significant variance reduction compared to traditional Multiple Importance Sampling (MIS).

## Overview

ReSTIR DI addresses the challenge of many-light rendering by:

1. **Reservoir Sampling**: Using weighted reservoir sampling to maintain one representative light sample per pixel
2. **Temporal Reuse**: Reusing samples from previous frames to increase effective sample count
3. **Spatial Reuse**: Sharing samples between neighboring pixels with similar geometry
4. **Alias Tables**: Efficient O(1) light sampling using Walker's alias method

## Key Benefits

- **Variance Reduction**: Target ≥40% variance reduction vs MIS-only at 64 spp
- **Scalability**: Performance independent of light count (after preprocessing)
- **Quality**: Maintains unbiased results with proper Jacobian corrections
- **GPU Friendly**: Designed for parallel execution on modern GPUs

## API Reference

### Python API

```python
from forge3d.lighting import RestirDI, RestirConfig, LightSample, LightType

# Create ReSTIR instance
config = RestirConfig(
    initial_candidates=32,      # Number of initial light candidates per pixel
    spatial_neighbors=4,        # Number of spatial neighbors for reuse
    spatial_radius=16.0,        # Spatial search radius in pixels
    temporal_neighbors=1,       # Number of temporal samples to reuse
    max_temporal_age=20,        # Maximum age for temporal samples
    bias_correction=True        # Enable bias correction for unbiased results
)

restir = RestirDI(config)
```

#### Adding Lights

```python
# Add individual lights
light_idx = restir.add_light(
    position=(x, y, z),
    intensity=1.5,
    light_type=LightType.POINT,
    weight=2.0  # Sampling weight
)

# Or set multiple lights at once
lights = [
    LightSample(position=(0, 5, 0), intensity=10.0, light_type=LightType.POINT),
    LightSample(position=(0, 1, 0), intensity=1.0, light_type=LightType.DIRECTIONAL)
]
weights = [5.0, 1.0]  # Higher weight = more likely to be sampled
restir.set_lights(lights, weights)
```

#### Rendering

```python
# Prepare G-buffer (from your rasterizer/raytracer)
g_buffer = {
    'depth': depth_array,       # (H, W) depth values
    'normal': normal_array,     # (H, W, 3) world-space normals
    'world_pos': pos_array      # (H, W, 3) world positions
}

# Render with ReSTIR
image = restir.render_frame(
    width=1920, height=1080,
    camera_params=camera_dict,
    g_buffer=g_buffer,
    output_format="rgba"
)
```

#### Variance Analysis

```python
# Compare against reference (e.g., MIS-only)
reduction = restir.calculate_variance_reduction(reference_image, restir_image)
print(f"Variance reduction: {reduction:.1f}%")  # Target: ≥40%
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_candidates` | 32 | Number of initial light samples per pixel |
| `temporal_neighbors` | 1 | Number of temporal samples to reuse |
| `spatial_neighbors` | 4 | Number of spatial neighbors for reuse |
| `spatial_radius` | 16.0 | Spatial search radius (pixels) |
| `max_temporal_age` | 20 | Maximum frame age for temporal reuse |
| `bias_correction` | True | Enable bias correction for unbiased results |
| `depth_threshold` | 0.1 | Depth similarity threshold for reuse |
| `normal_threshold` | 0.9 | Normal similarity threshold (cosine) |

## Technical Details

### Alias Table Construction

The alias table enables O(1) sampling from weighted light distributions using Walker's alias method:

```
Construction: O(n) time, O(n) space
Sampling: O(1) time per sample
```

### Reservoir Operations

Each pixel maintains a reservoir containing:
- Current light sample
- Weight sum (w_sum)
- Sample count (M)
- Final weight for shading

### Temporal Reuse

Temporal reuse propagates samples across frames using motion vectors:

1. **Geometry Validation**: Check depth and normal similarity
2. **Jacobian Calculation**: Account for visibility/geometric changes
3. **Reservoir Combination**: Merge temporal sample with current reservoir

### Spatial Reuse

Spatial reuse shares samples between neighboring pixels:

1. **Neighbor Selection**: Use structured/random patterns within radius
2. **Similarity Testing**: Validate geometric compatibility
3. **Multi-pass**: Multiple spatial passes for better quality

### Bias Correction

Bias correction ensures unbiased results by:
- Recomputing normalization factors
- Accounting for all possible sample sources
- Maintaining correct PDF estimates

## Performance Considerations

### Memory Usage

- **Light Storage**: ~40 bytes per light (position, direction, intensity, params)
- **Alias Table**: 8 bytes per light (probability + alias index)
- **Reservoirs**: ~64 bytes per pixel (sample + metadata)
- **G-buffers**: Standard requirements (depth, normal, position)

### GPU Memory Layout

```
Bind Group 0: ReSTIR parameters and light data
  - binding 0: RestirParams (uniform)
  - binding 1: Light array (storage, read)
  - binding 2: Alias table (storage, read)

Bind Group 1: Temporal data
  - binding 0: Previous reservoirs (storage, read)
  - binding 1: Current reservoirs (storage, read_write)
  - binding 2: Motion vectors (storage, read)

Bind Group 2: Spatial data
  - binding 0: Input reservoirs (storage, read)
  - binding 1: Output reservoirs (storage, read_write)
  - binding 2: G-buffer data (storage, read)
```

### Performance Tips

1. **Light Count**: Performance is largely independent of light count after alias table construction
2. **Spatial Radius**: Larger radius improves quality but increases cost
3. **Sample Count**: More initial candidates improve quality with linear cost increase
4. **Bias Correction**: Can be disabled for performance if bias is acceptable

## Limitations

1. **Dynamic Lights**: Light changes require alias table reconstruction
2. **Memory**: Requires additional GPU memory for reservoirs and temporal data
3. **Latency**: Temporal reuse introduces one frame of latency
4. **Complexity**: More complex than traditional sampling methods

## Integration Example

```python
import numpy as np
from forge3d.lighting import create_test_scene

# Create a test scene with many lights
restir = create_test_scene(
    num_lights=1000,
    scene_bounds=(20.0, 20.0, 10.0),
    intensity_range=(0.1, 5.0),
    seed=42
)

print(f"Created scene with {restir.num_lights} lights")

# Get statistics
stats = restir.get_statistics()
print(f"Configuration: {stats['config']}")

# Sample a light
light_idx, pdf = restir.sample_light(0.5, 0.3)
print(f"Sampled light {light_idx} with PDF {pdf:.4f}")
```

## References

- Bitterli, B., et al. "Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting." ACM TOG 2020.
- Walker, A. J. "An efficient method for generating discrete random variables with general distributions." ACM TOMS 1977.

## See Also

- [Path Tracing API](path_tracing.md)
- [GPU Memory Guide](../memory/gpu_memory_guide.rst)
- [WGSL Bind Group Layouts](wgsl_bind_group_layouts.md)
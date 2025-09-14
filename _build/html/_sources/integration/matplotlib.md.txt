# Matplotlib Integration

This document describes forge3d's integration with matplotlib for colormap handling, data normalization, and visualization display.

## Overview

forge3d provides seamless integration with matplotlib through three main modules:

- **`forge3d.adapters.mpl_cmap`**: Colormap and normalization adapters
- **`forge3d.helpers.mpl_display`**: Display helpers for RGBA buffers  
- **Optional dependency handling**: Graceful degradation when matplotlib is unavailable

## Installation

Matplotlib integration is optional. To enable it:

```bash
pip install matplotlib
```

forge3d will automatically detect matplotlib availability and enable integration features.

## Colormap Integration

### Basic Usage

Convert matplotlib colormaps to forge3d format:

```python
import forge3d as f3d
from forge3d.adapters.mpl_cmap import matplotlib_to_forge3d_colormap

# Convert matplotlib colormap to forge3d LUT
lut = matplotlib_to_forge3d_colormap('viridis')
print(lut.shape)  # (256, 4) - RGBA uint8

# Use with terrain rendering
renderer = f3d.Renderer(512, 512)
heightmap = create_heightmap()  # Your heightmap data
renderer.add_terrain(heightmap, (1.0, 1.0), 1.0, 'viridis')
```

### Supported Colormaps

All matplotlib colormaps are supported, including:

- **Sequential**: viridis, plasma, inferno, magma, cividis
- **Diverging**: coolwarm, bwr, seismic, RdYlBu
- **Qualitative**: tab10, Set1, Set2, tab20
- **Reversed variants**: Add '_r' suffix (e.g., 'viridis_r')

```python
from forge3d.adapters.mpl_cmap import get_matplotlib_colormap_names

# List all available colormaps
colormaps = get_matplotlib_colormap_names()
print(f"Available colormaps: {len(colormaps)}")
```

### Custom Colormap Objects

You can use matplotlib Colormap objects directly:

```python
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Create custom colormap
colors = ['red', 'yellow', 'green', 'blue']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

# Convert to forge3d format
lut = matplotlib_to_forge3d_colormap(custom_cmap, n_colors=128)
```

## Data Normalization

### Linear Normalization

```python
from forge3d.adapters.mpl_cmap import matplotlib_normalize

# Basic linear normalization
data = np.array([0, 25, 50, 75, 100])
normalized = matplotlib_normalize(data, vmin=0, vmax=100)
print(normalized)  # [0.0, 0.25, 0.5, 0.75, 1.0]

# Auto-range normalization
normalized_auto = matplotlib_normalize(data)  # Uses data.min() and data.max()
```

### Logarithmic Normalization

For data with wide dynamic ranges:

```python
from forge3d.adapters.mpl_cmap import LogNormAdapter

# Logarithmic normalization
data = np.array([1, 10, 100, 1000, 10000])
log_norm = LogNormAdapter(vmin=1, vmax=10000)
normalized = log_norm(data)
print(normalized)  # [0.0, 0.25, 0.5, 0.75, 1.0]

# Inverse transformation
recovered = log_norm.inverse(normalized)
print(np.allclose(recovered, data))  # True
```

### Power Law Normalization

For gamma correction and power-law scaling:

```python
from forge3d.adapters.mpl_cmap import PowerNormAdapter

# Power normalization (gamma correction)
data = np.array([0, 0.25, 0.5, 0.75, 1.0])
power_norm = PowerNormAdapter(gamma=2.2, vmin=0, vmax=1)
normalized = power_norm(data)

# Result is power-law scaled
```

### Boundary Normalization

For discrete color mapping:

```python
from forge3d.adapters.mpl_cmap import BoundaryNormAdapter

# Define discrete levels
boundaries = [0, 10, 20, 30, 40, 50]
boundary_norm = BoundaryNormAdapter(boundaries, ncolors=5)

# Data is mapped to discrete levels
data = np.array([5, 15, 25, 35, 45])
normalized = boundary_norm(data)
```

## Display Helpers

### Basic Display

Display forge3d RGBA buffers in matplotlib figures:

```python
import matplotlib.pyplot as plt
from forge3d.helpers.mpl_display import imshow_rgba

# Render scene
renderer = f3d.Renderer(512, 512)
rgba = renderer.render_triangle_rgba()

# Display in matplotlib
fig, ax = plt.subplots()
im = imshow_rgba(ax, rgba)
ax.set_title('forge3d Output')
plt.show()
```

### Custom Extents and DPI

```python
# Display with custom coordinate system
extent = (0, 100, 0, 75)  # (left, right, bottom, top)
im = imshow_rgba(ax, rgba, extent=extent, dpi=150)

ax.set_xlabel('Distance (m)')
ax.set_ylabel('Distance (m)')
```

### Subplot Comparisons

Display multiple images in subplots:

```python
from forge3d.helpers.mpl_display import imshow_rgba_subplots

# Create multiple renderings
rgba_list = [
    renderer1.render_rgba(),
    renderer2.render_rgba(),
    renderer3.render_rgba()
]

titles = ['Viridis', 'Plasma', 'Coolwarm']

# Create subplot comparison
fig, images = imshow_rgba_subplots(
    rgba_list, 
    titles=titles, 
    ncols=3,
    figsize=(12, 4)
)
plt.show()
```

### Saving Comparisons

Save multi-panel comparisons directly:

```python
from forge3d.helpers.mpl_display import save_rgba_comparison

save_rgba_comparison(
    rgba_list,
    'comparison.png',
    titles=titles,
    dpi=300
)
```

## Accuracy and Performance

### Colormap Accuracy

forge3d colormap conversion maintains high accuracy:

- **SSIM ≥ 0.999** vs matplotlib reference on 1024×32 color ramps
- **Max channel difference ≤ 1e-7** for linear normalization
- **Exact match** for boundary normalization

### Normalization Accuracy

All normalization adapters maintain parity with matplotlib:

```python
import matplotlib.colors as mcolors
import numpy as np

# Test accuracy
data = np.array([1, 10, 100, 1000])

# matplotlib reference
mpl_norm = mcolors.LogNorm(vmin=1, vmax=1000)
mpl_result = mpl_norm(data)

# forge3d adapter
forge3d_norm = LogNormAdapter(vmin=1, vmax=1000)
forge3d_result = forge3d_norm(data)

# Verify accuracy
max_diff = np.max(np.abs(forge3d_result - mpl_result))
print(f"Max difference: {max_diff:.2e}")  # Should be < 1e-7
```

### Zero-Copy Display

Display helpers optimize for zero-copy when possible:

- **C-contiguous uint8 arrays**: No copies for display
- **Float32 arrays**: Single copy for dtype conversion
- **Memory warnings**: Issued for non-contiguous arrays

```python
from forge3d.helpers.mpl_display import validate_rgba_array
import numpy as np

# Check array properties
rgba = renderer.render_rgba()
validated = validate_rgba_array(rgba)

print(f"C-contiguous: {rgba.flags['C_CONTIGUOUS']}")
print(f"Memory usage: {rgba.nbytes / 1024**2:.1f} MB")
```

## Error Handling and Validation

### Optional Dependency Handling

forge3d gracefully handles missing matplotlib:

```python
from forge3d.adapters.mpl_cmap import is_matplotlib_available

if is_matplotlib_available():
    # Use matplotlib features
    lut = matplotlib_to_forge3d_colormap('viridis')
else:
    # Fallback behavior
    print("Matplotlib not available, using built-in colormaps")
```

### Array Validation

Display helpers validate input arrays:

```python
# Invalid arrays raise clear errors
try:
    invalid_2d = np.zeros((100, 100))  # Missing channel dimension
    imshow_rgba(ax, invalid_2d)
except ValueError as e:
    print(f"Validation error: {e}")
    # "rgba must be 3D array (H, W, C), got 2D shape (100, 100)"
```

### Supported Formats

Display helpers support multiple formats:

- **RGB/RGBA**: Both 3 and 4 channel arrays
- **uint8/float32/float64**: Automatic conversion and validation
- **Value ranges**: 
  - uint8: [0, 255]
  - float: [0.0, 1.0] (values outside range trigger warnings)

## Examples

### Complete Terrain Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
import forge3d as f3d
from forge3d.adapters.mpl_cmap import matplotlib_normalize, LogNormAdapter
from forge3d.helpers.mpl_display import imshow_rgba

# Create synthetic DEM
size = 256
x, y = np.mgrid[0:size, 0:size]
heightmap = np.sin(x/20) * np.cos(y/15) + 0.5 * np.random.randn(size, size)
heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())

# Apply log normalization for enhanced contrast
log_norm = LogNormAdapter(vmin=0.01, vmax=1.0)
heightmap_log = log_norm(heightmap.clip(0.01, 1.0))

# Render with different colormaps
renderer = f3d.Renderer(512, 512)

colormaps = ['viridis', 'terrain', 'plasma']
rgba_scenes = []

for cmap in colormaps:
    renderer.add_terrain(heightmap_log.astype(np.float32), (1.0, 1.0), 2.0, cmap)
    rgba_scenes.append(renderer.render_rgba())

# Display comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (rgba, cmap) in enumerate(zip(rgba_scenes, colormaps)):
    extent = (0, 10, 0, 10)  # 10km x 10km area
    imshow_rgba(axes[i], rgba, extent=extent)
    axes[i].set_title(f'{cmap.title()} Colormap')
    axes[i].set_xlabel('Distance (km)')
    axes[i].set_ylabel('Distance (km)')

plt.tight_layout()
plt.show()
```

### Performance Comparison

```python
import time
from forge3d.helpers.mpl_display import imshow_rgba_subplots

# Generate test data
sizes = [(256, 256), (512, 512), (1024, 1024)]
rgba_arrays = []
timings = []

for h, w in sizes:
    # Create test pattern
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = (np.arange(h)[:, None] * 255 // h).astype(np.uint8)
    rgba[:, :, 1] = (np.arange(w)[None, :] * 255 // w).astype(np.uint8)
    rgba[:, :, 3] = 255
    
    # Time display operation
    fig, ax = plt.subplots()
    start = time.perf_counter()
    im = imshow_rgba(ax, rgba)
    elapsed = time.perf_counter() - start
    
    rgba_arrays.append(rgba)
    timings.append(elapsed)
    plt.close(fig)

# Display timing results
titles = [f'{h}×{w}\n{t*1000:.1f}ms' for (h, w), t in zip(sizes, timings)]

fig, images = imshow_rgba_subplots(
    rgba_arrays,
    titles=titles,
    ncols=3,
    figsize=(12, 4)
)
fig.suptitle('Display Performance Comparison')
plt.show()

print("Performance Results:")
for (h, w), timing in zip(sizes, timings):
    pixels = h * w
    rate = pixels / timing / 1e6
    print(f"  {h}×{w}: {timing*1000:.1f}ms ({rate:.1f} Mpix/s)")
```

## Best Practices

### Memory Efficiency

1. **Use C-contiguous arrays** for optimal performance
2. **Avoid unnecessary copies** by validating array properties
3. **Consider array size** for large visualizations

```python
# Check array properties
rgba = renderer.render_rgba()
print(f"C-contiguous: {rgba.flags['C_CONTIGUOUS']}")
print(f"Writeable: {rgba.flags['WRITEABLE']}")

# Make contiguous if needed (creates copy)
if not rgba.flags['C_CONTIGUOUS']:
    rgba = np.ascontiguousarray(rgba)
```

### Colormap Selection

1. **Use perceptually uniform colormaps** (viridis, plasma, cividis) for scientific data
2. **Choose appropriate normalization** based on data distribution
3. **Consider colorblind-friendly options** (viridis, cividis)

### Error Handling

1. **Check matplotlib availability** before using integration features
2. **Validate array formats** with provided helpers
3. **Handle edge cases** (empty arrays, extreme values)

```python
from forge3d.adapters.mpl_cmap import is_matplotlib_available
from forge3d.helpers.mpl_display import validate_rgba_array

def safe_display(rgba):
    """Safely display RGBA array with error handling."""
    try:
        if not is_matplotlib_available():
            print("Matplotlib not available")
            return None
            
        validated = validate_rgba_array(rgba)
        
        fig, ax = plt.subplots()
        im = imshow_rgba(ax, validated)
        return fig, ax, im
        
    except Exception as e:
        print(f"Display failed: {e}")
        return None
```

## API Reference

### forge3d.adapters.mpl_cmap

- `matplotlib_to_forge3d_colormap(cmap, n_colors=256)`: Convert colormap to LUT
- `matplotlib_normalize(data, norm=None, vmin=None, vmax=None, clip=False)`: Normalize data
- `get_matplotlib_colormap_names()`: List available colormap names
- `is_matplotlib_available()`: Check matplotlib availability
- `LogNormAdapter`, `PowerNormAdapter`, `BoundaryNormAdapter`: Normalization classes
- `create_matplotlib_normalizer(norm_type, **kwargs)`: Normalizer factory

### forge3d.helpers.mpl_display  

- `imshow_rgba(ax, rgba, extent=None, dpi=None, **kwargs)`: Display RGBA array
- `imshow_rgba_subplots(rgba_arrays, titles=None, **kwargs)`: Multi-panel display
- `save_rgba_comparison(rgba_arrays, output_path, **kwargs)`: Save comparison
- `validate_rgba_array(rgba, name="rgba")`: Validate array format
- `setup_matplotlib_backend(backend=None)`: Configure backend
- `quick_show(rgba, title="forge3d Output")`: Quick display utility

## Troubleshooting

### Common Issues

**ImportError: matplotlib required**
- Install matplotlib: `pip install matplotlib`
- Check availability: `is_matplotlib_available()`

**ValueError: array must be 3D**
- Ensure RGBA array has shape (H, W, C) where C is 3 or 4
- Use `validate_rgba_array()` to check format

**Warning: not C-contiguous**
- Array may cause performance issues
- Use `np.ascontiguousarray()` if the array will be reused

**Display backend issues**
- Use Agg backend for headless operation: `setup_matplotlib_backend('Agg')`
- Check available backends with `matplotlib.get_available_backends()`

### Performance Issues

**Slow display of large arrays**
- Check array contiguity with `rgba.flags['C_CONTIGUOUS']`
- Consider downsampling for interactive display
- Use appropriate interpolation method

**Memory usage concerns**
- Monitor array size: `rgba.nbytes / 1024**2` MB
- Be aware of copies created during dtype conversion
- Use memory profiling tools for large datasets
# Zero-Copy NumPy Interoperability

This document describes forge3d's zero-copy NumPy interoperability features, which minimize memory copies when transferring data between Python and the Rust graphics core.

## Overview

Zero-copy interoperability allows NumPy arrays to share memory directly with forge3d's Rust implementation, eliminating unnecessary data copies that can impact performance and memory usage. This is particularly important for:

- **RGBA output**: Render results returned as NumPy arrays
- **Height input**: Heightmap data passed from NumPy to terrain rendering
- **Large datasets**: Where copying would be expensive

## Guaranteed Zero-Copy Pathways

### RGBA Output

When rendering operations return RGBA data, forge3d uses zero-copy pathways where possible:

```python
import forge3d as f3d
import numpy as np

# Method 1: Via Renderer instance
renderer = f3d.Renderer(512, 512)
rgba_array = renderer.render_triangle_rgba()  # Zero-copy when possible

# Method 2: Via module function  
rgba_array = f3d.render_triangle_rgba(512, 512)  # Zero-copy when possible
```

**Guarantees:**
- Returned arrays are always C-contiguous (`RGBA.flags['C_CONTIGUOUS']` is `True`)
- Data type is always `uint8`
- Shape is always `(height, width, 4)` for RGBA data
- Memory backing may be reused across calls for performance

### Height Input

When passing heightmap data to terrain operations, zero-copy is used for compliant inputs:

```python
import numpy as np
import forge3d as f3d

# Create C-contiguous float32 heightmap (zero-copy path)
heightmap = np.random.rand(256, 256).astype(np.float32)
assert heightmap.flags['C_CONTIGUOUS']  # Required for zero-copy

renderer = f3d.Renderer(1024, 1024)
renderer.add_terrain(heightmap, spacing=(1.0, 1.0), exaggeration=2.0, colormap="viridis")
```

**Requirements for Zero-Copy:**
- Array must be **C-contiguous** (row-major layout)
- Data type must be `float32` (preferred) or `float64`
- Array must be 2-dimensional with shape `(height, width)`

**Copy Scenarios:**
- Non-contiguous arrays require conversion to contiguous format
- `float64` arrays are converted to `float32` internally
- Invalid shapes or data types trigger validation errors

## Data Type and Layout Requirements

### Supported Input Types

| Data Type | Layout | Zero-Copy | Notes |
|-----------|--------|-----------|-------|
| `float32` | C-contiguous | ✓ Yes | Preferred for heightmaps |
| `float64` | C-contiguous | ⚠ Copy | Converted to `float32` |
| `float32` | Non-contiguous | ⚠ Copy | Made contiguous first |
| Other types | Any | ❌ Error | Not supported |

### Output Guarantees

| Operation | Data Type | Layout | Shape | Zero-Copy |
|-----------|-----------|---------|--------|-----------|
| RGBA render | `uint8` | C-contiguous | `(H, W, 4)` | When possible |
| Height readback | `float32` | C-contiguous | `(H, W)` | Always |

## Validation and Debugging

### Using Validation Helpers

The `forge3d._validate` module provides utilities for checking zero-copy compatibility:

```python
from forge3d._validate import ptr, is_c_contiguous, check_zero_copy_compatibility

# Check array properties
heightmap = np.ones((100, 100), dtype=np.float32)
print(f"Pointer: 0x{ptr(heightmap):x}")
print(f"C-contiguous: {is_c_contiguous(heightmap)}")

# Comprehensive compatibility check
compat = check_zero_copy_compatibility(heightmap)
if not compat['compatible']:
    print("Issues:", compat['issues'])
```

### Pointer Validation

To verify zero-copy behavior, compare memory pointers:

```python
from forge3d._validate import ptr, validate_zero_copy_path

input_array = np.ones((64, 64), dtype=np.float32)
input_ptr = ptr(input_array)

# After processing...
output_array = some_operation(input_array)
output_ptr = ptr(output_array)

# Check if same backing store is used (when expected)
try:
    validate_zero_copy_path("operation", input_ptr, output_ptr)
    print("✓ Zero-copy validated")
except RuntimeError as e:
    print(f"⚠ Copy detected: {e}")
```

## Performance Profiling

Use the provided profiler tool to measure zero-copy performance:

```bash
# Basic profiling
python python/tools/profile_copies.py --render-size 1024x1024

# Custom terrain size
python python/tools/profile_copies.py --render-size 512x512 --terrain-size 256x256

# Verbose output with detailed metrics
python python/tools/profile_copies.py --render-size 1024x1024 --verbose
```

The profiler will:
- Measure operation timing
- Validate pointer consistency
- Report any unexpected copies
- Provide performance metrics

## Common Pitfalls and Solutions

### Non-Contiguous Arrays

**Problem**: Transpose or slice operations create non-contiguous views:
```python
heightmap = np.ones((100, 100), dtype=np.float32)
transposed = heightmap.T  # Non-contiguous!
# This will trigger a copy during add_terrain()
```

**Solution**: Use `np.ascontiguousarray()` when needed:
```python
heightmap_fixed = np.ascontiguousarray(transposed)
assert heightmap_fixed.flags['C_CONTIGUOUS']
```

### Wrong Data Types

**Problem**: Using unsupported data types:
```python
heightmap_int = np.ones((100, 100), dtype=np.int32)  # Not supported
# This will raise a TypeError
```

**Solution**: Convert to supported types:
```python
heightmap_float = heightmap_int.astype(np.float32)
```

### Memory Layout Assumptions

**Problem**: Assuming all NumPy arrays are zero-copy compatible:
```python
# This might not be zero-copy depending on how it was created
some_array = complex_numpy_operation()
```

**Solution**: Always validate before critical operations:
```python
from forge3d._validate import check_zero_copy_compatibility

compat = check_zero_copy_compatibility(some_array)
if not compat['compatible']:
    print("Fixing compatibility issues...")
    some_array = np.ascontiguousarray(some_array, dtype=np.float32)
```

## Testing Zero-Copy Behavior

The test suite includes comprehensive zero-copy validation:

```bash
# Run zero-copy specific tests
pytest -xvs tests/test_numpy_interop.py

# Run with verbose output to see pointer validation
pytest -xvs tests/test_numpy_interop.py::TestZeroCopyRGBAOutput
```

### Writing Zero-Copy Tests

When writing tests that depend on zero-copy behavior:

```python
from forge3d._validate import ptr, validate_zero_copy_path

def test_my_zero_copy_operation():
    input_data = np.ones((64, 64), dtype=np.float32)
    input_ptr = ptr(input_data)
    
    # Perform operation
    result = my_operation(input_data)
    
    # Validate zero-copy if expected
    if should_be_zero_copy:
        result_ptr = ptr(result)
        validate_zero_copy_path("my_operation", input_ptr, result_ptr)
```

## Implementation Notes

### Buffer Reuse

forge3d may reuse GPU readback buffers across operations for performance:
- First render might allocate a new buffer
- Subsequent renders of the same size may reuse the buffer
- Buffer size changes trigger reallocation

### Memory Alignment

GPU operations require specific memory alignment:
- Texture data uses 256-byte row alignment for GPU compatibility
- NumPy arrays returned to Python strip padding for convenience
- Internal operations handle alignment transparently

### Cross-Platform Consistency

Zero-copy behavior is consistent across supported platforms:
- Windows (DirectX/Vulkan backends)
- Linux (Vulkan/OpenGL backends)
- macOS (Metal backend)

Pointer validation and performance characteristics may vary slightly between platforms but zero-copy guarantees remain the same.

## Error Handling

### Common Error Messages

| Error | Cause | Solution |
|--------|--------|-----------|
| "heightmap must be C-contiguous" | Non-contiguous array | Use `np.ascontiguousarray()` |
| "heightmap must be a 2-D NumPy array" | Wrong dimensionality | Reshape or fix input data |
| "dtype=float32\|float64" | Wrong data type | Convert with `.astype()` |
| "Zero-copy validation failed" | Unexpected copy | Check input compatibility |

### Debugging Tips

1. **Check contiguity**: Always verify `array.flags['C_CONTIGUOUS']`
2. **Validate data types**: Ensure `float32` for heightmaps, expect `uint8` for RGBA
3. **Use validation helpers**: Leverage `forge3d._validate` module
4. **Profile operations**: Use the profiler tool to identify performance issues
5. **Read error messages**: They provide specific guidance on fixes

## Best Practices

### For Application Developers

1. **Pre-validate inputs**: Check array compatibility before expensive operations
2. **Use appropriate data types**: Prefer `float32` for numerical data
3. **Maintain contiguity**: Be careful with array slicing and transposition
4. **Profile regularly**: Use the profiler tool during development

### For Library Integration

1. **Document requirements**: Clearly specify zero-copy requirements
2. **Provide fallbacks**: Handle both zero-copy and copy scenarios gracefully  
3. **Validate at boundaries**: Check compatibility at API entry points
4. **Test thoroughly**: Include zero-copy validation in test suites

This documentation should be updated as zero-copy features evolve and new pathways are added.
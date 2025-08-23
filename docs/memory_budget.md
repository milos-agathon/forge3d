# Memory Budget Management

This document describes forge3d's memory budget tracking and enforcement system, which ensures efficient GPU memory usage and prevents excessive memory consumption.

## Overview

forge3d implements a comprehensive memory budget system that:

- **Tracks all GPU resource allocations** (buffers and textures)
- **Enforces a 512 MiB limit** on host-visible memory usage
- **Provides detailed metrics** for monitoring and debugging
- **Prevents out-of-memory scenarios** before they occur

The system focuses primarily on **host-visible memory** as this is typically the most constrained resource in GPU environments.

## Budget Limits

### Default Budget: 512 MiB

forge3d enforces a default budget limit of **512 MiB (536,870,912 bytes)** for host-visible memory allocations. This limit is:

- **Conservative and widely compatible** across different GPU types
- **Focused on host-visible resources** (CPU-accessible GPU memory)
- **Automatically enforced** at allocation time
- **Configurable at the global level** (though not exposed in Python API)

### What Counts Towards the Budget

| Resource Type | Counted | Notes |
|---------------|---------|-------|
| **Readback Buffers** | ✓ Yes | Used for rendering output, height texture readback |
| **Uniform Buffers** | ✓ Yes | When created with MAP_READ/MAP_WRITE usage |
| **Staging Buffers** | ✓ Yes | Temporary buffers for data transfer |
| **Vertex/Index Buffers** | ❌ No | GPU-only memory, not host-visible |
| **Render Textures** | ❌ No | GPU-only memory, not host-visible |
| **Height Textures** | ❌ No | GPU-only memory, not host-visible |

## Memory Metrics API

### Getting Memory Metrics

Use the `get_memory_metrics()` method to retrieve current memory usage:

```python
import forge3d as f3d

renderer = f3d.Renderer(512, 512)
metrics = renderer.get_memory_metrics()

print(f"Host-visible memory: {metrics['host_visible_bytes']} bytes")
print(f"Budget utilization: {metrics['utilization_ratio']:.1%}")
print(f"Within budget: {metrics['within_budget']}")
```

### Metrics Dictionary Structure

```python
{
    # Resource counts
    'buffer_count': int,          # Number of tracked buffers
    'texture_count': int,         # Number of tracked textures
    
    # Memory usage in bytes
    'buffer_bytes': int,          # Total buffer memory
    'texture_bytes': int,         # Total texture memory  
    'host_visible_bytes': int,    # Host-visible memory (budget target)
    'total_bytes': int,           # Sum of buffer_bytes + texture_bytes
    
    # Budget status
    'limit_bytes': int,           # Budget limit (512 MiB)
    'within_budget': bool,        # Whether usage is within limit
    'utilization_ratio': float,   # host_visible_bytes / limit_bytes
}
```

### Example Metrics Output

```python
{
    'buffer_count': 3,
    'texture_count': 2, 
    'buffer_bytes': 2097152,      # ~2 MB buffers
    'texture_bytes': 4194304,     # ~4 MB textures
    'host_visible_bytes': 1048576, # ~1 MB host-visible
    'total_bytes': 6291456,       # ~6 MB total
    'limit_bytes': 536870912,     # 512 MiB limit
    'within_budget': True,
    'utilization_ratio': 0.00195  # ~0.2% of budget used
}
```

## Budget Enforcement

### Automatic Checking

Budget limits are enforced automatically before allocating resources that count towards the budget:

```python
renderer = f3d.Renderer(4096, 4096)  # Large render target
# If this would exceed budget, raises RuntimeError with budget details
```

### Error Messages

When budget limits are exceeded, forge3d raises a `RuntimeError` with detailed information:

```python
RuntimeError: Memory budget exceeded: current 450000000 bytes + requested 100000000 bytes would exceed limit of 536870912 bytes (host-visible: 450000000 bytes)
```

Error messages include:
- **Current usage** before the failed allocation
- **Requested size** that would exceed the budget
- **Budget limit** that would be violated
- **Host-visible breakdown** for debugging

## Monitoring Memory Usage

### Real-Time Monitoring

Monitor memory usage during operations:

```python
import forge3d as f3d

renderer = f3d.Renderer(1024, 1024)

def print_memory_status():
    metrics = renderer.get_memory_metrics()
    print(f"Memory: {metrics['host_visible_bytes']:,} bytes "
          f"({metrics['utilization_ratio']:.1%} of budget)")

print_memory_status()  # Initial state

# Add terrain
heightmap = np.random.rand(512, 512).astype(np.float32)
renderer.add_terrain(heightmap, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")
print_memory_status()  # After terrain setup

# Upload to GPU
renderer.upload_height_r32f()
print_memory_status()  # After GPU upload

# Render
rgba = renderer.render_triangle_rgba()
print_memory_status()  # After rendering
```

### Performance Impact Monitoring

Track how memory usage affects performance:

```python
import time
import forge3d as f3d

def benchmark_with_metrics(width, height):
    renderer = f3d.Renderer(width, height)
    
    start_time = time.time()
    rgba = renderer.render_triangle_rgba()
    render_time = time.time() - start_time
    
    metrics = renderer.get_memory_metrics()
    
    return {
        'render_time': render_time,
        'memory_usage': metrics['host_visible_bytes'],
        'budget_utilization': metrics['utilization_ratio']
    }

# Test different sizes
for size in [256, 512, 1024, 2048]:
    result = benchmark_with_metrics(size, size)
    print(f"{size}x{size}: {result['render_time']:.3f}s, "
          f"{result['memory_usage']:,} bytes ({result['budget_utilization']:.1%})")
```

## Memory Budget Scenarios

### Normal Usage Patterns

Most typical forge3d usage stays well within the 512 MiB budget:

| Operation | Typical Memory Usage | Budget Impact |
|-----------|---------------------|---------------|
| Renderer(256, 256) | ~256 KB readback buffer | < 0.1% |
| Renderer(1024, 1024) | ~4 MB readback buffer | < 1% |
| Renderer(2048, 2048) | ~16 MB readback buffer | ~3% |
| Height texture (512x512) | GPU-only, no budget impact | 0% |
| Triangle geometry | GPU-only, no budget impact | 0% |

### High Memory Scenarios

Operations that use significant portions of the budget:

```python
# Large render targets with multiple operations
renderer = f3d.Renderer(4096, 4096)  # ~64 MB readback buffer

# Multiple terrain operations
for i in range(10):
    heightmap = np.random.rand(1024, 1024).astype(np.float32)  
    # Each terrain creates temporary GPU resources
```

### Budget-Friendly Practices

**Optimize render target sizes:**
```python
# Instead of always using large sizes
renderer = f3d.Renderer(4096, 4096)  # High memory usage

# Use appropriate sizes for your needs
renderer = f3d.Renderer(1024, 1024)  # More efficient
```

**Monitor budget utilization:**
```python
def check_budget_before_operation(renderer, operation_name):
    metrics = renderer.get_memory_metrics()
    if metrics['utilization_ratio'] > 0.8:  # 80% threshold
        print(f"Warning: High memory usage before {operation_name}: "
              f"{metrics['utilization_ratio']:.1%}")
    return metrics['within_budget']
```

**Reuse renderers when possible:**
```python
# Instead of creating multiple renderers
renderers = [f3d.Renderer(512, 512) for _ in range(10)]  # High memory

# Reuse a single renderer
renderer = f3d.Renderer(512, 512)
for operation in operations:
    result = renderer.render_triangle_rgba()  # Memory efficient
```

## Debugging Memory Issues

### Investigating Budget Violations

When encountering budget errors:

1. **Check current usage:**
   ```python
   metrics = renderer.get_memory_metrics()
   print(f"Current usage: {metrics['host_visible_bytes']:,} bytes")
   print(f"Budget limit: {metrics['limit_bytes']:,} bytes")
   ```

2. **Identify large allocations:**
   ```python
   # Large render targets are the most common cause
   width, height = 4096, 4096
   readback_size = width * height * 4  # RGBA bytes
   aligned_row = ((width * 4 + 255) // 256) * 256  # Row alignment
   actual_size = aligned_row * height
   print(f"Render target {width}x{height} needs {actual_size:,} bytes")
   ```

3. **Track allocations over time:**
   ```python
   operations = []
   
   def track_operation(name, func, *args, **kwargs):
       metrics_before = renderer.get_memory_metrics()
       result = func(*args, **kwargs)
       metrics_after = renderer.get_memory_metrics()
       
       operations.append({
           'name': name,
           'memory_delta': metrics_after['host_visible_bytes'] - metrics_before['host_visible_bytes'],
           'final_usage': metrics_after['host_visible_bytes']
       })
       return result
   ```

### Common Memory Issues

**Issue: Budget exceeded on renderer creation**
```python
# Problem: Render target too large
renderer = f3d.Renderer(8192, 8192)  # ~256 MB just for readback buffer
```
**Solution:** Use smaller render targets or divide work into tiles

**Issue: Budget exceeded during terrain upload**
```python
# Problem: Multiple large terrain textures 
for i in range(10):
    large_heightmap = np.random.rand(2048, 2048).astype(np.float32)
    renderer.add_terrain(large_heightmap, ...)
    renderer.upload_height_r32f()
```
**Solution:** Process terrains sequentially and clear unused data

**Issue: Gradual memory accumulation**
```python
# Problem: Creating many renderers without cleanup
renderers = []
for size in range(100, 1000, 100):
    renderers.append(f3d.Renderer(size, size))  # Accumulates memory
```
**Solution:** Reuse renderer instances or explicitly manage lifecycle

## Implementation Details

### Thread Safety

The memory tracker uses atomic operations for thread-safe tracking across multiple GPU contexts and Python threads.

### Global vs. Per-Renderer Tracking

Memory tracking is **global** across all renderer instances:
- All renderers share the same 512 MiB budget
- Creating multiple renderers accumulates their memory usage
- Metrics reflect total system usage, not per-renderer usage

### Memory Alignment

GPU memory allocations include platform-specific alignment:
- **Readback buffers**: 256-byte row alignment for GPU compatibility  
- **Texture uploads**: Row padding for multi-row transfers
- **Reported sizes**: Reflect actual allocated sizes, including alignment

### Performance Impact

Memory tracking has minimal performance overhead:
- **Atomic counters**: Lock-free tracking operations
- **Allocation-time checks**: Budget validation only during allocation
- **No runtime overhead**: Zero cost during render operations

This system ensures predictable memory usage while maintaining high performance for typical forge3d workloads.
# GPU Profiling and Performance Analysis

This document describes the GPU profiling and performance measurement capabilities in forge3d, including timestamp queries, debug markers, and integration with external profiling tools.

## Overview

The forge3d GPU timing system provides:

- **Timestamp Queries**: Measure GPU execution time for rendering passes
- **Debug Markers**: Label GPU work for external profilers (RenderDoc, Nsight Graphics, RGP)
- **Pipeline Statistics**: Count vertex/fragment invocations when available
- **Minimal Overhead**: < 1% frame time impact when enabled
- **Python Integration**: Access timing data through the `forge3d.gpu_metrics` module

## Quick Start

### Basic GPU Timing

```python
import forge3d
import forge3d.gpu_metrics as metrics

# Create renderer with GPU profiling enabled
renderer = forge3d.Renderer(width=1920, height=1080)

# Configure GPU timing
config = metrics.create_default_config()
renderer.enable_gpu_profiling(config)

# Render with timing
rgba_data = renderer.render_rgba()

# Get performance metrics
gpu_metrics = renderer.get_gpu_metrics()
print(f"Total GPU time: {gpu_metrics.total_gpu_time_ms:.2f} ms")

# Access individual pass timings
hdr_time = gpu_metrics.get_timing_by_name('hdr_render')
if hdr_time:
    print(f"HDR render: {hdr_time.gpu_time_ms:.2f} ms")
```

### External Profiler Integration

```python
# Enable debug markers for RenderDoc/Nsight
debug_config = metrics.GpuTimingConfig(
    enable_timestamps=True,
    enable_debug_markers=True,  # Shows in profiler
    label_prefix="MyApp"
)

renderer.enable_gpu_profiling(debug_config)
```

## GPU Timing Configuration

### Configuration Options

The `GpuTimingConfig` class provides several options:

```python
config = metrics.GpuTimingConfig(
    enable_timestamps=True,         # GPU timestamp queries
    enable_pipeline_stats=False,    # Vertex/fragment counts (often unsupported)
    enable_debug_markers=True,      # RenderDoc/Nsight markers  
    label_prefix="forge3d",         # Marker label prefix
    max_queries_per_frame=32        # Query budget per frame
)
```

### Predefined Configurations

```python
# Default configuration (balanced)
config = metrics.create_default_config()

# Minimal overhead (production)
config = metrics.create_minimal_config()  # All profiling disabled

# Debug configuration (development)
config = metrics.create_debug_config()    # All features enabled
```

### Device Capability Validation

```python
# Check device support
device_info = forge3d.device_probe()
features = device_info['features']

has_timestamps = 'TIMESTAMP_QUERY' in features
has_pipeline_stats = 'PIPELINE_STATISTICS_QUERY' in features

# Validate configuration
warnings = metrics.validate_config(config, {
    'timestamps': has_timestamps,
    'pipeline_stats': has_pipeline_stats
})

if warnings:
    print("Configuration warnings:")
    for warning in warnings:
        print(f"  - {warning}")
```

## Timing Scopes and Measurements

### Built-in Timing Scopes

forge3d automatically instruments the following rendering passes:

| Scope Name | Description |
|------------|-------------|
| `hdr_render` | HDR rendering pass |
| `hdr_tonemap` | HDR tone mapping |
| `hdr_offscreen_tonemap` | HDR offscreen tone mapping |
| `terrain_lod_update` | Terrain LOD updates |
| `vector_indirect_culling` | Vector indirect culling |
| `postfx_chain` | Post-processing effects chain |
| `bloom_brightpass` | Bloom bright pass |
| `bloom_blur_h` | Bloom horizontal blur |
| `bloom_blur_v` | Bloom vertical blur |

### Accessing Timing Results

```python
# Get all timings as dictionary
timings = gpu_metrics.get_timings_dict()
print(f"HDR render: {timings['hdr_render']:.2f} ms")
print(f"Tone mapping: {timings['hdr_tonemap']:.2f} ms")

# Get specific timing result with metadata
hdr_result = gpu_metrics.get_timing_by_name('hdr_render')
if hdr_result and hdr_result.timestamp_valid:
    print(f"HDR GPU time: {hdr_result.gpu_time_ms:.2f} ms")
    if hdr_result.pipeline_stats:
        stats = hdr_result.pipeline_stats
        print(f"  Vertices: {stats.get('vertex_invocations', 0):,}")
        print(f"  Fragments: {stats.get('fragment_invocations', 0):,}")
```

### Custom Timing Scopes

For custom rendering code, use the Rust timing API directly:

```rust
use crate::core::gpu_timing::GpuTimingManager;

// In your render method
let timing_scope = timing_manager.begin_scope(encoder, "my_custom_pass");
// ... GPU work ...
timing_manager.end_scope(encoder, timing_scope);

// Or use the convenience macro
gpu_time!(timing_manager, encoder, "my_custom_pass", {
    // ... GPU work ...
});
```

## External Profiler Integration

### RenderDoc

1. Enable debug markers in your configuration
2. Launch RenderDoc and attach to your application
3. Capture a frame - forge3d timing scopes will appear as labeled regions

```python
config = metrics.GpuTimingConfig(enable_debug_markers=True)
renderer.enable_gpu_profiling(config)
```

### NVIDIA Nsight Graphics

1. Configure timing as above
2. Launch Nsight Graphics and profile your application
3. View timing markers in the GPU timeline

### AMD Radeon GPU Profiler (RGP)

1. Enable markers and run your application
2. Use RGP to capture and analyze GPU traces
3. forge3d markers will appear in the wavefront timeline

## Performance Considerations

### Overhead Measurements

GPU timing adds minimal overhead:

- **Timestamp queries**: ~1 μs per query
- **Debug markers**: ~2 μs per marker  
- **Total overhead**: < 1% of frame time at 60 FPS

```python
# Estimate overhead for your configuration
query_count = 8  # Typical for full pipeline
overhead_ms = metrics.estimate_timing_overhead(query_count, enable_markers=True)
print(f"Estimated overhead: {overhead_ms:.4f} ms")

# At 60 FPS (16.67ms budget)
frame_budget = 16.67
overhead_percent = (overhead_ms / frame_budget) * 100
print(f"Overhead: {overhead_percent:.2f}% of frame budget")
```

### Production Guidelines

- Use `create_minimal_config()` for production builds
- Enable timing only during development and profiling sessions
- Limit `max_queries_per_frame` to avoid GPU query pool exhaustion
- Consider conditional compilation for release builds

```python
import os

# Enable profiling only in debug builds
if os.environ.get('DEBUG_GPU_TIMING'):
    config = metrics.create_debug_config()
    renderer.enable_gpu_profiling(config)
else:
    # No profiling overhead in production
    pass
```

## Troubleshooting

### Common Issues

**Q: Timestamps return 0.0ms or invalid results**

A: Check that your device supports `TIMESTAMP_QUERY` feature:

```python
device_info = forge3d.device_probe()
if 'TIMESTAMP_QUERY' not in device_info['features']:
    print("Device does not support timestamp queries")
```

**Q: Pipeline statistics are always empty**

A: Pipeline statistics require `PIPELINE_STATISTICS_QUERY` feature, which is often unsupported. Disable in production:

```python
config.enable_pipeline_stats = False  # Usually not supported
```

**Q: Debug markers not visible in profiler**

A: Ensure markers are enabled and you're capturing the correct process:

```python
config.enable_debug_markers = True
config.label_prefix = "MyApp"  # Use distinctive prefix
```

### Validation and Debugging

```python
# Check configuration validity
warnings = metrics.validate_config(config, device_features)
if warnings:
    print("Configuration issues:")
    for warning in warnings:
        print(f"  {warning}")

# Monitor timing overhead
gpu_metrics = renderer.get_gpu_metrics()
summary = gpu_metrics.get_summary()
print(f"Timing overhead: {summary['timestamp_overhead_ms']:.4f} ms")
```

## API Reference

### Configuration Classes

- `GpuTimingConfig`: Main configuration class
- `TimingResult`: Individual timing measurement
- `GpuMetrics`: Collection of timing results

### Utility Functions

- `create_default_config()`: Balanced configuration
- `create_minimal_config()`: No profiling overhead  
- `create_debug_config()`: All features enabled
- `estimate_timing_overhead()`: Overhead estimation
- `validate_config()`: Configuration validation

### Integration Methods

- `Renderer.enable_gpu_profiling(config)`: Enable timing
- `Renderer.get_gpu_metrics()`: Get timing results
- `Renderer.disable_gpu_profiling()`: Disable timing

## Best Practices

1. **Development vs Production**: Use debug config for development, minimal for production
2. **Selective Timing**: Enable only the passes you're analyzing
3. **Overhead Awareness**: Monitor timing overhead in performance-critical applications
4. **External Tools**: Use RenderDoc/Nsight for detailed GPU analysis
5. **Validation**: Always validate configuration against device capabilities

## Examples

See the `examples/gpu_profiling_demo.py` for a complete example of GPU timing integration, including configuration options, timing analysis, and performance monitoring.
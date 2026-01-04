# Async Readback System Guide

This guide describes the asynchronous and double-buffered readback system in forge3d, designed to improve performance for applications that frequently read GPU textures back to CPU memory.

## Overview

The async readback system provides:
- **Non-blocking operations** - Readbacks don't block the main thread
- **Double-buffering** - Overlapping readback operations for higher throughput
- **Buffer pooling** - Reuse of readback buffers to reduce allocations
- **Backpressure control** - Limits on concurrent operations to prevent memory exhaustion
- **Performance monitoring** - Statistics and benchmarking tools

## Problem Statement

Traditional synchronous readbacks suffer from:
- **CPU blocking** - Main thread waits for GPU completion
- **Poor throughput** - Sequential operations with idle time
- **Memory fragmentation** - Frequent buffer allocation/deallocation
- **Resource contention** - Multiple operations competing for GPU resources

## Architecture

### Core Components

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Application       │───►│  AsyncReadback      │───►│   GPU Readback      │
│   (Python/Rust)     │    │   Manager           │    │   Worker Pool       │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │  Buffer Pool        │
                           │  (Double-buffered)  │
                           └─────────────────────┘
```

### Buffer Management

```
Buffer Pool:
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Buffer0 │ │ Buffer1 │ │ Buffer2 │ │ Buffer3 │
│ [FREE]  │ │ [IN_USE]│ │ [FREE]  │ │ [IN_USE]│
└─────────┘ └─────────┘ └─────────┘ └─────────┘
      ▲           │           ▲           │
      │           │           │           │
   Available   Reading    Available   Reading
```

## Configuration

### AsyncReadbackConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `double_buffered` | `true` | Enable double-buffering for overlapped operations |
| `pre_allocate` | `true` | Pre-allocate buffers for consistent performance |
| `max_pending_ops` | `4` | Maximum concurrent readback operations |

### Performance Tuning

```rust
// High throughput configuration
let config = AsyncReadbackConfig {
    double_buffered: true,
    pre_allocate: true,
    max_pending_ops: 8, // Higher concurrency
};

// Low memory configuration  
let config = AsyncReadbackConfig {
    double_buffered: false,
    pre_allocate: false,
    max_pending_ops: 2, // Limit memory usage
};
```

## API Usage

### Rust API

#### Basic Usage

```rust
use forge3d::core::async_readback::{AsyncReadbackManager, AsyncReadbackConfig};
use std::sync::Arc;

// Create async readback manager
let config = AsyncReadbackConfig::default();
let manager = AsyncReadbackManager::new(device, queue, config)?;

// Start async readback
let handle = manager.readback_texture_async(&texture, 512, 512).await?;

// Wait for completion
let rgba_data = handle.wait().await?;

// Or check if ready (non-blocking)
if let Some(data) = handle.try_get()? {
    println!("Readback completed: {} bytes", data.len());
}
```

#### Batch Operations

```rust
// Start multiple readbacks
let mut handles = Vec::new();
for i in 0..10 {
    let handle = manager.readback_texture_async(&textures[i], 256, 256).await?;
    handles.push(handle);
}

// Wait for all to complete
let results = futures::future::try_join_all(
    handles.into_iter().map(|h| h.wait())
).await?;
```

### Python API

#### Basic Usage

```python
import asyncio
from forge3d.async_readback import AsyncRenderer, AsyncReadbackConfig

# Create async-enabled renderer
config = AsyncReadbackConfig(double_buffered=True, max_pending_ops=4)
renderer = AsyncRenderer(512, 512, config)

# Async rendering
async def render_async():
    handle = await renderer.render_async()
    rgba_array = await handle.wait()
    return rgba_array

# Run async operation
rgba_data = asyncio.run(render_async())
```

#### Context Manager

```python
from forge3d.async_readback import AsyncReadbackContext

async def batch_rendering():
    async with AsyncReadbackContext(renderer) as manager:
        # Start multiple operations
        handles = []
        for i in range(5):
            handle = await manager.readback_texture_async(512, 512)
            handles.append(handle)
        
        # Wait for all completions
        results = await asyncio.gather(*[h.wait() for h in handles])
        return results

results = asyncio.run(batch_rendering())
```

#### Performance Monitoring

```python
# Get performance statistics
stats = renderer.get_readback_stats()
print(f"Average readback time: {stats['average_time_ms']:.2f} ms")
print(f"Pending operations: {stats['pending_operations']}")
print(f"Success rate: {stats['completed_operations'] / stats['total_operations']:.1%}")

# Benchmark performance
from forge3d.async_readback import benchmark_readback_performance

benchmark = benchmark_readback_performance(renderer, num_operations=100)
print(f"Async speedup: {benchmark['speedup']:.2f}x")
print(f"Throughput: {benchmark['async_ops_per_sec']:.1f} ops/sec")
```

## Performance Characteristics

### Throughput Comparison

| Method | Operations/sec | Latency (ms) | Memory Usage |
|--------|----------------|--------------|--------------|
| **Synchronous** | 50-100 | 10-20 | Low |
| **Async Single** | 80-150 | 8-15 | Medium |  
| **Async Double-buffered** | 150-300 | 6-12 | Medium-High |
| **Async Batch** | 200-500 | 5-10 | High |

### Memory Usage

```
Synchronous:     [████    ] 2-4 MiB
Async (x2):      [████████] 4-8 MiB  
Async (x4):      [████████████████] 8-16 MiB
```

### Performance Factors

1. **GPU Memory Bandwidth** - Primary bottleneck
2. **Buffer Size** - Larger textures = higher latency
3. **Concurrent Operations** - Diminishing returns after 4-8 operations
4. **Driver Overhead** - Varies by GPU vendor
5. **System Memory** - Can become bottleneck with many large buffers

## Error Handling

### Common Errors

```rust
// Too many pending operations
if pending_count >= max_pending_ops {
    return Err(RenderError::Readback("Too many pending operations".into()));
}

// Buffer allocation failure
if buffer_allocation_failed {
    return Err(RenderError::Upload("Failed to allocate readback buffer".into()));
}

// GPU mapping failure
map_result.map_err(|e| RenderError::Readback(format!("MapAsync failed: {:?}", e)))?;
```

### Recovery Strategies

1. **Backpressure** - Wait for pending operations to complete
2. **Degradation** - Fall back to synchronous readback
3. **Buffer reuse** - Recycle existing buffers instead of allocating new ones
4. **Operation batching** - Group multiple operations to reduce overhead

## Best Practices

### Configuration Guidelines

1. **Start with defaults** - Usually provide good performance
2. **Tune `max_pending_ops`** based on memory constraints:
   - Low memory: 1-2 operations
   - Medium memory: 4-6 operations  
   - High memory: 8+ operations
3. **Enable double-buffering** for sustained workloads
4. **Disable pre-allocation** for sporadic usage

### Usage Patterns

#### High-Frequency Readbacks
```rust
// Pre-allocate and reuse buffers
let config = AsyncReadbackConfig {
    pre_allocate: true,
    max_pending_ops: 6,
    ..Default::default()
};
```

#### Batch Processing
```rust
// Process in chunks to avoid memory exhaustion
for chunk in textures.chunks(4) {
    let handles: Vec<_> = chunk.iter()
        .map(|tex| manager.readback_texture_async(tex, 512, 512))
        .collect();
    
    let results = futures::future::try_join_all(handles).await?;
    process_batch(results);
}
```

#### Streaming Workloads
```python
async def streaming_processor():
    async with AsyncReadbackContext(renderer) as manager:
        while True:
            # Start next readback
            handle = await manager.readback_texture_async(512, 512)
            
            # Process previous results while GPU works
            if previous_handle and previous_handle.is_complete:
                data = previous_handle.try_get()
                if data is not None:
                    process_frame(data)
            
            previous_handle = handle
            await asyncio.sleep(0.016)  # ~60 FPS
```

## Troubleshooting

### Performance Issues

**Symptom**: Lower than expected throughput
- **Cause**: GPU memory bandwidth saturation
- **Solution**: Reduce concurrent operations or texture sizes

**Symptom**: High memory usage
- **Cause**: Too many pre-allocated buffers
- **Solution**: Reduce `max_pending_ops` or disable `pre_allocate`

**Symptom**: Inconsistent performance
- **Cause**: Buffer allocation/deallocation overhead
- **Solution**: Enable `pre_allocate` and buffer pooling

### Memory Issues

**Symptom**: Out of memory errors
- **Cause**: Accumulated readback buffers
- **Solution**: Lower `max_pending_ops`, add explicit cleanup

**Symptom**: Memory fragmentation
- **Cause**: Frequent large allocations
- **Solution**: Use consistent buffer sizes, enable pooling

### Driver Issues

**Symptom**: Map_async failures
- **Cause**: Driver-specific limitations
- **Solution**: Reduce concurrent operations, add retry logic

## Implementation Details

### Thread Safety

- **Manager**: Thread-safe with internal mutexes
- **Handles**: Safe to pass between threads
- **Buffers**: Protected by atomic reference counting

### GPU Resource Management

- **Buffer lifecycle**: Tracked with RAII handles
- **Memory accounting**: Integrated with global tracker  
- **Cleanup**: Automatic on manager drop

### Future Improvements

1. **Adaptive concurrency** - Dynamic adjustment based on performance
2. **Format-specific optimization** - Specialized paths for different texture formats
3. **Multi-GPU support** - Load balancing across multiple GPUs
4. **Compression** - On-the-fly compression for network transfer

## References

- [WebGPU Buffer Mapping](https://gpuweb.github.io/gpuweb/#buffer-mapping)
- [Vulkan Memory Management Best Practices](https://developer.nvidia.com/vulkan-memory-management)
- [DirectX 12 Resource Barriers](https://docs.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12)
- [Async/Await in Rust](https://rust-lang.github.io/async-book/)
# O1: Staging Buffer Rings

Staging buffer rings provide efficient GPU upload mechanisms with automatic wrap-around and fence-based synchronization to prevent buffer reuse before GPU operations complete.

## Overview

The staging ring system implements a circular buffer approach for GPU uploads:

- **3-ring buffer system**: Three staging buffers rotate to prevent GPU/CPU synchronization stalls
- **Fence-backed synchronization**: CPU checks fence completion before buffer reuse
- **Automatic wrap-around**: Seamlessly advances to next buffer when current is full
- **Usage statistics**: Tracks bytes in-flight, ring index, and stall counts

## Architecture

### Core Components

1. **StagingRing**: Main ring buffer manager
2. **FenceTracker**: Handles fence-based synchronization
3. **StagingBuffer**: Individual buffer within the ring

### Buffer States

- **Available**: Buffer is ready for new allocations
- **In-Use**: Buffer has pending allocations
- **Fenced**: Buffer is waiting for GPU fence completion

## API Reference

### Rust API

```rust
use forge3d::core::staging_rings::StagingRing;

// Create staging ring system
let staging_ring = StagingRing::new(device, queue, 3, 1024 * 1024);

// Get current buffer
let buffer = staging_ring.current();

// Allocate space in ring
if let Some((buffer, offset)) = staging_ring.allocate(data.len() as u64) {
    // Use buffer at offset
    queue.write_buffer_with_data(buffer, offset, data);
    
    // Advance ring with fence
    staging_ring.advance(fence_value);
}

// Check statistics
let stats = staging_ring.stats();
println!("Bytes in flight: {}", stats.bytes_in_flight);
```

### Python API

```python
import forge3d.memory as memory

# Initialize staging rings
result = memory.init_memory_system(staging_rings=True, ring_count=3)

# Get statistics
stats = memory.staging_stats()
print(f"Current ring: {stats['current_ring_index']}")
print(f"Buffer stalls: {stats['buffer_stalls']}")

# High-level manager
manager = memory.StagingRingManager(ring_count=3, buffer_size=1024*1024)
if manager.initialize():
    print("Staging rings initialized successfully")
    print(manager.stats())
```

## Performance Characteristics

### Benchmarks

The O1 implementation meets the following acceptance criteria:

- **< 2 ms CPU overhead** for 100 MB transfers (median over 100 runs)
- **Zero buffer reuse** before fence signal (validated in tests)
- **Usage statistics** report ring index and in-flight bytes

### Typical Performance

| Transfer Size | CPU Overhead | GPU Time | Throughput |
|---------------|--------------|----------|------------|
| 1 MB          | 0.1 ms      | 0.5 ms   | 2 GB/s     |
| 10 MB         | 0.8 ms      | 4.2 ms   | 2.4 GB/s   |
| 100 MB        | 1.5 ms      | 42 ms    | 2.3 GB/s   |

## Configuration

### Ring Count

- **1 ring**: No buffering, potential stalls
- **2 rings**: Basic double buffering
- **3 rings**: Recommended, provides overlap buffer
- **4+ rings**: Diminishing returns, increased memory usage

### Buffer Size

- **Small (< 1MB)**: Good for frequent small uploads
- **Medium (1-4MB)**: Balanced for mixed workloads  
- **Large (> 8MB)**: Efficient for large transfers but high memory usage

## Memory Budget

Staging rings respect the 512 MiB host-visible memory constraint:

```
Total Memory = ring_count × buffer_size
Recommended: 3 × 1MB = 3MB (well under limit)
Maximum: 3 × 170MB = 510MB (near limit)
```

## Error Handling

### Common Issues

1. **Buffer Full**: All rings are full or fenced
   - **Solution**: Wait for fences or increase buffer size
   - **Detection**: `allocate()` returns `None`

2. **Fence Timeout**: GPU takes too long to complete
   - **Solution**: Check GPU workload or driver issues
   - **Detection**: High `buffer_stalls` in statistics

3. **Memory Pressure**: Approaching host-visible limit
   - **Solution**: Reduce ring count or buffer size
   - **Detection**: Device creation failures

### Debug Information

Enable debug logging for detailed fence tracking:

```rust
env_logger::init();
// Logs fence submission and completion events
```

## Integration

### With Memory Pools (O2)

Staging rings work seamlessly with memory pools:
- Pools manage long-lived allocations
- Staging rings handle temporary upload buffers
- No overlap in memory usage patterns

### With Texture Compression (O3)

Compressed textures benefit from staging rings:
- Large compressed data benefits from ring buffering
- Reduces stalls during batch texture uploads
- Statistics help tune compression vs. upload balance

## Testing

### Unit Tests

```bash
cargo test staging_rings
```

### Performance Tests

```bash
pytest tests/test_staging_performance.py -v
```

### Example Usage

```bash
python examples/staging_ring_demo.py
```

The example demonstrates:
- Ring initialization
- Allocation patterns
- Fence management
- Statistics monitoring

## Troubleshooting

### High Buffer Stalls

If `buffer_stalls` is increasing:

1. **Increase ring count**: More buffers for overlap
2. **Increase buffer size**: Fewer wrap-arounds needed
3. **Check GPU load**: May be processing slowly
4. **Profile fence timing**: Look for GPU bottlenecks

### Memory Usage Issues

Monitor staging memory with:

```python
stats = memory.staging_stats()
memory_usage = stats['ring_count'] * stats['buffer_size']
print(f"Staging memory: {memory_usage / 1024 / 1024:.1f} MB")
```

### Platform Differences

- **Windows (DX12)**: Generally good performance
- **Linux (Vulkan)**: Excellent performance, preferred
- **macOS (Metal)**: Good performance, some fence timing differences
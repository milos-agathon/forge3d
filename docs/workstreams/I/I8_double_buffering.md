# I8: Double-Buffering for Per-Frame Data

**Status**: Implemented  
**Feature Flag**: `wsI_double_buf`  
**Files**: `src/core/double_buffer.rs`, `tests/wsI_i8_pingpong.rs`

## Overview

Double-buffering (ping-pong buffers) eliminates GPU-CPU synchronization stalls when updating per-frame uniform and storage buffers. Instead of writing to GPU-in-use buffers, we rotate between 2 or 3 buffers, ensuring the GPU reads from one buffer while the CPU writes to another.

## Key Benefits

- **Eliminates Stalls**: No waiting for GPU to finish using buffers before writing
- **Improved Throughput**: CPU and GPU work independently on different buffers  
- **Reduced Latency**: Eliminates blocking waits during buffer updates
- **Validation Clean**: No hazards from writing to in-flight buffers

## Architecture

### Core Components

```rust
pub struct DoubleBuffer {
    buffers: Vec<Buffer>,        // 2 or 3 buffers for rotation
    current_write: usize,        // Index for CPU writes
    current_read: usize,         // Index for GPU reads/binding
    config: DoubleBufferConfig,  // Configuration
    metrics: Option<...>,        // Performance tracking
}
```

### Configuration Options

```rust
// Double-buffering (N=2) - standard approach
let config = DoubleBufferConfig::uniform(256);

// Triple-buffering (N=3) - for high-frequency updates  
let config = DoubleBufferConfig::uniform(256)
    .with_triple_buffering()
    .with_metrics();
```

## Usage Patterns

### Basic Double-Buffering

```rust
use forge3d::core::double_buffer::{DoubleBuffer, DoubleBufferConfig};

// Create double-buffer for per-frame globals
let config = DoubleBufferConfig::uniform(256);
let mut globals_buffer = DoubleBuffer::new(&device, config, "globals", Some(&registry))?;

// Per-frame update cycle
for frame in 0..frame_count {
    // 1. Update data in write buffer (safe - GPU not using it)
    let frame_data = compute_frame_data(frame);
    globals_buffer.write_typed(&queue, &frame_data, 0)?;
    
    // 2. Bind current read buffer for GPU
    render_pass.set_bind_group(0, &bind_group_using(globals_buffer.current_buffer()), &[]);
    
    // 3. Render with current buffer
    render_pass.draw(...);
    
    // 4. Swap buffers for next frame
    globals_buffer.swap();
}
```

### Triple-Buffering for High Frequency

```rust
// Triple-buffering prevents stalls even with very frequent updates
let config = DoubleBufferConfig::uniform(256)
    .with_triple_buffering()
    .with_metrics();

let mut high_freq_buffer = DoubleBuffer::new(&device, config, "high_freq", None)?;

// Can handle updates every frame without contention
for frame in 0..1000 {
    high_freq_buffer.write_typed(&queue, &high_frequency_data, 0)?;
    // GPU uses buffer from 2 frames ago, safe from contention
    high_freq_buffer.swap();
}
```

### Buffer Pool Management

```rust
use forge3d::core::double_buffer::DoubleBufferPool;

let mut pool = DoubleBufferPool::new();

// Add multiple double-buffers to pool
let globals_config = DoubleBufferConfig::uniform(256).with_metrics();
let globals_buffer = DoubleBuffer::new(&device, globals_config, "globals", None)?;
pool.add_buffer("globals".to_string(), globals_buffer);

let objects_config = DoubleBufferConfig::storage(1024).with_metrics();  
let objects_buffer = DoubleBuffer::new(&device, objects_config, "objects", None)?;
pool.add_buffer("objects".to_string(), objects_buffer);

// Update all buffers
if let Some(globals) = pool.get_buffer_mut("globals") {
    globals.write_typed(&queue, &global_data, 0)?;
}

// Swap all buffers together
pool.swap_all();
```

## Buffer Rotation Strategies

### Double-Buffering (N=2)
```
Frame 0: CPU writes Buffer A, GPU reads Buffer B
Frame 1: CPU writes Buffer B, GPU reads Buffer A  
Frame 2: CPU writes Buffer A, GPU reads Buffer B
```

### Triple-Buffering (N=3)  
```
Frame 0: CPU writes Buffer A, GPU reads Buffer C
Frame 1: CPU writes Buffer B, GPU reads Buffer A
Frame 2: CPU writes Buffer C, GPU reads Buffer B
Frame 3: CPU writes Buffer A, GPU reads Buffer C
```

## When to Use Triple-Buffering

Triple-buffering helps when:

1. **High Update Frequency**: Updating buffers multiple times per frame
2. **Long GPU Work**: GPU takes multiple frames to process commands
3. **Variable Frame Times**: Inconsistent timing between CPU and GPU
4. **Latency Critical**: Need minimum latency between update and display

```rust
// Use triple-buffering for these scenarios:
let config = if high_frequency_updates || variable_timing {
    DoubleBufferConfig::uniform(size).with_triple_buffering()
} else {
    DoubleBufferConfig::uniform(size) // Double is sufficient
};
```

## Performance Validation

### 300-Frame Stress Test

The implementation includes a comprehensive validation test:

```rust
#[tokio::test]
async fn test_300_frame_headless_validation() {
    // Creates 300 frames with continuous buffer updates
    // Validates: 0 hazards, 0 stalls, correct metrics
    
    for frame in 0..300 {
        globals_buffer.write_typed(&queue, &frame_data, 0)?;
        // Render pass using current_buffer()
        globals_buffer.swap();
    }
    
    assert_eq!(validation_errors, 0);
    assert_eq!(stall_count, 0);
}
```

### Performance Metrics

```rust
if let Some(metrics) = double_buffer.metrics() {
    println!("Swaps: {}", metrics.swap_count);
    println!("Writes: {}", metrics.write_count);  
    println!("Bytes: {}", metrics.bytes_written);
    println!("Avg interval: {:.2} frames", metrics.avg_swap_interval);
    println!("Stalls avoided: {}", metrics.stalls_avoided);
}
```

## Memory Management

### Buffer Configuration

```rust
// Uniform buffers (smaller, frequent updates)
let uniform_config = DoubleBufferConfig::uniform(256);

// Storage buffers (larger, less frequent updates)  
let storage_config = DoubleBufferConfig::storage(4096);

// Custom configuration
let custom_config = DoubleBufferConfig {
    size: 1024,
    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    buffer_count: 3,
    enable_metrics: true,
};
```

### Memory Tracking Integration

```rust
let registry = ResourceRegistry::new();
let double_buffer = DoubleBuffer::new(&device, config, "label", Some(&registry))?;

// Automatically tracks allocation in memory budget
let stats = registry.get_stats();
assert!(stats.within_budget);
```

## Best Practices

### 1. Buffer Sizing
- Size buffers for worst-case per-frame data
- Align to GPU requirements (typically 256 bytes)
- Consider memory budget constraints

### 2. Update Patterns
```rust
// ✅ Good: Update, bind, render, swap
buffer.write_typed(&queue, &data, 0)?;
render_pass.set_bind_group(0, &bind_group, &[]);
render_pass.draw(...);
buffer.swap();

// ❌ Bad: Swap before binding
buffer.swap();
render_pass.set_bind_group(0, &bind_group, &[]); // Wrong buffer!
```

### 3. Synchronization Points
```rust
// Occasional sync to prevent unbounded queue growth
if frame % 100 == 0 {
    buffer.sync(&device); // Poll for completion
}
```

### 4. Error Handling
```rust
match buffer.write_data(&queue, &data, offset) {
    Ok(()) => { /* Success */ },
    Err(RenderError::Upload(msg)) => {
        eprintln!("Buffer write failed: {}", msg);
        // Consider buffer resize or error recovery
    },
}
```

## Validation and Testing

### Key Tests
- **Creation**: Verify buffer count and initial state
- **Swapping**: Ensure proper index rotation
- **Writing**: Validate data writes and bounds checking
- **300-Frame**: Stress test with validation layers enabled
- **CPU Wait Reduction**: Measure performance improvement

### Running Tests
```bash
# Run double-buffer tests
cargo test --features wsI_double_buf wsI_i8_pingpong

# Run with validation layers (if available)
RUST_LOG=warn cargo test --features wsI_double_buf test_300_frame_headless_validation
```

## Integration Examples

### Render Loop Integration

```rust
struct Renderer {
    globals_buffer: DoubleBuffer,
    objects_buffer: DoubleBuffer,
    // ... other state
}

impl Renderer {
    fn render_frame(&mut self, frame_data: &FrameData) -> Result<(), RenderError> {
        // Update per-frame data
        self.globals_buffer.write_typed(&self.queue, &frame_data.globals, 0)?;
        self.objects_buffer.write_typed(&self.queue, &frame_data.objects, 0)?;
        
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                // ... setup
            });
            
            // Bind current buffers (GPU reads while CPU writes to other buffers)
            render_pass.set_bind_group(0, &self.create_globals_bind_group(), &[]);
            render_pass.set_bind_group(1, &self.create_objects_bind_group(), &[]);
            
            // Render with current buffer data
            render_pass.draw(...);
        }
        
        self.queue.submit(Some(encoder.finish()));
        
        // Swap for next frame
        self.globals_buffer.swap();
        self.objects_buffer.swap();
        
        Ok(())
    }
}
```

## Common Pitfalls

1. **Binding Wrong Buffer**: Always use `current_buffer()` for GPU binding
2. **Missing Swaps**: Forget to call `swap()` after render, causing stalls
3. **Over-Synchronization**: Calling `sync()` too frequently defeats the purpose
4. **Buffer Size Mismatch**: Writing more data than buffer can hold
5. **Feature Flag Issues**: Missing conditional compilation for feature-gated code

## Debugging Tips

```rust
// Check buffer safety
assert!(buffer.is_write_safe());

// Monitor metrics
if let Some(metrics) = buffer.metrics() {
    if metrics.stalls_avoided == 0 {
        println!("Warning: No stalls detected - may not be helping");
    }
}

// Validate indices
println!("Write index: {}, Read index: {}", 
         buffer.current_write_index(), buffer.current_read_index());
```
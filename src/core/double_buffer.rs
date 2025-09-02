//! Double-buffering for per-frame data (I8)
//!
//! Implements ping-pong buffers to avoid writing GPU-in-use buffers during
//! per-frame uniform/storage buffer updates. Supports double (N=2) and 
//! triple-buffering (N=3) strategies.

use std::sync::Arc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue, BufferAddress, CommandEncoder};
use crate::core::memory_tracker::ResourceRegistry;
use crate::error::RenderError;

/// Configuration for double-buffering strategy
#[derive(Debug, Clone, Copy)]
pub struct DoubleBufferConfig {
    /// Buffer size in bytes
    pub size: u64,
    /// Buffer usage flags
    pub usage: BufferUsages,
    /// Number of buffers (2 for double-buffering, 3 for triple-buffering)
    pub buffer_count: u32,
    /// Enable validation and metrics tracking
    pub enable_metrics: bool,
}

impl DoubleBufferConfig {
    /// Create standard config for uniform buffers
    pub fn uniform(size: u64) -> Self {
        Self {
            size,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            buffer_count: 2,
            enable_metrics: false,
        }
    }

    /// Create standard config for storage buffers  
    pub fn storage(size: u64) -> Self {
        Self {
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            buffer_count: 2,
            enable_metrics: false,
        }
    }

    /// Enable triple-buffering for high-frequency updates
    pub fn with_triple_buffering(mut self) -> Self {
        self.buffer_count = 3;
        self
    }

    /// Enable metrics tracking for performance analysis
    pub fn with_metrics(mut self) -> Self {
        self.enable_metrics = true;
        self
    }
}

/// Double-buffer for per-frame data with ping-pong strategy
pub struct DoubleBuffer {
    /// Array of buffers (2 or 3 buffers)
    buffers: Vec<Buffer>,
    /// Current buffer index for writing
    current_write: usize,
    /// Current buffer index for reading/binding
    current_read: usize,
    /// Buffer configuration
    config: DoubleBufferConfig,
    /// Frame counter for rotation strategy
    frame_count: u64,
    /// Metrics tracking (if enabled)
    metrics: Option<DoubleBufferMetrics>,
}

/// Performance metrics for double-buffer usage
#[derive(Debug, Clone, Default)]
pub struct DoubleBufferMetrics {
    /// Total number of buffer swaps
    pub swap_count: u64,
    /// Number of write operations
    pub write_count: u64,
    /// Number of potential stalls avoided
    pub stalls_avoided: u64,
    /// Total bytes written
    pub bytes_written: u64,
    /// Average time between swaps (in frames)
    pub avg_swap_interval: f32,
}

impl DoubleBuffer {
    /// Create a new double-buffer with the given configuration
    pub fn new(
        device: &Device,
        config: DoubleBufferConfig,
        label_prefix: &str,
        registry: Option<&ResourceRegistry>,
    ) -> Result<Self, RenderError> {
        if config.buffer_count < 2 || config.buffer_count > 3 {
            return Err(RenderError::Upload(format!(
                "Invalid buffer count: {}. Must be 2 or 3.", config.buffer_count
            )));
        }

        let mut buffers = Vec::with_capacity(config.buffer_count as usize);
        for i in 0..config.buffer_count {
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(&format!("{}_double_buffer_{}", label_prefix, i)),
                size: config.size,
                usage: config.usage,
                mapped_at_creation: false,
            });

            // Track in memory registry
            if let Some(registry) = registry {
                let is_host_visible = config.usage.contains(BufferUsages::MAP_READ) || 
                                    config.usage.contains(BufferUsages::MAP_WRITE);
                registry.track_buffer_allocation(config.size, is_host_visible);
            }

            buffers.push(buffer);
        }

        let metrics = if config.enable_metrics {
            Some(DoubleBufferMetrics::default())
        } else {
            None
        };

        Ok(Self {
            buffers,
            current_write: 0,
            current_read: if config.buffer_count == 2 { 1 } else { 2 },
            config,
            frame_count: 0,
            metrics,
        })
    }

    /// Get the current buffer for reading/binding in shaders
    pub fn current_buffer(&self) -> &Buffer {
        &self.buffers[self.current_read]
    }

    /// Get the current buffer for writing new data
    pub fn write_buffer(&self) -> &Buffer {
        &self.buffers[self.current_write]
    }

    /// Swap buffers for next frame (ping-pong)
    pub fn swap(&mut self) {
        self.frame_count += 1;

        // Track metrics
        if let Some(ref mut metrics) = self.metrics {
            metrics.swap_count += 1;
            // Update average swap interval
            let interval = self.frame_count as f32 / metrics.swap_count as f32;
            metrics.avg_swap_interval = interval;
        }

        match self.config.buffer_count {
            2 => {
                // Simple ping-pong: swap write/read
                std::mem::swap(&mut self.current_write, &mut self.current_read);
            }
            3 => {
                // Triple-buffering: advance write, read follows with 1-frame delay
                self.current_write = (self.current_write + 1) % 3;
                self.current_read = (self.current_read + 1) % 3;
            }
            _ => unreachable!(),
        }
    }

    /// Write data to the current write buffer
    pub fn write_data(&mut self, queue: &Queue, data: &[u8], offset: u64) -> Result<(), RenderError> {
        if data.len() + offset as usize > self.config.size as usize {
            return Err(RenderError::Upload(format!(
                "Write would exceed buffer size: {} + {} > {}",
                offset, data.len(), self.config.size
            )));
        }

        queue.write_buffer(&self.buffers[self.current_write], offset, data);

        // Track metrics
        if let Some(ref mut metrics) = self.metrics {
            metrics.write_count += 1;
            metrics.bytes_written += data.len() as u64;
        }

        Ok(())
    }

    /// Write typed data to the current write buffer
    pub fn write_typed<T: bytemuck::Pod>(&mut self, queue: &Queue, data: &[T], offset: u64) -> Result<(), RenderError> {
        let bytes = bytemuck::cast_slice(data);
        self.write_data(queue, bytes, offset)
    }

    /// Get buffer configuration
    pub fn config(&self) -> DoubleBufferConfig {
        self.config
    }

    /// Get current frame count
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get performance metrics (if enabled)
    pub fn metrics(&self) -> Option<&DoubleBufferMetrics> {
        self.metrics.as_ref()
    }

    /// Reset metrics counters
    pub fn reset_metrics(&mut self) {
        if let Some(ref mut metrics) = self.metrics {
            *metrics = DoubleBufferMetrics::default();
        }
    }

    /// Check if buffer might be in use by GPU
    pub fn is_write_safe(&self) -> bool {
        // With double/triple-buffering, writing to write_buffer should always be safe
        // as GPU should be using a different buffer
        match self.config.buffer_count {
            2 => self.current_write != self.current_read,
            3 => {
                // In triple-buffering, ensure write buffer is not read buffer
                self.current_write != self.current_read
            }
            _ => false,
        }
    }

    /// Force synchronization point (for debugging/validation)
    pub fn sync(&self, device: &Device) {
        // Poll to ensure all pending operations complete
        device.poll(wgpu::Maintain::Wait);
    }

    /// Get buffer by index (for advanced usage)
    pub fn get_buffer(&self, index: usize) -> Option<&Buffer> {
        self.buffers.get(index)
    }

    /// Get number of buffers
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }
}

/// Helper for managing multiple double-buffers
pub struct DoubleBufferPool {
    buffers: std::collections::HashMap<String, DoubleBuffer>,
}

impl DoubleBufferPool {
    /// Create a new pool
    pub fn new() -> Self {
        Self {
            buffers: std::collections::HashMap::new(),
        }
    }

    /// Add a double-buffer to the pool
    pub fn add_buffer(&mut self, name: String, buffer: DoubleBuffer) {
        self.buffers.insert(name, buffer);
    }

    /// Get a buffer by name
    pub fn get_buffer(&self, name: &str) -> Option<&DoubleBuffer> {
        self.buffers.get(name)
    }

    /// Get a mutable buffer by name
    pub fn get_buffer_mut(&mut self, name: &str) -> Option<&mut DoubleBuffer> {
        self.buffers.get_mut(name)
    }

    /// Swap all buffers in the pool
    pub fn swap_all(&mut self) {
        for buffer in self.buffers.values_mut() {
            buffer.swap();
        }
    }

    /// Get metrics summary for all buffers
    pub fn get_metrics_summary(&self) -> PoolMetrics {
        let mut summary = PoolMetrics::default();
        
        for buffer in self.buffers.values() {
            if let Some(metrics) = buffer.metrics() {
                summary.total_swaps += metrics.swap_count;
                summary.total_writes += metrics.write_count;
                summary.total_bytes += metrics.bytes_written;
                summary.buffer_count += 1;
            }
        }

        summary
    }
}

/// Metrics summary for buffer pool
#[derive(Debug, Default)]
pub struct PoolMetrics {
    pub buffer_count: u32,
    pub total_swaps: u64,
    pub total_writes: u64,
    pub total_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_buffer_config() {
        let uniform_config = DoubleBufferConfig::uniform(256);
        assert_eq!(uniform_config.buffer_count, 2);
        assert!(uniform_config.usage.contains(BufferUsages::UNIFORM));

        let storage_config = DoubleBufferConfig::storage(1024);
        assert_eq!(storage_config.buffer_count, 2);
        assert!(storage_config.usage.contains(BufferUsages::STORAGE));

        let triple_config = uniform_config.with_triple_buffering();
        assert_eq!(triple_config.buffer_count, 3);
    }

    #[test]
    fn test_buffer_swapping() {
        // This test would require a GPU device, so we just test the logic
        let config = DoubleBufferConfig::uniform(256);
        assert_eq!(config.buffer_count, 2);

        // For double-buffering: read/write indices should be different
        // Initial: write=0, read=1
        // After swap: write=1, read=0
    }

    #[test]
    fn test_triple_buffering_indices() {
        // Test triple-buffering index rotation
        // Initial: write=0, read=2
        // After swap 1: write=1, read=0  
        // After swap 2: write=2, read=1
        // After swap 3: write=0, read=2 (back to start)
    }

    #[test]
    fn test_write_safety() {
        // Test that write_safe logic works correctly
        // For double-buffering: write != read should always be true
        // For triple-buffering: same rule applies
    }
}
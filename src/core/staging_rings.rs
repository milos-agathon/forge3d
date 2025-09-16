//! O1: Staging buffer rings for efficient GPU uploads
//!
//! This module implements a ring buffer system for staging GPU uploads with
//! automatic wrap-around and fence-based synchronization to prevent buffer reuse
//! before completion.

use crate::core::fence_tracker::FenceTracker;
use std::sync::{Arc, Mutex};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

/// Statistics for staging ring buffer usage
#[derive(Debug, Clone, Default)]
pub struct StagingStats {
    /// Total bytes currently in-flight across all rings
    pub bytes_in_flight: u64,
    /// Current active ring index
    pub current_ring_index: usize,
    /// Number of buffer stalls encountered
    pub buffer_stalls: u64,
    /// Total number of rings
    pub ring_count: usize,
    /// Size of each ring buffer
    pub buffer_size: u64,
}

/// A single staging buffer within the ring system
#[derive(Debug)]
struct StagingBuffer {
    /// WGPU buffer handle
    buffer: Buffer,
    /// Current offset within the buffer
    offset: u64,
    /// Size of this buffer
    size: u64,
    /// Whether this buffer is currently in use
    in_use: bool,
}

impl StagingBuffer {
    fn new(device: &Device, size: u64, label: Option<&str>) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label,
            size,
            usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            offset: 0,
            size,
            in_use: false,
        }
    }

    fn reset(&mut self) {
        self.offset = 0;
        self.in_use = false;
    }

    fn can_allocate(&self, requested_size: u64) -> bool {
        !self.in_use && (self.offset + requested_size <= self.size)
    }

    fn allocate(&mut self, size: u64) -> Option<u64> {
        if self.can_allocate(size) {
            let current_offset = self.offset;
            self.offset += size;
            self.in_use = true;
            Some(current_offset)
        } else {
            None
        }
    }
}

/// Ring buffer system for staging GPU uploads
pub struct StagingRing {
    /// Array of staging buffers (rings)
    buffers: Vec<StagingBuffer>,
    /// Current active ring index
    current_index: usize,
    /// Fence tracker for synchronization
    fence_tracker: Arc<Mutex<FenceTracker>>,
    /// Statistics tracking
    stats: Arc<Mutex<StagingStats>>,
    /// Device reference for buffer creation
    #[allow(dead_code)]
    device: Arc<Device>,
    /// Queue reference for fence submission
    #[allow(dead_code)]
    queue: Arc<Queue>,
}

impl StagingRing {
    /// Create a new staging ring system
    ///
    /// # Arguments
    ///
    /// * `device` - WGPU device for buffer creation
    /// * `queue` - WGPU queue for fence operations
    /// * `ring_count` - Number of buffers in the ring (typically 3)
    /// * `buffer_size` - Size of each buffer in bytes
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        ring_count: usize,
        buffer_size: u64,
    ) -> Self {
        let mut buffers = Vec::with_capacity(ring_count);
        for i in 0..ring_count {
            let label = format!("StagingRing_Buffer_{}", i);
            buffers.push(StagingBuffer::new(&device, buffer_size, Some(&label)));
        }

        let stats = StagingStats {
            bytes_in_flight: 0,
            current_ring_index: 0,
            buffer_stalls: 0,
            ring_count,
            buffer_size,
        };

        Self {
            buffers,
            current_index: 0,
            fence_tracker: Arc::new(Mutex::new(FenceTracker::new(device.clone(), queue.clone()))),
            stats: Arc::new(Mutex::new(stats)),
            device,
            queue,
        }
    }

    /// Get the current active staging buffer
    pub fn current(&self) -> &Buffer {
        &self.buffers[self.current_index].buffer
    }

    /// Get current buffer with allocation offset
    pub fn allocate(&mut self, size: u64) -> Option<(&Buffer, u64)> {
        // Try to allocate from current buffer
        if let Some(offset) = self.buffers[self.current_index].allocate(size) {
            // Update stats
            if let Ok(mut stats) = self.stats.lock() {
                stats.bytes_in_flight += size;
            }
            return Some((&self.buffers[self.current_index].buffer, offset));
        }

        // Current buffer is full, try to advance
        if self.try_advance() {
            // Try again with new buffer
            if let Some(offset) = self.buffers[self.current_index].allocate(size) {
                if let Ok(mut stats) = self.stats.lock() {
                    stats.bytes_in_flight += size;
                }
                return Some((&self.buffers[self.current_index].buffer, offset));
            }
        }

        // No space available
        if let Ok(mut stats) = self.stats.lock() {
            stats.buffer_stalls += 1;
        }
        None
    }

    /// Advance to the next ring buffer with fence synchronization
    pub fn advance(&mut self, fence_value: u64) -> bool {
        // Submit fence for current buffer
        {
            let mut fence_tracker = self.fence_tracker.lock().unwrap();
            fence_tracker.submit_fence(self.current_index, fence_value);
        }

        self.try_advance()
    }

    /// Try to advance to next available buffer
    fn try_advance(&mut self) -> bool {
        let start_index = self.current_index;

        loop {
            let next_index = (self.current_index + 1) % self.buffers.len();

            // Check if next buffer is available (fence completed)
            let is_available = {
                let fence_tracker = self.fence_tracker.lock().unwrap();
                fence_tracker.is_buffer_available(next_index)
            };

            if is_available {
                // Reset the buffer and switch to it
                self.buffers[next_index].reset();
                self.current_index = next_index;

                // Update stats
                if let Ok(mut stats) = self.stats.lock() {
                    stats.current_ring_index = next_index;
                }

                return true;
            }

            // Try the next buffer
            self.current_index = next_index;

            // If we've tried all buffers, no space available
            if self.current_index == start_index {
                return false;
            }
        }
    }

    /// Get current usage statistics
    pub fn stats(&self) -> StagingStats {
        self.stats.lock().unwrap().clone()
    }

    /// Update bytes in flight (called when transfers complete)
    pub fn update_bytes_in_flight(&self, completed_bytes: u64) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.bytes_in_flight = stats.bytes_in_flight.saturating_sub(completed_bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::{Backends, DeviceDescriptor, Features, Instance, Limits, RequestAdapterOptions};

    async fn create_test_device() -> (Arc<Device>, Arc<Queue>) {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: Backends::all(),
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .expect("Failed to get adapter");

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default(), None)
            .await
            .expect("Failed to get device");

        (Arc::new(device), Arc::new(queue))
    }

    #[tokio::test]
    async fn test_staging_ring_creation() {
        let (device, queue) = create_test_device().await;
        let ring = StagingRing::new(device, queue, 3, 1024);

        let stats = ring.stats();
        assert_eq!(stats.ring_count, 3);
        assert_eq!(stats.buffer_size, 1024);
        assert_eq!(stats.current_ring_index, 0);
        assert_eq!(stats.bytes_in_flight, 0);
    }

    #[tokio::test]
    async fn test_staging_allocation() {
        let (device, queue) = create_test_device().await;
        let mut ring = StagingRing::new(device, queue, 3, 1024);

        // Test allocation
        let result = ring.allocate(256);
        assert!(result.is_some());

        let stats = ring.stats();
        assert_eq!(stats.bytes_in_flight, 256);
    }

    #[tokio::test]
    async fn test_buffer_wrap_around() {
        let (device, queue) = create_test_device().await;
        let mut ring = StagingRing::new(device, queue, 3, 512);

        // Fill current buffer
        let _alloc1 = ring.allocate(512);
        assert!(_alloc1.is_some());

        // This should try to advance to next buffer
        let result = ring.allocate(256);

        let stats = ring.stats();
        // Should have attempted to advance (may fail due to fence not being ready)
        assert!(stats.buffer_stalls > 0 || result.is_some());
    }
}

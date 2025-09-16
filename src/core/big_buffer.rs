//! Big buffer pattern for per-object data (I7)
//!
//! Implements a single large STORAGE buffer with dynamic offset addressing
//! to reduce bind group churn when rendering many objects. Uses 64-byte
//! alignment and provides RAII allocation management.

use crate::core::memory_tracker::ResourceRegistry;
use crate::error::RenderError;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use wgpu::{Buffer, BufferAddress, BufferDescriptor, BufferUsages, Device};

/// Size of each allocation block in bytes (64-byte aligned for WGSL std140)
pub const BIG_BUFFER_BLOCK_SIZE: u64 = 64;

/// Maximum size of the big buffer in bytes (128 MiB)
pub const BIG_BUFFER_MAX_SIZE: u64 = 128 * 1024 * 1024;

/// Handle to an allocated block within the big buffer
#[derive(Debug, Clone)]
pub struct BigBufferBlock {
    /// Offset in bytes from the start of the buffer
    pub offset: u32,
    /// Size in bytes of this block
    pub size: u32,
    /// Index for WGSL indexing (offset / BIG_BUFFER_BLOCK_SIZE)
    pub index: u32,
    /// Weak reference to the allocator for cleanup
    allocator: Weak<Mutex<BigBufferAllocator>>,
}

impl Drop for BigBufferBlock {
    fn drop(&mut self) {
        if let Some(allocator) = self.allocator.upgrade() {
            if let Ok(mut allocator) = allocator.lock() {
                allocator.deallocate_block(self.offset);
            }
        }
    }
}

/// Internal allocator state for managing blocks within the big buffer
struct BigBufferAllocator {
    /// Free blocks ordered by offset
    free_blocks: Vec<(u32, u32)>, // (offset, size)
    /// Allocated blocks
    allocated_blocks: HashMap<u32, u32>, // offset -> size
    /// Total buffer size
    #[allow(dead_code)]
    total_size: u32,
}

impl BigBufferAllocator {
    fn new(size: u32) -> Self {
        Self {
            free_blocks: vec![(0, size)],
            allocated_blocks: HashMap::new(),
            total_size: size,
        }
    }

    /// Allocate a block of the given size, returns offset and actual size
    fn allocate_block(&mut self, size: u32) -> Result<(u32, u32), RenderError> {
        // Align size to BIG_BUFFER_BLOCK_SIZE
        let aligned_size = ((size + BIG_BUFFER_BLOCK_SIZE as u32 - 1)
            / BIG_BUFFER_BLOCK_SIZE as u32)
            * BIG_BUFFER_BLOCK_SIZE as u32;

        // Find first fit
        for i in 0..self.free_blocks.len() {
            let (offset, block_size) = self.free_blocks[i];
            if block_size >= aligned_size {
                // Remove this free block
                self.free_blocks.remove(i);

                // If there's leftover space, add it back as a free block
                if block_size > aligned_size {
                    let remaining_offset = offset + aligned_size;
                    let remaining_size = block_size - aligned_size;
                    self.free_blocks.push((remaining_offset, remaining_size));
                    // Keep free blocks sorted by offset
                    self.free_blocks.sort_by_key(|&(o, _)| o);
                }

                self.allocated_blocks.insert(offset, aligned_size);
                return Ok((offset, aligned_size));
            }
        }

        Err(RenderError::Upload(format!(
            "BigBuffer allocation failed: requested {} bytes (aligned to {}), {} bytes available",
            size,
            aligned_size,
            self.total_free_bytes()
        )))
    }

    /// Deallocate a block at the given offset
    fn deallocate_block(&mut self, offset: u32) {
        if let Some(size) = self.allocated_blocks.remove(&offset) {
            // Add back to free blocks
            self.free_blocks.push((offset, size));
            // Sort by offset
            self.free_blocks.sort_by_key(|&(o, _)| o);
            // Merge adjacent free blocks
            self.merge_free_blocks();
        }
    }

    /// Merge adjacent free blocks to reduce fragmentation
    fn merge_free_blocks(&mut self) {
        if self.free_blocks.is_empty() {
            return;
        }

        let mut merged = Vec::new();
        let mut current = self.free_blocks[0];

        for &(offset, size) in &self.free_blocks[1..] {
            if current.0 + current.1 == offset {
                // Adjacent blocks, merge them
                current.1 += size;
            } else {
                // Not adjacent, add current to merged and start new current
                merged.push(current);
                current = (offset, size);
            }
        }
        merged.push(current);

        self.free_blocks = merged;
    }

    /// Get total free bytes
    fn total_free_bytes(&self) -> u32 {
        self.free_blocks.iter().map(|&(_, size)| size).sum()
    }
}

/// Big buffer for efficient per-object data storage
pub struct BigBuffer {
    /// The underlying GPU buffer
    buffer: Buffer,
    /// Allocator for managing blocks within the buffer
    allocator: Arc<Mutex<BigBufferAllocator>>,
    /// Size of the buffer in bytes
    size: u32,
}

impl BigBuffer {
    /// Create a new big buffer with the specified size
    pub fn new(
        device: &Device,
        size: u32,
        registry: Option<&ResourceRegistry>,
    ) -> Result<Self, RenderError> {
        if size == 0 || size > BIG_BUFFER_MAX_SIZE as u32 {
            return Err(RenderError::Upload(format!(
                "BigBuffer size must be between 1 and {} bytes, got {}",
                BIG_BUFFER_MAX_SIZE, size
            )));
        }

        // Ensure size is aligned to BIG_BUFFER_BLOCK_SIZE
        let aligned_size = ((size + BIG_BUFFER_BLOCK_SIZE as u32 - 1)
            / BIG_BUFFER_BLOCK_SIZE as u32)
            * BIG_BUFFER_BLOCK_SIZE as u32;

        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("BigBuffer"),
            size: aligned_size as BufferAddress,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Track allocation in registry
        if let Some(registry) = registry {
            registry.track_buffer_allocation(aligned_size as u64, false);
        }

        let allocator = Arc::new(Mutex::new(BigBufferAllocator::new(aligned_size)));

        Ok(Self {
            buffer,
            allocator,
            size: aligned_size,
        })
    }

    /// Allocate a block for per-object data
    pub fn allocate_block(&self, size: u32) -> Result<BigBufferBlock, RenderError> {
        let mut allocator = self
            .allocator
            .lock()
            .map_err(|_| RenderError::Upload("BigBuffer allocator lock poisoned".to_string()))?;

        let (offset, actual_size) = allocator.allocate_block(size)?;
        let index = offset / BIG_BUFFER_BLOCK_SIZE as u32;

        Ok(BigBufferBlock {
            offset,
            size: actual_size,
            index,
            allocator: Arc::downgrade(&self.allocator),
        })
    }

    /// Get the underlying GPU buffer
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get buffer size in bytes
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> BigBufferStats {
        let allocator = self.allocator.lock().unwrap();
        let free_bytes = allocator.total_free_bytes();
        let used_bytes = self.size - free_bytes;

        BigBufferStats {
            total_bytes: self.size,
            used_bytes,
            free_bytes,
            allocated_blocks: allocator.allocated_blocks.len() as u32,
            free_blocks: allocator.free_blocks.len() as u32,
            fragmentation_ratio: if self.size > 0 {
                allocator.free_blocks.len() as f32
                    / (self.size / BIG_BUFFER_BLOCK_SIZE as u32) as f32
            } else {
                0.0
            },
        }
    }
}

/// Memory statistics for the big buffer
#[derive(Debug, Clone)]
pub struct BigBufferStats {
    pub total_bytes: u32,
    pub used_bytes: u32,
    pub free_bytes: u32,
    pub allocated_blocks: u32,
    pub free_blocks: u32,
    pub fragmentation_ratio: f32,
}

/// Helper for dynamic offset addressing
pub fn calculate_dynamic_offset(block: &BigBufferBlock) -> u32 {
    block.offset
}

/// Helper for index addressing (for WGSL array indexing)
pub fn calculate_index_address(block: &BigBufferBlock) -> u32 {
    block.index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment() {
        assert_eq!(BIG_BUFFER_BLOCK_SIZE, 64);

        // Test aligned sizes
        assert_eq!(((63 + 63) / 64) * 64, 64);
        assert_eq!(((64 + 63) / 64) * 64, 64);
        assert_eq!(((65 + 63) / 64) * 64, 128);
    }

    #[test]
    fn test_allocator_basic() {
        let mut allocator = BigBufferAllocator::new(256);

        // Allocate first block
        let (offset1, size1) = allocator.allocate_block(32).unwrap();
        assert_eq!(offset1, 0);
        assert_eq!(size1, 64); // Aligned to BIG_BUFFER_BLOCK_SIZE

        // Allocate second block
        let (offset2, size2) = allocator.allocate_block(32).unwrap();
        assert_eq!(offset2, 64);
        assert_eq!(size2, 64);

        // Deallocate first block
        allocator.deallocate_block(offset1);

        // Should be able to allocate in the freed space
        let (offset3, size3) = allocator.allocate_block(32).unwrap();
        assert_eq!(offset3, 0);
        assert_eq!(size3, 64);
    }

    #[test]
    fn test_allocator_merge() {
        let mut allocator = BigBufferAllocator::new(256);

        // Allocate all blocks
        let (o1, _) = allocator.allocate_block(64).unwrap();
        let (o2, _) = allocator.allocate_block(64).unwrap();
        let (o3, _) = allocator.allocate_block(64).unwrap();
        let (o4, _) = allocator.allocate_block(64).unwrap();

        // Free middle blocks
        allocator.deallocate_block(o2);
        allocator.deallocate_block(o3);

        // Should merge into one block
        assert_eq!(allocator.free_blocks.len(), 1);
        assert_eq!(allocator.free_blocks[0], (64, 128));
    }

    #[test]
    fn test_index_calculation() {
        let block = BigBufferBlock {
            offset: 128,
            size: 64,
            index: 2,
            allocator: Weak::new(),
        };

        assert_eq!(calculate_dynamic_offset(&block), 128);
        assert_eq!(calculate_index_address(&block), 2);
    }
}

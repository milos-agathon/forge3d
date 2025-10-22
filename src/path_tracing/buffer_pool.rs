// src/path_tracing/buffer_pool.rs
// Buffer pool for reusing GPU allocations across tiles to minimize overhead

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

/// Buffer pool key (identifies buffer type and size)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferKey {
    pub usage: BufferUsageFlags,
    pub size: u64,
}

/// Simplified buffer usage flags for pool key
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferUsageFlags(u32);

impl BufferUsageFlags {
    pub const VERTEX: Self = Self(1 << 0);
    pub const INDEX: Self = Self(1 << 1);
    pub const STORAGE: Self = Self(1 << 2);
    pub const UNIFORM: Self = Self(1 << 3);
    pub const STAGING: Self = Self(1 << 4);
    pub const COPY_SRC: Self = Self(1 << 5);
    pub const COPY_DST: Self = Self(1 << 6);
    
    pub fn from_wgpu(usage: BufferUsages) -> Self {
        let mut flags = 0u32;
        if usage.contains(BufferUsages::VERTEX) { flags |= 1 << 0; }
        if usage.contains(BufferUsages::INDEX) { flags |= 1 << 1; }
        if usage.contains(BufferUsages::STORAGE) { flags |= 1 << 2; }
        if usage.contains(BufferUsages::UNIFORM) { flags |= 1 << 3; }
        if usage.contains(BufferUsages::MAP_READ) { flags |= 1 << 4; }
        if usage.contains(BufferUsages::COPY_SRC) { flags |= 1 << 5; }
        if usage.contains(BufferUsages::COPY_DST) { flags |= 1 << 6; }
        Self(flags)
    }
    
    pub fn to_wgpu(self) -> BufferUsages {
        let mut usage = BufferUsages::empty();
        if self.0 & (1 << 0) != 0 { usage |= BufferUsages::VERTEX; }
        if self.0 & (1 << 1) != 0 { usage |= BufferUsages::INDEX; }
        if self.0 & (1 << 2) != 0 { usage |= BufferUsages::STORAGE; }
        if self.0 & (1 << 3) != 0 { usage |= BufferUsages::UNIFORM; }
        if self.0 & (1 << 4) != 0 { usage |= BufferUsages::MAP_READ; }
        if self.0 & (1 << 5) != 0 { usage |= BufferUsages::COPY_SRC; }
        if self.0 & (1 << 6) != 0 { usage |= BufferUsages::COPY_DST; }
        usage
    }
}

/// Pooled buffer with metadata
struct PooledBuffer {
    buffer: Arc<Buffer>,
    in_use: bool,
}

/// Buffer pool for reusing allocations across tiles
pub struct BufferPool {
    pools: HashMap<BufferKey, Vec<PooledBuffer>>,
    device: std::sync::Arc<Device>,
    total_allocated: u64,
    hit_count: usize,
    miss_count: usize,
}

impl BufferPool {
    pub fn new(device: std::sync::Arc<Device>) -> Self {
        Self {
            pools: HashMap::new(),
            device,
            total_allocated: 0,
            hit_count: 0,
            miss_count: 0,
        }
    }
    
    /// Acquire a buffer from the pool or create a new one
    pub fn acquire(&mut self, usage: BufferUsages, size: u64, label: Option<&str>) -> Arc<Buffer> {
        let key = BufferKey {
            usage: BufferUsageFlags::from_wgpu(usage),
            size,
        };
        
        // Try to find an available buffer in the pool
        if let Some(pool) = self.pools.get_mut(&key) {
            for entry in pool.iter_mut() {
                if !entry.in_use {
                    entry.in_use = true;
                    self.hit_count += 1;
                    return Arc::clone(&entry.buffer);
                }
            }
        }
        
        // No available buffer, create a new one
        self.miss_count += 1;
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label,
            size,
            usage,
            mapped_at_creation: false,
        });
        
        self.total_allocated += size;
        
        // Wrap in Arc and add to pool
        let buffer_arc = Arc::new(buffer);
        let pool = self.pools.entry(key).or_insert_with(Vec::new);
        pool.push(PooledBuffer {
            buffer: Arc::clone(&buffer_arc),
            in_use: true,
        });
        
        buffer_arc
    }
    
    /// Release a buffer back to the pool
    pub fn release(&mut self, usage: BufferUsages, size: u64) {
        let key = BufferKey {
            usage: BufferUsageFlags::from_wgpu(usage),
            size,
        };
        
        if let Some(pool) = self.pools.get_mut(&key) {
            // Mark first in-use buffer as available
            for entry in pool.iter_mut() {
                if entry.in_use {
                    entry.in_use = false;
                    break;
                }
            }
        }
    }
    
    /// Release all buffers (mark as not in use)
    pub fn release_all(&mut self) {
        for pool in self.pools.values_mut() {
            for entry in pool.iter_mut() {
                entry.in_use = false;
            }
        }
    }
    
    /// Print pool statistics
    pub fn print_stats(&self) {
        let total_mb = self.total_allocated as f64 / (1024.0 * 1024.0);
        let hit_rate = if self.hit_count + self.miss_count > 0 {
            (self.hit_count as f64) / ((self.hit_count + self.miss_count) as f64) * 100.0
        } else {
            0.0
        };
        
        eprintln!("[BufferPool] Stats:");
        eprintln!("  Total allocated: {:.2} MB", total_mb);
        eprintln!("  Hit rate: {:.1}% ({} hits, {} misses)", 
            hit_rate, self.hit_count, self.miss_count);
        eprintln!("  Pool count: {}", self.pools.len());
    }
    
    /// Clear the pool (drop all buffers)
    pub fn clear(&mut self) {
        self.pools.clear();
        self.total_allocated = 0;
        self.hit_count = 0;
        self.miss_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_buffer_usage_flags_conversion() {
        let usage = BufferUsages::VERTEX | BufferUsages::COPY_DST;
        let flags = BufferUsageFlags::from_wgpu(usage);
        let converted = flags.to_wgpu();
        assert!(converted.contains(BufferUsages::VERTEX));
        assert!(converted.contains(BufferUsages::COPY_DST));
    }
}

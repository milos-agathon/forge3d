use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};
use super::types::{DefragStats, MemoryPoolStats};

/// A single allocated block from a memory pool
#[derive(Debug, Clone)]
pub struct PoolBlock {
    /// Unique ID for this block
    pub id: u64,
    /// Size of the block in bytes
    pub size: u64,
    /// Buffer offset within the pool
    pub offset: u64,
    /// Reference count
    ref_count: Arc<Mutex<u32>>,
    /// Pool ID this block belongs to
    pool_id: u8,
    /// Weak reference to the pool manager
    pool_manager: std::sync::Weak<Mutex<MemoryPoolManager>>,
}

impl PoolBlock {
    /// Increment reference count
    pub fn add_ref(&self) {
        if let Ok(mut count) = self.ref_count.lock() {
            *count += 1;
        }
    }

    /// Decrement reference count, returns true if count reaches zero
    pub fn release(&self) -> bool {
        if let Ok(mut count) = self.ref_count.lock() {
            if *count > 0 {
                *count -= 1;
                *count == 0
            } else {
                true
            }
        } else {
            true
        }
    }

    /// Get current reference count
    pub fn ref_count(&self) -> u32 {
        self.ref_count.lock().map(|count| *count).unwrap_or(0)
    }
}

impl Drop for PoolBlock {
    fn drop(&mut self) {
        // Return block to pool when dropped with zero references
        if self.release() {
            if let Some(manager) = self.pool_manager.upgrade() {
                if let Ok(mut manager) = manager.lock() {
                    manager.return_block(self.pool_id, self.offset, self.size);
                }
            }
        }
    }
}

/// A single memory pool for a specific size bucket
struct MemoryPool {
    /// GPU buffer for this pool
    #[allow(dead_code)]
    buffer: Buffer,
    /// Size of each allocation in this pool
    allocation_size: u64,
    /// Total pool size
    total_size: u64,
    /// Free blocks (offset, size)
    free_blocks: Vec<(u64, u64)>,
    /// Allocated blocks (offset -> (size, ref_count))
    allocated_blocks: HashMap<u64, (u64, Arc<Mutex<u32>>)>,
    /// Next unique block ID
    next_block_id: u64,
}

impl MemoryPool {
    fn new(device: &Device, allocation_size: u64, pool_size: u64, pool_id: u8) -> Self {
        // Ensure 64-byte alignment
        let aligned_size = ((allocation_size + 63) / 64) * 64;
        let aligned_pool_size = ((pool_size + 63) / 64) * 64;

        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("MemoryPool_{}", pool_id)),
            size: aligned_pool_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Initially, the entire buffer is one free block
        let free_blocks = vec![(0, aligned_pool_size)];

        Self {
            buffer,
            allocation_size: aligned_size,
            total_size: aligned_pool_size,
            free_blocks,
            allocated_blocks: HashMap::new(),
            next_block_id: 1,
        }
    }

    fn allocate_block(&mut self) -> Option<(u64, u64, Arc<Mutex<u32>>)> {
        // Find a free block that can fit our allocation
        for i in 0..self.free_blocks.len() {
            let (offset, size) = self.free_blocks[i];
            if size >= self.allocation_size {
                // Remove this free block
                self.free_blocks.remove(i);

                // If there's leftover space, add it back
                if size > self.allocation_size {
                    let remaining_offset = offset + self.allocation_size;
                    let remaining_size = size - self.allocation_size;
                    self.free_blocks.push((remaining_offset, remaining_size));
                    self.free_blocks.sort_by_key(|&(o, _)| o);
                }

                let block_id = self.next_block_id;
                self.next_block_id += 1;

                let ref_count = Arc::new(Mutex::new(1));
                self.allocated_blocks
                    .insert(offset, (self.allocation_size, ref_count.clone()));

                return Some((block_id, offset, ref_count));
            }
        }
        None
    }

    fn free_block(&mut self, offset: u64) -> bool {
        if let Some((size, _)) = self.allocated_blocks.remove(&offset) {
            // Add back to free blocks
            self.free_blocks.push((offset, size));
            self.free_blocks.sort_by_key(|&(o, _)| o);
            self.merge_free_blocks();
            true
        } else {
            false
        }
    }

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
                merged.push(current);
                current = (offset, size);
            }
        }
        merged.push(current);

        self.free_blocks = merged;
    }

    fn fragmentation_ratio(&self) -> f32 {
        if self.total_size == 0 {
            return 0.0;
        }

        let used_bytes: u64 = self.allocated_blocks.values().map(|(size, _)| *size).sum();
        if used_bytes == 0 {
            return 0.0;
        }

        // Fragmentation is roughly the ratio of free block count to theoretical minimum
        let free_space: u64 = self.free_blocks.iter().map(|(_, size)| *size).sum();
        if free_space == 0 {
            return 0.0;
        }

        // More free blocks = more fragmentation
        (self.free_blocks.len() as f32) / ((self.total_size / self.allocation_size) as f32).max(1.0)
    }
}

/// Manager for multiple memory pools with different size buckets
pub struct MemoryPoolManager {
    /// Individual pools for different size buckets
    pools: Vec<MemoryPool>,
    /// Size buckets (power of two from 64B to 8MB)
    size_buckets: Vec<u64>,
    /// Pool statistics
    stats: MemoryPoolStats,
}

impl MemoryPoolManager {
    /// Create a new memory pool manager with power-of-two size buckets
    pub fn new(device: &Device) -> Self {
        // Create power-of-two buckets from 64B to 8MB
        let size_buckets: Vec<u64> = (6..24).map(|i| 1u64 << i).collect(); // 2^6 to 2^23

        let mut pools = Vec::new();
        for (pool_id, &bucket_size) in size_buckets.iter().enumerate() {
            let pool_size = bucket_size * 1024; // 1024 blocks per pool
            let pool = MemoryPool::new(device, bucket_size, pool_size, pool_id as u8);
            pools.push(pool);
        }

        let pool_count = size_buckets.len() as u32;

        Self {
            pools,
            size_buckets,
            stats: MemoryPoolStats {
                pool_count,
                ..Default::default()
            },
        }
    }

    /// Allocate a block from the appropriate size bucket
    pub fn allocate_bucket(&mut self, size: u32) -> Result<PoolBlock, String> {
        let size = size as u64;

        // Find the appropriate bucket (smallest that fits)
        let bucket_index = self
            .size_buckets
            .iter()
            .position(|&bucket_size| bucket_size >= size)
            .ok_or_else(|| format!("Allocation size {} exceeds maximum bucket size", size))?;

        // Try to allocate from the bucket
        if let Some((id, offset, ref_count)) = self.pools[bucket_index].allocate_block() {
            self.stats.total_allocated += self.size_buckets[bucket_index];
            self.stats.active_blocks += 1;

            Ok(PoolBlock {
                id,
                size: self.size_buckets[bucket_index],
                offset,
                ref_count,
                pool_id: bucket_index as u8,
                pool_manager: std::sync::Weak::new(), // Initialize empty weak reference
            })
        } else {
            Err(format!(
                "Failed to allocate {} bytes from pool bucket {}",
                size, bucket_index
            ))
        }
    }

    /// Return a block to its pool (called by PoolBlock::drop)
    pub(crate) fn return_block(&mut self, pool_id: u8, offset: u64, size: u64) {
        if let Some(pool) = self.pools.get_mut(pool_id as usize) {
            if pool.free_block(offset) {
                self.stats.total_freed += size;
                self.stats.active_blocks = self.stats.active_blocks.saturating_sub(1);
            }
        }
    }

    /// Perform defragmentation on all pools
    pub fn defragment(&mut self) -> DefragStats {
        let start_time = std::time::Instant::now();
        let mut stats = DefragStats::default();

        // Calculate fragmentation before
        let frag_before: f32 = self
            .pools
            .iter()
            .map(|pool| pool.fragmentation_ratio())
            .sum::<f32>()
            / self.pools.len() as f32;

        stats.fragmentation_before = frag_before;

        // Perform defragmentation on each pool
        for pool in &mut self.pools {
            // Merge free blocks (basic defragmentation)
            let blocks_before = pool.free_blocks.len();
            pool.merge_free_blocks();
            let blocks_after = pool.free_blocks.len();

            stats.blocks_moved += (blocks_before - blocks_after) as u32;

            // Calculate bytes compacted (rough estimate)
            let free_bytes: u64 = pool.free_blocks.iter().map(|(_, size)| *size).sum();
            stats.bytes_compacted += free_bytes / pool.free_blocks.len().max(1) as u64;
        }

        // Calculate fragmentation after
        let frag_after: f32 = self
            .pools
            .iter()
            .map(|pool| pool.fragmentation_ratio())
            .sum::<f32>()
            / self.pools.len() as f32;

        stats.fragmentation_after = frag_after;
        stats.time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        stats
    }

    /// Get current memory pool statistics
    pub fn get_stats(&mut self) -> MemoryPoolStats {
        // Update fragmentation ratio
        let total_frag: f32 = self
            .pools
            .iter()
            .map(|pool| pool.fragmentation_ratio())
            .sum::<f32>();

        self.stats.fragmentation_ratio = total_frag / self.pools.len() as f32;

        // Find largest free block
        self.stats.largest_free_block = self
            .pools
            .iter()
            .flat_map(|pool| &pool.free_blocks)
            .map(|(_, size)| *size)
            .max()
            .unwrap_or(0);

        self.stats.clone()
    }
}

/// Global memory pool manager instance
static GLOBAL_POOL_MANAGER: std::sync::OnceLock<Arc<Mutex<MemoryPoolManager>>> =
    std::sync::OnceLock::new();

/// Initialize global memory pool manager
pub fn init_global_pools(device: &Device) -> Arc<Mutex<MemoryPoolManager>> {
    GLOBAL_POOL_MANAGER
        .get_or_init(|| Arc::new(Mutex::new(MemoryPoolManager::new(device))))
        .clone()
}

/// Get reference to global memory pool manager
pub fn global_pools() -> Option<&'static Arc<Mutex<MemoryPoolManager>>> {
    GLOBAL_POOL_MANAGER.get()
}

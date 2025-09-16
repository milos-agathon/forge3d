use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, TextureFormat};

/// Global memory tracking registry for GPU resources
pub struct ResourceRegistry {
    // Atomic counters for thread-safe tracking
    buffer_count: AtomicU32,
    texture_count: AtomicU32,
    buffer_bytes: AtomicU64,
    texture_bytes: AtomicU64,
    host_visible_bytes: AtomicU64,

    // Budget limit (512 MiB)
    budget_limit: u64,
}

/// Memory metrics returned to Python
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    pub buffer_count: u32,
    pub texture_count: u32,
    pub buffer_bytes: u64,
    pub texture_bytes: u64,
    pub host_visible_bytes: u64,
    pub total_bytes: u64,
    pub limit_bytes: u64,
    pub within_budget: bool,
    pub utilization_ratio: f64,
}

impl ResourceRegistry {
    /// Create new registry with 512 MiB budget limit
    pub fn new() -> Self {
        Self {
            buffer_count: AtomicU32::new(0),
            texture_count: AtomicU32::new(0),
            buffer_bytes: AtomicU64::new(0),
            texture_bytes: AtomicU64::new(0),
            host_visible_bytes: AtomicU64::new(0),
            budget_limit: 512 * 1024 * 1024, // 512 MiB
        }
    }

    /// Track a buffer allocation
    pub fn track_buffer_allocation(&self, size: u64, is_host_visible: bool) {
        self.buffer_count.fetch_add(1, Ordering::Relaxed);
        self.buffer_bytes.fetch_add(size, Ordering::Relaxed);

        if is_host_visible {
            self.host_visible_bytes.fetch_add(size, Ordering::Relaxed);
        }
    }

    /// Free a buffer allocation
    pub fn free_buffer_allocation(&self, size: u64, is_host_visible: bool) {
        self.buffer_count.fetch_sub(1, Ordering::Relaxed);
        self.buffer_bytes.fetch_sub(size, Ordering::Relaxed);

        if is_host_visible {
            self.host_visible_bytes.fetch_sub(size, Ordering::Relaxed);
        }
    }

    /// Track a texture allocation
    pub fn track_texture_allocation(&self, width: u32, height: u32, format: TextureFormat) {
        let size = calculate_texture_size(width, height, format);

        self.texture_count.fetch_add(1, Ordering::Relaxed);
        self.texture_bytes.fetch_add(size, Ordering::Relaxed);
        // Textures are typically not host-visible in our usage
    }

    /// Free a texture allocation
    pub fn free_texture_allocation(&self, width: u32, height: u32, format: TextureFormat) {
        let size = calculate_texture_size(width, height, format);

        self.texture_count.fetch_sub(1, Ordering::Relaxed);
        self.texture_bytes.fetch_sub(size, Ordering::Relaxed);
    }

    /// Get current memory metrics
    pub fn get_metrics(&self) -> MemoryMetrics {
        let buffer_count = self.buffer_count.load(Ordering::Relaxed);
        let texture_count = self.texture_count.load(Ordering::Relaxed);
        let buffer_bytes = self.buffer_bytes.load(Ordering::Relaxed);
        let texture_bytes = self.texture_bytes.load(Ordering::Relaxed);
        let host_visible_bytes = self.host_visible_bytes.load(Ordering::Relaxed);
        let total_bytes = buffer_bytes + texture_bytes;
        let within_budget = host_visible_bytes <= self.budget_limit;
        let utilization_ratio = host_visible_bytes as f64 / self.budget_limit as f64;

        MemoryMetrics {
            buffer_count,
            texture_count,
            buffer_bytes,
            texture_bytes,
            host_visible_bytes,
            total_bytes,
            limit_bytes: self.budget_limit,
            within_budget,
            utilization_ratio,
        }
    }

    /// Get budget limit
    pub fn get_budget_limit(&self) -> u64 {
        self.budget_limit
    }

    /// Check if allocation would exceed budget
    pub fn check_budget(&self, additional_host_visible: u64) -> Result<(), String> {
        let current = self.host_visible_bytes.load(Ordering::Relaxed);
        if current + additional_host_visible > self.budget_limit {
            return Err(format!(
                "Memory budget exceeded: current {} bytes + requested {} bytes would exceed limit of {} bytes",
                current, additional_host_visible, self.budget_limit
            ));
        }
        Ok(())
    }
}

/// Calculate texture size in bytes based on dimensions and format
///
/// Supports all WebGPU texture formats with accurate byte-per-pixel calculations.
/// For compressed formats, this calculates uncompressed size as an approximation.
fn calculate_texture_size(width: u32, height: u32, format: TextureFormat) -> u64 {
    let bytes_per_pixel = match format {
        // 8-bit formats
        TextureFormat::R8Unorm
        | TextureFormat::R8Snorm
        | TextureFormat::R8Uint
        | TextureFormat::R8Sint => 1,

        // 16-bit formats (2 bytes)
        TextureFormat::Rg8Unorm
        | TextureFormat::Rg8Snorm
        | TextureFormat::Rg8Uint
        | TextureFormat::Rg8Sint => 2,
        TextureFormat::R16Uint | TextureFormat::R16Sint | TextureFormat::R16Float => 2,
        TextureFormat::Depth16Unorm => 2,

        // 32-bit formats (4 bytes)
        TextureFormat::Rgba8Unorm
        | TextureFormat::Rgba8UnormSrgb
        | TextureFormat::Rgba8Snorm
        | TextureFormat::Rgba8Uint
        | TextureFormat::Rgba8Sint => 4,
        TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb => 4,
        TextureFormat::Rgb10a2Unorm | TextureFormat::Rgb10a2Uint => 4,
        TextureFormat::Rg11b10Float => 4,
        TextureFormat::Rg16Uint | TextureFormat::Rg16Sint | TextureFormat::Rg16Float => 4,
        TextureFormat::R32Uint | TextureFormat::R32Sint | TextureFormat::R32Float => 4,
        TextureFormat::Depth32Float => 4,
        TextureFormat::Depth24Plus => 4, // Usually 32-bit internally
        TextureFormat::Depth24PlusStencil8 => 4, // 24-bit depth + 8-bit stencil

        // 64-bit formats (8 bytes)
        TextureFormat::Rgba16Uint | TextureFormat::Rgba16Sint | TextureFormat::Rgba16Float => 8,
        TextureFormat::Rg32Uint | TextureFormat::Rg32Sint | TextureFormat::Rg32Float => 8,
        TextureFormat::Depth32FloatStencil8 => 8, // 32-bit depth + 8-bit stencil + padding

        // 128-bit formats (16 bytes)
        TextureFormat::Rgba32Uint | TextureFormat::Rgba32Sint | TextureFormat::Rgba32Float => 16,

        // Compressed formats (approximate uncompressed size for memory estimation)
        // BC1 (DXT1) - 4:1 compression ratio, 4x4 blocks with 8 bytes per block
        TextureFormat::Bc1RgbaUnorm | TextureFormat::Bc1RgbaUnormSrgb => {
            return calculate_compressed_texture_size(width, height, 8, 4);
        }

        // BC2 (DXT3) - 2:1 compression ratio, 4x4 blocks with 16 bytes per block
        TextureFormat::Bc2RgbaUnorm | TextureFormat::Bc2RgbaUnormSrgb => {
            return calculate_compressed_texture_size(width, height, 16, 4);
        }

        // BC3 (DXT5) - 2:1 compression ratio, 4x4 blocks with 16 bytes per block
        TextureFormat::Bc3RgbaUnorm | TextureFormat::Bc3RgbaUnormSrgb => {
            return calculate_compressed_texture_size(width, height, 16, 4);
        }

        // BC4 - Single channel, 4x4 blocks with 8 bytes per block
        TextureFormat::Bc4RUnorm | TextureFormat::Bc4RSnorm => {
            return calculate_compressed_texture_size(width, height, 8, 4);
        }

        // BC5 - Two channel, 4x4 blocks with 16 bytes per block
        TextureFormat::Bc5RgUnorm | TextureFormat::Bc5RgSnorm => {
            return calculate_compressed_texture_size(width, height, 16, 4);
        }

        // BC6H - HDR compression, 4x4 blocks with 16 bytes per block
        TextureFormat::Bc6hRgbUfloat | TextureFormat::Bc6hRgbFloat => {
            return calculate_compressed_texture_size(width, height, 16, 4);
        }

        // BC7 - High quality compression, 4x4 blocks with 16 bytes per block
        TextureFormat::Bc7RgbaUnorm | TextureFormat::Bc7RgbaUnormSrgb => {
            return calculate_compressed_texture_size(width, height, 16, 4);
        }

        // ETC2 compression formats - 4x4 blocks
        TextureFormat::Etc2Rgb8Unorm | TextureFormat::Etc2Rgb8UnormSrgb => {
            return calculate_compressed_texture_size(width, height, 8, 4);
        }
        TextureFormat::Etc2Rgb8A1Unorm | TextureFormat::Etc2Rgb8A1UnormSrgb => {
            return calculate_compressed_texture_size(width, height, 8, 4);
        }
        TextureFormat::Etc2Rgba8Unorm | TextureFormat::Etc2Rgba8UnormSrgb => {
            return calculate_compressed_texture_size(width, height, 16, 4);
        }
        TextureFormat::EacR11Unorm | TextureFormat::EacR11Snorm => {
            return calculate_compressed_texture_size(width, height, 8, 4);
        }
        TextureFormat::EacRg11Unorm | TextureFormat::EacRg11Snorm => {
            return calculate_compressed_texture_size(width, height, 16, 4);
        }

        // ASTC compression formats - variable block sizes (using most common 4x4 blocks)
        TextureFormat::Astc { .. } => {
            // ASTC block size varies, but 4x4 with 16 bytes per block is most common
            return calculate_compressed_texture_size(width, height, 16, 4);
        }

        // Fallback for any formats not explicitly handled
        _ => {
            // Conservative estimate: assume 4 bytes per pixel for unknown formats
            // This prevents underestimating memory usage
            4
        }
    };

    (width as u64) * (height as u64) * bytes_per_pixel
}

/// Calculate compressed texture size in bytes
///
/// # Parameters
/// - `width`, `height`: Texture dimensions in pixels
/// - `bytes_per_block`: Number of bytes per compression block
/// - `block_size`: Size of compression block (e.g., 4 for 4x4 blocks)
fn calculate_compressed_texture_size(
    width: u32,
    height: u32,
    bytes_per_block: u64,
    block_size: u32,
) -> u64 {
    let blocks_x = (width + block_size - 1) / block_size;
    let blocks_y = (height + block_size - 1) / block_size;
    (blocks_x as u64) * (blocks_y as u64) * bytes_per_block
}

/// Check if buffer usage indicates host-visible memory
pub fn is_host_visible_usage(usage: BufferUsages) -> bool {
    usage.contains(BufferUsages::MAP_READ) || usage.contains(BufferUsages::MAP_WRITE)
}

/// Global singleton registry instance
static GLOBAL_REGISTRY: std::sync::OnceLock<ResourceRegistry> = std::sync::OnceLock::new();

/// Get reference to global memory tracker
pub fn global_tracker() -> &'static ResourceRegistry {
    GLOBAL_REGISTRY.get_or_init(|| ResourceRegistry::new())
}

/// O2: Memory Pool System with size-bucket allocation
///
/// Provides efficient memory allocation using power-of-two size buckets
/// with reference counting and defragmentation capabilities.

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

/// Statistics from defragmentation operation
#[derive(Debug, Clone, Default)]
pub struct DefragStats {
    /// Number of blocks moved during defragmentation
    pub blocks_moved: u32,
    /// Total bytes compacted
    pub bytes_compacted: u64,
    /// Time taken in milliseconds
    pub time_ms: f64,
    /// Fragmentation ratio before defrag (0.0-1.0)
    pub fragmentation_before: f32,
    /// Fragmentation ratio after defrag (0.0-1.0)
    pub fragmentation_after: f32,
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

/// Statistics for the memory pool system
#[derive(Debug, Clone, Default)]
pub struct MemoryPoolStats {
    /// Total bytes allocated from pools
    pub total_allocated: u64,
    /// Total bytes freed back to pools
    pub total_freed: u64,
    /// Current fragmentation ratio (0.0-1.0)
    pub fragmentation_ratio: f32,
    /// Number of currently active blocks
    pub active_blocks: u32,
    /// Number of memory pools
    pub pool_count: u32,
    /// Size of largest free block
    pub largest_free_block: u64,
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
    fn return_block(&mut self, pool_id: u8, offset: u64, size: u64) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_basic_operations() {
        let registry = ResourceRegistry::new();

        // Initial state
        let metrics = registry.get_metrics();
        assert_eq!(metrics.buffer_count, 0);
        assert_eq!(metrics.buffer_bytes, 0);
        assert!(metrics.within_budget);

        // Track allocation
        registry.track_buffer_allocation(1024, true);
        let metrics = registry.get_metrics();
        assert_eq!(metrics.buffer_count, 1);
        assert_eq!(metrics.buffer_bytes, 1024);
        assert_eq!(metrics.host_visible_bytes, 1024);

        // Free allocation
        registry.free_buffer_allocation(1024, true);
        let metrics = registry.get_metrics();
        assert_eq!(metrics.buffer_count, 0);
        assert_eq!(metrics.buffer_bytes, 0);
        assert_eq!(metrics.host_visible_bytes, 0);
    }

    #[test]
    fn test_budget_checking() {
        let registry = ResourceRegistry::new();

        // Should pass within budget
        assert!(registry.check_budget(100 * 1024 * 1024).is_ok());

        // Should fail when exceeding budget
        assert!(registry.check_budget(600 * 1024 * 1024).is_err());
    }

    #[test]
    fn test_host_visible_detection() {
        assert!(is_host_visible_usage(BufferUsages::MAP_READ));
        assert!(is_host_visible_usage(BufferUsages::MAP_WRITE));
        assert!(is_host_visible_usage(
            BufferUsages::COPY_DST | BufferUsages::MAP_READ
        ));
        assert!(!is_host_visible_usage(BufferUsages::VERTEX));
        assert!(!is_host_visible_usage(BufferUsages::INDEX));
    }

    #[test]
    fn test_texture_format_sizes() {
        use super::calculate_texture_size;

        // Test 8-bit formats (1 byte per pixel)
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::R8Unorm),
            16 * 16 * 1
        );

        // Test 16-bit formats (2 bytes per pixel)
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Rg8Unorm),
            16 * 16 * 2
        );
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::R16Float),
            16 * 16 * 2
        );
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Depth16Unorm),
            16 * 16 * 2
        );

        // Test 32-bit formats (4 bytes per pixel)
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Rgba8Unorm),
            16 * 16 * 4
        );
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Rgba8UnormSrgb),
            16 * 16 * 4
        );
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Bgra8Unorm),
            16 * 16 * 4
        );
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::R32Float),
            16 * 16 * 4
        );
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Depth32Float),
            16 * 16 * 4
        );
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Depth24Plus),
            16 * 16 * 4
        );
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Depth24PlusStencil8),
            16 * 16 * 4
        );

        // Test 64-bit formats (8 bytes per pixel)
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Rgba16Float),
            16 * 16 * 8
        );
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Rg32Float),
            16 * 16 * 8
        );
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Depth32FloatStencil8),
            16 * 16 * 8
        );

        // Test 128-bit formats (16 bytes per pixel)
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Rgba32Float),
            16 * 16 * 16
        );
    }

    #[test]
    fn test_compressed_texture_sizes() {
        use super::calculate_texture_size;

        // Test BC1 (8 bytes per 4x4 block)
        // 16x16 = 4x4 blocks total, each block is 8 bytes
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Bc1RgbaUnorm),
            4 * 4 * 8
        );

        // Test BC3/BC5 (16 bytes per 4x4 block)
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Bc3RgbaUnorm),
            4 * 4 * 16
        );
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Bc5RgUnorm),
            4 * 4 * 16
        );

        // Test non-aligned size (17x17 requires 5x5 blocks)
        assert_eq!(
            calculate_texture_size(17, 17, TextureFormat::Bc1RgbaUnorm),
            5 * 5 * 8
        );

        // Test ETC2 formats
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Etc2Rgb8Unorm),
            4 * 4 * 8
        );
        assert_eq!(
            calculate_texture_size(16, 16, TextureFormat::Etc2Rgba8Unorm),
            4 * 4 * 16
        );
    }

    #[test]
    fn test_compressed_texture_size_calculation() {
        use super::calculate_compressed_texture_size;

        // Perfect alignment: 16x16 with 4x4 blocks
        assert_eq!(calculate_compressed_texture_size(16, 16, 8, 4), 4 * 4 * 8);

        // Non-aligned: 15x15 should round up to 4x4 blocks (16x16)
        assert_eq!(calculate_compressed_texture_size(15, 15, 8, 4), 4 * 4 * 8);

        // Non-aligned: 17x17 should round up to 5x5 blocks (20x20)
        assert_eq!(calculate_compressed_texture_size(17, 17, 8, 4), 5 * 5 * 8);

        // Different block size: 8x8 blocks
        assert_eq!(calculate_compressed_texture_size(16, 16, 16, 8), 2 * 2 * 16);
    }

    #[test]
    fn test_memory_accounting_accuracy() {
        let registry = ResourceRegistry::new();

        // Test various texture formats for accurate accounting
        let test_cases = [
            (TextureFormat::R8Unorm, 1024, 1024, 1024 * 1024 * 1),
            (TextureFormat::Rg8Unorm, 512, 512, 512 * 512 * 2),
            (TextureFormat::Rgba8Unorm, 256, 256, 256 * 256 * 4),
            (TextureFormat::R16Float, 512, 512, 512 * 512 * 2),
            (TextureFormat::Rgba16Float, 128, 128, 128 * 128 * 8),
            (TextureFormat::R32Float, 256, 256, 256 * 256 * 4),
            (TextureFormat::Rgba32Float, 64, 64, 64 * 64 * 16),
        ];

        let mut expected_total = 0;
        for (format, width, height, expected_size) in test_cases.iter() {
            registry.track_texture_allocation(*width, *height, *format);
            expected_total += expected_size;

            let metrics = registry.get_metrics();
            assert_eq!(metrics.texture_bytes, expected_total);
        }
    }
}

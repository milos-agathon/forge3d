use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use wgpu::{BufferUsages, TextureFormat};
use super::types::MemoryMetrics;

/// Global memory tracking registry for GPU resources
pub struct ResourceRegistry {
    // Atomic counters for thread-safe tracking
    buffer_count: AtomicU32,
    texture_count: AtomicU32,
    buffer_bytes: AtomicU64,
    texture_bytes: AtomicU64,
    host_visible_bytes: AtomicU64,
    resident_tiles: AtomicU32,
    resident_tile_bytes: AtomicU64,
    staging_bytes_in_flight: AtomicU64,
    staging_ring_count: AtomicU32,
    staging_buffer_size: AtomicU64,
    staging_buffer_stalls: AtomicU64,

    // Budget limit (512 MiB)
    budget_limit: u64,
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
            resident_tiles: AtomicU32::new(0),
            resident_tile_bytes: AtomicU64::new(0),
            staging_bytes_in_flight: AtomicU64::new(0),
            staging_ring_count: AtomicU32::new(0),
            staging_buffer_size: AtomicU64::new(0),
            staging_buffer_stalls: AtomicU64::new(0),
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

    /// Update the number of resident virtual texture tiles tracked globally
    pub fn set_resident_tiles(&self, count: u32, tile_bytes: u64) {
        self.resident_tiles.store(count, Ordering::Relaxed);
        self.resident_tile_bytes
            .store(tile_bytes, Ordering::Relaxed);
    }

    /// Clear resident tile telemetry when virtual textures are torn down
    pub fn clear_resident_tiles(&self) {
        self.set_resident_tiles(0, 0);
    }

    /// Update staging ring telemetry aggregated from GPU upload rings
    pub fn set_staging_stats(
        &self,
        bytes_in_flight: u64,
        ring_count: usize,
        buffer_size: u64,
        stalls: u64,
    ) {
        self.staging_bytes_in_flight
            .store(bytes_in_flight, Ordering::Relaxed);
        self.staging_ring_count
            .store(ring_count as u32, Ordering::Relaxed);
        self.staging_buffer_size
            .store(buffer_size, Ordering::Relaxed);
        self.staging_buffer_stalls.store(stalls, Ordering::Relaxed);
    }

    /// Reset staging telemetry when rings are dropped or disabled
    pub fn clear_staging_stats(&self) {
        self.set_staging_stats(0, 0, 0, 0);
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
        let resident_tiles = self.resident_tiles.load(Ordering::Relaxed);
        let resident_tile_bytes = self.resident_tile_bytes.load(Ordering::Relaxed);
        let staging_bytes_in_flight = self.staging_bytes_in_flight.load(Ordering::Relaxed);
        let staging_ring_count = self.staging_ring_count.load(Ordering::Relaxed);
        let staging_buffer_size = self.staging_buffer_size.load(Ordering::Relaxed);
        let staging_buffer_stalls = self.staging_buffer_stalls.load(Ordering::Relaxed);

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
            resident_tiles,
            resident_tile_bytes,
            staging_bytes_in_flight,
            staging_ring_count,
            staging_buffer_size,
            staging_buffer_stalls,
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
pub fn calculate_texture_size(width: u32, height: u32, format: TextureFormat) -> u64 {
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
pub fn calculate_compressed_texture_size(
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

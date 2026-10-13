use wgpu::{BufferUsages, TextureFormat};

/// Resident estimate for one 2D mip, without transfer-row padding.
pub fn calculate_texture_size(width: u32, height: u32, format: TextureFormat) -> u64 {
    crate::core::texture_layout::resident_2d_size(width, height, format)
}

pub fn calculate_compressed_texture_size(
    width: u32,
    height: u32,
    bytes_per_block: u64,
    block_size: u32,
) -> u64 {
    crate::core::texture_layout::square_block_size(width, height, bytes_per_block, block_size)
}

pub fn is_host_visible_usage(usage: BufferUsages) -> bool {
    usage.contains(BufferUsages::MAP_READ) || usage.contains(BufferUsages::MAP_WRITE)
}

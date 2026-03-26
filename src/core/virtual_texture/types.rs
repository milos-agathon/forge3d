use bytemuck::{Pod, Zeroable};

/// Virtual texture configuration
#[derive(Debug, Clone)]
pub struct VirtualTextureConfig {
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    pub tile_border: u32,
    pub max_mip_levels: u32,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub format: wgpu::TextureFormat,
    pub use_feedback: bool,
}

impl Default for VirtualTextureConfig {
    fn default() -> Self {
        Self {
            width: 16384,
            height: 16384,
            tile_size: 128,
            tile_border: 4,
            max_mip_levels: 8,
            atlas_width: 2048,
            atlas_height: 2048,
            format: wgpu::TextureFormat::Rgba8Unorm,
            use_feedback: true,
        }
    }
}

/// Page table entry for virtual texture addressing
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct PageTableEntry {
    pub atlas_u: f32,
    pub atlas_v: f32,
    pub is_resident: u32,
    pub mip_bias: f32,
}

/// Virtual texture statistics
#[derive(Debug, Clone, Default)]
pub struct VirtualTextureStats {
    pub total_pages: u32,
    pub resident_pages: u32,
    pub cache_hits: u32,
    pub cache_misses: u32,
    pub tiles_streamed: u32,
    pub memory_usage: u64,
    pub avg_load_time_ms: f32,
}

/// Camera information for tile visibility calculation
#[derive(Debug, Clone, Copy)]
pub struct CameraInfo {
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub fov_degrees: f32,
    pub aspect_ratio: f32,
    pub near_plane: f32,
    pub far_plane: f32,
}

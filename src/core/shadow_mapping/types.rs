use wgpu::TextureFormat;

/// PCF quality settings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcfQuality {
    /// No filtering (single sample)
    None = 1,
    /// 3x3 PCF kernel
    Low = 3,
    /// 5x5 PCF kernel
    Medium = 5,
    /// 7x7 PCF kernel or Poisson disk sampling
    High = 7,
}

/// Shadow mapping configuration
#[derive(Debug, Clone)]
pub struct ShadowMappingConfig {
    /// Resolution of each shadow map
    pub shadow_map_size: u32,

    /// PCF quality setting
    pub pcf_quality: PcfQuality,

    /// Depth bias to prevent shadow acne
    pub depth_bias: f32,

    /// Slope-scaled bias factor
    pub slope_bias: f32,

    /// Shadow debug visualization mode:
    ///   0 = disabled
    ///   1 = cascade boundary overlay (color-coded by cascade)
    ///   2 = raw shadow visibility (grayscale)
    /// Set via FORGE3D_TERRAIN_SHADOW_DEBUG env var: "cascades" or "raw"
    pub debug_mode: u32,

    /// Shadow map format (D24Plus or D32Float)
    pub depth_format: TextureFormat,
}

impl Default for ShadowMappingConfig {
    fn default() -> Self {
        Self {
            shadow_map_size: 1024,
            pcf_quality: PcfQuality::Medium,
            depth_bias: 0.005,
            slope_bias: 1.0,
            debug_mode: 0,
            depth_format: TextureFormat::Depth24Plus,
        }
    }
}

/// CSM uniform data for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CsmUniforms {
    /// Light direction in world space
    pub light_direction: [f32; 4],

    /// Light view matrix  
    pub light_view: [[f32; 4]; 4],

    /// Shadow cascade data (up to 4 cascades)
    pub cascades: [CsmCascadeData; 4],

    /// Number of active cascades
    pub cascade_count: u32,

    /// PCF kernel size
    pub pcf_kernel_size: u32,

    /// Depth bias to prevent acne
    pub depth_bias: f32,

    /// Slope-scaled bias
    pub slope_bias: f32,

    /// Shadow map resolution
    pub shadow_map_size: f32,

    /// Debug visualization mode
    pub debug_mode: u32,

    /// Peter-panning prevention offset
    pub peter_panning_offset: f32,

    /// PCSS light radius (optional softness control)
    pub pcss_light_radius: f32,
}

/// GPU representation of a shadow cascade
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CsmCascadeData {
    /// Light-space projection matrix
    pub light_projection: [[f32; 4]; 4],

    /// Combined light_view_proj matrix (projection * view)
    /// Pre-computed for efficiency and to ensure consistency with shadow depth pass
    pub light_view_proj: [[f32; 4]; 4],

    /// Near plane distance
    pub near_distance: f32,

    /// Far plane distance
    pub far_distance: f32,

    /// Texel size in world space
    pub texel_size: f32,

    /// Padding for alignment
    pub _padding: f32,
}

/// Shadow atlas information for debugging
#[derive(Debug)]
pub struct ShadowAtlasInfo {
    /// Number of cascades
    pub cascade_count: u32,

    /// Atlas dimensions (width, height, depth)
    pub atlas_dimensions: (u32, u32, u32),

    /// Individual cascade resolutions
    pub cascade_resolutions: Vec<u32>,

    /// Memory usage in bytes
    pub memory_usage: u64,
}

/// Statistics from shadow map generation
#[derive(Debug)]
pub struct ShadowStats {
    /// Number of draw calls for shadow generation
    pub draw_calls: u32,

    /// Number of triangles rendered to shadow maps
    pub triangles_rendered: u64,

    /// Time taken for shadow map generation (ms)
    pub generation_time_ms: f32,

    /// GPU memory usage for shadow maps (bytes)
    pub memory_usage_bytes: u64,
}

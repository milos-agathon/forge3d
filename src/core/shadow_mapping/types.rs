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
/// Layout must match WGSL struct in terrain_pbr_pom.wgsl
/// Expected size: 704 bytes (16-byte aligned)
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

    /// Cascade blend range (0.0 = no blend, 0.1 = 10% blend at boundaries)
    pub cascade_blend_range: f32,

    /// P6.2: Active shadow technique (Hard=0, PCF=1, PCSS=2)
    pub technique: u32,

    /// Padding for 16-byte alignment (struct ends at offset 696 + 8 = 704, which is 16-byte aligned)
    pub _padding: [f32; 2],
}

// Compile-time assertion: CsmUniforms must be exactly 704 bytes to match WGSL layout
const _: () = assert!(
    std::mem::size_of::<CsmUniforms>() == 704,
    "CsmUniforms size mismatch with WGSL"
);

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

// Compile-time assertion: CsmCascadeData must be exactly 144 bytes (2 mat4x4 + 4 floats)
const _: () = assert!(
    std::mem::size_of::<CsmCascadeData>() == 144,
    "CsmCascadeData size mismatch with WGSL"
);

#[cfg(test)]
mod layout_lock_tests {
    use super::*;

    /// Helper macro to compute field offset without external crates
    macro_rules! offset_of {
        ($type:ty, $field:ident) => {{
            let uninit = std::mem::MaybeUninit::<$type>::uninit();
            let base_ptr = uninit.as_ptr() as usize;
            let field_ptr = unsafe { std::ptr::addr_of!((*uninit.as_ptr()).$field) } as usize;
            field_ptr - base_ptr
        }};
    }

    #[test]
    fn test_csm_uniforms_size() {
        // WGSL struct expects exactly 704 bytes
        assert_eq!(std::mem::size_of::<CsmUniforms>(), 704);
    }

    #[test]
    fn test_csm_cascade_data_size() {
        // WGSL ShadowCascade: 2 mat4x4 (128) + 4 floats (16) = 144 bytes
        assert_eq!(std::mem::size_of::<CsmCascadeData>(), 144);
    }

    #[test]
    fn test_csm_uniforms_critical_field_offsets() {
        // These offsets must match WGSL struct layout in terrain_pbr_pom.wgsl:
        // light_direction: vec4<f32>        @ offset 0
        // light_view: mat4x4<f32>           @ offset 16
        // cascades: array<ShadowCascade, 4> @ offset 80 (16 + 64)
        // cascade_count: u32                @ offset 656 (80 + 4*144)
        // pcf_kernel_size: u32              @ offset 660
        // technique: u32                    @ offset 692

        assert_eq!(
            offset_of!(CsmUniforms, light_direction),
            0,
            "light_direction offset"
        );
        assert_eq!(offset_of!(CsmUniforms, light_view), 16, "light_view offset");
        assert_eq!(offset_of!(CsmUniforms, cascades), 80, "cascades offset");
        assert_eq!(
            offset_of!(CsmUniforms, cascade_count),
            656,
            "cascade_count offset"
        );
        assert_eq!(
            offset_of!(CsmUniforms, pcf_kernel_size),
            660,
            "pcf_kernel_size offset"
        );
        assert_eq!(offset_of!(CsmUniforms, technique), 692, "technique offset");
        assert_eq!(offset_of!(CsmUniforms, _padding), 696, "_padding offset");
    }

    #[test]
    fn test_csm_cascade_data_field_offsets() {
        // WGSL ShadowCascade layout:
        // light_projection: mat4x4<f32>  @ offset 0
        // light_view_proj: mat4x4<f32>   @ offset 64
        // near_distance: f32             @ offset 128
        // far_distance: f32              @ offset 132
        // texel_size: f32                @ offset 136
        // _padding: f32                  @ offset 140

        assert_eq!(
            offset_of!(CsmCascadeData, light_projection),
            0,
            "light_projection offset"
        );
        assert_eq!(
            offset_of!(CsmCascadeData, light_view_proj),
            64,
            "light_view_proj offset"
        );
        assert_eq!(
            offset_of!(CsmCascadeData, near_distance),
            128,
            "near_distance offset"
        );
        assert_eq!(
            offset_of!(CsmCascadeData, far_distance),
            132,
            "far_distance offset"
        );
        assert_eq!(
            offset_of!(CsmCascadeData, texel_size),
            136,
            "texel_size offset"
        );
        assert_eq!(offset_of!(CsmCascadeData, _padding), 140, "_padding offset");
    }
}

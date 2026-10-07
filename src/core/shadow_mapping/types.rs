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

// Contract A (terrain/mesh shadow uniforms: 864 bytes, 144-byte cascades) is
// defined once in `crate::shadows::csm_types`. Re-export both names so every
// module path shares a single layout instead of drifting copies.
pub use crate::shadows::{CsmUniforms, ShadowCascade as CsmCascadeData};

// Compile-time size check: the struct must match the WGSL storage layout.
const _: () = assert!(
    std::mem::size_of::<CsmUniforms>() == 864,
    "CsmUniforms size mismatch with WGSL"
);

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
    use crate::shader_sources::csm_wgsl_layout::{layout, COPIES};

    /// Every WGSL copy of `ShadowCascade` + `CsmUniforms` must match the Rust
    /// ABI, field for field and byte for byte.
    ///
    /// The expected offsets are derived from the WGSL type declarations rather
    /// than hand-written, because a hand-written table can agree with a stale
    /// copy of itself and prove nothing. Deriving them closes the drift this
    /// lock exists for: adding, removing or resizing a field in all four WGSL
    /// copies leaves them "identical" to each other while silently moving the
    /// GPU layout away from the Rust struct. This also pins both total sizes,
    /// superseding separate size assertions.
    #[test]
    fn rust_csm_layout_matches_every_wgsl_copy() {
        const CASCADE_BYTES: usize = std::mem::size_of::<CsmCascadeData>();
        const UNIFORM_BYTES: usize = std::mem::size_of::<CsmUniforms>();

        /// Compiler-reported offsets in WGSL field order. An unknown name
        /// panics, so a new WGSL field cannot slip past uncompared.
        fn rust_offsets(
            wgsl: &[(String, usize)],
            offset_of_field: fn(&str) -> usize,
        ) -> Vec<(String, usize)> {
            wgsl.iter()
                .map(|(name, _)| (name.clone(), offset_of_field(name)))
                .collect()
        }

        /// Compiler-reported offset of a WGSL-named `ShadowCascade` field.
        fn cascade_offset(name: &str) -> usize {
            match name {
                "light_projection" => std::mem::offset_of!(CsmCascadeData, light_projection),
                "light_view_proj" => std::mem::offset_of!(CsmCascadeData, light_view_proj),
                "near_distance" => std::mem::offset_of!(CsmCascadeData, near_distance),
                "far_distance" => std::mem::offset_of!(CsmCascadeData, far_distance),
                "texel_size" => std::mem::offset_of!(CsmCascadeData, texel_size),
                other => panic!("WGSL ShadowCascade field `{other}` has no Rust counterpart"),
            }
        }

        /// Compiler-reported offset of a WGSL-named `CsmUniforms` field.
        fn uniform_offset(name: &str) -> usize {
            match name {
                "light_direction" => std::mem::offset_of!(CsmUniforms, light_direction),
                "light_view" => std::mem::offset_of!(CsmUniforms, light_view),
                "cascades" => std::mem::offset_of!(CsmUniforms, cascades),
                "cascade_count" => std::mem::offset_of!(CsmUniforms, cascade_count),
                "pcf_kernel_size" => std::mem::offset_of!(CsmUniforms, pcf_kernel_size),
                "depth_bias" => std::mem::offset_of!(CsmUniforms, depth_bias),
                "slope_bias" => std::mem::offset_of!(CsmUniforms, slope_bias),
                "shadow_map_size" => std::mem::offset_of!(CsmUniforms, shadow_map_size),
                "debug_mode" => std::mem::offset_of!(CsmUniforms, debug_mode),
                "evsm_positive_exp" => std::mem::offset_of!(CsmUniforms, evsm_positive_exp),
                "evsm_negative_exp" => std::mem::offset_of!(CsmUniforms, evsm_negative_exp),
                "peter_panning_offset" => std::mem::offset_of!(CsmUniforms, peter_panning_offset),
                "enable_unclipped_depth" => std::mem::offset_of!(CsmUniforms, enable_unclipped_depth),
                "depth_clip_factor" => std::mem::offset_of!(CsmUniforms, depth_clip_factor),
                "technique" => std::mem::offset_of!(CsmUniforms, technique),
                "technique_flags" => std::mem::offset_of!(CsmUniforms, technique_flags),
                "technique_params" => std::mem::offset_of!(CsmUniforms, technique_params),
                "technique_reserved" => std::mem::offset_of!(CsmUniforms, technique_reserved),
                "cascade_blend_range" => std::mem::offset_of!(CsmUniforms, cascade_blend_range),
                other => panic!("WGSL CsmUniforms field `{other}` has no Rust counterpart"),
            }
        }

        for (file, text) in COPIES {
            let (wgsl_cascade, cascade_bytes) = layout(text, "ShadowCascade", CASCADE_BYTES);
            assert_eq!(
                cascade_bytes, CASCADE_BYTES,
                "{file}: WGSL ShadowCascade is {cascade_bytes} bytes, Rust is {CASCADE_BYTES}",
            );
            assert_eq!(
                wgsl_cascade,
                rust_offsets(&wgsl_cascade, cascade_offset),
                "{file}: ShadowCascade layout differs from the Rust struct",
            );

            let (wgsl_uniforms, uniform_bytes) = layout(text, "CsmUniforms", CASCADE_BYTES);
            assert_eq!(
                uniform_bytes, UNIFORM_BYTES,
                "{file}: WGSL CsmUniforms is {uniform_bytes} bytes, Rust is {UNIFORM_BYTES}",
            );
            assert_eq!(
                wgsl_uniforms,
                rust_offsets(&wgsl_uniforms, uniform_offset),
                "{file}: CsmUniforms layout differs from the Rust struct",
            );
        }
    }
}

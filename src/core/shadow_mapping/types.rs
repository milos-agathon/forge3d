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
        // Keep this lockstep with the WGSL layout comments in terrain_pbr_pom.wgsl.
        assert_eq!(std::mem::size_of::<CsmUniforms>(), 864);
    }

    #[test]
    fn test_csm_cascade_data_size() {
        // WGSL ShadowCascade: 2 mat4x4 (128) + 4 floats (16) = 144 bytes
        assert_eq!(std::mem::size_of::<CsmCascadeData>(), 144);
    }

    /// Every WGSL copy of `ShadowCascade` + `CsmUniforms` must match the Rust
    /// ABI, field for field and byte for byte.
    ///
    /// The expected offsets are derived from the WGSL type declarations rather
    /// than hand-written, because a hand-written table can agree with a stale
    /// copy of itself and prove nothing. Deriving them closes the drift this
    /// lock exists for: adding, removing or resizing a field in all four WGSL
    /// copies leaves them "identical" to each other while silently moving the
    /// GPU layout away from the Rust struct.
    #[test]
    fn rust_csm_layout_matches_every_wgsl_copy() {
        use crate::shader_sources::csm_wgsl_layout::{layout, struct_fields, COPIES};

        const CASCADE_BYTES: usize = std::mem::size_of::<CsmCascadeData>();

        /// WGSL spells a padding run as consecutive `_padN*` members while Rust
        /// uses one array per run, so compare the run start — that is the offset
        /// the layout actually pins. A lone `_padding` field (as in
        /// `ShadowCascade`) is not a run and keeps its own name.
        fn canonical(fields: Vec<(String, String)>, offsets: Vec<usize>) -> Vec<(String, usize)> {
            let mut out: Vec<(String, usize)> = Vec::new();
            let mut run = 0usize;
            let mut prev_in_run = false;
            for ((name, _), offset) in fields.into_iter().zip(offsets) {
                let in_run = name.len() > 4
                    && name.starts_with("_pad")
                    && name.as_bytes()[4].is_ascii_digit();
                if in_run {
                    if !prev_in_run {
                        run += 1;
                        out.push((format!("_padding{run}"), offset));
                    }
                } else {
                    out.push((name, offset));
                }
                prev_in_run = in_run;
            }
            out
        }

        /// Compiler-reported offset of a WGSL-named `CsmUniforms` field. An
        /// unknown name panics, so a new WGSL field cannot slip past uncompared.
        fn rust_uniform_offset(name: &str) -> usize {
            match name {
                "light_direction" => offset_of!(CsmUniforms, light_direction),
                "light_view" => offset_of!(CsmUniforms, light_view),
                "cascades" => offset_of!(CsmUniforms, cascades),
                "cascade_count" => offset_of!(CsmUniforms, cascade_count),
                "pcf_kernel_size" => offset_of!(CsmUniforms, pcf_kernel_size),
                "depth_bias" => offset_of!(CsmUniforms, depth_bias),
                "slope_bias" => offset_of!(CsmUniforms, slope_bias),
                "shadow_map_size" => offset_of!(CsmUniforms, shadow_map_size),
                "debug_mode" => offset_of!(CsmUniforms, debug_mode),
                "evsm_positive_exp" => offset_of!(CsmUniforms, evsm_positive_exp),
                "evsm_negative_exp" => offset_of!(CsmUniforms, evsm_negative_exp),
                "peter_panning_offset" => offset_of!(CsmUniforms, peter_panning_offset),
                "enable_unclipped_depth" => offset_of!(CsmUniforms, enable_unclipped_depth),
                "depth_clip_factor" => offset_of!(CsmUniforms, depth_clip_factor),
                "technique" => offset_of!(CsmUniforms, technique),
                "technique_flags" => offset_of!(CsmUniforms, technique_flags),
                "technique_params" => offset_of!(CsmUniforms, technique_params),
                "technique_reserved" => offset_of!(CsmUniforms, technique_reserved),
                "cascade_blend_range" => offset_of!(CsmUniforms, cascade_blend_range),
                "_padding1" => offset_of!(CsmUniforms, _padding1),
                "_padding2" => offset_of!(CsmUniforms, _padding2),
                other => panic!("WGSL CsmUniforms field `{other}` has no Rust counterpart"),
            }
        }

        /// Compiler-reported offset of a WGSL-named `ShadowCascade` field.
        fn rust_cascade_offset(name: &str) -> usize {
            match name {
                "light_projection" => offset_of!(CsmCascadeData, light_projection),
                "light_view_proj" => offset_of!(CsmCascadeData, light_view_proj),
                "near_distance" => offset_of!(CsmCascadeData, near_distance),
                "far_distance" => offset_of!(CsmCascadeData, far_distance),
                "texel_size" => offset_of!(CsmCascadeData, texel_size),
                "_padding" => offset_of!(CsmCascadeData, _padding),
                other => panic!("WGSL ShadowCascade field `{other}` has no Rust counterpart"),
            }
        }

        for (file, text) in COPIES {
            let (cascade_offsets, cascade_size) = layout(text, "ShadowCascade", CASCADE_BYTES);
            let wgsl_cascade = canonical(struct_fields(text, "ShadowCascade"), cascade_offsets);
            assert_eq!(
                cascade_size,
                CASCADE_BYTES,
                "{file}: WGSL ShadowCascade is {cascade_size} bytes, Rust is {CASCADE_BYTES}",
            );
            let rust_cascade: Vec<(String, usize)> = wgsl_cascade
                .iter()
                .map(|(name, _)| (name.clone(), rust_cascade_offset(name)))
                .collect();
            assert_eq!(
                wgsl_cascade, rust_cascade,
                "{file}: ShadowCascade layout differs from the Rust struct",
            );

            let (uniform_offsets, uniform_size) = layout(text, "CsmUniforms", CASCADE_BYTES);
            let wgsl_uniforms = canonical(struct_fields(text, "CsmUniforms"), uniform_offsets);
            assert_eq!(
                uniform_size,
                std::mem::size_of::<CsmUniforms>(),
                "{file}: WGSL CsmUniforms is {uniform_size} bytes, Rust is {}",
                std::mem::size_of::<CsmUniforms>(),
            );
            let rust_uniforms: Vec<(String, usize)> = wgsl_uniforms
                .iter()
                .map(|(name, _)| (name.clone(), rust_uniform_offset(name)))
                .collect();
            assert_eq!(
                wgsl_uniforms, rust_uniforms,
                "{file}: CsmUniforms layout differs from the Rust struct",
            );
        }
    }
}

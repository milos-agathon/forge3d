//! Shadow mapping types and structures
//!
//! Defines the core types for cascaded shadow maps including configuration,
//! cascade data, uniforms, directional light, and statistics.

use glam::Vec3;

/// Parse shadow debug mode from FORGE3D_TERRAIN_SHADOW_DEBUG environment variable.
///
/// Returns:
///   0 = disabled (default)
///   1 = cascade boundary overlay ("cascades")
///   2 = raw shadow visibility ("raw")
pub fn parse_shadow_debug_env() -> u32 {
    match std::env::var("FORGE3D_TERRAIN_SHADOW_DEBUG").as_deref() {
        Ok("cascades") | Ok("1") => 1,
        Ok("raw") | Ok("2") => 2,
        _ => 0,
    }
}

/// Configuration for cascaded shadow maps
#[derive(Debug, Clone)]
pub struct CsmConfig {
    /// Number of cascade levels (typically 2-4)
    pub cascade_count: u32,
    /// Shadow map resolution per cascade
    pub shadow_map_size: u32,
    /// Far plane distance for camera
    pub camera_far: f32,
    /// Near plane distance for camera  
    pub camera_near: f32,
    /// Lambda factor for cascade split scheme (0.0 = uniform, 1.0 = logarithmic)
    pub lambda: f32,
    /// Bias to prevent shadow acne
    pub depth_bias: f32,
    /// Slope-scaled bias for angled surfaces
    pub slope_bias: f32,
    /// PCF filter kernel size (1, 3, 5, or 7)
    pub pcf_kernel_size: u32,
}

impl Default for CsmConfig {
    fn default() -> Self {
        Self {
            cascade_count: 4,
            shadow_map_size: 2048,
            camera_far: 1000.0,
            camera_near: 0.1,
            lambda: 0.5,
            depth_bias: 0.0001,
            slope_bias: 0.001,
            pcf_kernel_size: 3,
        }
    }
}

/// Fog shadow cascade: matches `volumetric.wgsl`'s `FogShadowCascade` (80 bytes).
///
/// Distinct from `shadows::ShadowCascade`, the terrain/mesh cascade (144 bytes),
/// which also carries a combined `light_view_proj`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FogShadowCascade {
    /// Light-space projection matrix for this cascade
    pub light_projection: [[f32; 4]; 4],
    /// Far plane distance for this cascade
    pub far_distance: f32,
    /// Near plane distance for this cascade  
    pub near_distance: f32,
    /// Texel size in world space
    pub texel_size: f32,
    /// Padding for alignment
    pub _padding: f32,
}

/// Fog CSM uniform data sent to the GPU.
///
/// Covers the whole of `volumetric.wgsl`'s `FogCsmUniforms`; the trailing
/// fields and padding here are unused by the fog shader. Distinct from
/// `shadows::CsmUniforms`, the terrain/mesh contract: 608 bytes with 80-byte
/// cascades against 864 with 144-byte cascades. `fog_layout_lock_tests` is what
/// keeps the two apart.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FogCsmUniforms {
    /// Light direction in world space
    pub light_direction: [f32; 4],
    /// Light view matrix (world to light space)
    pub light_view: [[f32; 4]; 4],
    /// Shadow cascades data
    pub cascades: [FogShadowCascade; 4],
    /// Number of active cascades
    pub cascade_count: u32,
    /// PCF kernel size
    pub pcf_kernel_size: u32,
    /// Depth bias
    pub depth_bias: f32,
    /// Slope-scaled bias
    pub slope_bias: f32,
    /// Shadow map texture array size
    pub shadow_map_size: f32,
    /// Debug visualization mode (0=off, 1=cascade colors)
    pub debug_mode: u32,
    /// P0.2/M3: EVSM exponents
    pub evsm_positive_exp: f32,
    pub evsm_negative_exp: f32,
    /// Peter-panning prevention offset
    pub peter_panning_offset: f32,
    /// Enable unclipped depth
    pub enable_unclipped_depth: u32,
    /// Depth clip factor
    pub depth_clip_factor: f32,
    /// P0.2/M3: Active shadow technique (Hard=0, PCF=1, PCSS=2, VSM=3, EVSM=4, MSM=5)
    pub technique: u32,
    /// Technique feature flags
    pub technique_flags: u32,
    /// Padding to align technique_params to 16-byte boundary
    pub _padding1: [f32; 3],
    /// Technique parameters: [pcss_blocker_radius, pcss_filter_radius, moment_bias, light_size]
    pub technique_params: [f32; 4],
    /// Reserved for future technique parameters
    pub technique_reserved: [f32; 4],
    /// Cascade blend range (0.0 = no blend, 0.1 = 10% blend at boundaries)
    pub cascade_blend_range: f32,
    /// Trailing padding; the fog shader (`FogCsmUniforms`) reads only the leading fields.
    pub _padding2: [f32; 27],
}

// Compile-time size checks: the cascade must match `volumetric.wgsl` exactly,
// and the uniform buffer must stay large enough to cover the shader's 432-byte
// struct. Neither was asserted anywhere before.
const _: () = assert!(std::mem::size_of::<FogShadowCascade>() == 80);
const _: () = assert!(std::mem::size_of::<FogCsmUniforms>() == 608);

/// The fog contract had no layout verification at all, so `volumetric.wgsl` could
/// drift from it silently. Derive the expectations from the shader instead, the
/// same way `core::shadow_mapping::layout_lock_tests` does for the terrain one.
#[cfg(test)]
mod fog_layout_lock_tests {
    use super::*;
    use crate::shader_sources::csm_wgsl_layout::layout;

    const FOG_WGSL: &str = include_str!("../../shaders/volumetric.wgsl");

    /// Compiler-reported offsets in WGSL field order. An unknown name panics, so
    /// a new WGSL field cannot slip past uncompared.
    fn rust_offsets(
        wgsl: &[(String, usize)],
        offset_of_field: fn(&str) -> usize,
    ) -> Vec<(String, usize)> {
        wgsl.iter()
            .map(|(name, _)| (name.clone(), offset_of_field(name)))
            .collect()
    }

    /// The shader stops at `debug_mode`, so it declares a strict prefix of the
    /// Rust struct; the total may only grow on the Rust side.
    #[test]
    fn fog_rust_layout_covers_volumetric_wgsl() {
        const CASCADE_BYTES: usize = std::mem::size_of::<FogShadowCascade>();

        fn cascade_offset(name: &str) -> usize {
            match name {
                "light_projection" => std::mem::offset_of!(FogShadowCascade, light_projection),
                "far_distance" => std::mem::offset_of!(FogShadowCascade, far_distance),
                "near_distance" => std::mem::offset_of!(FogShadowCascade, near_distance),
                "texel_size" => std::mem::offset_of!(FogShadowCascade, texel_size),
                other => {
                    panic!("WGSL FogShadowCascade field `{other}` has no Rust counterpart")
                }
            }
        }

        fn uniform_offset(name: &str) -> usize {
            match name {
                "light_direction" => std::mem::offset_of!(FogCsmUniforms, light_direction),
                "light_view" => std::mem::offset_of!(FogCsmUniforms, light_view),
                "cascades" => std::mem::offset_of!(FogCsmUniforms, cascades),
                "cascade_count" => std::mem::offset_of!(FogCsmUniforms, cascade_count),
                "pcf_kernel_size" => std::mem::offset_of!(FogCsmUniforms, pcf_kernel_size),
                "depth_bias" => std::mem::offset_of!(FogCsmUniforms, depth_bias),
                "slope_bias" => std::mem::offset_of!(FogCsmUniforms, slope_bias),
                "shadow_map_size" => std::mem::offset_of!(FogCsmUniforms, shadow_map_size),
                "debug_mode" => std::mem::offset_of!(FogCsmUniforms, debug_mode),
                other => panic!("WGSL FogCsmUniforms field `{other}` has no Rust counterpart"),
            }
        }

        let (wgsl_cascade, cascade_bytes) = layout(FOG_WGSL, "FogShadowCascade", CASCADE_BYTES);
        assert_eq!(
            cascade_bytes, CASCADE_BYTES,
            "WGSL FogShadowCascade is {cascade_bytes} bytes, Rust is {CASCADE_BYTES}",
        );
        assert_eq!(
            wgsl_cascade,
            rust_offsets(&wgsl_cascade, cascade_offset),
            "FogShadowCascade layout differs from the Rust struct",
        );

        let (wgsl_uniforms, uniform_bytes) = layout(FOG_WGSL, "FogCsmUniforms", CASCADE_BYTES);
        assert!(
            uniform_bytes <= std::mem::size_of::<FogCsmUniforms>(),
            "WGSL FogCsmUniforms needs {uniform_bytes} bytes, Rust has {}",
            std::mem::size_of::<FogCsmUniforms>(),
        );
        assert_eq!(
            wgsl_uniforms,
            rust_offsets(&wgsl_uniforms, uniform_offset),
            "FogCsmUniforms layout differs from the Rust struct",
        );
    }
}

/// Directional light configuration for shadow casting
#[derive(Debug, Clone)]
pub struct DirectionalLight {
    /// Light direction (normalized, pointing towards light source)
    pub direction: Vec3,
    /// Light color and intensity
    pub color: Vec3,
    /// Light intensity multiplier
    pub intensity: f32,
    /// Enable shadow casting
    pub cast_shadows: bool,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.0, -1.0, 0.3).normalize(),
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 3.0,
            cast_shadows: true,
        }
    }
}

/// Shadow mapping statistics and debugging info
#[derive(Debug, Clone)]
pub struct ShadowStats {
    /// Number of active cascades
    pub cascade_count: u32,
    /// Shadow map resolution per cascade  
    pub shadow_map_size: u32,
    /// Total memory usage in bytes
    pub memory_usage: u64,
    /// Light direction
    pub light_direction: Vec3,
    /// Cascade split distances
    pub split_distances: Vec<f32>,
    /// Texel sizes per cascade
    pub texel_sizes: Vec<f32>,
}

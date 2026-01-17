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

/// Shadow cascade data
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowCascade {
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

/// CSM uniform buffer data sent to GPU
/// P0.2/M3: Must match shader std140 layout (816 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CsmUniforms {
    /// Light direction in world space
    pub light_direction: [f32; 4],
    /// Light view matrix (world to light space)
    pub light_view: [[f32; 4]; 4],
    /// Shadow cascades data
    pub cascades: [ShadowCascade; 4],
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
    /// Padding for std430 alignment (storage buffer) - 27 floats to reach 864 total bytes
    pub _padding2: [f32; 27],
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

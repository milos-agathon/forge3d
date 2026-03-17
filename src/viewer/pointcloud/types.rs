use bytemuck::{Pod, Zeroable};

/// 3D point instance for GPU rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PointInstance3D {
    pub position: [f32; 3],
    pub elevation_norm: f32,
    pub rgb: [f32; 3],
    pub intensity: f32,
    pub size: f32,
    pub _pad: [f32; 3],
}

/// Point cloud uniforms.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PointCloudUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub viewport_size: [f32; 2],
    pub point_size: f32,
    pub color_mode: u32,
    pub has_rgb: u32,
    pub has_intensity: u32,
    pub _pad: [u32; 2],
}

/// Color mode for point cloud rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    Elevation = 0,
    Rgb = 1,
    Intensity = 2,
}

impl ColorMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "rgb" => Self::Rgb,
            "intensity" => Self::Intensity,
            _ => Self::Elevation,
        }
    }
}

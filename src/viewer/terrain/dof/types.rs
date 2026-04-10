/// DoF uniforms for the blur shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DofUniforms {
    pub screen_dims: [f32; 4],
    pub dof_params: [f32; 4],
    pub dof_params2: [f32; 4],
    pub camera_params: [f32; 4],
}

/// Depth of Field configuration
#[derive(Debug, Clone)]
pub struct DofConfig {
    pub focus_distance: f32,
    pub f_stop: f32,
    pub focal_length: f32,
    pub quality: u32,
    pub max_blur_radius: f32,
    pub blur_strength: f32,
    pub tilt_pitch: f32,
    pub tilt_yaw: f32,
}

impl Default for DofConfig {
    fn default() -> Self {
        Self {
            focus_distance: 500.0,
            f_stop: 5.6,
            focal_length: 50.0,
            quality: 8,
            max_blur_radius: 32.0,
            blur_strength: 25.0,
            tilt_pitch: 0.0,
            tilt_yaw: 0.0,
        }
    }
}

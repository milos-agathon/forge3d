// src/viewer/render/fog.rs
// Fog rendering helpers for the interactive viewer

use glam::Mat4;
use wgpu::{Buffer, Queue};

use super::super::viewer_types::{FogCameraUniforms, VolumetricUniformsStd140};

/// Convert Mat4 to array for uniforms
#[inline]
fn mat4_to_arr(m: Mat4) -> [[f32; 4]; 4] {
    let c = m.to_cols_array();
    [
        [c[0], c[1], c[2], c[3]],
        [c[4], c[5], c[6], c[7]],
        [c[8], c[9], c[10], c[11]],
        [c[12], c[13], c[14], c[15]],
    ]
}

/// Fog camera parameters
pub struct FogCameraParams {
    pub view_matrix: Mat4,
    pub proj_matrix: Mat4,
    pub eye: glam::Vec3,
    pub znear: f32,
    pub zfar: f32,
}

/// Update fog camera uniforms
pub fn update_fog_camera(queue: &Queue, fog_camera_buf: &Buffer, params: &FogCameraParams) {
    let inv_view = params.view_matrix.inverse();
    let inv_proj = params.proj_matrix.inverse();
    let fog_cam = FogCameraUniforms {
        view: mat4_to_arr(params.view_matrix),
        proj: mat4_to_arr(params.proj_matrix),
        inv_view: mat4_to_arr(inv_view),
        inv_proj: mat4_to_arr(inv_proj),
        view_proj: mat4_to_arr(params.proj_matrix * params.view_matrix),
        eye_position: [params.eye.x, params.eye.y, params.eye.z],
        near: params.znear,
        far: params.zfar,
        _pad: [0.0; 3],
    };
    queue.write_buffer(fog_camera_buf, 0, bytemuck::bytes_of(&fog_cam));
}

/// Fog volumetric parameters
pub struct FogVolumetricParams {
    pub density: f32,
    pub phase_g: f32,
    pub steps: u32,
    pub zfar: f32,
    pub sun_direction: [f32; 3],
    pub sun_intensity: f32,
    pub temporal_alpha: f32,
    pub use_shadows: bool,
    pub frame_index: u32,
}

/// Update fog volumetric uniforms
pub fn update_fog_params(queue: &Queue, fog_params_buf: &Buffer, params: &FogVolumetricParams) {
    let fog_params_packed = VolumetricUniformsStd140 {
        density: params.density.max(0.0),
        height_falloff: 0.1,
        phase_g: params.phase_g.clamp(-0.999, 0.999),
        max_steps: params.steps,
        start_distance: 0.1,
        max_distance: params.zfar,
        _pad_a0: 0.0,
        _pad_a1: 0.0,
        scattering_color: [1.0, 1.0, 1.0],
        absorption: 1.0,
        sun_direction: params.sun_direction,
        sun_intensity: params.sun_intensity.max(0.0),
        ambient_color: [0.2, 0.25, 0.3],
        temporal_alpha: params.temporal_alpha.clamp(0.0, 0.9),
        use_shadows: if params.use_shadows { 1 } else { 0 },
        jitter_strength: 0.8,
        frame_index: params.frame_index,
        _pad0: 0,
    };
    queue.write_buffer(fog_params_buf, 0, bytemuck::bytes_of(&fog_params_packed));
}

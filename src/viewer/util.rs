// src/viewer/util.rs
// Utility functions for the viewer module

use glam::Mat4;

/// Convert a glam Mat4 to a 4x4 array for GPU uniforms
#[inline]
pub fn mat4_to_arr4(m: Mat4) -> [[f32; 4]; 4] {
    let c = m.to_cols_array();
    [
        [c[0], c[1], c[2], c[3]],
        [c[4], c[5], c[6], c[7]],
        [c[8], c[9], c[10], c[11]],
        [c[12], c[13], c[14], c[15]],
    ]
}

/// Build a perspective projection matrix from viewer config
#[inline]
pub fn build_perspective(width: u32, height: u32, fov_deg: f32, znear: f32, zfar: f32) -> Mat4 {
    let aspect = width as f32 / height as f32;
    let fov = fov_deg.to_radians();
    Mat4::perspective_rh(fov, aspect, znear, zfar)
}

/// Build CameraParams for GI from matrices and eye position
/// P1.1: Added prev_view_proj for motion vector computation
/// P1.2: Added jitter_offset for TAA
#[inline]
pub fn build_camera_params(
    model_view: Mat4,
    proj: Mat4,
    eye: glam::Vec3,
    prev_view_proj: Mat4,
    frame_index: u32,
    jitter_offset: [f32; 2],
) -> crate::core::screen_space_effects::CameraParams {
    let inv_model_view = model_view.inverse();
    let inv_proj = proj.inverse();
    crate::core::screen_space_effects::CameraParams {
        view_matrix: mat4_to_arr4(model_view),
        inv_view_matrix: mat4_to_arr4(inv_model_view),
        proj_matrix: mat4_to_arr4(proj),
        inv_proj_matrix: mat4_to_arr4(inv_proj),
        prev_view_proj_matrix: mat4_to_arr4(prev_view_proj),
        camera_pos: [eye.x, eye.y, eye.z],
        frame_index,
        jitter_offset,
        _pad_jitter: [0.0, 0.0],
    }
}

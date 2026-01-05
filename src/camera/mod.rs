//! Camera math module for T2.1: view/projection matrices with Python API
//!
//! Provides right-handed, Y-up, -Z forward camera math (standard GL-style look-at).
//! Supports both "wgpu" (0..1 Z) and "gl" (-1..1 Z) clip spaces.

pub mod dof;
pub mod validation;

use glam::{Mat4, Vec3, Vec4Swizzles};
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::Bound;

// Re-export validation functions for internal use
pub use validation::{
    validate_aspect, validate_camera_params, validate_clip_space, validate_far, validate_fovy,
    validate_near, validate_ortho_bottom_top, validate_ortho_left_right, validate_up_not_colinear,
    validate_vec3_finite,
};

// Re-export DOF functions
pub use dof::{
    camera_aperture_to_f_stop, camera_circle_of_confusion, camera_depth_of_field_range,
    camera_dof_params, camera_f_stop_to_aperture, camera_hyperfocal_distance,
    create_camera_dof_params,
};

/// Returns the GL→WGPU depth conversion matrix.
/// Maps GL clip-space Z [-1,1] to WGPU/Vulkan/Metal [0,1].
#[inline]
fn gl_to_wgpu() -> Mat4 {
    Mat4::from_cols_array(&[
        1.0, 0.0, 0.0, 0.0, // column 0
        0.0, 1.0, 0.0, 0.0, // column 1
        0.0, 0.0, 0.5, 0.0, // column 2
        0.0, 0.0, 0.5, 1.0, // column 3
    ])
}

/// Converts a Mat4 to a NumPy array with shape (4,4) and dtype float32, C-contiguous
fn mat4_to_numpy<'py>(py: Python<'py>, mat: Mat4) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let data = mat.to_cols_array_2d();

    // Create a flattened array in row-major order for NumPy
    let flat: Vec<f32> = (0..4)
        .flat_map(|row| (0..4).map(move |col| data[col][row]))
        .collect();

    let array = PyArray2::from_vec2_bound(
        py,
        &[
            flat[0..4].to_vec(),
            flat[4..8].to_vec(),
            flat[8..12].to_vec(),
            flat[12..16].to_vec(),
        ],
    )?;

    Ok(array)
}

/// Compute view matrix using right-handed, Y-up, -Z forward convention
#[pyfunction]
#[pyo3(text_signature = "(eye, target, up)")]
pub fn camera_look_at<'py>(
    py: Python<'py>,
    eye: (f32, f32, f32),
    target: (f32, f32, f32),
    up: (f32, f32, f32),
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let eye_vec = Vec3::new(eye.0, eye.1, eye.2);
    let target_vec = Vec3::new(target.0, target.1, target.2);
    let up_vec = Vec3::new(up.0, up.1, up.2);

    // Validate inputs
    validate_vec3_finite(eye_vec, "eye")?;
    validate_vec3_finite(target_vec, "target")?;
    validate_vec3_finite(up_vec, "up")?;
    validate_up_not_colinear(eye_vec, target_vec, up_vec)?;

    let view_matrix = Mat4::look_at_rh(eye_vec, target_vec, up_vec);
    mat4_to_numpy(py, view_matrix)
}

/// Compute perspective projection matrix
#[pyfunction]
#[pyo3(text_signature = "(fovy_deg, aspect, znear, zfar, clip_space='wgpu')")]
pub fn camera_perspective<'py>(
    py: Python<'py>,
    fovy_deg: f32,
    aspect: f32,
    znear: f32,
    zfar: f32,
    clip_space: Option<String>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let clip_space = clip_space.as_deref().unwrap_or("wgpu");

    // Validate inputs
    validate_fovy(fovy_deg)?;
    validate_aspect(aspect)?;
    validate_near(znear)?;
    validate_far(zfar, znear)?;
    validate_clip_space(clip_space)?;

    let fovy_rad = fovy_deg.to_radians();

    // Always start with GL projection
    let proj_gl = Mat4::perspective_rh_gl(fovy_rad, aspect, znear, zfar);

    let proj_matrix = match clip_space {
        "gl" => proj_gl,
        "wgpu" => gl_to_wgpu() * proj_gl,
        _ => unreachable!(), // Already validated
    };

    mat4_to_numpy(py, proj_matrix)
}

/// Compute orthographic projection matrix
#[pyfunction]
#[pyo3(text_signature = "(left, right, bottom, top, znear, zfar, clip_space='wgpu')")]
pub fn camera_orthographic<'py>(
    py: Python<'py>,
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    znear: f32,
    zfar: f32,
    clip_space: Option<String>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let clip_space = clip_space.as_deref().unwrap_or("wgpu");

    // Validate inputs
    validate_ortho_left_right(left, right)?;
    validate_ortho_bottom_top(bottom, top)?;
    validate_near(znear)?;
    validate_far(zfar, znear)?;
    validate_clip_space(clip_space)?;

    let w = right - left;
    let h = top - bottom;
    let d = zfar - znear;

    let proj_gl = Mat4::from_cols_array(&[
        2.0 / w,
        0.0,
        0.0,
        0.0, // column 0
        0.0,
        2.0 / h,
        0.0,
        0.0, // column 1
        0.0,
        0.0,
        -2.0 / d,
        0.0, // column 2
        -(right + left) / w,
        -(top + bottom) / h,
        -(zfar + znear) / d,
        1.0, // column 3
    ]);

    let proj_matrix = match clip_space {
        "gl" => proj_gl,
        "wgpu" => gl_to_wgpu() * proj_gl,
        _ => unreachable!(), // Already validated
    };

    mat4_to_numpy(py, proj_matrix)
}

/// Compute combined view-projection matrix
#[pyfunction]
#[pyo3(text_signature = "(eye, target, up, fovy_deg, aspect, znear, zfar, clip_space='wgpu')")]
pub fn camera_view_proj<'py>(
    py: Python<'py>,
    eye: (f32, f32, f32),
    target: (f32, f32, f32),
    up: (f32, f32, f32),
    fovy_deg: f32,
    aspect: f32,
    znear: f32,
    zfar: f32,
    clip_space: Option<String>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let clip_space = clip_space.as_deref().unwrap_or("wgpu");

    let eye_vec = Vec3::new(eye.0, eye.1, eye.2);
    let target_vec = Vec3::new(target.0, target.1, target.2);
    let up_vec = Vec3::new(up.0, up.1, up.2);

    // Validate inputs
    validate_vec3_finite(eye_vec, "eye")?;
    validate_vec3_finite(target_vec, "target")?;
    validate_vec3_finite(up_vec, "up")?;
    validate_up_not_colinear(eye_vec, target_vec, up_vec)?;
    validate_fovy(fovy_deg)?;
    validate_aspect(aspect)?;
    validate_near(znear)?;
    validate_far(zfar, znear)?;
    validate_clip_space(clip_space)?;

    let view_matrix = Mat4::look_at_rh(eye_vec, target_vec, up_vec);

    let fovy_rad = fovy_deg.to_radians();
    let proj_gl = Mat4::perspective_rh_gl(fovy_rad, aspect, znear, zfar);

    let proj_matrix = match clip_space {
        "gl" => proj_gl,
        "wgpu" => gl_to_wgpu() * proj_gl,
        _ => unreachable!(), // Already validated
    };

    let view_proj_matrix = proj_matrix * view_matrix;
    mat4_to_numpy(py, view_proj_matrix)
}

/// Helper function to create perspective matrix for WGPU clip space
pub fn perspective_wgpu(fovy_rad: f32, aspect: f32, znear: f32, zfar: f32) -> Mat4 {
    let proj_gl = Mat4::perspective_rh_gl(fovy_rad, aspect, znear, zfar);
    gl_to_wgpu() * proj_gl
}

/// Extract camera world position from view matrix
pub fn camera_world_position_from_view(view_matrix: Mat4) -> Vec3 {
    let inv_view = view_matrix.inverse();
    inv_view.w_axis.xyz()
}

//! Camera math module for T2.1: view/projection matrices with Python API
//! 
//! Provides right-handed, Y-up, -Z forward camera math (standard GL-style look-at).
//! Supports both "wgpu" (0..1 Z) and "gl" (-1..1 Z) clip spaces.

use pyo3::prelude::*;
use pyo3::Bound; // needed for Bound<'py, PyArray2<f32>> return types
use numpy::PyArray2;
use glam::{Mat4, Vec3};

/// Returns the GL→WGPU depth conversion matrix.
/// Maps GL clip-space Z [-1,1] to WGPU/Vulkan/Metal [0,1].
#[inline]
fn gl_to_wgpu() -> Mat4 {
    Mat4::from_cols_array(&[
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.5,
        0.0, 0.0, 0.0, 1.0,
    ])
}

/// Error messages matching the exact strings specified in task requirements
const ERROR_FOVY: &str = "fovy_deg must be finite and in (0, 180)";
const ERROR_NEAR: &str = "znear must be finite and > 0";
const ERROR_FAR: &str = "zfar must be finite and > znear";
const ERROR_ASPECT: &str = "aspect must be finite and > 0";
const ERROR_VECFINITE: &str = "eye/target/up components must be finite";
const ERROR_UPCOLINEAR: &str = "up vector must not be colinear with view direction";
const ERROR_CLIP: &str = "clip_space must be 'wgpu' or 'gl'";

/// Validates all components of a Vec3 are finite
fn validate_vec3_finite(v: Vec3, _param_name: &str) -> PyResult<()> {
    if !v.is_finite() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(ERROR_VECFINITE));
    }
    Ok(())
}

/// Validates field of view angle
fn validate_fovy(fovy_deg: f32) -> PyResult<()> {
    if !fovy_deg.is_finite() || fovy_deg <= 0.0 || fovy_deg >= 180.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(ERROR_FOVY));
    }
    Ok(())
}

/// Validates near plane distance  
fn validate_near(znear: f32) -> PyResult<()> {
    if !znear.is_finite() || znear <= 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(ERROR_NEAR));
    }
    Ok(())
}

/// Validates far plane distance relative to near
fn validate_far(zfar: f32, znear: f32) -> PyResult<()> {
    if !zfar.is_finite() || zfar <= znear {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(ERROR_FAR));
    }
    Ok(())
}

/// Validates aspect ratio
fn validate_aspect(aspect: f32) -> PyResult<()> {
    if !aspect.is_finite() || aspect <= 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(ERROR_ASPECT));
    }
    Ok(())
}

/// Validates clip space parameter
fn validate_clip_space(clip_space: &str) -> PyResult<()> {
    match clip_space {
        "wgpu" | "gl" => Ok(()),
        _ => Err(pyo3::exceptions::PyRuntimeError::new_err(ERROR_CLIP)),
    }
}

/// Validates that up vector is not colinear with view direction
fn validate_up_not_colinear(eye: Vec3, target: Vec3, up: Vec3) -> PyResult<()> {
    let view_dir = (target - eye).normalize_or_zero();
    let up_norm = up.normalize_or_zero();
    
    // Check if cross product is near zero (vectors are parallel)
    let cross = view_dir.cross(up_norm);
    if cross.length_squared() < 1e-6 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(ERROR_UPCOLINEAR));
    }
    Ok(())
}

/// Converts a Mat4 to a NumPy array with shape (4,4) and dtype float32, C-contiguous
fn mat4_to_numpy<'py>(py: Python<'py>, mat: Mat4) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // glam Mat4 is column-major, but we want to return it as a (4,4) array
    // where the indexing matches mathematical conventions
    let data = mat.to_cols_array_2d();
    
    // Create a flattened array in row-major order for NumPy
    let flat: Vec<f32> = (0..4).flat_map(|row| {
        (0..4).map(move |col| data[col][row])
    }).collect();
    
    let array = PyArray2::from_vec2_bound(py, &vec![
        flat[0..4].to_vec(),
        flat[4..8].to_vec(), 
        flat[8..12].to_vec(),
        flat[12..16].to_vec(),
    ])?;
    
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

/// Helper function to validate camera parameters (for internal use)
pub fn validate_camera_params(
    eye: Vec3,
    target: Vec3,
    up: Vec3,
    fovy_deg: f32,
    znear: f32,
    zfar: f32,
) -> PyResult<()> {
    validate_vec3_finite(eye, "eye")?;
    validate_vec3_finite(target, "target")?;
    validate_vec3_finite(up, "up")?;
    validate_up_not_colinear(eye, target, up)?;
    validate_fovy(fovy_deg)?;
    validate_near(znear)?;
    validate_far(zfar, znear)?;
    Ok(())
}
//! Camera parameter validation utilities
//!
//! Provides validation functions for camera parameters used in view/projection
//! matrix construction and depth-of-field calculations.

use glam::Vec3;
use pyo3::prelude::*;

/// Error messages matching the exact strings specified in task requirements
pub const ERROR_FOVY: &str = "fovy_deg must be finite and in (0, 180)";
pub const ERROR_NEAR: &str = "znear must be finite and > 0";
pub const ERROR_FAR: &str = "zfar must be finite and > znear";
pub const ERROR_ASPECT: &str = "aspect must be finite and > 0";
pub const ERROR_VECFINITE: &str = "eye/target/up components must be finite";
pub const ERROR_UPCOLINEAR: &str = "up vector must not be colinear with view direction";
pub const ERROR_CLIP: &str = "clip_space must be 'wgpu' or 'gl'";
pub const ERROR_ORTHO_LEFT_RIGHT: &str = "left must be finite and < right";
pub const ERROR_ORTHO_BOTTOM_TOP: &str = "bottom must be finite and < top";
pub const ERROR_APERTURE: &str = "aperture must be finite and > 0";
pub const ERROR_FOCUS_DISTANCE: &str = "focus_distance must be finite and > 0";
pub const ERROR_FOCAL_LENGTH: &str = "focal_length must be finite and > 0";

/// Validates all components of a Vec3 are finite
pub fn validate_vec3_finite(v: Vec3, _param_name: &str) -> PyResult<()> {
    if !v.is_finite() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(ERROR_VECFINITE));
    }
    Ok(())
}

/// Validates field of view angle
pub fn validate_fovy(fovy_deg: f32) -> PyResult<()> {
    if !fovy_deg.is_finite() || fovy_deg <= 0.0 || fovy_deg >= 180.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(ERROR_FOVY));
    }
    Ok(())
}

/// Validates near plane distance  
pub fn validate_near(znear: f32) -> PyResult<()> {
    if !znear.is_finite() || znear <= 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(ERROR_NEAR));
    }
    Ok(())
}

/// Validates far plane distance relative to near
pub fn validate_far(zfar: f32, znear: f32) -> PyResult<()> {
    if !zfar.is_finite() || zfar <= znear {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(ERROR_FAR));
    }
    Ok(())
}

/// Validates aspect ratio
pub fn validate_aspect(aspect: f32) -> PyResult<()> {
    if !aspect.is_finite() || aspect <= 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(ERROR_ASPECT));
    }
    Ok(())
}

/// Validates clip space parameter
pub fn validate_clip_space(clip_space: &str) -> PyResult<()> {
    match clip_space {
        "wgpu" | "gl" => Ok(()),
        _ => Err(pyo3::exceptions::PyRuntimeError::new_err(ERROR_CLIP)),
    }
}

/// Validates that up vector is not colinear with view direction
pub fn validate_up_not_colinear(eye: Vec3, target: Vec3, up: Vec3) -> PyResult<()> {
    let view_dir = (target - eye).normalize_or_zero();
    let up_norm = up.normalize_or_zero();

    // Check if cross product is near zero (vectors are parallel)
    let cross = view_dir.cross(up_norm);
    if cross.length_squared() < 1e-6 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(ERROR_UPCOLINEAR));
    }
    Ok(())
}

/// Validates orthographic left and right parameters
pub fn validate_ortho_left_right(left: f32, right: f32) -> PyResult<()> {
    if !left.is_finite() || !right.is_finite() || left >= right {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            ERROR_ORTHO_LEFT_RIGHT,
        ));
    }
    Ok(())
}

/// Validates orthographic bottom and top parameters
pub fn validate_ortho_bottom_top(bottom: f32, top: f32) -> PyResult<()> {
    if !bottom.is_finite() || !top.is_finite() || bottom >= top {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            ERROR_ORTHO_BOTTOM_TOP,
        ));
    }
    Ok(())
}

/// Validates aperture parameter for DOF
pub fn validate_aperture(aperture: f32) -> PyResult<()> {
    if !aperture.is_finite() || aperture <= 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(ERROR_APERTURE));
    }
    Ok(())
}

/// Validates focus distance for DOF
pub fn validate_focus_distance(focus_distance: f32) -> PyResult<()> {
    if !focus_distance.is_finite() || focus_distance <= 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            ERROR_FOCUS_DISTANCE,
        ));
    }
    Ok(())
}

/// Validates focal length for DOF
pub fn validate_focal_length(focal_length: f32) -> PyResult<()> {
    if !focal_length.is_finite() || focal_length <= 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            ERROR_FOCAL_LENGTH,
        ));
    }
    Ok(())
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

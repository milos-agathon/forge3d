//! Depth of Field (DOF) camera utilities
//!
//! Provides DOF parameter creation, f-stop conversions, hyperfocal distance,
//! and circle of confusion calculations for physically-based camera rendering.

use crate::camera::validation::{
    validate_aperture, validate_focal_length, validate_focus_distance,
};
use crate::core::dof::CameraDofParams;
use pyo3::prelude::*;

/// Create DOF parameters with validation
#[pyfunction]
#[pyo3(
    text_signature = "(aperture, focus_distance, focal_length, auto_focus=False, auto_focus_speed=2.0)"
)]
pub fn camera_dof_params(
    aperture: f32,
    focus_distance: f32,
    focal_length: f32,
    auto_focus: Option<bool>,
    auto_focus_speed: Option<f32>,
) -> PyResult<(f32, f32, f32, bool, f32)> {
    // Validate DOF parameters
    validate_aperture(aperture)?;
    validate_focus_distance(focus_distance)?;
    validate_focal_length(focal_length)?;

    let auto_focus = auto_focus.unwrap_or(false);
    let auto_focus_speed = auto_focus_speed.unwrap_or(2.0);

    if !auto_focus_speed.is_finite() || auto_focus_speed <= 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "auto_focus_speed must be finite and > 0",
        ));
    }

    Ok((
        aperture,
        focus_distance,
        focal_length,
        auto_focus,
        auto_focus_speed,
    ))
}

/// Convert f-stop to aperture value (reciprocal)
#[pyfunction]
#[pyo3(text_signature = "(f_stop)")]
pub fn camera_f_stop_to_aperture(f_stop: f32) -> PyResult<f32> {
    if !f_stop.is_finite() || f_stop <= 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "f_stop must be finite and > 0",
        ));
    }
    Ok(1.0 / f_stop)
}

/// Convert aperture value to f-stop
#[pyfunction]
#[pyo3(text_signature = "(aperture)")]
pub fn camera_aperture_to_f_stop(aperture: f32) -> PyResult<f32> {
    validate_aperture(aperture)?;
    Ok(1.0 / aperture)
}

/// Calculate hyperfocal distance for DOF
#[pyfunction]
#[pyo3(text_signature = "(focal_length, f_stop, circle_of_confusion=0.03)")]
pub fn camera_hyperfocal_distance(
    focal_length: f32,
    f_stop: f32,
    circle_of_confusion: Option<f32>,
) -> PyResult<f32> {
    validate_focal_length(focal_length)?;
    if !f_stop.is_finite() || f_stop <= 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "f_stop must be finite and > 0",
        ));
    }

    let coc = circle_of_confusion.unwrap_or(0.03); // Default for 35mm film
    if !coc.is_finite() || coc <= 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "circle_of_confusion must be finite and > 0",
        ));
    }

    Ok((focal_length * focal_length) / (f_stop * coc) + focal_length)
}

/// Calculate depth of field range (near and far distances)
#[pyfunction]
#[pyo3(text_signature = "(focal_length, f_stop, focus_distance, circle_of_confusion=0.03)")]
pub fn camera_depth_of_field_range(
    focal_length: f32,
    f_stop: f32,
    focus_distance: f32,
    circle_of_confusion: Option<f32>,
) -> PyResult<(f32, f32)> {
    validate_focal_length(focal_length)?;
    validate_focus_distance(focus_distance)?;
    if !f_stop.is_finite() || f_stop <= 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "f_stop must be finite and > 0",
        ));
    }

    let coc = circle_of_confusion.unwrap_or(0.03); // Default for 35mm film
    if !coc.is_finite() || coc <= 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "circle_of_confusion must be finite and > 0",
        ));
    }

    let h = (focal_length * focal_length) / (f_stop * coc) + focal_length;

    let near = (h * focus_distance) / (h + focus_distance - focal_length);
    let far = if focus_distance < (h - focal_length) {
        (h * focus_distance) / (h - focus_distance + focal_length)
    } else {
        f32::INFINITY
    };

    Ok((near, far))
}

/// Calculate circle of confusion for a given depth and camera parameters
#[pyfunction]
#[pyo3(text_signature = "(depth, focal_length, aperture, focus_distance, sensor_size=36.0)")]
pub fn camera_circle_of_confusion(
    depth: f32,
    focal_length: f32,
    aperture: f32,
    focus_distance: f32,
    sensor_size: Option<f32>,
) -> PyResult<f32> {
    if !depth.is_finite() || depth <= 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "depth must be finite and > 0",
        ));
    }
    validate_focal_length(focal_length)?;
    validate_aperture(aperture)?;
    validate_focus_distance(focus_distance)?;

    let sensor_size = sensor_size.unwrap_or(36.0); // 35mm full frame sensor
    if !sensor_size.is_finite() || sensor_size <= 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "sensor_size must be finite and > 0",
        ));
    }

    let object_distance = depth;
    let distance_diff = (object_distance - focus_distance).abs();
    let denominator = object_distance * (focus_distance + focal_length);

    if denominator < 0.001 {
        return Ok(0.0);
    }

    let coc = (aperture * focal_length * distance_diff) / denominator;
    Ok(coc * sensor_size) // Convert to millimeters
}

/// Create CameraDofParams from validated inputs
pub fn create_camera_dof_params(
    aperture: f32,
    focus_distance: f32,
    focal_length: f32,
    auto_focus: bool,
    auto_focus_speed: f32,
) -> PyResult<CameraDofParams> {
    validate_aperture(aperture)?;
    validate_focus_distance(focus_distance)?;
    validate_focal_length(focal_length)?;

    Ok(CameraDofParams {
        aperture,
        focus_distance,
        focal_length,
        auto_focus,
        auto_focus_speed,
    })
}

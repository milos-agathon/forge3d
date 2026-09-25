//! Thin native entry points for the public `forge3d.astro` package.
#![cfg(feature = "extension-module")]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::astro::{self, time::UtcDateTime, Body};
use crate::geo::units::{Angle, Degree};

#[pyfunction]
pub fn astro_body_position(
    body: &str,
    datetime_utc: &str,
    latitude_deg: f64,
    longitude_deg: f64,
) -> PyResult<(f64, f64, f64)> {
    let body = Body::parse(body).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let utc = UtcDateTime::parse(datetime_utc).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let position = astro::body_position(
        body,
        utc,
        Angle::<Degree>::new(latitude_deg),
        Angle::<Degree>::new(longitude_deg),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        position.azimuth.value(),
        position.altitude.value(),
        position.distance_km,
    ))
}

#[pyfunction]
pub fn astro_moon_phase(datetime_utc: &str) -> PyResult<f64> {
    let utc = UtcDateTime::parse(datetime_utc).map_err(|e| PyValueError::new_err(e.to_string()))?;
    astro::moon_phase(utc).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub fn astro_moon_phase_at(
    datetime_utc: &str,
    latitude_deg: f64,
    longitude_deg: f64,
) -> PyResult<f64> {
    let utc = UtcDateTime::parse(datetime_utc).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let position = astro::body_position(
        Body::Moon,
        utc,
        Angle::<Degree>::new(latitude_deg),
        Angle::<Degree>::new(longitude_deg),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(position
        .illuminated_fraction
        .expect("Moon position has a phase"))
}

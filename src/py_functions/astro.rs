//! Thin native entry points for the public `forge3d.astro` package.
#![cfg(feature = "extension-module")]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::astro::{self, time::UtcDateTime, Body};
use crate::geo::units::{Angle, Degree};
use glam::DVec3;

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
pub fn astro_body_position_refracted(
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
        position.refracted_altitude.value(),
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

/// Internal acceptance probe used by the Python SIDERA oracle. Values are
/// GMST radians, refraction arcminutes at a true 5° altitude, and the two
/// reduction ablations in arcminutes. This is not a user-facing sky API.
#[pyfunction]
pub fn astro_validation_metrics(
    datetime_utc: &str,
    latitude_deg: f64,
    longitude_deg: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    let value = || -> anyhow::Result<(f64, f64, f64, f64)> {
        let utc = UtcDateTime::parse(datetime_utc)?;
        let latitude = Angle::<Degree>::new(latitude_deg);
        let longitude = Angle::<Degree>::new(longitude_deg);
        let jd_tt = utc.jd_tt()?;
        let gmst = astro::time::gmst_utc(utc)?;
        let true_altitude = 5.0_f64.to_radians();
        let refraction_arcmin =
            (astro::frames::refraction_saemundsson(true_altitude) - true_altitude).to_degrees()
                * 60.0;
        let precession = astro::frames::precession_iau2006(jd_tt);
        let precession_arcmin = astro::catalog::stars()?
            .into_iter()
            .map(|star| {
                let (sr, cr) = star.ra_j2000_rad.sin_cos();
                let (sd, cd) = star.dec_j2000_rad.sin_cos();
                let j2000 = DVec3::new(cd * cr, cd * sr, sd);
                j2000.angle_between(precession * j2000).to_degrees() * 60.0
            })
            .fold(0.0_f64, f64::max);
        let moon = astro::frames::ecliptic_j2000_to_true_equatorial(
            astro::moon::geocentric_ecliptic_j2000(jd_tt)?,
            jd_tt,
        )?;
        let (dpsi, deps) = astro::frames::nutation_iau2000b(jd_tt)?;
        let theta = astro::time::gast_utc(
            utc,
            dpsi,
            astro::frames::mean_obliquity_iau2006(jd_tt) + deps,
        )? + longitude.radians();
        let observer = astro::frames::observer_equatorial_km(latitude.radians(), theta, 0.0);
        let parallax_arcmin = moon.angle_between(moon - observer).to_degrees() * 60.0;
        Ok((gmst, refraction_arcmin, precession_arcmin, parallax_arcmin))
    };
    value().map_err(|e| PyValueError::new_err(e.to_string()))
}

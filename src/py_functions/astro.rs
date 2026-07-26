use super::super::*;
use crate::{
    astro::{self, time::UtcDateTime, Body, Observer},
    geo::units::{Angle, Degree},
};
use glam::{DMat3, DVec3};

fn inputs(
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    height_m: f64,
) -> PyResult<(UtcDateTime, Observer)> {
    let utc = UtcDateTime::new(year, month, day, hour, minute, second)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let observer = Observer::new(
        Angle::new(latitude_deg),
        Angle::new(longitude_deg),
        height_m,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((utc, observer))
}

#[pyfunction]
#[pyo3(signature = (
    body, year, month, day, hour, minute, second, latitude_deg, longitude_deg,
    height_m=0.0, refraction=false
))]
pub(crate) fn astro_body_position(
    body: &str,
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    height_m: f64,
    refraction: bool,
) -> PyResult<(f64, f64, f64)> {
    let (utc, observer) = inputs(
        year,
        month,
        day,
        hour,
        minute,
        second,
        latitude_deg,
        longitude_deg,
        height_m,
    )?;
    let position = astro::body_position(
        Body::parse(body).map_err(|error| PyValueError::new_err(error.to_string()))?,
        utc,
        observer,
        refraction,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        position.azimuth.value(),
        position.altitude.value(),
        position.distance_au,
    ))
}

#[pyfunction]
pub(crate) fn astro_moon_phase(
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
) -> PyResult<(f64, f64, f64)> {
    let utc = UtcDateTime::new(year, month, day, hour, minute, second)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let phase = astro::moon_phase(utc).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        phase.illuminated_fraction,
        phase.phase_angle.value(),
        phase.apparent_semidiameter_arcsec,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    year, month, day, hour, minute, second, latitude_deg, longitude_deg, height_m=0.0
))]
pub(crate) fn sky_set_observation(
    py: Python<'_>,
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    height_m: f64,
) -> PyResult<Py<PyDict>> {
    let (utc, observer) = inputs(
        year,
        month,
        day,
        hour,
        minute,
        second,
        latitude_deg,
        longitude_deg,
        height_m,
    )?;
    let observation = astro::observation::set_observation(utc, observer)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let result = PyDict::new_bound(py);
    result.set_item("revision", observation.revision)?;
    result.set_item("star_count", observation.stars.len())?;
    for (body, position) in observation.bodies {
        result.set_item(
            body.name(),
            (
                position.azimuth.value(),
                position.altitude.value(),
                position.distance_au,
            ),
        )?;
    }
    result.set_item(
        "moon_phase",
        (
            observation.moon_phase.illuminated_fraction,
            observation.moon_phase.phase_angle.value(),
            observation.moon_phase.apparent_semidiameter_arcsec,
        ),
    )?;
    Ok(result.unbind())
}

#[pyfunction]
pub(crate) fn astro_validation_metrics(py: Python<'_>) -> PyResult<Py<PyDict>> {
    use astro::time::{julian_day_tt, julian_day_ut1};

    let gmst_cases = [
        (
            UtcDateTime::new(2000, 1, 1, 0, 0, 0.0),
            1.744_793_167_080_565_2,
        ),
        (
            UtcDateTime::new(2026, 7, 26, 22, 0, 0.0),
            4.792_809_523_685_919,
        ),
        (
            UtcDateTime::new(2050, 12, 31, 23, 0, 0.0),
            1.493_405_560_994_384_4,
        ),
    ];
    let mut gmst_max_seconds = 0.0_f64;
    for (utc, expected) in gmst_cases {
        let utc = utc.map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let actual = astro::frames::gmst(
            julian_day_ut1(utc).map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
            julian_day_tt(utc).map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
        );
        gmst_max_seconds =
            gmst_max_seconds.max((actual - expected).abs() * 86_400.0 / std::f64::consts::TAU);
    }

    let refraction_arcminutes =
        (astro::frames::refract_altitude(Angle::<Degree>::new(5.0)).value() - 5.0) * 60.0;
    let utc = UtcDateTime::new(2026, 7, 26, 22, 0, 0.0)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let jd_tt = julian_day_tt(utc).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let precession_arcminutes = DVec3::X
        .angle_between(astro::frames::precess_j2000_to_date(DVec3::X, jd_tt))
        .to_degrees()
        * 60.0;
    let observer = Observer::new(Angle::new(52.3676), Angle::new(4.9041), 0.0)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let sidereal = astro::frames::gast(
        julian_day_ut1(utc).map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
        jd_tt,
    );
    let lunar = astro::moon::geocentric_ecliptic(jd_tt)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let (dpsi, deps, obliquity) = astro::frames::nutation(jd_tt);
    let geocentric = DMat3::from_rotation_x(obliquity + deps)
        * DMat3::from_rotation_z(dpsi)
        * lunar.ecliptic_of_date_au;
    let topocentric = astro::frames::geocentric_to_topocentric(geocentric, observer, sidereal);
    let geo_horizontal = astro::frames::equatorial_to_horizontal(geocentric, observer, sidereal);
    let topo_horizontal = astro::frames::equatorial_to_horizontal(topocentric, observer, sidereal);
    let parallax_arcminutes = horizontal_separation_arcsec(
        geo_horizontal.0.value(),
        geo_horizontal.1.value(),
        topo_horizontal.0.value(),
        topo_horizontal.1.value(),
    ) / 60.0;

    let result = PyDict::new_bound(py);
    result.set_item("gmst_max_seconds", gmst_max_seconds)?;
    result.set_item(
        "refraction_error_arcminutes",
        (refraction_arcminutes - 9.674_126_604).abs(),
    )?;
    result.set_item("precession_arcminutes", precession_arcminutes)?;
    result.set_item("lunar_parallax_arcminutes", parallax_arcminutes)?;
    Ok(result.unbind())
}

fn horizontal_separation_arcsec(az_a: f64, alt_a: f64, az_b: f64, alt_b: f64) -> f64 {
    let delta_azimuth = (az_a - az_b).to_radians();
    let (alt_a, alt_b) = (alt_a.to_radians(), alt_b.to_radians());
    (alt_a.sin() * alt_b.sin() + alt_a.cos() * alt_b.cos() * delta_azimuth.cos())
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
        * 3_600.0
}

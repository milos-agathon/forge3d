//! IAU 2006 precession, IAU 2000B nutation, topocentric parallax and refraction.
//!
//! Rotation polynomials and the 77-term nutation table follow the BSD-licensed
//! ERFA implementation of the IAU standards. The table is generated from
//! `nut00b.c` by `tools/sidera_assets.py`; see `data/sidera/ERFA-LICENSE`.

use anyhow::{ensure, Result};
use glam::{DMat3, DVec3};

use super::time::{gast_utc, UtcDateTime};

const ARCSEC: f64 = std::f64::consts::PI / 648_000.0;
const C_AU_PER_DAY: f64 = 173.144_632_674_240;
const NUTATION: &[u8] = include_bytes!("../../data/sidera/nut00b.bin");

fn rx(a: f64) -> DMat3 {
    let (s, c) = a.sin_cos();
    DMat3::from_cols(DVec3::X, DVec3::new(0.0, c, s), DVec3::new(0.0, -s, c))
}

fn rz(a: f64) -> DMat3 {
    let (s, c) = a.sin_cos();
    DMat3::from_cols(DVec3::new(c, s, 0.0), DVec3::new(-s, c, 0.0), DVec3::Z)
}

pub fn mean_obliquity_iau2006(jd_tt: f64) -> f64 {
    let t = (jd_tt - 2_451_545.0) / 36_525.0;
    (84381.406
        + t * (-46.836769
            + t * (-0.0001831 + t * (0.00200340 + t * (-0.000000576 - 0.0000000434 * t)))))
        * ARCSEC
}

pub fn nutation_iau2000b(jd_tt: f64) -> Result<(f64, f64)> {
    ensure!(&NUTATION[0..4] == b"N00B", "invalid IAU 2000B asset");
    let count = u32::from_le_bytes(NUTATION[4..8].try_into()?) as usize;
    ensure!(
        count == 77 && NUTATION.len() == 8 + count * 53,
        "invalid IAU 2000B terms"
    );
    let t = (jd_tt - 2_451_545.0) / 36_525.0;
    let argument_arcsec = [
        485868.249036 + 1717915923.2178 * t,
        1287104.79305 + 129596581.0481 * t,
        335779.526232 + 1739527262.8478 * t,
        1072260.70369 + 1602961601.2090 * t,
        450160.398036 - 6962890.5431 * t,
    ];
    let argument = argument_arcsec.map(|v| v.rem_euclid(1_296_000.0) * ARCSEC);
    let mut dp = 0.0;
    let mut de = 0.0;
    for i in (0..count).rev() {
        let o = 8 + i * 53;
        let angle = (0..5)
            .map(|k| (NUTATION[o + k] as i8 as f64) * argument[k])
            .sum::<f64>();
        let coefficient = |k: usize| {
            f64::from_le_bytes(
                NUTATION[o + 5 + k * 8..o + 13 + k * 8]
                    .try_into()
                    .expect("nutation coefficient width"),
            )
        };
        let (s, c) = angle.sin_cos();
        dp += (coefficient(0) + coefficient(1) * t) * s + coefficient(2) * c;
        de += (coefficient(3) + coefficient(4) * t) * c + coefficient(5) * s;
    }
    // ERFA nut00b planetary bias, in milliarcseconds.
    Ok((
        (dp * 1e-7 - 0.000135) * ARCSEC,
        (de * 1e-7 + 0.000388) * ARCSEC,
    ))
}

fn fw_angles(jd_tt: f64) -> (f64, f64, f64, f64) {
    let t = (jd_tt - 2_451_545.0) / 36_525.0;
    let gamma = (-0.052928
        + t * (10.556378
            + t * (0.4932044 + t * (-0.00031238 + t * (-0.000002788 + 0.0000000260 * t)))))
        * ARCSEC;
    let phi = (84381.412819
        + t * (-46.811016
            + t * (0.0511268 + t * (0.00053289 + t * (-0.000000440 - 0.0000000176 * t)))))
        * ARCSEC;
    let psi = (-0.041775
        + t * (5038.481484
            + t * (1.5584175 + t * (-0.00018522 + t * (-0.000026452 - 0.0000000148 * t)))))
        * ARCSEC;
    (gamma, phi, psi, mean_obliquity_iau2006(jd_tt))
}

fn fw_matrix(gamma: f64, phi: f64, psi: f64, epsilon: f64) -> DMat3 {
    // Equivalent to ERFA fw2m (its positive axis rotations use the opposite
    // sign convention from glam's active column-vector rotations).
    rx(epsilon) * rz(psi) * rx(-phi) * rz(-gamma)
}

pub fn precession_iau2006(jd_tt: f64) -> DMat3 {
    let (gamma, phi, psi, epsilon) = fw_angles(jd_tt);
    fw_matrix(gamma, phi, psi, epsilon)
}

pub fn precession_nutation_iau2006(jd_tt: f64) -> Result<DMat3> {
    let (gamma, phi, psi, epsilon) = fw_angles(jd_tt);
    let (dpsi, deps) = nutation_iau2000b(jd_tt)?;
    Ok(fw_matrix(gamma, phi, psi + dpsi, epsilon + deps))
}

pub fn ecliptic_date_to_true_equatorial(v: DVec3, jd_tt: f64) -> Result<DVec3> {
    let (dpsi, deps) = nutation_iau2000b(jd_tt)?;
    Ok(rx(mean_obliquity_iau2006(jd_tt) + deps) * rz(dpsi) * v)
}

pub fn ecliptic_j2000_to_true_equatorial(v: DVec3, jd_tt: f64) -> Result<DVec3> {
    Ok(precession_nutation_iau2006(jd_tt)? * rx(mean_obliquity_iau2006(2_451_545.0)) * v)
}

pub fn annual_aberration(unit_direction: DVec3, earth_velocity_au_per_day: DVec3) -> DVec3 {
    // First-order Lorentz aberration: u + beta - (u·beta)u, with beta=v/c.
    // See ERFA eraAb and Explanatory Supplement (Urban & Seidelmann 2013),
    // Eq. 7.40; higher-order terms are outside this reduction's model.
    let beta = earth_velocity_au_per_day / C_AU_PER_DAY;
    (unit_direction + beta - unit_direction.dot(beta) * unit_direction).normalize()
}

#[derive(Clone, Copy, Debug)]
pub struct Topocentric {
    pub azimuth_rad: f64,
    pub altitude_rad: f64,
    pub distance_km: f64,
}

pub fn observer_equatorial_km(latitude_rad: f64, local_sidereal_rad: f64, height_km: f64) -> DVec3 {
    // WGS84 geodetic ellipsoid, a=6378.137 km, f=1/298.257223563.
    let a = 6_378.137;
    let f = 1.0 / 298.257_223_563;
    let e2 = f * (2.0 - f);
    let (sin_lat, cos_lat) = latitude_rad.sin_cos();
    let normal = a / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    let (sin_theta, cos_theta) = local_sidereal_rad.sin_cos();
    DVec3::new(
        (normal + height_km) * cos_lat * cos_theta,
        (normal + height_km) * cos_lat * sin_theta,
        (normal * (1.0 - e2) + height_km) * sin_lat,
    )
}

/// Rotate a true-equatorial direction into the renderer's local horizon axes:
/// +x east, +y up, +z south.
pub fn true_equatorial_direction_to_render_horizon(
    direction: DVec3,
    utc: UtcDateTime,
    latitude_rad: f64,
    longitude_rad: f64,
) -> Result<DVec3> {
    ensure!(
        direction.is_finite() && direction.length_squared() > 0.0,
        "invalid equatorial direction"
    );
    ensure!(
        latitude_rad.is_finite()
            && longitude_rad.is_finite()
            && latitude_rad.abs() <= std::f64::consts::FRAC_PI_2
            && longitude_rad.abs() <= std::f64::consts::PI,
        "invalid observer angles"
    );
    let jd_tt = utc.jd_tt()?;
    let (dpsi, deps) = nutation_iau2000b(jd_tt)?;
    let theta = gast_utc(utc, dpsi, mean_obliquity_iau2006(jd_tt) + deps)? + longitude_rad;
    let direction = direction.normalize();
    let (sin_theta, cos_theta) = theta.sin_cos();
    let (sin_latitude, cos_latitude) = latitude_rad.sin_cos();
    let east = -sin_theta * direction.x + cos_theta * direction.y;
    let north = -sin_latitude * cos_theta * direction.x - sin_latitude * sin_theta * direction.y
        + cos_latitude * direction.z;
    let up = cos_latitude * cos_theta * direction.x
        + cos_latitude * sin_theta * direction.y
        + sin_latitude * direction.z;
    Ok(DVec3::new(east, up, -north).normalize())
}

pub fn topocentric(
    body_equatorial_km: DVec3,
    utc: UtcDateTime,
    latitude_rad: f64,
    longitude_rad: f64,
    observer_height_km: f64,
) -> Result<Topocentric> {
    ensure!(
        latitude_rad.is_finite()
            && longitude_rad.is_finite()
            && latitude_rad.abs() <= std::f64::consts::FRAC_PI_2
            && longitude_rad.abs() <= std::f64::consts::PI,
        "invalid observer angles"
    );
    let jd_tt = utc.jd_tt()?;
    let (dpsi, deps) = nutation_iau2000b(jd_tt)?;
    let theta = gast_utc(utc, dpsi, mean_obliquity_iau2006(jd_tt) + deps)? + longitude_rad;
    let observer = observer_equatorial_km(latitude_rad, theta, observer_height_km);
    let relative = body_equatorial_km - observer;
    let (s, c) = theta.sin_cos();
    let (sl, cl) = latitude_rad.sin_cos();
    let east = -s * relative.x + c * relative.y;
    let north = -sl * c * relative.x - sl * s * relative.y + cl * relative.z;
    let up = cl * c * relative.x + cl * s * relative.y + sl * relative.z;
    Ok(Topocentric {
        azimuth_rad: east.atan2(north).rem_euclid(std::f64::consts::TAU),
        altitude_rad: up.atan2(east.hypot(north)),
        distance_km: relative.length(),
    })
}

pub fn refraction_saemundsson(true_altitude_rad: f64) -> f64 {
    // Saemundsson 1986, standard atmosphere (1010 hPa, 10 °C): additive
    // correction to true altitude in arcminutes. Below -1° the empirical
    // formula is outside its domain and no correction is applied.
    let h = true_altitude_rad.to_degrees();
    if h < -1.0 || h > 90.0 {
        return true_altitude_rad;
    }
    let correction_arcmin = 1.02 / (h + 10.3 / (h + 5.11)).to_radians().tan();
    true_altitude_rad + (correction_arcmin / 60.0).to_radians()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precession_and_nutation_match_erfa_reference_at_j2000() {
        let (dp, de) = nutation_iau2000b(2_451_545.0).unwrap();
        assert!((dp / ARCSEC + 13.93166388897).abs() < 0.001);
        assert!((de / ARCSEC + 5.76941707729).abs() < 0.001);
        assert!((precession_iau2006(2_451_545.0) * DVec3::X).distance(DVec3::X) < 1e-6);
    }

    #[test]
    fn standard_refraction_at_five_degrees() {
        let correction_arcmin =
            (refraction_saemundsson(5f64.to_radians()) - 5f64.to_radians()).to_degrees() * 60.0;
        assert!((correction_arcmin - 9.674).abs() < 0.2);
        // Bennett uses apparent altitude. Invert the Saemundsson true-altitude
        // mapping before comparing the two standard-atmosphere formulas.
        let apparent = 5.0_f64.to_radians();
        let mut low = 3.0_f64.to_radians();
        let mut high = apparent;
        for _ in 0..50 {
            let midpoint = (low + high) * 0.5;
            if refraction_saemundsson(midpoint) < apparent {
                low = midpoint;
            } else {
                high = midpoint;
            }
        }
        let saemundsson_arcmin = (apparent - (low + high) * 0.5).to_degrees() * 60.0;
        let bennett_arcmin = 1.0 / (5.0_f64 + 7.31 / (5.0 + 4.4)).to_radians().tan();
        assert!((saemundsson_arcmin - bennett_arcmin).abs() < 0.2);
    }
}

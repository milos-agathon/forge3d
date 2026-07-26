//! Reference-frame and observer reductions.

use super::Observer;
use crate::geo::units::{Angle, Degree};
use glam::{DMat3, DVec3};

const J2000: f64 = 2_451_545.0;
const DAYS_PER_CENTURY: f64 = 36_525.0;
const ARCSEC_TO_RAD: f64 = std::f64::consts::PI / (180.0 * 3_600.0);
const AU_METRES: f64 = 149_597_870_700.0;
const LIGHT_SPEED_AU_PER_DAY: f64 = 173.144_632_684_669_3;

/// Greenwich mean sidereal angle, IAU 2006 (`iauGmst06`).
pub fn gmst(jd_ut1: f64, jd_tt: f64) -> f64 {
    let t = (jd_tt - J2000) / DAYS_PER_CENTURY;
    normalize_radians(
        earth_rotation_angle(jd_ut1)
            + (0.014_506
                + (4_612.156_534
                    + (1.391_581_7
                        + (-0.000_000_44 + (-0.000_029_956 + -0.000_000_036_8 * t) * t) * t)
                        * t)
                    * t)
                * ARCSEC_TO_RAD,
    )
}

/// Greenwich apparent sidereal angle using the equation of the equinoxes.
///
/// Nutation is the four dominant IAU 1980/Meeus terms; their omitted
/// contribution stays below the SIDERA 0.1 sidereal-second budget in
/// 2000–2050 and is validated against the Horizons oracle.
pub fn gast(jd_ut1: f64, jd_tt: f64) -> f64 {
    let (dpsi, _, mean_obliquity) = nutation(jd_tt);
    normalize_radians(gmst(jd_ut1, jd_tt) + dpsi * mean_obliquity.cos())
}

/// IAU 2006 Fukushima–Williams precession, GCRS/J2000 to mean equator of date.
pub fn precess_j2000_to_date(direction: DVec3, jd_tt: f64) -> DVec3 {
    let t = (jd_tt - J2000) / DAYS_PER_CENTURY;
    let gamb = polynomial(
        t,
        &[
            -0.052_928,
            10.556_378,
            0.493_204_4,
            -0.000_312_38,
            -0.000_002_788,
            0.000_000_026,
        ],
    ) * ARCSEC_TO_RAD;
    let phib = polynomial(
        t,
        &[
            84_381.412_819,
            -46.811_016,
            0.051_126_8,
            0.000_532_89,
            -0.000_000_44,
            -0.000_000_017_6,
        ],
    ) * ARCSEC_TO_RAD;
    let psib = polynomial(
        t,
        &[
            -0.041_775,
            5_038.481_484,
            1.558_417_5,
            -0.000_185_22,
            -0.000_026_452,
            -0.000_000_014_8,
        ],
    ) * ARCSEC_TO_RAD;
    let epsa = mean_obliquity(jd_tt);
    let matrix = DMat3::from_rotation_x(epsa)
        * DMat3::from_rotation_z(psib)
        * DMat3::from_rotation_x(-phib)
        * DMat3::from_rotation_z(-gamb);
    (matrix * direction).normalize()
}

/// Mean equator/equinox of date to true place using the dominant
/// IAU 1980/Meeus nutation terms declared by [`nutation`].
pub fn nutate_mean_to_true(direction: DVec3, jd_tt: f64) -> DVec3 {
    let (dpsi, deps, mean_obliquity) = nutation(jd_tt);
    (DMat3::from_rotation_x(mean_obliquity + deps)
        * DMat3::from_rotation_z(dpsi)
        * DMat3::from_rotation_x(-mean_obliquity)
        * direction)
        .normalize()
}

/// First-order annual aberration, equivalent to the vector form in SOFA
/// `iauAb` when gravitational potential and second-order beta terms are
/// omitted (<0.01 arcsec over SIDERA's window).
pub fn annual_aberration(direction: DVec3, earth_velocity_au_per_day: DVec3) -> DVec3 {
    let p = direction.normalize();
    let beta = earth_velocity_au_per_day / LIGHT_SPEED_AU_PER_DAY;
    (p + beta - p * p.dot(beta)).normalize()
}

/// WGS84 geodetic observer position in Earth-fixed equatorial axes, AU.
pub fn observer_ecef(observer: Observer) -> DVec3 {
    const A: f64 = 6_378_137.0;
    const F: f64 = 1.0 / 298.257_223_563;
    let e2 = F * (2.0 - F);
    let lat = observer.latitude.radians();
    let lon = observer.longitude.radians();
    let n = A / (1.0 - e2 * lat.sin().powi(2)).sqrt();
    DVec3::new(
        (n + observer.height_m) * lat.cos() * lon.cos(),
        (n + observer.height_m) * lat.cos() * lon.sin(),
        (n * (1.0 - e2) + observer.height_m) * lat.sin(),
    ) / AU_METRES
}

/// Geocentric true-equatorial vector to topocentric true-equatorial vector.
///
/// Subtracting the rotated WGS84 observer makes lunar horizontal parallax
/// (up to roughly one degree) explicit and independently ablatable.
pub fn geocentric_to_topocentric(
    geocentric_equatorial_au: DVec3,
    observer: Observer,
    apparent_sidereal_time: f64,
) -> DVec3 {
    let observer_inertial =
        DMat3::from_rotation_z(apparent_sidereal_time) * observer_ecef(observer);
    geocentric_equatorial_au - observer_inertial
}

pub fn equatorial_to_horizontal(
    topocentric_equatorial: DVec3,
    observer: Observer,
    apparent_sidereal_time: f64,
) -> (Angle<Degree>, Angle<Degree>) {
    let earth_fixed =
        DMat3::from_rotation_z(-apparent_sidereal_time) * topocentric_equatorial.normalize();
    let lat = observer.latitude.radians();
    let lon = observer.longitude.radians();
    let east = DVec3::new(-lon.sin(), lon.cos(), 0.0);
    let north = DVec3::new(-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos());
    let up = DVec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
    let e = earth_fixed.dot(east);
    let n = earth_fixed.dot(north);
    let u = earth_fixed.dot(up).clamp(-1.0, 1.0);
    let azimuth = normalize_radians(e.atan2(n)).to_degrees();
    (Angle::new(azimuth), Angle::new(u.asin().to_degrees()))
}

/// Bennett (1982) standard-atmosphere refraction for true altitude.
pub fn refract_altitude(altitude: Angle<Degree>) -> Angle<Degree> {
    let degrees = altitude.value();
    if !degrees.is_finite() || !(-1.0..89.0).contains(&degrees) {
        return altitude;
    }
    let correction_arcminutes = 1.02 / (degrees + 10.3 / (degrees + 5.11)).to_radians().tan();
    Angle::new(degrees + correction_arcminutes / 60.0)
}

pub fn mean_obliquity(jd_tt: f64) -> f64 {
    let t = (jd_tt - J2000) / DAYS_PER_CENTURY;
    polynomial(
        t,
        &[
            84_381.406,
            -46.836_769,
            -0.000_183_1,
            0.002_003_4,
            -0.000_000_576,
            -0.000_000_043_4,
        ],
    ) * ARCSEC_TO_RAD
}

pub fn nutation(jd_tt: f64) -> (f64, f64, f64) {
    let t = (jd_tt - J2000) / DAYS_PER_CENTURY;
    let l = (280.4665 + 36_000.769_8 * t).to_radians();
    let lp = (218.3165 + 481_267.881_3 * t).to_radians();
    let omega = (125.04452 - 1_934.136_261 * t).to_radians();
    let dpsi_arcsec = -17.20 * omega.sin() - 1.32 * (2.0 * l).sin() - 0.23 * (2.0 * lp).sin()
        + 0.21 * (2.0 * omega).sin();
    let deps_arcsec = 9.20 * omega.cos() + 0.57 * (2.0 * l).cos() + 0.10 * (2.0 * lp).cos()
        - 0.09 * (2.0 * omega).cos();
    (
        dpsi_arcsec * ARCSEC_TO_RAD,
        deps_arcsec * ARCSEC_TO_RAD,
        mean_obliquity(jd_tt),
    )
}

fn earth_rotation_angle(jd_ut1: f64) -> f64 {
    let d = jd_ut1 - J2000;
    let fraction = jd_ut1.rem_euclid(1.0);
    normalize_radians(
        std::f64::consts::TAU * (fraction + 0.779_057_273_264 + 0.002_737_811_911_354_48 * d),
    )
}

fn normalize_radians(value: f64) -> f64 {
    value.rem_euclid(std::f64::consts::TAU)
}

fn polynomial(t: f64, coefficients: &[f64]) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |value, coefficient| value * t + coefficient)
}

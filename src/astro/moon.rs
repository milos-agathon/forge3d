//! Meeus-class 60-term lunar longitude, distance, and latitude theory.
//!
//! The complete tables from *Astronomical Algorithms*, chapter 47, are
//! committed in compact form. SIDERA declares a 30 arcsecond maximum
//! topocentric position error over 2000–2050 and gates it against Horizons.

use super::AstroError;
use glam::DVec3;
use std::io::{Cursor, Read};
use std::sync::OnceLock;

const DATA: &[u8] = include_bytes!("../../assets/astro/moon_terms.bin");
const AU_KM: f64 = 149_597_870.7;
const J2000: f64 = 2_451_545.0;

#[derive(Clone, Copy, Debug)]
pub struct LunarCoordinates {
    pub ecliptic_of_date_au: DVec3,
    pub distance_km: f64,
}

#[derive(Debug)]
struct Theory {
    longitude_radius: Vec<LongitudeRadiusTerm>,
    latitude: Vec<LatitudeTerm>,
}

#[derive(Debug)]
struct LongitudeRadiusTerm {
    argument: [i8; 4],
    longitude: i32,
    radius: i32,
}

#[derive(Debug)]
struct LatitudeTerm {
    argument: [i8; 4],
    latitude: i32,
}

pub fn geocentric_ecliptic(jd_tt: f64) -> Result<LunarCoordinates, AstroError> {
    let t = (jd_tt - J2000) / 36_525.0;
    let l_prime = polynomial(
        t,
        &[
            218.316_447_7,
            481_267.881_234_21,
            -0.001_578_6,
            1.0 / 538_841.0,
            -1.0 / 65_194_000.0,
        ],
    );
    let d = polynomial(
        t,
        &[
            297.850_192_1,
            445_267.111_403_4,
            -0.001_881_9,
            1.0 / 545_868.0,
            -1.0 / 113_065_000.0,
        ],
    );
    let m = polynomial(
        t,
        &[
            357.529_109_2,
            35_999.050_290_9,
            -0.000_153_6,
            1.0 / 24_490_000.0,
        ],
    );
    let m_prime = polynomial(
        t,
        &[
            134.963_396_4,
            477_198.867_505_5,
            0.008_741_4,
            1.0 / 69_699.0,
            -1.0 / 14_712_000.0,
        ],
    );
    let f = polynomial(
        t,
        &[
            93.272_095,
            483_202.017_523_3,
            -0.003_653_9,
            -1.0 / 3_526_000.0,
            1.0 / 863_310_000.0,
        ],
    );
    let arguments = [d, m, m_prime, f].map(f64::to_radians);
    let eccentricity = 1.0 - 0.002_516 * t - 0.000_007_4 * t * t;
    let theory = theory()?;
    let mut longitude_sum = 0.0;
    let mut radius_sum = 0.0;
    for term in &theory.longitude_radius {
        let argument = dot(term.argument, arguments);
        let scale = eccentricity.powi(i32::from(term.argument[1].unsigned_abs()));
        longitude_sum += f64::from(term.longitude) * scale * argument.sin();
        radius_sum += f64::from(term.radius) * scale * argument.cos();
    }
    let a1 = (119.75 + 131.849 * t).to_radians();
    let a2 = (53.09 + 479_264.290 * t).to_radians();
    longitude_sum +=
        3_958.0 * a1.sin() + 1_962.0 * (l_prime - f).to_radians().sin() + 318.0 * a2.sin();

    let mut latitude_sum = 0.0;
    for term in &theory.latitude {
        let argument = dot(term.argument, arguments);
        let scale = eccentricity.powi(i32::from(term.argument[1].unsigned_abs()));
        latitude_sum += f64::from(term.latitude) * scale * argument.sin();
    }
    let a3 = (313.45 + 481_266.484 * t).to_radians();
    latitude_sum += -2_235.0 * l_prime.to_radians().sin()
        + 382.0 * a3.sin()
        + 175.0 * (a1 - f.to_radians()).sin()
        + 175.0 * (a1 + f.to_radians()).sin()
        + 127.0 * (l_prime - m_prime).to_radians().sin()
        - 115.0 * (l_prime + m_prime).to_radians().sin();

    let longitude = (l_prime + longitude_sum / 1_000_000.0).to_radians();
    let latitude = (latitude_sum / 1_000_000.0).to_radians();
    let distance_km = 385_000.56 + radius_sum / 1_000.0;
    let (sin_l, cos_l) = longitude.sin_cos();
    let (sin_b, cos_b) = latitude.sin_cos();
    Ok(LunarCoordinates {
        ecliptic_of_date_au: DVec3::new(cos_b * cos_l, cos_b * sin_l, sin_b)
            * (distance_km / AU_KM),
        distance_km,
    })
}

fn dot(coefficients: [i8; 4], arguments: [f64; 4]) -> f64 {
    coefficients
        .iter()
        .zip(arguments)
        .map(|(coefficient, argument)| f64::from(*coefficient) * argument)
        .sum()
}

fn polynomial(t: f64, coefficients: &[f64]) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |value, coefficient| value * t + coefficient)
        .rem_euclid(360.0)
}

fn theory() -> Result<&'static Theory, AstroError> {
    static THEORY: OnceLock<Result<Theory, AstroError>> = OnceLock::new();
    THEORY.get_or_init(parse).as_ref().map_err(Clone::clone)
}

fn parse() -> Result<Theory, AstroError> {
    let mut input = Cursor::new(DATA);
    let mut magic = [0; 8];
    input
        .read_exact(&mut magic)
        .map_err(|_| AstroError::InvalidData("moon_terms.bin"))?;
    if &magic != b"F3DMOON1" {
        return Err(AstroError::InvalidData("moon_terms.bin"));
    }
    let longitude_count = read_u32(&mut input)? as usize;
    let latitude_count = read_u32(&mut input)? as usize;
    let mut longitude_radius = Vec::with_capacity(longitude_count);
    for _ in 0..longitude_count {
        longitude_radius.push(LongitudeRadiusTerm {
            argument: read_arguments(&mut input)?,
            longitude: read_i32(&mut input)?,
            radius: read_i32(&mut input)?,
        });
    }
    let mut latitude = Vec::with_capacity(latitude_count);
    for _ in 0..latitude_count {
        latitude.push(LatitudeTerm {
            argument: read_arguments(&mut input)?,
            latitude: read_i32(&mut input)?,
        });
    }
    if input.position() != DATA.len() as u64 {
        return Err(AstroError::InvalidData("moon_terms.bin"));
    }
    Ok(Theory {
        longitude_radius,
        latitude,
    })
}

fn read_arguments(input: &mut Cursor<&[u8]>) -> Result<[i8; 4], AstroError> {
    let mut bytes = [0; 4];
    input
        .read_exact(&mut bytes)
        .map_err(|_| AstroError::InvalidData("moon_terms.bin"))?;
    Ok(bytes.map(|value| value as i8))
}

fn read_u32(input: &mut Cursor<&[u8]>) -> Result<u32, AstroError> {
    let mut bytes = [0; 4];
    input
        .read_exact(&mut bytes)
        .map_err(|_| AstroError::InvalidData("moon_terms.bin"))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32(input: &mut Cursor<&[u8]>) -> Result<i32, AstroError> {
    let mut bytes = [0; 4];
    input
        .read_exact(&mut bytes)
        .map_err(|_| AstroError::InvalidData("moon_terms.bin"))?;
    Ok(i32::from_le_bytes(bytes))
}

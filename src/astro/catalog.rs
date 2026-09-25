//! Yale Bright Star Catalogue, fifth edition: J2000 RA/Dec, V and B−V.
//!
//! The compact file contains 9,096 records with usable J2000 position and V
//! magnitude. Source and epoch are recorded in `data/sidera/MANIFEST.md`.

use anyhow::{ensure, Result};
use glam::DVec3;

use super::time::{gast_utc, UtcDateTime};
use super::{frames, vsop};

const DATA: &[u8] = include_bytes!("../../data/sidera/ybsc5.bin");
const STAR_WIDTH: usize = 16;

#[derive(Clone, Copy, Debug)]
pub struct Star {
    pub ra_j2000_rad: f64,
    pub dec_j2000_rad: f64,
    pub v_magnitude: f32,
    pub b_minus_v: f32,
}

pub fn stars() -> Result<Vec<Star>> {
    ensure!(&DATA[0..4] == b"YBS5", "invalid Yale catalog asset magic");
    let count = u32::from_le_bytes(DATA[4..8].try_into()?) as usize;
    ensure!(
        count == 9_096 && DATA.len() == 8 + count * STAR_WIDTH,
        "invalid Yale catalog asset length"
    );
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let offset = 8 + i * STAR_WIDTH;
        let value = |n: usize| {
            f32::from_le_bytes(
                DATA[offset + n * 4..offset + n * 4 + 4]
                    .try_into()
                    .expect("star record width"),
            )
        };
        out.push(Star {
            ra_j2000_rad: value(0) as f64,
            dec_j2000_rad: value(1) as f64,
            v_magnitude: value(2),
            b_minus_v: value(3),
        });
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
pub struct VisibleStar {
    pub azimuth_rad: f64,
    pub altitude_rad: f64,
    pub relative_irradiance: f32,
    pub b_minus_v: f32,
}

pub fn visible_stars(
    utc: UtcDateTime,
    latitude_rad: f64,
    longitude_rad: f64,
) -> Result<Vec<VisibleStar>> {
    utc.validate()?;
    let jd_tt = utc.jd_tt()?;
    let matrix = frames::precession_nutation_iau2006(jd_tt)?;
    let earth_velocity =
        frames::ecliptic_date_to_true_equatorial(vsop::earth_velocity_ecliptic(jd_tt)?, jd_tt)?;
    let (dpsi, deps) = frames::nutation_iau2000b(jd_tt)?;
    let theta = gast_utc(utc, dpsi, frames::mean_obliquity_iau2006(jd_tt) + deps)? + longitude_rad;
    let (st, ct) = theta.sin_cos();
    let (sl, cl) = latitude_rad.sin_cos();
    let mut out = Vec::new();
    for star in stars()? {
        let (sr, cr) = star.ra_j2000_rad.sin_cos();
        let (sd, cd) = star.dec_j2000_rad.sin_cos();
        let true_direction =
            frames::annual_aberration(matrix * DVec3::new(cd * cr, cd * sr, sd), earth_velocity);
        let east = -st * true_direction.x + ct * true_direction.y;
        let north =
            -sl * ct * true_direction.x - sl * st * true_direction.y + cl * true_direction.z;
        let up = cl * ct * true_direction.x + cl * st * true_direction.y + sl * true_direction.z;
        if up > 0.0 {
            out.push(VisibleStar {
                azimuth_rad: east.atan2(north).rem_euclid(std::f64::consts::TAU),
                altitude_rad: up.atan2(east.hypot(north)),
                relative_irradiance: 10.0_f32.powf(-0.4 * star.v_magnitude),
                b_minus_v: star.b_minus_v,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_nearly_nine_thousand_finite_stars() {
        let entries = stars().unwrap();
        assert_eq!(entries.len(), 9_096);
        assert!(entries.iter().all(|star| star.ra_j2000_rad.is_finite()
            && star.dec_j2000_rad.is_finite()
            && star.v_magnitude.is_finite()
            && star.b_minus_v.is_finite()));
        for (index, star) in entries.iter().enumerate() {
            let offset = 8 + index * STAR_WIDTH;
            let encoded = [
                star.ra_j2000_rad as f32,
                star.dec_j2000_rad as f32,
                star.v_magnitude,
                star.b_minus_v,
            ];
            assert_eq!(
                bytemuck::cast_slice::<f32, u8>(&encoded),
                &DATA[offset..offset + STAR_WIDTH]
            );
        }
    }
}

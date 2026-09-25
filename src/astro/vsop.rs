//! VSOP87D heliocentric L/B/R series from IMCCE's published coefficients.
//!
//! The binary is a byte-exact representation of all 25,766 source terms;
//! precision is limited by the source theory and subsequent frame reductions,
//! not by a hidden term cutoff. T is Julian millennia of TDB from J2000. For
//! this interval we use TT as TDB; their periodic offset is below 2 ms.

use anyhow::{ensure, Result};
use glam::DVec3;

use super::Body;

const DATA: &[u8] = include_bytes!("../../data/sidera/vsop87d.bin");
const TERM_WIDTH: usize = 27;
const AU_LIGHT_DAYS: f64 = 0.005_775_518_331_09;

fn term(offset: usize) -> (usize, usize, usize, f64, f64, f64) {
    let body = DATA[offset] as usize;
    let variable = DATA[offset + 1] as usize;
    let power = DATA[offset + 2] as usize;
    let value = |start: usize| {
        f64::from_le_bytes(
            DATA[offset + start..offset + start + 8]
                .try_into()
                .expect("VSOP term width"),
        )
    };
    (body, variable, power, value(3), value(11), value(19))
}

pub fn heliocentric_lbr(body: usize, jd_tt: f64) -> Result<[f64; 3]> {
    ensure!(body < 6 && jd_tt.is_finite(), "invalid VSOP body or time");
    ensure!(&DATA[0..4] == b"V87D", "invalid VSOP asset magic");
    let count = u32::from_le_bytes(DATA[4..8].try_into()?) as usize;
    ensure!(
        DATA.len() == 8 + count * TERM_WIDTH,
        "invalid VSOP asset length"
    );
    let t = (jd_tt - 2_451_545.0) / 365_250.0;
    let mut sums = [0.0; 3];
    for i in 0..count {
        let (target, variable, power, amplitude, phase, frequency) = term(8 + i * TERM_WIDTH);
        if target == body {
            sums[variable] += amplitude * (phase + frequency * t).cos() * t.powi(power as i32);
        }
    }
    sums[0] = sums[0].rem_euclid(std::f64::consts::TAU);
    Ok(sums)
}

pub fn heliocentric_xyz(body: usize, jd_tt: f64) -> Result<DVec3> {
    let [longitude, latitude, radius] = heliocentric_lbr(body, jd_tt)?;
    let cb = latitude.cos();
    Ok(DVec3::new(
        radius * cb * longitude.cos(),
        radius * cb * longitude.sin(),
        radius * latitude.sin(),
    ))
}

pub fn geocentric_ecliptic(body: Body, jd_tt: f64) -> Result<DVec3> {
    let earth = heliocentric_xyz(0, jd_tt)?;
    let index = match body {
        Body::Sun => return Ok(-earth),
        Body::Mercury => 1,
        Body::Venus => 2,
        Body::Mars => 3,
        Body::Jupiter => 4,
        Body::Saturn => 5,
        Body::Moon => anyhow::bail!("lunar position uses ELP82B"),
    };
    // Geometric light time is solved against the retarded planetary position.
    let mut light_days = 0.0;
    let mut relative = DVec3::ZERO;
    for _ in 0..3 {
        relative = heliocentric_xyz(index, jd_tt - light_days)? - earth;
        light_days = relative.length() * AU_LIGHT_DAYS;
    }
    Ok(relative)
}

pub fn earth_velocity_ecliptic(jd_tt: f64) -> Result<DVec3> {
    const HALF_STEP_DAYS: f64 = 0.001;
    Ok((heliocentric_xyz(0, jd_tt + HALF_STEP_DAYS)?
        - heliocentric_xyz(0, jd_tt - HALF_STEP_DAYS)?)
        / (2.0 * HALF_STEP_DAYS))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earth_distance_at_j2000() {
        let [_, _, radius] = heliocentric_lbr(0, 2_451_545.0).unwrap();
        assert!((radius - 0.9833).abs() < 0.001);
    }
}

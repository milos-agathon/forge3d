//! ELP2000-82B lunar theory, using IMCCE's complete published term set.
//!
//! The source's 36 tables are deterministically compiled to `elp82b.bin` by
//! `tools/sidera_assets.py`. This evaluator follows IMCCE's `elp82b_1`
//! spherical sums and its ecliptic J2000 frame rotation. Output is geocentric
//! kilometres in the mean J2000 ecliptic frame.

use anyhow::{ensure, Result};
use glam::DVec3;

const DATA: &[u8] = include_bytes!("../../data/sidera/elp82b.bin");
const MAIN_WIDTH: usize = 49;
const SECONDARY_WIDTH: usize = 26;
const RAD: f64 = 648_000.0 / std::f64::consts::PI;

fn read_f64(offset: usize) -> f64 {
    f64::from_le_bytes(
        DATA[offset..offset + 8]
            .try_into()
            .expect("ELP coefficient width"),
    )
}

pub fn geocentric_ecliptic_j2000(jd_tt: f64) -> Result<DVec3> {
    ensure!(jd_tt.is_finite(), "non-finite lunar time");
    ensure!(&DATA[0..4] == b"ELP8", "invalid ELP asset magic");
    let main_count = u32::from_le_bytes(DATA[4..8].try_into()?) as usize;
    let secondary_count = u32::from_le_bytes(DATA[8..12].try_into()?) as usize;
    ensure!(
        DATA.len() == 12 + main_count * MAIN_WIDTH + secondary_count * SECONDARY_WIDTH,
        "invalid ELP asset length"
    );
    let t = (jd_tt - 2_451_545.0) / 36_525.0;
    let powers = [1.0, t, t * t, t * t * t, t * t * t * t];
    let mut sums = [0.0; 3];
    for i in 0..main_count {
        let offset = 12 + i * MAIN_WIDTH;
        let variable = DATA[offset] as usize;
        ensure!(variable < 3, "invalid ELP main variable");
        let amplitude = read_f64(offset + 1);
        let phase = (0..5)
            .map(|k| read_f64(offset + 9 + k * 8) * powers[k])
            .sum::<f64>();
        sums[variable] += amplitude * phase.sin();
    }
    for i in 0..secondary_count {
        let offset = 12 + main_count * MAIN_WIDTH + i * SECONDARY_WIDTH;
        let variable = DATA[offset] as usize;
        let time_power = DATA[offset + 1] as usize;
        ensure!(
            variable < 3 && time_power < 3,
            "invalid ELP secondary index"
        );
        let amplitude = read_f64(offset + 2);
        let phase = read_f64(offset + 10) + read_f64(offset + 18) * t;
        sums[variable] += amplitude * powers[time_power] * phase.sin();
    }
    let mean_longitude = (218.0_f64 + 18.0 / 60.0 + 59.95571 / 3600.0).to_radians()
        + (1732559343.73604 * t - 5.8883 * t * t + 0.006604 * t.powi(3) - 0.00003169 * t.powi(4))
            / RAD;
    let longitude = sums[0] / RAD + mean_longitude;
    let latitude = sums[1] / RAD;
    let distance = sums[2] * (384747.9806448954 / 384747.9806743165);
    ensure!(
        distance > 300_000.0 && distance < 450_000.0,
        "ELP lunar distance outside physical range"
    );
    let (sl, cl) = longitude.sin_cos();
    let (sb, cb) = latitude.sin_cos();
    let x = distance * cb * cl;
    let y = distance * cb * sl;
    let z = distance * sb;
    // IMCCE elp82b_1: rotate its mean ecliptic longitude/latitude to the
    // inertial J2000 ecliptic frame with the fitted pw/qw precession terms.
    let pw = t
        * (0.10180391e-4
            + t * (0.47020439e-6 + t * (-0.5417367e-9 + t * (-0.2507948e-11 + t * 0.463486e-14))));
    let qw = t
        * (-0.113469002e-3
            + t * (0.12372674e-6 + t * (0.1265417e-8 + t * (-0.1371808e-11 - t * 0.320334e-14))));
    let root = 2.0 * (1.0 - pw * pw - qw * qw).sqrt();
    let pq = 2.0 * pw * qw;
    let p2 = 1.0 - 2.0 * pw * pw;
    let q2 = 1.0 - 2.0 * qw * qw;
    let p = pw * root;
    let q = qw * root;
    Ok(DVec3::new(
        p2 * x + pq * y + p * z,
        pq * x + q2 * y - q * z,
        -p * x + q * y + (p2 + q2 - 1.0) * z,
    ))
}

pub fn apparent_semidiameter_radians(distance_km: f64) -> Result<f64> {
    ensure!(
        distance_km.is_finite() && distance_km > 1_737.4,
        "invalid Moon range"
    );
    Ok((1_737.4 / distance_km).asin())
}

pub fn illuminated_fraction(sun_from_moon: DVec3, observer_from_moon: DVec3) -> Result<f64> {
    ensure!(
        sun_from_moon.length_squared() > 0.0 && observer_from_moon.length_squared() > 0.0,
        "invalid lunar phase geometry"
    );
    let cosine = sun_from_moon
        .normalize()
        .dot(observer_from_moon.normalize())
        .clamp(-1.0, 1.0);
    Ok((1.0 + cosine) * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moon_at_j2000_is_finite_and_nearby() {
        let v = geocentric_ecliptic_j2000(2_451_545.0).unwrap();
        assert!(v.is_finite());
        assert!((350_000.0..420_000.0).contains(&v.length()));
    }
}

//! SIDERA: bounded, offline astronomical positions for fixed Earth observers.
//!
//! The analytic models and committed coefficient sets are defined for UTC
//! 2000-01-01 through 2050-12-31. Requests outside that interval fail.

pub mod catalog;
pub mod frames;
pub mod moon;
pub mod render;
pub mod time;
pub mod vsop;

use anyhow::Result;
use glam::DVec3;

use crate::geo::units::{Angle, Degree};

const AU_KM: f64 = 149_597_870.700;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Body {
    Sun,
    Moon,
    Mercury,
    Venus,
    Mars,
    Jupiter,
    Saturn,
}

impl Body {
    pub fn parse(name: &str) -> anyhow::Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "sun" => Ok(Self::Sun),
            "moon" => Ok(Self::Moon),
            "mercury" => Ok(Self::Mercury),
            "venus" => Ok(Self::Venus),
            "mars" => Ok(Self::Mars),
            "jupiter" => Ok(Self::Jupiter),
            "saturn" => Ok(Self::Saturn),
            _ => anyhow::bail!("unsupported SIDERA body: {name}"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BodyPosition {
    pub azimuth: Angle<Degree>,
    /// Airless apparent topocentric altitude; suitable for Horizons APPARENT=AIRLESS.
    pub altitude: Angle<Degree>,
    /// Saemundsson standard-atmosphere correction, reported separately.
    pub refracted_altitude: Angle<Degree>,
    pub distance_km: f64,
    pub apparent_semidiameter: Option<Angle<Degree>>,
    pub illuminated_fraction: Option<f64>,
}

fn sun_or_planet_vector(body: Body, jd_tt: f64) -> Result<DVec3> {
    let ecliptic = vsop::geocentric_ecliptic(body, jd_tt)?;
    let velocity = vsop::earth_velocity_ecliptic(jd_tt)?;
    let direction = frames::annual_aberration(ecliptic.normalize(), velocity);
    let equatorial = frames::ecliptic_date_to_true_equatorial(direction, jd_tt)?;
    Ok(equatorial * ecliptic.length() * AU_KM)
}

pub fn body_position(
    body: Body,
    utc: time::UtcDateTime,
    latitude: Angle<Degree>,
    longitude: Angle<Degree>,
) -> Result<BodyPosition> {
    utc.validate()?;
    let jd_tt = utc.jd_tt()?;
    let vector = if body == Body::Moon {
        frames::ecliptic_j2000_to_true_equatorial(moon::geocentric_ecliptic_j2000(jd_tt)?, jd_tt)?
    } else {
        sun_or_planet_vector(body, jd_tt)?
    };
    let topocentric =
        frames::topocentric(vector, utc, latitude.radians(), longitude.radians(), 0.0)?;
    let moon_fraction = if body == Body::Moon {
        let (dpsi, deps) = frames::nutation_iau2000b(jd_tt)?;
        let theta = time::gast_utc(utc, dpsi, frames::mean_obliquity_iau2006(jd_tt) + deps)?
            + longitude.radians();
        let observer = frames::observer_equatorial_km(latitude.radians(), theta, 0.0);
        let sun = sun_or_planet_vector(Body::Sun, jd_tt)?;
        Some(moon::illuminated_fraction(sun - vector, observer - vector)?)
    } else {
        None
    };
    let semidiameter = if body == Body::Moon {
        Some(Angle::new(
            moon::apparent_semidiameter_radians(topocentric.distance_km)?.to_degrees(),
        ))
    } else {
        None
    };
    Ok(BodyPosition {
        azimuth: Angle::new(topocentric.azimuth_rad.to_degrees()),
        altitude: Angle::new(topocentric.altitude_rad.to_degrees()),
        refracted_altitude: Angle::new(
            frames::refraction_saemundsson(topocentric.altitude_rad).to_degrees(),
        ),
        distance_km: topocentric.distance_km,
        apparent_semidiameter: semidiameter,
        illuminated_fraction: moon_fraction,
    })
}

pub fn moon_phase(utc: time::UtcDateTime) -> Result<f64> {
    utc.validate()?;
    let jd_tt = utc.jd_tt()?;
    let moon =
        frames::ecliptic_j2000_to_true_equatorial(moon::geocentric_ecliptic_j2000(jd_tt)?, jd_tt)?;
    let sun = sun_or_planet_vector(Body::Sun, jd_tt)?;
    moon::illuminated_fraction(sun - moon, -moon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizons_oracle_summary() {
        let data = include_str!("../../tests/data/horizons_vectors.dat");
        let mut worst: [(f64, String); 7] = std::array::from_fn(|_| (0.0, String::new()));
        let mut worst_phase = (0.0_f64, String::new());
        let mut worst_semi = (0.0_f64, String::new());
        let mut count = 0;
        for row in data
            .lines()
            .filter(|row| !row.starts_with('#') && !row.starts_with("utc,"))
        {
            let fields: Vec<_> = row.split(',').collect();
            assert_eq!(fields.len(), 10);
            let utc = time::UtcDateTime::parse(fields[0]).unwrap();
            let lat: f64 = fields[2].parse().unwrap();
            let lon: f64 = fields[3].parse().unwrap();
            let body = Body::parse(fields[4]).unwrap();
            let actual = body_position(body, utc, Angle::new(lat), Angle::new(lon)).unwrap();
            let az: f64 = fields[5].parse::<f64>().unwrap().to_radians();
            let alt: f64 = fields[6].parse::<f64>().unwrap().to_radians();
            let az_actual = actual.azimuth.radians();
            let alt_actual = actual.altitude.radians();
            let cosine = (alt.sin() * alt_actual.sin()
                + alt.cos() * alt_actual.cos() * (az - az_actual).cos())
            .clamp(-1.0, 1.0);
            let error = cosine.acos().to_degrees() * 3600.0;
            let index = match body {
                Body::Sun => 0,
                Body::Moon => 1,
                Body::Mercury => 2,
                Body::Venus => 3,
                Body::Mars => 4,
                Body::Jupiter => 5,
                Body::Saturn => 6,
            };
            if error > worst[index].0 {
                worst[index] = (error, format!("{} {} {}", fields[0], fields[1], fields[4]));
            }
            if body == Body::Moon {
                let phase_error = (actual.illuminated_fraction.unwrap()
                    - fields[8].parse::<f64>().unwrap())
                .abs();
                if phase_error > worst_phase.0 {
                    worst_phase = (phase_error, format!("{} {}", fields[0], fields[1]));
                }
                let semi_error = (actual.apparent_semidiameter.unwrap().value() * 3600.0
                    - fields[9].parse::<f64>().unwrap())
                .abs();
                if semi_error > worst_semi.0 {
                    worst_semi = (semi_error, format!("{} {}", fields[0], fields[1]));
                }
            }
            count += 1;
        }
        println!("Horizons {count} rows, worst arcsec by body: {worst:?}; phase {worst_phase:?}; semi arcsec {worst_semi:?}");
        assert_eq!(count, 280);
        for (index, (error, where_at)) in worst.iter().enumerate() {
            let limit = match index {
                0 => 10.0,
                1 => 30.0,
                _ => 60.0,
            };
            assert!(
                *error < limit,
                "body index {index}: {error} arcsec at {where_at}"
            );
        }
        assert!(worst_phase.0 < 0.005, "Moon phase error: {worst_phase:?}");
        assert!(
            worst_semi.0 < 1.0,
            "Moon semidiameter error: {worst_semi:?}"
        );
    }

    #[test]
    fn precession_and_lunar_parallax_are_load_bearing() {
        let utc = time::UtcDateTime::parse("2026-09-25T22:00:00Z").unwrap();
        let jd_tt = utc.jd_tt().unwrap();
        let precession = frames::precession_iau2006(jd_tt);
        let precession_arcmin = catalog::stars()
            .unwrap()
            .into_iter()
            .map(|star| {
                let (sr, cr) = star.ra_j2000_rad.sin_cos();
                let (sd, cd) = star.dec_j2000_rad.sin_cos();
                let j2000 = DVec3::new(cd * cr, cd * sr, sd);
                j2000.angle_between(precession * j2000).to_degrees() * 60.0
            })
            .fold(0.0_f64, f64::max);
        let moon = frames::ecliptic_j2000_to_true_equatorial(
            moon::geocentric_ecliptic_j2000(jd_tt).unwrap(),
            jd_tt,
        )
        .unwrap();
        let (dpsi, deps) = frames::nutation_iau2000b(jd_tt).unwrap();
        let theta = time::gast_utc(utc, dpsi, frames::mean_obliquity_iau2006(jd_tt) + deps)
            .unwrap()
            + 4.9_f64.to_radians();
        let observer = frames::observer_equatorial_km(52.37_f64.to_radians(), theta, 0.0);
        let parallax_arcmin = moon.angle_between(moon - observer).to_degrees() * 60.0;
        println!("precession ablation {precession_arcmin:.3} arcmin; lunar parallax ablation {parallax_arcmin:.3} arcmin");
        assert!(precession_arcmin > 20.0);
        assert!(parallax_arcmin > 30.0);
    }
}

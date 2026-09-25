//! Display mapping for SIDERA's fixed observation sky.
//!
//! Catalog V flux is relative to a V=0 star (`10^(-0.4 V)`). B−V is mapped
//! through Ballesteros' color-temperature approximation, then sampled as a
//! three-wavelength Planck emitter for a display RGB approximation. The
//! color-temperature relation is Ballesteros (2012), arXiv:1201.1809.
//! Twilight ramp is an explicitly visual civil-to-astronomical (-6° to -18°)
//! fade, not a radiative-transfer or sky-glow prediction. Lunar directional
//! strength uses Krisciunas & Schaefer (1991), Eq. 9 phase-magnitude fit,
//! PASP 103, 1033 (NASA ADS 1991PASP..103.1033K),
//! with that paper's Eq. 3 air mass and a declared V-band extinction setting;
//! it is a relative lighting term.

use anyhow::Result;

use super::{body_position, catalog, time::UtcDateTime, Body, BodyPosition};
use crate::geo::units::{Angle, Degree};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyInstance {
    pub direction_radius: [f32; 4],
    pub color_flux: [f32; 4],
    /// xyz: direction to Sun for lunar terminator; w: 0 star, 1 planet, 2 Moon.
    pub sun_kind: [f32; 4],
    /// xyz: IAU lunar north pole in renderer horizon axes (+x east, +y up,
    /// +z south); unused otherwise.
    pub moon_pole: [f32; 4],
}

pub struct Observation {
    pub sun: BodyPosition,
    pub moon: BodyPosition,
    pub moonlight_relative: f32,
    pub instances: Vec<SkyInstance>,
}

fn direction(azimuth: f64, altitude: f64) -> [f32; 3] {
    let (sa, ca) = azimuth.sin_cos();
    let (sh, ch) = altitude.sin_cos();
    [(ch * sa) as f32, sh as f32, (-ch * ca) as f32]
}

fn planck(wavelength_nm: f64, temperature_k: f64) -> f64 {
    let wavelength_m = wavelength_nm * 1e-9;
    let c2 = 0.014_387_768_77;
    1.0 / (wavelength_m.powi(5) * (c2 / (wavelength_m * temperature_k)).exp_m1())
}

fn star_rgb(bv: f32) -> [f32; 3] {
    let bv = f64::from(bv).clamp(-0.4, 2.0);
    let temperature = 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62));
    let samples = [650.0, 550.0, 460.0];
    let raw = samples.map(|lambda| planck(lambda, temperature) / planck(lambda, 6500.0));
    let maximum = raw.iter().copied().fold(0.0_f64, f64::max);
    raw.map(|value| (value / maximum) as f32)
}

pub fn moonlight_relative(illuminated_fraction: f64, altitude_rad: f64) -> f32 {
    if altitude_rad <= 0.0 {
        return 0.0;
    }
    let phase_deg = (2.0 * illuminated_fraction - 1.0)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    let magnitude = -12.73 + 0.026 * phase_deg + 4e-9 * phase_deg.powi(4);
    let flux = 10.0_f64.powf(-0.4 * (magnitude + 12.73));
    let zenith = std::f64::consts::FRAC_PI_2 - altitude_rad;
    // Krisciunas & Schaefer (1991), Eq. 3: X(Z)=(1-0.96 sin² Z)^(-1/2).
    let air_mass = (1.0 - 0.96 * zenith.sin().powi(2)).powf(-0.5);
    // KS91's V-band extinction parameter is site dependent. 0.172 mag per
    // air mass is a declared representative clear-sky display setting.
    (flux * 10.0_f64.powf(-0.4 * 0.172 * air_mass) * altitude_rad.sin()) as f32
}

pub fn prepare(
    utc: UtcDateTime,
    latitude: Angle<Degree>,
    longitude: Angle<Degree>,
) -> Result<Observation> {
    let sun = body_position(Body::Sun, utc, latitude, longitude)?;
    let moon = body_position(Body::Moon, utc, latitude, longitude)?;
    let sun_dir = direction(sun.azimuth.radians(), sun.refracted_altitude.radians());
    // Match the smoothstep ramp in sky.wgsl, evaluated on solar depression.
    let depression = (-sun.altitude.value() - 6.0) / 12.0;
    let t = depression.clamp(0.0, 1.0) as f32;
    let twilight = t * t * (3.0 - 2.0 * t);
    let mut instances = Vec::new();
    for star in catalog::visible_stars(utc, latitude.radians(), longitude.radians())? {
        let dir = direction(star.azimuth_rad, star.altitude_rad);
        let rgb = star_rgb(star.b_minus_v);
        instances.push(SkyInstance {
            direction_radius: [dir[0], dir[1], dir[2], 0.000_55],
            color_flux: [rgb[0], rgb[1], rgb[2], star.relative_irradiance * twilight],
            sun_kind: [sun_dir[0], sun_dir[1], sun_dir[2], 0.0],
            moon_pole: [0.0; 4],
        });
    }
    for (body, color, magnitude) in [
        (Body::Mercury, [0.84, 0.83, 0.78], -0.3),
        (Body::Venus, [1.0, 0.95, 0.85], -4.0),
        (Body::Mars, [1.0, 0.48, 0.31], 0.5),
        (Body::Jupiter, [0.98, 0.88, 0.72], -2.0),
        (Body::Saturn, [0.94, 0.84, 0.66], 0.8),
    ] {
        let p = body_position(body, utc, latitude, longitude)?;
        if p.refracted_altitude.value() > 0.0 {
            let dir = direction(p.azimuth.radians(), p.refracted_altitude.radians());
            instances.push(SkyInstance {
                direction_radius: [dir[0], dir[1], dir[2], 0.000_7],
                color_flux: [
                    color[0],
                    color[1],
                    color[2],
                    10.0_f32.powf(-0.4 * magnitude) * twilight,
                ],
                sun_kind: [sun_dir[0], sun_dir[1], sun_dir[2], 1.0],
                moon_pole: [0.0; 4],
            });
        }
    }
    if moon.refracted_altitude.value() > 0.0 {
        let dir = direction(moon.azimuth.radians(), moon.refracted_altitude.radians());
        let jd_tt = utc.jd_tt()?;
        let pole_true_equatorial = super::frames::precession_nutation_iau2006(jd_tt)?
            * super::moon::north_pole_icrf(jd_tt)?;
        let pole = super::frames::true_equatorial_direction_to_render_horizon(
            pole_true_equatorial,
            utc,
            latitude.radians(),
            longitude.radians(),
        )?;
        instances.push(SkyInstance {
            direction_radius: [
                dir[0],
                dir[1],
                dir[2],
                moon.apparent_semidiameter.expect("Moon radius").radians() as f32,
            ],
            color_flux: [1.0, 0.96, 0.86, twilight],
            sun_kind: [sun_dir[0], sun_dir[1], sun_dir[2], 2.0],
            moon_pole: [pole.x as f32, pole.y as f32, pole.z as f32, 0.0],
        });
    }
    Ok(Observation {
        sun,
        moon,
        moonlight_relative: moonlight_relative(
            moon.illuminated_fraction.expect("Moon phase"),
            moon.refracted_altitude.radians(),
        ),
        instances,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn night_observation_contains_catalog_and_moon() {
        let utc = UtcDateTime::parse("2026-09-25T22:00:00Z").unwrap();
        let observation = prepare(utc, Angle::new(52.37), Angle::new(4.9)).unwrap();
        assert!(observation.instances.len() > 4_000);
        assert!(observation
            .instances
            .iter()
            .any(|point| point.sun_kind[3] == 2.0));
        assert!(observation.moonlight_relative > 0.0);
    }

    #[test]
    fn ks91_phase_and_horizon_control_moonlight() {
        let altitude = 35.0_f64.to_radians();
        let full = moonlight_relative(1.0, altitude);
        let quarter = moonlight_relative(0.5, altitude);
        let new_moon = moonlight_relative(0.0, altitude);
        assert!(full > quarter && quarter > new_moon);
        assert_eq!(moonlight_relative(1.0, -altitude), 0.0);
    }
}

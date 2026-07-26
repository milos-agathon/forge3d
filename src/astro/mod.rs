//! CPU astronomical coordinates for the closed 2000–2050 SIDERA window.
//!
//! The analytic theories are deliberately bounded: callers get an error
//! outside the interval validated against the committed Horizons oracle.

pub mod catalog;
pub mod frames;
pub mod moon;
pub mod observation;
pub mod time;
pub mod vsop;

use crate::geo::units::{Angle, Degree};
use glam::{DMat3, DVec3};
use std::fmt;

const LIGHT_SPEED_AU_PER_DAY: f64 = 173.144_632_684_669_3;
const MOON_RADIUS_KM: f64 = 1_737.4;

pub const MIN_YEAR: i32 = 2000;
pub const MAX_YEAR: i32 = 2050;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    pub fn parse(name: &str) -> Result<Self, AstroError> {
        match name.to_ascii_lowercase().as_str() {
            "sun" => Ok(Self::Sun),
            "moon" => Ok(Self::Moon),
            "mercury" => Ok(Self::Mercury),
            "venus" => Ok(Self::Venus),
            "mars" => Ok(Self::Mars),
            "jupiter" => Ok(Self::Jupiter),
            "saturn" => Ok(Self::Saturn),
            _ => Err(AstroError::UnknownBody(name.to_string())),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Sun => "sun",
            Self::Moon => "moon",
            Self::Mercury => "mercury",
            Self::Venus => "venus",
            Self::Mars => "mars",
            Self::Jupiter => "jupiter",
            Self::Saturn => "saturn",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Observer {
    latitude: Angle<Degree>,
    longitude: Angle<Degree>,
    height_m: f64,
}

impl Observer {
    pub fn new(
        latitude: Angle<Degree>,
        longitude: Angle<Degree>,
        height_m: f64,
    ) -> Result<Self, AstroError> {
        if !latitude.value().is_finite()
            || !(-90.0..=90.0).contains(&latitude.value())
            || !longitude.value().is_finite()
            || !(-180.0..=180.0).contains(&longitude.value())
            || !height_m.is_finite()
        {
            return Err(AstroError::InvalidObserver);
        }
        Ok(Self {
            latitude,
            longitude,
            height_m,
        })
    }

    pub fn latitude(self) -> Angle<Degree> {
        self.latitude
    }

    pub fn longitude(self) -> Angle<Degree> {
        self.longitude
    }

    pub fn height_m(self) -> f64 {
        self.height_m
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BodyPosition {
    pub azimuth: Angle<Degree>,
    pub altitude: Angle<Degree>,
    pub distance_au: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct MoonPhase {
    pub illuminated_fraction: f64,
    pub phase_angle: Angle<Degree>,
    pub apparent_semidiameter_arcsec: f64,
}

pub fn body_position(
    body: Body,
    utc: time::UtcDateTime,
    observer: Observer,
    refraction: bool,
) -> Result<BodyPosition, AstroError> {
    let jd_tt = time::julian_day_tt(utc)?;
    let sidereal = frames::gast(time::julian_day_ut1(utc)?, jd_tt);
    let geocentric = apparent_geocentric_equatorial(body, jd_tt)?;
    let topocentric = frames::geocentric_to_topocentric(geocentric, observer, sidereal);
    let (azimuth, true_altitude) =
        frames::equatorial_to_horizontal(topocentric, observer, sidereal);
    Ok(BodyPosition {
        azimuth,
        altitude: if refraction {
            frames::refract_altitude(true_altitude)
        } else {
            true_altitude
        },
        distance_au: topocentric.length(),
    })
}

pub fn moon_phase(utc: time::UtcDateTime) -> Result<MoonPhase, AstroError> {
    let jd_tt = time::julian_day_tt(utc)?;
    let moon = moon::geocentric_ecliptic(jd_tt)?;
    let earth = vsop::heliocentric_ecliptic(vsop::VsopBody::Earth, jd_tt)?;
    let sun_from_earth = -earth;
    let moon_to_earth = -moon.ecliptic_of_date_au;
    let moon_to_sun = sun_from_earth - moon.ecliptic_of_date_au;
    let phase_angle = moon_to_earth.angle_between(moon_to_sun);
    Ok(MoonPhase {
        illuminated_fraction: (1.0 + phase_angle.cos()) * 0.5,
        phase_angle: Angle::new(phase_angle.to_degrees()),
        apparent_semidiameter_arcsec: (MOON_RADIUS_KM / moon.distance_km).asin().to_degrees()
            * 3_600.0,
    })
}

fn apparent_geocentric_equatorial(body: Body, jd_tt: f64) -> Result<DVec3, AstroError> {
    let earth = vsop::heliocentric_ecliptic(vsop::VsopBody::Earth, jd_tt)?;
    let ecliptic = match body {
        Body::Sun => -earth,
        Body::Moon => moon::geocentric_ecliptic(jd_tt)?.ecliptic_of_date_au,
        _ => {
            let planet = match body {
                Body::Mercury => vsop::VsopBody::Mercury,
                Body::Venus => vsop::VsopBody::Venus,
                Body::Mars => vsop::VsopBody::Mars,
                Body::Jupiter => vsop::VsopBody::Jupiter,
                Body::Saturn => vsop::VsopBody::Saturn,
                Body::Sun | Body::Moon => unreachable!(),
            };
            let mut light_time = 0.0;
            let mut geocentric = DVec3::ZERO;
            for _ in 0..3 {
                geocentric = vsop::heliocentric_ecliptic(planet, jd_tt - light_time)? - earth;
                light_time = geocentric.length() / LIGHT_SPEED_AU_PER_DAY;
            }
            geocentric
        }
    };
    let distance = ecliptic.length();
    let mean_obliquity = frames::mean_obliquity(jd_tt);
    let mean_equatorial = DMat3::from_rotation_x(mean_obliquity) * ecliptic;
    let true_equatorial = frames::nutate_mean_to_true(mean_equatorial, jd_tt);
    let velocity_ecliptic = vsop::earth_velocity(jd_tt)?;
    let velocity_equatorial = DMat3::from_rotation_x(mean_obliquity) * velocity_ecliptic;
    Ok(frames::annual_aberration(true_equatorial, velocity_equatorial) * distance)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AstroError {
    InvalidDateTime,
    OutsideValidityWindow,
    InvalidObserver,
    UnknownBody(String),
    InvalidData(&'static str),
}

impl fmt::Display for AstroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDateTime => f.write_str("invalid UTC date/time"),
            Self::OutsideValidityWindow => {
                write!(f, "ephemeris date must be within {MIN_YEAR}–{MAX_YEAR}")
            }
            Self::InvalidObserver => f.write_str("invalid WGS84 observer coordinates"),
            Self::UnknownBody(name) => write!(f, "unsupported astronomical body: {name}"),
            Self::InvalidData(name) => write!(f, "invalid committed astronomical data: {name}"),
        }
    }
}

impl std::error::Error for AstroError {}

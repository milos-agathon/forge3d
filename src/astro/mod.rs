//! CPU astronomical coordinates for the closed 2000–2050 SIDERA window.
//!
//! The analytic theories are deliberately bounded: callers get an error
//! outside the interval validated against the committed Horizons oracle.

pub mod frames;
pub mod time;

use crate::geo::units::{Angle, Degree};
use std::fmt;

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

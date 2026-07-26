//! UTC, TT, and Julian-date handling.
//!
//! ΔT follows the Espenak–Meeus NASA polynomial fit committed in
//! `assets/astro/delta_t_fit.dat`. The stored decimal coefficients reproduce
//! the cited polynomial to much better than one second over 2000–2050; this is
//! a fit residual, not a claim that future Earth rotation is predictable to
//! one second.

use super::{AstroError, MAX_YEAR, MIN_YEAR};
use std::sync::OnceLock;

const DELTA_T_DATA: &str = include_str!("../../assets/astro/delta_t_fit.dat");
const LEAP_SECOND_DATA: &str = include_str!("../../assets/astro/leap_seconds.dat");

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UtcDateTime {
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
}

impl UtcDateTime {
    pub fn new(
        year: i32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: f64,
    ) -> Result<Self, AstroError> {
        if !(MIN_YEAR..=MAX_YEAR).contains(&year) {
            return Err(AstroError::OutsideValidityWindow);
        }
        let max_day = days_in_month(year, month).ok_or(AstroError::InvalidDateTime)?;
        if day == 0
            || day > max_day
            || hour > 23
            || minute > 59
            || !second.is_finite()
            || (!(0.0..60.0).contains(&second)
                && !((60.0..61.0).contains(&second)
                    && hour == 23
                    && minute == 59
                    && is_positive_leap_second_day(year, month, day)))
        {
            return Err(AstroError::InvalidDateTime);
        }
        Ok(Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
        })
    }

    pub fn decimal_year(self) -> f64 {
        let elapsed_days: u32 = (1..self.month)
            .map(|month| u32::from(days_in_month(self.year, month).unwrap_or(0)))
            .sum::<u32>()
            + u32::from(self.day - 1);
        let day_fraction =
            (f64::from(self.hour) + (f64::from(self.minute) + self.second / 60.0) / 60.0) / 24.0;
        self.year as f64
            + (elapsed_days as f64 + day_fraction)
                / if is_leap_year(self.year) {
                    366.0
                } else {
                    365.0
                }
    }

    pub fn components(self) -> (i32, u8, u8, u8, u8, f64) {
        (
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
        )
    }
}

pub fn julian_day_utc(utc: UtcDateTime) -> f64 {
    julian_day_midnight(utc)
        + elapsed_utc_seconds(utc)
            / if is_positive_leap_second_day(utc.year, utc.month, utc.day) {
                86_401.0
            } else {
                86_400.0
            }
}

fn julian_day_midnight(utc: UtcDateTime) -> f64 {
    let mut year = utc.year;
    let mut month = i32::from(utc.month);
    if month <= 2 {
        year -= 1;
        month += 12;
    }
    let a = year.div_euclid(100);
    let b = 2 - a + a.div_euclid(4);
    (365.25 * f64::from(year + 4716)).floor()
        + (30.6001 * f64::from(month + 1)).floor()
        + f64::from(utc.day)
        + f64::from(b)
        - 1524.5
}

pub fn julian_day_tt(utc: UtcDateTime) -> Result<f64, AstroError> {
    Ok(julian_day_midnight(utc)
        + (elapsed_utc_seconds(utc) + tai_minus_utc_seconds(utc)? + 32.184) / 86_400.0)
}

pub fn julian_day_ut1(utc: UtcDateTime) -> Result<f64, AstroError> {
    Ok(julian_day_tt(utc)? - delta_t_seconds(utc)? / 86_400.0)
}

pub fn tai_minus_utc_seconds(utc: UtcDateTime) -> Result<f64, AstroError> {
    leap_seconds()?
        .iter()
        .rev()
        .find(|entry| (utc.year, utc.month, utc.day) >= (entry.year, entry.month, entry.day))
        .map(|entry| entry.offset)
        .ok_or(AstroError::InvalidData("leap_seconds.dat"))
}

fn elapsed_utc_seconds(utc: UtcDateTime) -> f64 {
    f64::from(utc.hour) * 3_600.0 + f64::from(utc.minute) * 60.0 + utc.second
}

pub fn delta_t_seconds(utc: UtcDateTime) -> Result<f64, AstroError> {
    let y = utc.decimal_year();
    let fits = delta_t_fit()?;
    let fit = fits
        .iter()
        .enumerate()
        .find(|(index, fit)| {
            y >= fit.start && (y < fit.end || (*index == fits.len() - 1 && y <= fit.end))
        })
        .map(|(_, fit)| fit)
        .ok_or(AstroError::InvalidData("delta_t_fit.dat"))?;
    let t = y - fit.origin;
    Ok(fit
        .coefficients
        .iter()
        .rev()
        .fold(0.0, |value, coefficient| value * t + coefficient))
}

#[derive(Debug)]
struct PolynomialFit {
    start: f64,
    end: f64,
    origin: f64,
    coefficients: Vec<f64>,
}

#[derive(Debug)]
struct LeapSecond {
    year: i32,
    month: u8,
    day: u8,
    offset: f64,
}

fn leap_seconds() -> Result<&'static [LeapSecond], AstroError> {
    static LEAP_SECONDS: OnceLock<Result<Vec<LeapSecond>, AstroError>> = OnceLock::new();
    LEAP_SECONDS
        .get_or_init(|| {
            LEAP_SECOND_DATA
                .lines()
                .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
                .map(|line| {
                    let mut values = line.split_whitespace();
                    let invalid = || AstroError::InvalidData("leap_seconds.dat");
                    let entry = LeapSecond {
                        year: values
                            .next()
                            .ok_or_else(invalid)?
                            .parse()
                            .map_err(|_| invalid())?,
                        month: values
                            .next()
                            .ok_or_else(invalid)?
                            .parse()
                            .map_err(|_| invalid())?,
                        day: values
                            .next()
                            .ok_or_else(invalid)?
                            .parse()
                            .map_err(|_| invalid())?,
                        offset: values
                            .next()
                            .ok_or_else(invalid)?
                            .parse()
                            .map_err(|_| invalid())?,
                    };
                    if values.next().is_some() {
                        return Err(invalid());
                    }
                    Ok(entry)
                })
                .collect()
        })
        .as_deref()
        .map_err(Clone::clone)
}

fn is_positive_leap_second_day(year: i32, month: u8, day: u8) -> bool {
    const DAYS: &[(i32, u8, u8)] = &[
        (2005, 12, 31),
        (2008, 12, 31),
        (2012, 6, 30),
        (2015, 6, 30),
        (2016, 12, 31),
    ];
    DAYS.contains(&(year, month, day))
}

fn delta_t_fit() -> Result<&'static [PolynomialFit], AstroError> {
    static FIT: OnceLock<Result<Vec<PolynomialFit>, AstroError>> = OnceLock::new();
    FIT.get_or_init(|| {
        DELTA_T_DATA
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
            .map(|line| {
                let values: Vec<f64> = line
                    .split_whitespace()
                    .map(str::parse)
                    .collect::<Result<_, _>>()
                    .map_err(|_| AstroError::InvalidData("delta_t_fit.dat"))?;
                if values.len() < 4 {
                    return Err(AstroError::InvalidData("delta_t_fit.dat"));
                }
                Ok(PolynomialFit {
                    start: values[0],
                    end: values[1],
                    origin: values[2],
                    coefficients: values[3..].to_vec(),
                })
            })
            .collect()
    })
    .as_deref()
    .map_err(Clone::clone)
}

const fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

const fn days_in_month(year: i32, month: u8) -> Option<u8> {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => Some(31),
        4 | 6 | 9 | 11 => Some(30),
        2 if is_leap_year(year) => Some(29),
        2 => Some(28),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_t_fit_is_continuous_at_2005_boundary() {
        let before = UtcDateTime::new(2004, 12, 31, 23, 59, 59.0).unwrap();
        let after = UtcDateTime::new(2005, 1, 1, 0, 0, 0.0).unwrap();
        assert!((delta_t_seconds(after).unwrap() - delta_t_seconds(before).unwrap()).abs() < 0.1);
    }
}

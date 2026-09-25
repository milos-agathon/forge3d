//! UTC, TT and UT1 conversion and IAU 2006 mean sidereal time.
//!
//! `delta_t.bin` is a monthly cubic fit to JPL Horizons TDB-UT, sampled every
//! five days over 2000–2050. The largest sampled fit residual is 1.1 µs.
//! TDB-TT itself varies by about 2 ms, which is inside the stated 1 s model
//! residual. Historical leap seconds are explicit; future UTC assumes no new
//! leap second after the last published entry and is a declared limitation.

use anyhow::{bail, ensure, Result};

const DELTA_T: &[u8] = include_bytes!("../../data/sidera/delta_t.bin");
const JD_J2000: f64 = 2_451_545.0;
const ARCSEC_TO_RAD: f64 = std::f64::consts::PI / 648_000.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UtcDateTime {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: f64,
}

impl UtcDateTime {
    pub fn parse(value: &str) -> Result<Self> {
        let value = value
            .strip_suffix('Z')
            .ok_or_else(|| anyhow::anyhow!("UTC datetime must end in Z"))?;
        let (date, clock) = value
            .split_once('T')
            .ok_or_else(|| anyhow::anyhow!("UTC datetime must contain T"))?;
        let mut date = date.split('-');
        let year: i32 = date
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing year"))?
            .parse()?;
        let month: u32 = date
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing month"))?
            .parse()?;
        let day: u32 = date
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing day"))?
            .parse()?;
        ensure!(date.next().is_none(), "extra UTC date field");
        let mut clock = clock.split(':');
        let hour: u32 = clock
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing hour"))?
            .parse()?;
        let minute: u32 = clock
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing minute"))?
            .parse()?;
        let second: f64 = clock
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing second"))?
            .parse()?;
        ensure!(clock.next().is_none(), "extra UTC clock field");
        let result = Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            (2000..=2050).contains(&self.year),
            "SIDERA supports UTC years 2000–2050"
        );
        ensure!((1..=12).contains(&self.month), "invalid UTC month");
        let days = days_in_month(self.year, self.month);
        ensure!((1..=days).contains(&self.day), "invalid UTC day");
        ensure!(
            self.hour < 24
                && self.minute < 60
                && self.second.is_finite()
                && (0.0..60.0).contains(&self.second),
            "invalid UTC clock time"
        );
        Ok(())
    }

    pub fn jd_utc(&self) -> Result<f64> {
        self.validate()?;
        Ok(julian_day(self.year, self.month, self.day)
            + (self.hour as f64 * 3600.0 + self.minute as f64 * 60.0 + self.second) / 86_400.0)
    }

    pub fn jd_tt(&self) -> Result<f64> {
        Ok(self.jd_utc()? + (tai_minus_utc(*self) + 32.184) / 86_400.0)
    }

    pub fn delta_t_seconds(&self) -> Result<f64> {
        self.validate()?;
        ensure!(&DELTA_T[0..4] == b"DT50", "invalid SIDERA delta-T asset");
        let start_year = u16::from_le_bytes([DELTA_T[4], DELTA_T[5]]) as i32;
        let count = u16::from_le_bytes([DELTA_T[6], DELTA_T[7]]) as usize;
        ensure!(
            start_year == 2000 && count == 612 && DELTA_T.len() == 8 + count * 32,
            "invalid SIDERA delta-T coefficient count"
        );
        let index = ((self.year - start_year) * 12 + self.month as i32 - 1) as usize;
        let begin = julian_day(self.year, self.month, 1);
        let end = if self.month == 12 {
            julian_day(self.year + 1, 1, 1)
        } else {
            julian_day(self.year, self.month + 1, 1)
        };
        let x = (self.jd_utc()? - begin) / (end - begin);
        let offset = 8 + index * 32;
        let coefficient = |n: usize| {
            f64::from_le_bytes(
                DELTA_T[offset + n * 8..offset + (n + 1) * 8]
                    .try_into()
                    .expect("fixed coefficient width"),
            )
        };
        Ok(coefficient(0) + x * (coefficient(1) + x * (coefficient(2) + x * coefficient(3))))
    }

    pub fn jd_ut1(&self) -> Result<f64> {
        Ok(self.jd_tt()? - self.delta_t_seconds()? / 86_400.0)
    }
}

pub fn julian_day(year: i32, month: u32, day: u32) -> f64 {
    let a = (14 - month as i32) / 12;
    let y = year + 4800 - a;
    let m = month as i32 + 12 * a - 3;
    (day as i32 + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045) as f64 - 0.5
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        4 | 6 | 9 | 11 => 30,
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        _ => 31,
    }
}

fn tai_minus_utc(utc: UtcDateTime) -> f64 {
    // IERS leap-second history. No published future event is invented.
    const LEAPS: &[(i32, u32, u32)] = &[
        (2006, 1, 1),
        (2009, 1, 1),
        (2012, 7, 1),
        (2015, 7, 1),
        (2017, 1, 1),
    ];
    32.0 + LEAPS
        .iter()
        .filter(|&&(y, m, d)| (utc.year, utc.month, utc.day) >= (y, m, d))
        .count() as f64
}

pub fn gmst_iau2006(jd_ut1: f64, jd_tt: f64) -> f64 {
    // IAU 2006 ERA plus P03 precession polynomial; SOFA/ERFA gmst06.
    let t = (jd_tt - JD_J2000) / 36_525.0;
    let era = std::f64::consts::TAU
        * (0.779_057_273_264_0 + 1.002_737_811_911_354_6 * (jd_ut1 - JD_J2000));
    let correction_arcsec = 0.014506
        + t * (4612.156534
            + t * (1.3915817 + t * (-0.00000044 + t * (-0.000029956 - 0.0000000368 * t))));
    (era + correction_arcsec * ARCSEC_TO_RAD).rem_euclid(std::f64::consts::TAU)
}

pub fn gmst_utc(utc: UtcDateTime) -> Result<f64> {
    Ok(gmst_iau2006(utc.jd_ut1()?, utc.jd_tt()?))
}

pub fn gast_utc(utc: UtcDateTime, nutation_longitude: f64, true_obliquity: f64) -> Result<f64> {
    if !nutation_longitude.is_finite() || !true_obliquity.is_finite() {
        bail!("non-finite nutation input")
    }
    Ok((gmst_utc(utc)? + nutation_longitude * true_obliquity.cos())
        .rem_euclid(std::f64::consts::TAU))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn j2000_and_window() {
        let epoch = UtcDateTime::parse("2000-01-01T12:00:00Z").unwrap();
        assert_eq!(epoch.jd_utc().unwrap(), JD_J2000);
        assert!(UtcDateTime::parse("1999-12-31T23:59:59Z").is_err());
        assert!(UtcDateTime::parse("2051-01-01T00:00:00Z").is_err());
        assert!(UtcDateTime::parse("2024-02-30T00:00:00Z").is_err());
    }

    #[test]
    fn fitted_delta_t_is_bounded() {
        let epoch = UtcDateTime::parse("2000-01-01T00:00:00Z").unwrap();
        assert!((epoch.delta_t_seconds().unwrap() - 64.183889).abs() < 0.1);
    }

    #[test]
    fn gmst_matches_erfa_iau_2006_reference_across_window() {
        // Independent ERFA gmst06 outputs, radians, at five representative
        // UT1/TT pairs spanning the supported interval.
        let cases = [
            (
                2_451_544.5,
                2_451_544.500_742_870_4,
                1.744_767_234_131_470_4,
            ),
            (
                2_456_018.5,
                2_456_018.500_766_018_4,
                3.311_834_022_343_855_4,
            ),
            (
                2_461_309.416_666_666_5,
                2_461_309.417_467_407,
                5.842_179_008_428_651,
            ),
            (
                2_465_502.916_666_666_5,
                2_465_502.917_467_407,
                5.725_455_760_970_902,
            ),
            (2_470_171.5, 2_470_171.500_800_741, 1.738_720_598_061_619_6),
        ];
        for (ut1, tt, reference) in cases {
            let seconds =
                (gmst_iau2006(ut1, tt) - reference).abs() * 86_400.0 / std::f64::consts::TAU;
            assert!(seconds < 0.1, "GMST error {seconds} seconds at UT1 {ut1}");
        }
    }
}

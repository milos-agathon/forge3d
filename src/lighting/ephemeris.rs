// src/lighting/ephemeris.rs
// P0.3/M2: Deterministic sun ephemeris calculation
// Computes solar azimuth and elevation from geographic coordinates and UTC datetime
// Algorithm based on NOAA Solar Calculator (pure math, no external dependencies)


/// Result of sun position calculation
#[derive(Debug, Clone, Copy)]
pub struct SunPosition {
    /// Solar azimuth in degrees (0 = North, 90 = East, 180 = South, 270 = West)
    pub azimuth: f64,
    /// Solar elevation/altitude in degrees (-90 to +90, negative = below horizon)
    pub elevation: f64,
}

impl SunPosition {
    /// Convert azimuth/elevation to a normalized direction vector (Y-up coordinate system)
    /// Returns [x, y, z] where y is up
    pub fn to_direction(&self) -> [f32; 3] {
        let az_rad = self.azimuth.to_radians();
        let el_rad = self.elevation.to_radians();
        
        let cos_el = el_rad.cos();
        let x = -az_rad.sin() * cos_el;
        let y = el_rad.sin();
        let z = -az_rad.cos() * cos_el;
        
        [x as f32, y as f32, z as f32]
    }
}

/// Calculate Julian Day from year, month, day, and fractional hour (UTC)
fn julian_day(year: i32, month: u32, day: u32, hour: f64) -> f64 {
    let (y, m) = if month <= 2 {
        (year - 1, month + 12)
    } else {
        (year, month)
    };
    
    let a = (y as f64 / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    
    let jd = (365.25 * (y as f64 + 4716.0)).floor()
        + (30.6001 * (m as f64 + 1.0)).floor()
        + day as f64
        + hour / 24.0
        + b
        - 1524.5;
    
    jd
}

/// Calculate Julian Century from Julian Day
fn julian_century(jd: f64) -> f64 {
    (jd - 2451545.0) / 36525.0
}

/// Calculate geometric mean longitude of the sun (degrees)
fn geom_mean_long_sun(t: f64) -> f64 {
    let l0 = 280.46646 + t * (36000.76983 + 0.0003032 * t);
    l0 % 360.0
}

/// Calculate geometric mean anomaly of the sun (degrees)
fn geom_mean_anomaly_sun(t: f64) -> f64 {
    357.52911 + t * (35999.05029 - 0.0001537 * t)
}

/// Calculate eccentricity of Earth's orbit
fn eccent_earth_orbit(t: f64) -> f64 {
    0.016708634 - t * (0.000042037 + 0.0000001267 * t)
}

/// Calculate sun equation of center (degrees)
fn sun_eq_of_center(t: f64) -> f64 {
    let m = geom_mean_anomaly_sun(t);
    let m_rad = m.to_radians();
    
    m_rad.sin() * (1.914602 - t * (0.004817 + 0.000014 * t))
        + (2.0 * m_rad).sin() * (0.019993 - 0.000101 * t)
        + (3.0 * m_rad).sin() * 0.000289
}

/// Calculate sun true longitude (degrees)
fn sun_true_long(t: f64) -> f64 {
    geom_mean_long_sun(t) + sun_eq_of_center(t)
}

/// Calculate sun apparent longitude (degrees)
fn sun_apparent_long(t: f64) -> f64 {
    let o = sun_true_long(t);
    o - 0.00569 - 0.00478 * (125.04 - 1934.136 * t).to_radians().sin()
}

/// Calculate mean obliquity of the ecliptic (degrees)
fn mean_obliq_ecliptic(t: f64) -> f64 {
    23.0 + (26.0 + (21.448 - t * (46.8150 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
}

/// Calculate obliquity correction (degrees)
fn obliq_corr(t: f64) -> f64 {
    let e0 = mean_obliq_ecliptic(t);
    e0 + 0.00256 * (125.04 - 1934.136 * t).to_radians().cos()
}

/// Calculate sun declination (degrees)
fn sun_declination(t: f64) -> f64 {
    let e = obliq_corr(t).to_radians();
    let lambda = sun_apparent_long(t).to_radians();
    
    (e.sin() * lambda.sin()).asin().to_degrees()
}

/// Calculate equation of time (minutes)
fn eq_of_time(t: f64) -> f64 {
    let e = obliq_corr(t).to_radians();
    let l0 = geom_mean_long_sun(t).to_radians();
    let ecc = eccent_earth_orbit(t);
    let m = geom_mean_anomaly_sun(t).to_radians();
    
    let y = (e / 2.0).tan().powi(2);
    
    let sin2l0 = (2.0 * l0).sin();
    let sinm = m.sin();
    let cos2l0 = (2.0 * l0).cos();
    let sin4l0 = (4.0 * l0).sin();
    let sin2m = (2.0 * m).sin();
    
    let etime = y * sin2l0 - 2.0 * ecc * sinm + 4.0 * ecc * y * sinm * cos2l0
        - 0.5 * y * y * sin4l0
        - 1.25 * ecc * ecc * sin2m;
    
    etime.to_degrees() * 4.0  // Convert to minutes
}

/// Calculate hour angle (degrees) for a given solar time
fn hour_angle(solar_time_minutes: f64) -> f64 {
    (solar_time_minutes / 4.0) - 180.0
}

/// Calculate solar zenith angle (degrees)
fn solar_zenith(lat: f64, decl: f64, ha: f64) -> f64 {
    let lat_rad = lat.to_radians();
    let decl_rad = decl.to_radians();
    let ha_rad = ha.to_radians();
    
    let cos_zenith = lat_rad.sin() * decl_rad.sin() + lat_rad.cos() * decl_rad.cos() * ha_rad.cos();
    cos_zenith.clamp(-1.0, 1.0).acos().to_degrees()
}

/// Calculate solar azimuth (degrees from north, clockwise)
fn solar_azimuth(lat: f64, zenith: f64, decl: f64, ha: f64) -> f64 {
    let lat_rad = lat.to_radians();
    let zenith_rad = zenith.to_radians();
    let decl_rad = decl.to_radians();
    
    let cos_az_num = lat_rad.sin() * zenith_rad.cos() - decl_rad.sin();
    let cos_az_den = lat_rad.cos() * zenith_rad.sin();
    
    let cos_az = if cos_az_den.abs() < 1e-10 {
        if cos_az_num >= 0.0 { 1.0 } else { -1.0 }
    } else {
        (cos_az_num / cos_az_den).clamp(-1.0, 1.0)
    };
    
    let az = cos_az.acos().to_degrees();
    
    if ha > 0.0 {
        (az + 180.0) % 360.0
    } else {
        (540.0 - az) % 360.0
    }
}

/// Calculate sun position for a given location and UTC datetime
///
/// # Arguments
/// * `latitude` - Observer latitude in degrees (-90 to 90, positive = North)
/// * `longitude` - Observer longitude in degrees (-180 to 180, positive = East)
/// * `year` - UTC year
/// * `month` - UTC month (1-12)
/// * `day` - UTC day of month (1-31)
/// * `hour` - UTC hour (0-23)
/// * `minute` - UTC minute (0-59)
/// * `second` - UTC second (0-59)
///
/// # Returns
/// `SunPosition` with azimuth (degrees from north) and elevation (degrees from horizon)
pub fn sun_position(
    latitude: f64,
    longitude: f64,
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> SunPosition {
    let lat = latitude.clamp(-90.0, 90.0);
    let lon = longitude.clamp(-180.0, 180.0);
    
    let fractional_hour = hour as f64 + minute as f64 / 60.0 + second as f64 / 3600.0;
    
    let jd = julian_day(year, month, day, fractional_hour);
    let t = julian_century(jd);
    
    let eqtime = eq_of_time(t);
    let decl = sun_declination(t);
    
    let time_offset = eqtime + 4.0 * lon;
    let true_solar_time = fractional_hour * 60.0 + time_offset;
    let true_solar_time = ((true_solar_time % 1440.0) + 1440.0) % 1440.0;
    
    let ha = hour_angle(true_solar_time);
    let zenith = solar_zenith(lat, decl, ha);
    let azimuth = solar_azimuth(lat, zenith, decl, ha);
    let elevation = 90.0 - zenith;
    
    SunPosition { azimuth, elevation }
}

/// Convenience function to calculate sun position from ISO 8601 datetime string
///
/// # Arguments
/// * `latitude` - Observer latitude in degrees
/// * `longitude` - Observer longitude in degrees
/// * `datetime_utc` - ISO 8601 datetime string (e.g., "2024-06-21T12:00:00")
///
/// # Returns
/// `Result<SunPosition, String>` with position or parsing error
pub fn sun_position_from_iso(
    latitude: f64,
    longitude: f64,
    datetime_utc: &str,
) -> Result<SunPosition, String> {
    let parts: Vec<&str> = datetime_utc.split('T').collect();
    if parts.len() != 2 {
        return Err(format!("Invalid datetime format: expected 'YYYY-MM-DDTHH:MM:SS', got '{}'", datetime_utc));
    }
    
    let date_parts: Vec<&str> = parts[0].split('-').collect();
    if date_parts.len() != 3 {
        return Err(format!("Invalid date format: expected 'YYYY-MM-DD', got '{}'", parts[0]));
    }
    
    let time_str = parts[1].trim_end_matches('Z');
    let time_parts: Vec<&str> = time_str.split(':').collect();
    if time_parts.len() < 2 {
        return Err(format!("Invalid time format: expected 'HH:MM' or 'HH:MM:SS', got '{}'", time_str));
    }
    
    let year: i32 = date_parts[0].parse().map_err(|_| format!("Invalid year: {}", date_parts[0]))?;
    let month: u32 = date_parts[1].parse().map_err(|_| format!("Invalid month: {}", date_parts[1]))?;
    let day: u32 = date_parts[2].parse().map_err(|_| format!("Invalid day: {}", date_parts[2]))?;
    
    let hour: u32 = time_parts[0].parse().map_err(|_| format!("Invalid hour: {}", time_parts[0]))?;
    let minute: u32 = time_parts[1].parse().map_err(|_| format!("Invalid minute: {}", time_parts[1]))?;
    let second: u32 = if time_parts.len() > 2 {
        time_parts[2].split('.').next().unwrap_or("0").parse().unwrap_or(0)
    } else {
        0
    };
    
    if month < 1 || month > 12 {
        return Err(format!("Month must be 1-12, got {}", month));
    }
    if day < 1 || day > 31 {
        return Err(format!("Day must be 1-31, got {}", day));
    }
    if hour > 23 {
        return Err(format!("Hour must be 0-23, got {}", hour));
    }
    if minute > 59 {
        return Err(format!("Minute must be 0-59, got {}", minute));
    }
    
    Ok(sun_position(latitude, longitude, year, month, day, hour, minute, second))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_summer_solstice_noon_equator() {
        let pos = sun_position(0.0, 0.0, 2024, 6, 21, 12, 0, 0);
        assert!(pos.elevation > 60.0, "Sun should be high at equator noon on solstice");
    }
    
    #[test]
    fn test_winter_north_pole() {
        let pos = sun_position(90.0, 0.0, 2024, 12, 21, 12, 0, 0);
        assert!(pos.elevation < 0.0, "Sun should be below horizon at North Pole in December");
    }
    
    #[test]
    fn test_iso_parsing() {
        let result = sun_position_from_iso(45.0, -122.0, "2024-06-21T12:00:00");
        assert!(result.is_ok());
        
        let result = sun_position_from_iso(45.0, -122.0, "invalid");
        assert!(result.is_err());
    }
}

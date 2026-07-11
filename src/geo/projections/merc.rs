// src/geo/projections/merc.rs
// Mercator Variant A (EPSG method 9804) and spherical Web Mercator
// ("Popular Visualisation Pseudo Mercator", EPSG method 1024).
// RELEVANT FILES: src/geo/projections/mod.rs, src/gis/crs.rs

use super::{lat_from_epsg_t, Ellipsoid, ProjError, ProjResult};

const WEB_MERCATOR_RADIUS: f64 = 6_378_137.0;
/// Latitude bound preserved from the pre-MENSURA web-mercator code so the
/// EPSG:1024 behaviour forge3d already relies on stays compatible.
pub const WEB_MERCATOR_MAX_LAT: f64 = 85.051_128_78;

/// Mercator Variant A (EPSG 9804): scale factor at the natural origin on the
/// equator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MercatorA {
    pub ellipsoid: Ellipsoid,
    /// Longitude of natural origin, degrees.
    pub lon0_deg: f64,
    /// Scale factor at natural origin.
    pub k0: f64,
    pub false_easting: f64,
    pub false_northing: f64,
}

impl MercatorA {
    pub fn forward(&self, lon_deg: f64, lat_deg: f64) -> ProjResult<(f64, f64)> {
        if lat_deg.abs() >= 90.0 || !lon_deg.is_finite() {
            return Err(ProjError::Domain(format!(
                "mercator input out of range: lon={lon_deg}, lat={lat_deg}"
            )));
        }
        let e = self.ellipsoid.e();
        let lat = lat_deg.to_radians();
        let es = e * lat.sin();
        let iso = (core::f64::consts::FRAC_PI_4 + lat / 2.0).tan().ln()
            + (e / 2.0) * ((1.0 - es) / (1.0 + es)).ln();
        Ok((
            self.false_easting
                + self.ellipsoid.a * self.k0 * (lon_deg - self.lon0_deg).to_radians(),
            self.false_northing + self.ellipsoid.a * self.k0 * iso,
        ))
    }

    pub fn inverse(&self, easting: f64, northing: f64) -> ProjResult<(f64, f64)> {
        let e = self.ellipsoid.e();
        let t = (-(northing - self.false_northing) / (self.ellipsoid.a * self.k0)).exp();
        let lat = lat_from_epsg_t(t, e)?;
        let lon = self.lon0_deg
            + ((easting - self.false_easting) / (self.ellipsoid.a * self.k0)).to_degrees();
        Ok((lon, lat.to_degrees()))
    }
}

/// Spherical Web Mercator forward (EPSG 1024 / CRS 3857).
pub fn web_mercator_forward(lon_deg: f64, lat_deg: f64) -> ProjResult<(f64, f64)> {
    if !(-WEB_MERCATOR_MAX_LAT..=WEB_MERCATOR_MAX_LAT).contains(&lat_deg) {
        return Err(ProjError::Domain(format!(
            "invalid_latitude_range: Web Mercator supports latitudes within +/-{WEB_MERCATOR_MAX_LAT}"
        )));
    }
    let x = WEB_MERCATOR_RADIUS * lon_deg.to_radians();
    let y = WEB_MERCATOR_RADIUS
        * (core::f64::consts::FRAC_PI_4 + lat_deg.to_radians() / 2.0)
            .tan()
            .ln();
    Ok((x, y))
}

/// Spherical Web Mercator inverse.
pub fn web_mercator_inverse(x: f64, y: f64) -> ProjResult<(f64, f64)> {
    if !x.is_finite() || !y.is_finite() {
        return Err(ProjError::Domain(
            "web mercator inverse input must be finite".to_string(),
        ));
    }
    let lon = (x / WEB_MERCATOR_RADIUS).to_degrees();
    let lat =
        (2.0 * (y / WEB_MERCATOR_RADIUS).exp().atan() - core::f64::consts::FRAC_PI_2).to_degrees();
    Ok((lon, lat))
}

#[cfg(test)]
mod tests {
    use super::super::{dms, BESSEL_1841};
    use super::*;

    /// EPSG Guidance Note 7-2, method 9804 worked example:
    /// Makassar / NEIEZ (Bessel 1841).
    fn neiez() -> MercatorA {
        MercatorA {
            ellipsoid: BESSEL_1841,
            lon0_deg: 110.0,
            k0: 0.997,
            false_easting: 3_900_000.0,
            false_northing: 900_000.0,
        }
    }

    #[test]
    fn epsg_g7_2_worked_example_forward() {
        let (e, n) = neiez().forward(120.0, dms(-3.0, 0.0, 0.0)).unwrap();
        assert!(
            (e - 5_009_726.583278828).abs() < 1e-3,
            "easting residual {} m",
            (e - 5_009_726.583278828).abs()
        );
        assert!(
            (n - 569_150.8186138709).abs() < 1e-3,
            "northing residual {} m",
            (n - 569_150.8186138709).abs()
        );
    }

    #[test]
    fn epsg_g7_2_worked_example_inverse() {
        let (lon, lat) = neiez()
            .inverse(5_009_726.583278828, 569_150.8186138709)
            .unwrap();
        assert!(
            (lon - 120.0).abs() < 9e-9,
            "lon residual {} deg",
            (lon - 120.0).abs()
        );
        assert!(
            (lat + 3.0).abs() < 9e-9,
            "lat residual {} deg",
            (lat + 3.0).abs()
        );
    }

    #[test]
    fn mercator_forward_inverse_roundtrip_is_exact() {
        // Machine-precision closure pins forward/inverse consistency to far
        // below 1 mm independently of the reference's print precision.
        let p = neiez();
        let mut worst = 0.0f64;
        for lat in [-70.0, -3.0, 0.0, 41.5, 84.0] {
            for lon in [95.0, 110.0, 120.0, 141.3] {
                let (e, n) = p.forward(lon, lat).unwrap();
                let (lon2, lat2) = p.inverse(e, n).unwrap();
                worst = worst.max((lon2 - lon).abs().max((lat2 - lat).abs()));
            }
        }
        assert!(worst < 1e-12, "worst roundtrip residual {worst} deg");
    }

    #[test]
    fn epsg_g7_2_web_mercator_worked_example_within_1mm() {
        // EPSG GN7-2 method 1024 worked example (WGS 84 / Pseudo-Mercator):
        // φ = 24°22'54.433"N, λ = 100°20'00.000"W → E = -11169055.58, N = 2800000.00.
        let lon = -dms(100.0, 20.0, 0.0);
        let lat = dms(24.0, 22.0, 54.433);
        let (e, n) = web_mercator_forward(lon, lat).unwrap();
        assert!(
            (e + 11_169_055.576258447).abs() < 1e-3,
            "easting residual {} m",
            (e + 11_169_055.58).abs()
        );
        assert!(
            (n - 2_800_000.003136157).abs() < 1e-3,
            "northing residual {} m",
            (n - 2_800_000.00).abs()
        );
        // Reverse worked example: E = -11169055.58, N = 2810000.00
        // → φ = 24°27'48.889"N, λ = 100°20'00.000"W.
        let (lon2, lat2) = web_mercator_inverse(-11_169_055.576258447, 2_810_000.0).unwrap();
        assert!(
            (lon2 + 100.33333333333333).abs() < 9e-9,
            "lon residual {lon2}"
        );
        assert!(
            (lat2 - 24.463580315801703).abs() < 9e-9,
            "lat residual {lat2}"
        );
    }

    #[test]
    fn web_mercator_matches_the_legacy_forge3d_constants() {
        // Behaviour-compatibility pin for the spherical EPSG:1024 path.
        let (x, y) = web_mercator_forward(1.0, 1.0).unwrap();
        assert!((x - 111_319.490_793_273_57).abs() < 1e-6);
        assert!((y - 111_325.142_866_385_04).abs() < 1e-6);
        let (lon, lat) = web_mercator_inverse(x, y).unwrap();
        assert!((lon - 1.0).abs() < 1e-13 && (lat - 1.0).abs() < 1e-13);
    }
}

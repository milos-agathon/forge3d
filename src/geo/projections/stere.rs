// src/geo/projections/stere.rs
// Polar Stereographic Variant A (EPSG method 9810).
// RELEVANT FILES: src/geo/projections/mod.rs, src/gis/crs.rs

use super::{epsg_t, lat_from_epsg_t, Ellipsoid, ProjError, ProjResult};

/// Polar Stereographic Variant A (EPSG 9810): natural origin at a pole with a
/// scale factor there.
#[derive(Clone, Copy, Debug)]
pub struct PolarStereographicA {
    pub ellipsoid: Ellipsoid,
    /// Latitude of natural origin: +90 (north aspect) or -90 (south aspect).
    pub lat0_deg: f64,
    /// Longitude of natural origin, degrees.
    pub lon0_deg: f64,
    /// Scale factor at the pole.
    pub k0: f64,
    pub false_easting: f64,
    pub false_northing: f64,
}

impl PolarStereographicA {
    fn north(&self) -> bool {
        self.lat0_deg > 0.0
    }

    fn rho(&self, lat_deg: f64) -> f64 {
        let e = self.ellipsoid.e();
        // For the north aspect EPSG evaluates t at +φ; the south aspect mirrors.
        let lat = if self.north() { lat_deg } else { -lat_deg };
        let t = epsg_t(lat.to_radians(), e);
        2.0 * self.ellipsoid.a * self.k0 * t
            / ((1.0 + e).powf(1.0 + e) * (1.0 - e).powf(1.0 - e)).sqrt()
    }

    pub fn forward(&self, lon_deg: f64, lat_deg: f64) -> ProjResult<(f64, f64)> {
        if self.lat0_deg.abs() != 90.0 {
            return Err(ProjError::Domain(
                "polar stereographic variant A requires a polar natural origin".to_string(),
            ));
        }
        if lat_deg.abs() > 90.0 || !lon_deg.is_finite() {
            return Err(ProjError::Domain(format!(
                "polar stereographic input out of range: lon={lon_deg}, lat={lat_deg}"
            )));
        }
        let rho = self.rho(lat_deg);
        let theta = (lon_deg - self.lon0_deg).to_radians();
        let (de, dn) = if self.north() {
            (rho * theta.sin(), -rho * theta.cos())
        } else {
            (rho * theta.sin(), rho * theta.cos())
        };
        Ok((self.false_easting + de, self.false_northing + dn))
    }

    pub fn inverse(&self, easting: f64, northing: f64) -> ProjResult<(f64, f64)> {
        let e = self.ellipsoid.e();
        let de = easting - self.false_easting;
        let dn = northing - self.false_northing;
        let rho = de.hypot(dn);
        let t = rho * ((1.0 + e).powf(1.0 + e) * (1.0 - e).powf(1.0 - e)).sqrt()
            / (2.0 * self.ellipsoid.a * self.k0);
        let lat_abs = if rho == 0.0 {
            core::f64::consts::FRAC_PI_2
        } else {
            lat_from_epsg_t(t, e)?
        };
        let (lat, lon) = if self.north() {
            (lat_abs, self.lon0_deg + de.atan2(-dn).to_degrees())
        } else {
            (-lat_abs, self.lon0_deg + de.atan2(dn).to_degrees())
        };
        Ok((lon, lat.to_degrees()))
    }
}

#[cfg(test)]
mod tests {
    use super::super::WGS84;
    use super::*;

    /// EPSG Guidance Note 7-2, method 9810 worked example:
    /// WGS 84 / UPS North.
    fn ups_north() -> PolarStereographicA {
        PolarStereographicA {
            ellipsoid: WGS84,
            lat0_deg: 90.0,
            lon0_deg: 0.0,
            k0: 0.994,
            false_easting: 2_000_000.0,
            false_northing: 2_000_000.0,
        }
    }

    #[test]
    fn epsg_g7_2_worked_example_forward() {
        // GN7-2 prints to 0.01 m (measured residual ≈ 2.6 mm, within the
        // reference's own rounding).
        let (e, n) = ups_north().forward(44.0, 73.0).unwrap();
        assert!(
            (e - 3_320_416.75).abs() < 1e-2,
            "easting residual {} m",
            (e - 3_320_416.75).abs()
        );
        assert!(
            (n - 632_668.43).abs() < 1e-2,
            "northing residual {} m",
            (n - 632_668.43).abs()
        );
    }

    #[test]
    fn epsg_g7_2_worked_example_inverse() {
        let (lon, lat) = ups_north().inverse(3_320_416.75, 632_668.43).unwrap();
        assert!(
            (lon - 44.0).abs() < 2e-7,
            "lon residual {} deg",
            (lon - 44.0).abs()
        );
        assert!(
            (lat - 73.0).abs() < 2e-7,
            "lat residual {} deg",
            (lat - 73.0).abs()
        );
    }

    #[test]
    fn polar_stereographic_roundtrip_is_exact() {
        let p = ups_north();
        let mut worst = 0.0f64;
        for lat in [60.0, 73.0, 84.5, 89.99] {
            for lon in [-179.0, -44.0, 0.0, 44.0, 133.7] {
                let (e, n) = p.forward(lon, lat).unwrap();
                let (lon2, lat2) = p.inverse(e, n).unwrap();
                let dlon = ((lon2 - lon + 540.0) % 360.0 - 180.0).abs();
                worst = worst.max(dlon.max((lat2 - lat).abs()));
            }
        }
        assert!(worst < 1e-11, "worst roundtrip residual {worst} deg");
    }

    #[test]
    fn pole_maps_to_false_origin_and_back() {
        let p = ups_north();
        let (e, n) = p.forward(0.0, 90.0).unwrap();
        assert!((e - 2_000_000.0).abs() < 1e-9 && (n - 2_000_000.0).abs() < 1e-9);
        let (_lon, lat) = p.inverse(e, n).unwrap();
        assert!((lat - 90.0).abs() < 1e-12);
    }
}

// src/geo/projections/lcc.rs
// Lambert Conic Conformal 2SP (EPSG method 9802).
// RELEVANT FILES: src/geo/projections/mod.rs, src/gis/crs.rs

use super::{epsg_m, epsg_t, lat_from_epsg_t, Ellipsoid, ProjError, ProjResult};

/// Lambert Conic Conformal (2SP), EPSG 9802 parameterization. All lengths in
/// metres; the projection axis unit is whatever `false_easting`/`false_northing`
/// and `ellipsoid.a` are expressed in (the G7-2 example uses US survey feet).
#[derive(Clone, Copy, Debug)]
pub struct LambertConformal2Sp {
    pub ellipsoid: Ellipsoid,
    /// Latitude of false origin, degrees.
    pub lat_f_deg: f64,
    /// Longitude of false origin, degrees.
    pub lon_f_deg: f64,
    /// First standard parallel, degrees.
    pub lat1_deg: f64,
    /// Second standard parallel, degrees.
    pub lat2_deg: f64,
    pub easting_f: f64,
    pub northing_f: f64,
}

struct LccConstants {
    n: f64,
    af: f64,
    r_f: f64,
}

impl LambertConformal2Sp {
    fn constants(&self) -> LccConstants {
        let e = self.ellipsoid.e();
        let e2 = self.ellipsoid.e2();
        let (p1, p2) = (self.lat1_deg.to_radians(), self.lat2_deg.to_radians());
        let m1 = epsg_m(p1, e2);
        let m2 = epsg_m(p2, e2);
        let t1 = epsg_t(p1, e);
        let t2 = epsg_t(p2, e);
        let tf = epsg_t(self.lat_f_deg.to_radians(), e);
        let n = (m1.ln() - m2.ln()) / (t1.ln() - t2.ln());
        let f = m1 / (n * t1.powf(n));
        let af = self.ellipsoid.a * f;
        let r_f = af * tf.powf(n);
        LccConstants { n, af, r_f }
    }

    pub fn forward(&self, lon_deg: f64, lat_deg: f64) -> ProjResult<(f64, f64)> {
        if lat_deg.abs() >= 90.0 || !lon_deg.is_finite() {
            return Err(ProjError::Domain(format!(
                "lambert conformal input out of range: lon={lon_deg}, lat={lat_deg}"
            )));
        }
        let c = self.constants();
        let t = epsg_t(lat_deg.to_radians(), self.ellipsoid.e());
        let r = c.af * t.powf(c.n);
        let theta = c.n * (lon_deg - self.lon_f_deg).to_radians();
        Ok((
            self.easting_f + r * theta.sin(),
            self.northing_f + c.r_f - r * theta.cos(),
        ))
    }

    pub fn inverse(&self, easting: f64, northing: f64) -> ProjResult<(f64, f64)> {
        let c = self.constants();
        let de = easting - self.easting_f;
        let dn = c.r_f - (northing - self.northing_f);
        let r = c.n.signum() * de.hypot(dn);
        if r == 0.0 {
            return Err(ProjError::Domain(
                "lambert conformal inverse undefined at the cone apex".to_string(),
            ));
        }
        let t = (r / c.af).powf(1.0 / c.n);
        let theta = de.atan2(dn);
        let lat = lat_from_epsg_t(t, self.ellipsoid.e())?;
        Ok((
            self.lon_f_deg + (theta / c.n).to_degrees(),
            lat.to_degrees(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{dms, CLARKE_1866};
    use super::*;

    /// US survey feet per metre (exactly 3937/1200 ftUS = 1 m... i.e. 1 ftUS = 1200/3937 m).
    const FT_US: f64 = 1200.0 / 3937.0;

    /// EPSG Guidance Note 7-2, method 9802 worked example:
    /// NAD27 / Texas South Central (Clarke 1866, US survey feet).
    fn texas_south_central() -> LambertConformal2Sp {
        LambertConformal2Sp {
            ellipsoid: CLARKE_1866,
            lat_f_deg: dms(27.0, 50.0, 0.0),
            lon_f_deg: -99.0,
            lat1_deg: dms(28.0, 23.0, 0.0),
            lat2_deg: dms(30.0, 17.0, 0.0),
            easting_f: 2_000_000.0 * FT_US,
            northing_f: 0.0,
        }
    }

    #[test]
    fn epsg_g7_2_worked_example_forward_within_1mm() {
        let (e, n) = texas_south_central()
            .forward(-96.0, dms(28.0, 30.0, 0.0))
            .unwrap();
        let expect_e = 2_963_503.91 * FT_US;
        let expect_n = 254_759.80 * FT_US;
        assert!(
            (e - expect_e).abs() < 1e-3,
            "easting residual {} m",
            (e - expect_e).abs()
        );
        assert!(
            (n - expect_n).abs() < 1e-3,
            "northing residual {} m",
            (n - expect_n).abs()
        );
    }

    #[test]
    fn epsg_g7_2_worked_example_inverse_within_1mm() {
        let (lon, lat) = texas_south_central()
            .inverse(2_963_503.91 * FT_US, 254_759.80 * FT_US)
            .unwrap();
        assert!((lon + 96.0).abs() < 1.2e-8, "lon residual {lon}");
        assert!(
            (lat - dms(28.0, 30.0, 0.0)).abs() < 1.2e-8,
            "lat residual {lat}"
        );
    }
}

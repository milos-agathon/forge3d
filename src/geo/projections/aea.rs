// src/geo/projections/aea.rs
// Albers Equal Area (EPSG method 9822).
// RELEVANT FILES: src/geo/projections/mod.rs, src/gis/crs.rs

use super::{epsg_m, Ellipsoid, ProjError, ProjResult};

/// Albers Equal Area, EPSG 9822 parameterization.
#[derive(Clone, Copy, Debug)]
pub struct AlbersEqualArea {
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

/// EPSG's α (an authalic-latitude helper, "q" in Snyder):
/// α(φ) = (1−e²)·[ sinφ/(1−e²sin²φ) − (1/2e)·ln((1−e sinφ)/(1+e sinφ)) ].
fn q_of(lat: f64, e: f64) -> f64 {
    let e2 = e * e;
    let s = lat.sin();
    (1.0 - e2) * (s / (1.0 - e2 * s * s) - (1.0 / (2.0 * e)) * ((1.0 - e * s) / (1.0 + e * s)).ln())
}

/// dα/dφ, for Newton inversion of α.
fn dq_dlat(lat: f64, e: f64) -> f64 {
    let e2 = e * e;
    let s = lat.sin();
    let d = 1.0 - e2 * s * s;
    2.0 * (1.0 - e2) * lat.cos() / (d * d)
}

struct AeaConstants {
    n: f64,
    c: f64,
    rho0: f64,
}

impl AlbersEqualArea {
    fn constants(&self) -> AeaConstants {
        let e = self.ellipsoid.e();
        let e2 = self.ellipsoid.e2();
        let (p1, p2) = (self.lat1_deg.to_radians(), self.lat2_deg.to_radians());
        let m1 = epsg_m(p1, e2);
        let m2 = epsg_m(p2, e2);
        let q1 = q_of(p1, e);
        let q2 = q_of(p2, e);
        let q0 = q_of(self.lat_f_deg.to_radians(), e);
        let n = (m1 * m1 - m2 * m2) / (q2 - q1);
        let c = m1 * m1 + n * q1;
        let rho0 = self.ellipsoid.a * (c - n * q0).sqrt() / n;
        AeaConstants { n, c, rho0 }
    }

    pub fn forward(&self, lon_deg: f64, lat_deg: f64) -> ProjResult<(f64, f64)> {
        if lat_deg.abs() > 90.0 || !lon_deg.is_finite() {
            return Err(ProjError::Domain(format!(
                "albers input out of range: lon={lon_deg}, lat={lat_deg}"
            )));
        }
        let k = self.constants();
        let q = q_of(lat_deg.to_radians(), self.ellipsoid.e());
        let rho = self.ellipsoid.a * (k.c - k.n * q).sqrt() / k.n;
        let theta = k.n * (lon_deg - self.lon_f_deg).to_radians();
        Ok((
            self.easting_f + rho * theta.sin(),
            self.northing_f + k.rho0 - rho * theta.cos(),
        ))
    }

    pub fn inverse(&self, easting: f64, northing: f64) -> ProjResult<(f64, f64)> {
        let e = self.ellipsoid.e();
        let k = self.constants();
        let de = easting - self.easting_f;
        let dn = k.rho0 - (northing - self.northing_f);
        // For a south-oriented cone (n < 0) both atan2 arguments flip sign.
        let theta = if k.n >= 0.0 {
            de.atan2(dn)
        } else {
            (-de).atan2(-dn)
        };
        let rho = de.hypot(dn);
        let q = (k.c - (rho * k.n / self.ellipsoid.a).powi(2)) / k.n;
        // Newton-invert α(φ) = q; α is monotone with a well-behaved derivative.
        let mut lat = (q / 2.0).clamp(-1.0, 1.0).asin();
        for _ in 0..25 {
            let delta = (q - q_of(lat, e)) / dq_dlat(lat, e);
            lat += delta;
            // 1e-15 rad ≈ 6 picometres; below one f64 ulp of a latitude.
            if delta.abs() < 1e-15 {
                let lon = self.lon_f_deg + (theta / k.n).to_degrees();
                return Ok((lon, lat.to_degrees()));
            }
        }
        Err(ProjError::Convergence(
            "albers latitude iteration did not converge".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{dms, GRS80};
    use super::*;

    /// EPSG method 9822 worked example (EPSG Dataset method text, rev. 2023-06-29):
    /// NAD83 / Great Lakes Albers (EPSG CRS 3174, GRS 1980). The September 2019
    /// GN7-2 PDF itself carries no numeric Albers example; this is the official
    /// registry example for the method.
    fn great_lakes() -> AlbersEqualArea {
        AlbersEqualArea {
            ellipsoid: GRS80,
            lat_f_deg: dms(45.0, 34.0, 8.3172),
            lon_f_deg: -dms(84.0, 27.0, 21.4380),
            lat1_deg: dms(42.0, 7.0, 21.9864),
            lat2_deg: dms(49.0, 0.0, 54.6480),
            easting_f: 1_000_000.0,
            northing_f: 1_000_000.0,
        }
    }

    #[test]
    fn epsg_worked_example_forward_within_1mm() {
        let (e, n) = great_lakes()
            .forward(-dms(78.0, 45.0, 0.0), dms(42.0, 45.0, 0.0))
            .unwrap();
        assert!(
            (e - 1_466_493.492).abs() < 1e-3,
            "easting residual {} m",
            (e - 1_466_493.492).abs()
        );
        assert!(
            (n - 702_903.006).abs() < 1e-3,
            "northing residual {} m",
            (n - 702_903.006).abs()
        );
    }

    #[test]
    fn epsg_worked_example_inverse_within_published_precision() {
        let (lon, lat) = great_lakes().inverse(1_466_493.492, 702_903.006).unwrap();
        // Inputs are mm-rounded → ~1.3e-8 deg positional uncertainty.
        assert!(
            (lon + dms(78.0, 45.0, 0.0)).abs() < 2e-8,
            "lon residual {} deg",
            (lon + dms(78.0, 45.0, 0.0)).abs()
        );
        assert!(
            (lat - dms(42.0, 45.0, 0.0)).abs() < 2e-8,
            "lat residual {} deg",
            (lat - dms(42.0, 45.0, 0.0)).abs()
        );
    }

    #[test]
    fn epsg_southern_hemisphere_example_forward_and_inverse() {
        // Second registry example: GRS 1967 Modified, south-oriented cone
        // (n < 0). The registry computed it from 9-decimal-rounded radian
        // inputs, which costs a few mm; 1 cm covers that reference rounding
        // (measured residuals ≈ 2.8 / 6.3 mm).
        let aea = AlbersEqualArea {
            ellipsoid: Ellipsoid::new(6_378_160.0, 298.25),
            lat_f_deg: -32.0,
            lon_f_deg: -60.0,
            lat1_deg: -5.0,
            lat2_deg: -42.0,
            easting_f: 0.0,
            northing_f: 0.0,
        };
        let lon = -dms(46.0, 0.0, 1.538);
        let lat = -dms(18.0, 30.0, 2.016);
        let (e, n) = aea.forward(lon, lat).unwrap();
        assert!(
            (e - 1_408_623.196).abs() < 1e-2,
            "easting residual {} m",
            (e - 1_408_623.196).abs()
        );
        assert!(
            (n - 1_507_641.482).abs() < 1e-2,
            "northing residual {} m",
            (n - 1_507_641.482).abs()
        );
        let (lon2, lat2) = aea.inverse(e, n).unwrap();
        assert!((lon2 - lon).abs() < 1e-12 && (lat2 - lat).abs() < 1e-12);
    }
}

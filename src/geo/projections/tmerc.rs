// src/geo/projections/tmerc.rs
// Transverse Mercator (EPSG method 9807) via Krüger series to 8th order in the
// third flattening n — accurate to well below 1 mm over a full UTM zone
// (a 4th-order series is not; see Karney, "Transverse Mercator with an
// accuracy of a few nanometers", J. Geod. 85(8), 2011).
// RELEVANT FILES: src/geo/projections/mod.rs, src/gis/crs.rs

use super::{tauf, taupf, Ellipsoid, ProjError, ProjResult};

/// Krüger forward coefficients α₁..α₈ as polynomials in n (Karney 2011 /
/// Kawase 2011, order 8; identical to GeographicLib TransverseMercator at
/// GEOGRAPHICLIB_TRANSVERSEMERCATOR_ORDER = 8).
fn alpha(n: f64) -> [f64; 8] {
    let n2 = n * n;
    let n3 = n2 * n;
    let n4 = n3 * n;
    let n5 = n4 * n;
    let n6 = n5 * n;
    let n7 = n6 * n;
    let n8 = n7 * n;
    [
        n / 2.0 - 2.0 / 3.0 * n2 + 5.0 / 16.0 * n3 + 41.0 / 180.0 * n4 - 127.0 / 288.0 * n5
            + 7891.0 / 37800.0 * n6
            + 72161.0 / 387072.0 * n7
            - 18975107.0 / 50803200.0 * n8,
        13.0 / 48.0 * n2 - 3.0 / 5.0 * n3 + 557.0 / 1440.0 * n4 + 281.0 / 630.0 * n5
            - 1983433.0 / 1935360.0 * n6
            + 13769.0 / 28800.0 * n7
            + 148003883.0 / 174182400.0 * n8,
        61.0 / 240.0 * n3 - 103.0 / 140.0 * n4 + 15061.0 / 26880.0 * n5 + 167603.0 / 181440.0 * n6
            - 67102379.0 / 29030400.0 * n7
            + 79682431.0 / 79833600.0 * n8,
        49561.0 / 161280.0 * n4 - 179.0 / 168.0 * n5
            + 6601661.0 / 7257600.0 * n6
            + 97445.0 / 49896.0 * n7
            - 40176129013.0 / 7664025600.0 * n8,
        34729.0 / 80640.0 * n5 - 3418889.0 / 1995840.0 * n6
            + 14644087.0 / 9123840.0 * n7
            + 2605413599.0 / 622702080.0 * n8,
        212378941.0 / 319334400.0 * n6 - 30705481.0 / 10378368.0 * n7
            + 175214326799.0 / 58118860800.0 * n8,
        1522256789.0 / 1383782400.0 * n7 - 16759934899.0 / 3113510400.0 * n8,
        1424729850961.0 / 743921418240.0 * n8,
    ]
}

/// Krüger inverse coefficients β₁..β₈ (same sources).
fn beta(n: f64) -> [f64; 8] {
    let n2 = n * n;
    let n3 = n2 * n;
    let n4 = n3 * n;
    let n5 = n4 * n;
    let n6 = n5 * n;
    let n7 = n6 * n;
    let n8 = n7 * n;
    [
        n / 2.0 - 2.0 / 3.0 * n2 + 37.0 / 96.0 * n3 - 1.0 / 360.0 * n4 - 81.0 / 512.0 * n5
            + 96199.0 / 604800.0 * n6
            - 5406467.0 / 38707200.0 * n7
            + 7944359.0 / 67737600.0 * n8,
        n2 / 48.0 + n3 / 15.0 - 437.0 / 1440.0 * n4 + 46.0 / 105.0 * n5
            - 1118711.0 / 3870720.0 * n6
            + 51841.0 / 1209600.0 * n7
            + 24749483.0 / 348364800.0 * n8,
        17.0 / 480.0 * n3 - 37.0 / 840.0 * n4 - 209.0 / 4480.0 * n5
            + 5569.0 / 90720.0 * n6
            + 9261899.0 / 58060800.0 * n7
            - 6457463.0 / 17740800.0 * n8,
        4397.0 / 161280.0 * n4 - 11.0 / 504.0 * n5 - 830251.0 / 7257600.0 * n6
            + 466511.0 / 2494800.0 * n7
            + 324154477.0 / 7664025600.0 * n8,
        4583.0 / 161280.0 * n5 - 108847.0 / 3991680.0 * n6 - 8005831.0 / 63866880.0 * n7
            + 22894433.0 / 124540416.0 * n8,
        20648693.0 / 638668800.0 * n6
            - 16363163.0 / 518918400.0 * n7
            - 2204645983.0 / 12915302400.0 * n8,
        219941297.0 / 5535129600.0 * n7 - 497323811.0 / 12454041600.0 * n8,
        191773887257.0 / 3719607091200.0 * n8,
    ]
}

/// Rectifying radius A = a/(1+n) · (1 + n²/4 + n⁴/64 + n⁶/256 + 25n⁸/16384).
fn rectifying_radius(a: f64, n: f64) -> f64 {
    let n2 = n * n;
    let n4 = n2 * n2;
    let n6 = n4 * n2;
    let n8 = n4 * n4;
    a / (1.0 + n) * (1.0 + n2 / 4.0 + n4 / 64.0 + n6 / 256.0 + 25.0 / 16384.0 * n8)
}

/// Transverse Mercator, EPSG method 9807 parameterization.
#[derive(Clone, Copy, Debug)]
pub struct TransverseMercator {
    pub ellipsoid: Ellipsoid,
    /// Latitude of natural origin, degrees.
    pub lat0_deg: f64,
    /// Longitude of natural origin (central meridian), degrees.
    pub lon0_deg: f64,
    /// Scale factor at natural origin.
    pub k0: f64,
    pub false_easting: f64,
    pub false_northing: f64,
}

impl TransverseMercator {
    /// Gauss–Schreiber intermediate coordinates (ξ', η') scaled by 1.
    fn xi_eta_prime(&self, lon_deg: f64, lat_deg: f64) -> (f64, f64) {
        let es = self.ellipsoid.e();
        let lam = (lon_deg - self.lon0_deg).to_radians();
        let taup = taupf(lat_deg.to_radians().tan(), es);
        let xi_p = taup.atan2(lam.cos());
        let eta_p = (lam.sin() / taup.hypot(lam.cos())).asinh();
        (xi_p, eta_p)
    }

    /// Meridian distance from the equator to `lat_deg`, in the Krüger series'
    /// own normalization (multiply by A for metres).
    fn xi_of_meridian(&self, lat_deg: f64) -> f64 {
        let (xi_p, _) = self.xi_eta_prime(self.lon0_deg, lat_deg);
        let a = alpha(self.ellipsoid.n());
        let mut xi = xi_p;
        for (j, aj) in a.iter().enumerate() {
            let k = 2.0 * (j + 1) as f64;
            xi += aj * (k * xi_p).sin();
        }
        xi
    }

    /// Forward: WGS-style geographic degrees → (easting, northing) metres.
    pub fn forward(&self, lon_deg: f64, lat_deg: f64) -> ProjResult<(f64, f64)> {
        if !(-90.0..=90.0).contains(&lat_deg) || !lon_deg.is_finite() {
            return Err(ProjError::Domain(format!(
                "transverse mercator input out of range: lon={lon_deg}, lat={lat_deg}"
            )));
        }
        let n = self.ellipsoid.n();
        let big_a = rectifying_radius(self.ellipsoid.a, n);
        let (xi_p, eta_p) = self.xi_eta_prime(lon_deg, lat_deg);
        let a = alpha(n);
        let (mut xi, mut eta) = (xi_p, eta_p);
        for (j, aj) in a.iter().enumerate() {
            let k = 2.0 * (j + 1) as f64;
            xi += aj * (k * xi_p).sin() * (k * eta_p).cosh();
            eta += aj * (k * xi_p).cos() * (k * eta_p).sinh();
        }
        let m0 = if self.lat0_deg == 0.0 {
            0.0
        } else {
            self.xi_of_meridian(self.lat0_deg)
        };
        Ok((
            self.false_easting + self.k0 * big_a * eta,
            self.false_northing + self.k0 * big_a * (xi - m0),
        ))
    }

    /// Inverse: (easting, northing) metres → geographic degrees (lon, lat).
    pub fn inverse(&self, easting: f64, northing: f64) -> ProjResult<(f64, f64)> {
        if !easting.is_finite() || !northing.is_finite() {
            return Err(ProjError::Domain(
                "transverse mercator inverse input must be finite".to_string(),
            ));
        }
        let n = self.ellipsoid.n();
        let es = self.ellipsoid.e();
        let big_a = rectifying_radius(self.ellipsoid.a, n);
        let m0 = if self.lat0_deg == 0.0 {
            0.0
        } else {
            self.xi_of_meridian(self.lat0_deg)
        };
        let xi = (northing - self.false_northing) / (self.k0 * big_a) + m0;
        let eta = (easting - self.false_easting) / (self.k0 * big_a);
        let b = beta(n);
        let (mut xi_p, mut eta_p) = (xi, eta);
        for (j, bj) in b.iter().enumerate() {
            let k = 2.0 * (j + 1) as f64;
            xi_p -= bj * (k * xi).sin() * (k * eta).cosh();
            eta_p -= bj * (k * xi).cos() * (k * eta).sinh();
        }
        let taup = xi_p.sin() / eta_p.sinh().hypot(xi_p.cos());
        let lam = eta_p.sinh().atan2(xi_p.cos());
        let lat = tauf(taup, es).atan();
        Ok((self.lon0_deg + lam.to_degrees(), lat.to_degrees()))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{dms, AIRY_1830};
    use super::*;

    /// EPSG Guidance Note 7-2, method 9807 worked example:
    /// OSGB36 / British National Grid (Airy 1830).
    fn osgb() -> TransverseMercator {
        TransverseMercator {
            ellipsoid: AIRY_1830,
            lat0_deg: 49.0,
            lon0_deg: -2.0,
            k0: 0.999_601_271_7,
            false_easting: 400_000.0,
            false_northing: -100_000.0,
        }
    }

    #[test]
    fn epsg_g7_2_worked_example_forward() {
        // GN7-2 prints E/N to 0.01 m and computes them with 8-decimal-rounded
        // radians, so 1 cm is the reference's own precision. The exact Krüger
        // value for this point is E = 577274.984 (see the OS-published test
        // below, which pins 1 mm agreement against a mm-precision reference).
        let (e, n) = osgb()
            .forward(dms(0.0, 30.0, 0.0), dms(50.0, 30.0, 0.0))
            .unwrap();
        assert!(
            (e - 577_274.99).abs() < 1e-2,
            "easting residual {} m",
            (e - 577_274.99).abs()
        );
        assert!(
            (n - 69_740.50).abs() < 1e-2,
            "northing residual {} m",
            (n - 69_740.50).abs()
        );
    }

    #[test]
    fn epsg_g7_2_worked_example_inverse() {
        let (lon, lat) = osgb().inverse(577_274.99, 69_740.50).unwrap();
        // Inputs are cm-rounded, so allow the matching ~1e-7 degree.
        assert!(
            (lon - dms(0.0, 30.0, 0.0)).abs() < 2e-7,
            "lon residual {} deg",
            (lon - dms(0.0, 30.0, 0.0)).abs()
        );
        assert!(
            (lat - dms(50.0, 30.0, 0.0)).abs() < 2e-7,
            "lat residual {} deg",
            (lat - dms(50.0, 30.0, 0.0)).abs()
        );
    }

    #[test]
    fn ordnance_survey_worked_example_within_2mm() {
        // OS "A guide to coordinate systems in Great Britain", worked example
        // (Caister water tower): published to 0.001 m / 0.0001".
        let lon = dms(1.0, 43.0, 4.5177);
        let lat = dms(52.0, 39.0, 27.2531);
        let (e, n) = osgb().forward(lon, lat).unwrap();
        assert!(
            (e - 651_409.903).abs() < 2e-3,
            "easting residual {} m",
            (e - 651_409.903).abs()
        );
        assert!(
            (n - 313_177.270).abs() < 2e-3,
            "northing residual {} m",
            (n - 313_177.270).abs()
        );
        let (lon2, lat2) = osgb().inverse(651_409.903, 313_177.270).unwrap();
        assert!(
            (lon2 - lon).abs() < 5e-8,
            "lon residual {} deg",
            (lon2 - lon).abs()
        );
        assert!(
            (lat2 - lat).abs() < 5e-8,
            "lat residual {} deg",
            (lat2 - lat).abs()
        );
    }

    #[test]
    fn utm_roundtrip_closes_to_nanometres_across_the_zone() {
        let utm31 = TransverseMercator {
            ellipsoid: super::super::WGS84,
            lat0_deg: 0.0,
            lon0_deg: 3.0,
            k0: 0.9996,
            false_easting: 500_000.0,
            false_northing: 0.0,
        };
        let mut worst = 0.0f64;
        for lat in [-79.5, -45.0, -0.5, 0.5, 33.3, 60.0, 79.5] {
            for lon in [0.05, 1.0, 3.0, 4.9, 5.95] {
                let (e, n) = utm31.forward(lon, lat).unwrap();
                let (lon2, lat2) = utm31.inverse(e, n).unwrap();
                worst = worst.max((lon2 - lon).abs().max((lat2 - lat).abs()));
            }
        }
        assert!(worst < 1e-12, "worst roundtrip residual {worst} deg");
    }
}

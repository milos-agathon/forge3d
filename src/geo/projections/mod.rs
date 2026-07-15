// src/geo/projections/mod.rs
// MENSURA: pure-Rust, f64, EPSG-conformant projection engine.
// Every method ships its EPSG Guidance Note 7-2 worked example as a unit test (≤ 1 mm).
// RELEVANT FILES: src/geo/projections/tmerc.rs, src/gis/crs.rs, src/geo/geodesic.rs

pub mod aea;
pub mod geocentric;
pub mod lcc;
pub mod merc;
pub mod stere;
pub mod tmerc;

use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum ProjError {
    #[error("projection domain error: {0}")]
    Domain(String),
    #[error("projection failed to converge: {0}")]
    Convergence(String),
    #[error("unsupported CRS: {0}")]
    Unsupported(String),
}

pub type ProjResult<T> = Result<T, ProjError>;

/// A reference ellipsoid defined by semi-major axis (metres) and flattening.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ellipsoid {
    /// Semi-major axis in metres.
    pub a: f64,
    /// Flattening.
    pub f: f64,
}

impl Ellipsoid {
    pub const fn new(a: f64, inv_f: f64) -> Self {
        Self { a, f: 1.0 / inv_f }
    }
    /// Semi-minor axis.
    pub fn b(&self) -> f64 {
        self.a * (1.0 - self.f)
    }
    /// First eccentricity squared.
    pub fn e2(&self) -> f64 {
        self.f * (2.0 - self.f)
    }
    /// First eccentricity.
    pub fn e(&self) -> f64 {
        self.e2().sqrt()
    }
    /// Second eccentricity squared.
    pub fn ep2(&self) -> f64 {
        self.e2() / (1.0 - self.e2())
    }
    /// Third flattening n = f / (2 - f).
    pub fn n(&self) -> f64 {
        self.f / (2.0 - self.f)
    }
    /// Prime-vertical radius of curvature at latitude (radians).
    pub fn prime_vertical(&self, lat_rad: f64) -> f64 {
        let s = lat_rad.sin();
        self.a / (1.0 - self.e2() * s * s).sqrt()
    }
}

/// WGS84 (EPSG:7030).
pub const WGS84: Ellipsoid = Ellipsoid::new(6_378_137.0, 298.257_223_563);
/// GRS 1980 (EPSG:7019).
pub const GRS80: Ellipsoid = Ellipsoid::new(6_378_137.0, 298.257_222_101);
/// Airy 1830 (EPSG:7001).
pub const AIRY_1830: Ellipsoid = Ellipsoid::new(6_377_563.396, 299.324_964_6);
/// Clarke 1866 (EPSG:7008). Defined by a and b; 1/f = a / (a - b).
pub const CLARKE_1866: Ellipsoid = Ellipsoid {
    a: 6_378_206.4,
    f: (6_378_206.4 - 6_356_583.8) / 6_378_206.4,
};
/// Bessel 1841 (EPSG:7004).
pub const BESSEL_1841: Ellipsoid = Ellipsoid::new(6_377_397.155, 299.152_812_8);
/// International 1924 (EPSG:7022).
pub const INTL_1924: Ellipsoid = Ellipsoid::new(6_378_388.0, 297.0);

/// Isometric-latitude helper τ' = taupf(τ): tangent of the conformal latitude
/// for tangent of geographic latitude τ, with es = first eccentricity.
/// Formulation follows GeographicLib for full f64 accuracy near the poles.
pub(crate) fn taupf(tau: f64, es: f64) -> f64 {
    let tau1 = (1.0 + tau * tau).sqrt();
    let sig = (es * (es * tau / tau1).atanh()).sinh();
    (1.0 + sig * sig).sqrt() * tau - sig * tau1
}

/// Inverse of `taupf`: recover τ = tan(φ) from τ' by Newton iteration.
pub(crate) fn tauf(taup: f64, es: f64) -> f64 {
    const NUMIT: usize = 8;
    let e2m = 1.0 - es * es;
    // Initial guess is exact for a sphere and very good otherwise.
    let mut tau = taup / e2m;
    let stol = 1e-14 * taup.abs().max(1.0);
    for _ in 0..NUMIT {
        let taupa = taupf(tau, es);
        let dtau = (taup - taupa) * (1.0 + e2m * tau * tau)
            / (e2m * (1.0 + tau * tau).sqrt() * (1.0 + taupa * taupa).sqrt());
        tau += dtau;
        if dtau.abs() < stol {
            break;
        }
    }
    tau
}

/// EPSG isometric parameter t = tan(π/4 − φ/2) / ((1 − e sinφ)/(1 + e sinφ))^(e/2).
pub(crate) fn epsg_t(lat: f64, e: f64) -> f64 {
    let es = e * lat.sin();
    (core::f64::consts::FRAC_PI_4 - lat / 2.0).tan() / ((1.0 - es) / (1.0 + es)).powf(e / 2.0)
}

/// EPSG grid-convergence helper m = cosφ / sqrt(1 − e² sin²φ).
pub(crate) fn epsg_m(lat: f64, e2: f64) -> f64 {
    let s = lat.sin();
    lat.cos() / (1.0 - e2 * s * s).sqrt()
}

/// Recover φ from EPSG's t by fixed-point iteration (converges quadratically
/// in e²; iterate to machine precision).
pub(crate) fn lat_from_epsg_t(t: f64, e: f64) -> ProjResult<f64> {
    let mut lat = core::f64::consts::FRAC_PI_2 - 2.0 * t.atan();
    for _ in 0..25 {
        let es = e * lat.sin();
        let next = core::f64::consts::FRAC_PI_2
            - 2.0 * (t * ((1.0 - es) / (1.0 + es)).powf(e / 2.0)).atan();
        let delta = (next - lat).abs();
        lat = next;
        if delta < 1e-16 {
            return Ok(lat);
        }
    }
    // 1e-16 rad ≈ 0.6 nm; anything not converged by 25 rounds is a domain bug.
    Err(ProjError::Convergence(
        "latitude iteration from isometric parameter did not converge".to_string(),
    ))
}

/// Projected CRSs the built-in engine can dispatch by bare EPSG code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EpsgProjection {
    /// Spherical Web Mercator (EPSG:3857, method 1024).
    WebMercator,
    /// WGS84 UTM zone (EPSG:326zz north / 327zz south, method 9807).
    Utm { zone: u8, north: bool },
}

/// Map an EPSG code to a supported built-in projection, if any.
pub fn epsg_projection(code: u32) -> Option<EpsgProjection> {
    match code {
        3857 => Some(EpsgProjection::WebMercator),
        32601..=32660 => Some(EpsgProjection::Utm {
            zone: (code - 32600) as u8,
            north: true,
        }),
        32701..=32760 => Some(EpsgProjection::Utm {
            zone: (code - 32700) as u8,
            north: false,
        }),
        _ => None,
    }
}

fn utm(zone: u8, north: bool) -> tmerc::TransverseMercator {
    tmerc::TransverseMercator {
        ellipsoid: WGS84,
        lat0_deg: 0.0,
        lon0_deg: f64::from(zone) * 6.0 - 183.0,
        k0: 0.9996,
        false_easting: 500_000.0,
        false_northing: if north { 0.0 } else { 10_000_000.0 },
    }
}

/// Forward-project WGS84 lon/lat degrees into a supported EPSG projected CRS.
pub fn epsg_forward(code: u32, lon_deg: f64, lat_deg: f64) -> Option<ProjResult<(f64, f64)>> {
    Some(match epsg_projection(code)? {
        EpsgProjection::WebMercator => merc::web_mercator_forward(lon_deg, lat_deg),
        EpsgProjection::Utm { zone, north } => utm(zone, north).forward(lon_deg, lat_deg),
    })
}

/// Inverse-project a supported EPSG projected CRS back to WGS84 lon/lat degrees.
pub fn epsg_inverse(code: u32, easting: f64, northing: f64) -> Option<ProjResult<(f64, f64)>> {
    Some(match epsg_projection(code)? {
        EpsgProjection::WebMercator => merc::web_mercator_inverse(easting, northing),
        EpsgProjection::Utm { zone, north } => utm(zone, north).inverse(easting, northing),
    })
}

#[cfg(test)]
pub(crate) fn dms(d: f64, m: f64, s: f64) -> f64 {
    d.signum() * (d.abs() + m / 60.0 + s / 3600.0)
}

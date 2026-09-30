// src/geo/projections/mod.rs
// MENSURA: pure-Rust, f64, EPSG-conformant projection engine.
// Every method ships its EPSG Guidance Note 7-2 worked example as a unit test (≤ 1 mm).
// RELEVANT FILES: src/geo/projections/tmerc.rs, src/gis/crs.rs, src/geo/geodesic.rs

pub mod aea;
pub mod eqc;
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

/// Explicit pure-Rust projection definition for callers that need a method
/// without pretending forge3d ships a complete EPSG registry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProjectionDefinition {
    Equirectangular(eqc::Equirectangular),
    PlanetocentricEquirectangular(eqc::Equirectangular),
    TransverseMercator(tmerc::TransverseMercator),
    LambertConformal2Sp(lcc::LambertConformal2Sp),
    AlbersEqualArea(aea::AlbersEqualArea),
    PolarStereographicA(stere::PolarStereographicA),
    PlanetocentricPolarStereographicA(stere::PolarStereographicA),
    MercatorA(merc::MercatorA),
    WebMercator,
}

impl ProjectionDefinition {
    pub fn forward(self, lon_deg: f64, lat_deg: f64) -> ProjResult<(f64, f64)> {
        match self {
            Self::Equirectangular(p) => p.forward(lon_deg, lat_deg),
            Self::PlanetocentricEquirectangular(p) => {
                let projection = eqc::Equirectangular {
                    ellipsoid: Ellipsoid {
                        a: p.ellipsoid.a,
                        f: 0.0,
                    },
                    ..p
                };
                projection.forward(
                    lon_deg,
                    planetocentric_to_planetographic_lat(p.ellipsoid, lat_deg),
                )
            }
            Self::TransverseMercator(p) => p.forward(lon_deg, lat_deg),
            Self::LambertConformal2Sp(p) => p.forward(lon_deg, lat_deg),
            Self::AlbersEqualArea(p) => p.forward(lon_deg, lat_deg),
            Self::PolarStereographicA(p) => p.forward(lon_deg, lat_deg),
            Self::PlanetocentricPolarStereographicA(p) => p.forward(
                lon_deg,
                planetocentric_to_planetographic_lat(p.ellipsoid, lat_deg),
            ),
            Self::MercatorA(p) => p.forward(lon_deg, lat_deg),
            Self::WebMercator => merc::web_mercator_forward(lon_deg, lat_deg),
        }
    }

    pub fn inverse(self, easting: f64, northing: f64) -> ProjResult<(f64, f64)> {
        match self {
            Self::Equirectangular(p) => p.inverse(easting, northing),
            Self::PlanetocentricEquirectangular(p) => {
                let projection = eqc::Equirectangular {
                    ellipsoid: Ellipsoid {
                        a: p.ellipsoid.a,
                        f: 0.0,
                    },
                    ..p
                };
                let (lon, lat) = projection.inverse(easting, northing)?;
                Ok((lon, planetographic_to_planetocentric_lat(p.ellipsoid, lat)))
            }
            Self::TransverseMercator(p) => p.inverse(easting, northing),
            Self::LambertConformal2Sp(p) => p.inverse(easting, northing),
            Self::AlbersEqualArea(p) => p.inverse(easting, northing),
            Self::PolarStereographicA(p) => p.inverse(easting, northing),
            Self::PlanetocentricPolarStereographicA(p) => {
                let (lon, lat) = p.inverse(easting, northing)?;
                Ok((lon, planetographic_to_planetocentric_lat(p.ellipsoid, lat)))
            }
            Self::MercatorA(p) => p.inverse(easting, northing),
            Self::WebMercator => merc::web_mercator_inverse(easting, northing),
        }
    }
}

fn planetocentric_to_planetographic_lat(ellipsoid: Ellipsoid, lat_deg: f64) -> f64 {
    let lat = lat_deg.to_radians();
    lat.sin()
        .atan2((1.0 - ellipsoid.e2()) * lat.cos())
        .to_degrees()
}

fn planetographic_to_planetocentric_lat(ellipsoid: Ellipsoid, lat_deg: f64) -> f64 {
    let lat = lat_deg.to_radians();
    ((1.0 - ellipsoid.e2()) * lat.sin())
        .atan2(lat.cos())
        .to_degrees()
}

/// Return the registered body for a supported IAU 2015 CRS code.
pub fn iau_body(code: u32) -> Option<&'static crate::geo::body::Body> {
    use crate::geo::body::{MARS, MOON};
    match code {
        30100 | 30110 | 30115 | 30130 | 30135 => Some(&MOON),
        49900 | 49902 | 49910 | 49912 | 49915 | 49917 | 49930 | 49932 | 49935 | 49937 => {
            Some(&MARS)
        }
        _ => None,
    }
}

/// Whether an IAU code is a planetocentric +East geographic CRS.
pub fn iau_geographic(body: &crate::geo::body::Body, code: u32) -> bool {
    iau_body(code) == Some(body) && matches!(code, 30100 | 49900 | 49902)
}

/// Ellipsoid or sphere declared by a supported IAU CRS code.
pub fn iau_ellipsoid(body: &crate::geo::body::Body, code: u32) -> Option<Ellipsoid> {
    if iau_body(code) != Some(body) {
        return None;
    }
    Some(if matches!(code, 49902 | 49912 | 49917 | 49932 | 49937) {
        body.ellipsoid
    } else {
        Ellipsoid {
            a: body.ellipsoid.a,
            f: 0.0,
        }
    })
}

/// Resolve the supported IAU 2015 planetary projection codes for `body`.
pub fn iau_projection(body: &crate::geo::body::Body, code: u32) -> Option<ProjectionDefinition> {
    let ellipsoid = iau_ellipsoid(body, code)?;
    match code {
        30110 | 49910 => Some(ProjectionDefinition::PlanetocentricEquirectangular(
            eqc::Equirectangular {
                ellipsoid,
                lat0_deg: 0.0,
                lon0_deg: 0.0,
                lat_ts_deg: 0.0,
                false_easting: 0.0,
                false_northing: 0.0,
            },
        )),
        49912 => Some(ProjectionDefinition::PlanetocentricEquirectangular(
            eqc::Equirectangular {
                ellipsoid,
                lat0_deg: 0.0,
                lon0_deg: 0.0,
                lat_ts_deg: 0.0,
                false_easting: 0.0,
                false_northing: 0.0,
            },
        )),
        30115 | 49915 => Some(ProjectionDefinition::PlanetocentricEquirectangular(
            eqc::Equirectangular {
                ellipsoid,
                lat0_deg: 0.0,
                lon0_deg: 180.0,
                lat_ts_deg: 0.0,
                false_easting: 0.0,
                false_northing: 0.0,
            },
        )),
        49917 => Some(ProjectionDefinition::PlanetocentricEquirectangular(
            eqc::Equirectangular {
                ellipsoid,
                lat0_deg: 0.0,
                lon0_deg: 180.0,
                lat_ts_deg: 0.0,
                false_easting: 0.0,
                false_northing: 0.0,
            },
        )),
        30130 | 49930 => Some(ProjectionDefinition::PlanetocentricPolarStereographicA(
            stere::PolarStereographicA {
                ellipsoid,
                lat0_deg: 90.0,
                lon0_deg: 0.0,
                k0: 1.0,
                false_easting: 0.0,
                false_northing: 0.0,
            },
        )),
        49932 => Some(ProjectionDefinition::PlanetocentricPolarStereographicA(
            stere::PolarStereographicA {
                ellipsoid,
                lat0_deg: 90.0,
                lon0_deg: 0.0,
                k0: 1.0,
                false_easting: 0.0,
                false_northing: 0.0,
            },
        )),
        30135 | 49935 => Some(ProjectionDefinition::PlanetocentricPolarStereographicA(
            stere::PolarStereographicA {
                ellipsoid,
                lat0_deg: -90.0,
                lon0_deg: 0.0,
                k0: 1.0,
                false_easting: 0.0,
                false_northing: 0.0,
            },
        )),
        49937 => Some(ProjectionDefinition::PlanetocentricPolarStereographicA(
            stere::PolarStereographicA {
                ellipsoid,
                lat0_deg: -90.0,
                lon0_deg: 0.0,
                k0: 1.0,
                false_easting: 0.0,
                false_northing: 0.0,
            },
        )),
        _ => None,
    }
}

/// Mars IAU codes that use the deliberately unsupported ographic/west-positive convention.
pub(crate) fn iau_is_mars_ographic(code: u32) -> bool {
    matches!(code, 49901 | 49911 | 49916 | 49931 | 49936)
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

/// Resolve an EPSG code to a fully-parameterized built-in [`ProjectionDefinition`].
///
/// This is the single authoritative EPSG → projection table consumed by
/// `src/gis/crs.rs`. It is a deliberately small *curated* set — one or a few
/// authoritative codes per method — not a claim to ship the full EPSG registry.
/// Every entry uses the WGS84/GRS80 datum family. The RGF93 v1 (2154) and NAD83
/// (5070) entries reach WGS 84 only through the published EPSG null
/// transformations named by [`epsg_geodetic_datum`] and
/// [`GeodeticDatum::operation_to_wgs84`] (EPSG:1671, 1 m; EPSG:1188, 4 m),
/// because MENSURA ships no NTv2/NADCON grid shift. Callers report those
/// operations; see `crate::gis::crs::datum_steps`. Unknown codes return
/// `None`, and the caller raises rather than passing coordinates through.
///
/// | EPSG | CRS | Method |
/// |------|-----|--------|
/// | 3857 | WGS 84 / Pseudo-Mercator | Web Mercator (1024) |
/// | 326zz / 327zz | WGS 84 / UTM zone N/S | Transverse Mercator (9807) |
/// | 3395 | WGS 84 / World Mercator | Mercator variant A (9804) |
/// | 2154 | RGF93 / Lambert-93 | Lambert Conic Conformal 2SP (9802) |
/// | 5070 | NAD83 / Conus Albers | Albers Equal Area (9822) |
/// | 5041 | WGS 84 / UPS North | Polar Stereographic variant A (9810) |
/// | 5042 | WGS 84 / UPS South | Polar Stereographic variant A (9810) |
pub fn epsg_projection_definition(code: u32) -> Option<ProjectionDefinition> {
    use ProjectionDefinition as P;
    Some(match code {
        3857 => P::WebMercator,
        32601..=32660 => P::TransverseMercator(utm((code - 32600) as u8, true)),
        32701..=32760 => P::TransverseMercator(utm((code - 32700) as u8, false)),
        // WGS 84 / World Mercator — Mercator variant A (EPSG method 9804).
        3395 => P::MercatorA(merc::MercatorA {
            ellipsoid: WGS84,
            lon0_deg: 0.0,
            k0: 1.0,
            false_easting: 0.0,
            false_northing: 0.0,
        }),
        // RGF93 / Lambert-93 — Lambert Conic Conformal 2SP (EPSG method 9802).
        2154 => P::LambertConformal2Sp(lcc::LambertConformal2Sp {
            ellipsoid: GRS80,
            lat_f_deg: 46.5,
            lon_f_deg: 3.0,
            lat1_deg: 49.0,
            lat2_deg: 44.0,
            easting_f: 700_000.0,
            northing_f: 6_600_000.0,
        }),
        // NAD83 / Conus Albers — Albers Equal Area (EPSG method 9822).
        5070 => P::AlbersEqualArea(aea::AlbersEqualArea {
            ellipsoid: GRS80,
            lat_f_deg: 23.0,
            lon_f_deg: -96.0,
            lat1_deg: 29.5,
            lat2_deg: 45.5,
            easting_f: 0.0,
            northing_f: 0.0,
        }),
        // WGS 84 / UPS North — Polar Stereographic variant A (EPSG method 9810).
        5041 => P::PolarStereographicA(stere::PolarStereographicA {
            ellipsoid: WGS84,
            lat0_deg: 90.0,
            lon0_deg: 0.0,
            k0: 0.994,
            false_easting: 2_000_000.0,
            false_northing: 2_000_000.0,
        }),
        // WGS 84 / UPS South — Polar Stereographic variant A (EPSG method 9810).
        5042 => P::PolarStereographicA(stere::PolarStereographicA {
            ellipsoid: WGS84,
            lat0_deg: -90.0,
            lon0_deg: 0.0,
            k0: 0.994,
            false_easting: 2_000_000.0,
            false_northing: 2_000_000.0,
        }),
        _ => return None,
    })
}

/// Geodetic datum a built-in EPSG CRS is referenced to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeodeticDatum {
    /// World Geodetic System 1984 (the engine's pivot datum).
    Wgs84,
    /// Reseau Geodesique Francais 1993 v1 (EPSG datum 6171), an ETRS89
    /// realization.
    Rgf93v1,
    /// North American Datum 1983 (EPSG datum 6269).
    Nad83,
}

impl GeodeticDatum {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Wgs84 => "WGS 84",
            Self::Rgf93v1 => "RGF93 v1",
            Self::Nad83 => "NAD83",
        }
    }

    /// The published EPSG operation from this datum to WGS 84, or `None` for
    /// WGS 84 itself. Both shipped operations are EPSG "Geocentric
    /// translations (geog2D domain)" (method 9603) with all-zero parameters:
    /// EPSG:1671 (RGF93 v1 to WGS 84 (1), accuracy 1 m) and EPSG:1188 (NAD83
    /// to WGS 84 (1), accuracy 4 m). They are null transformations, so they
    /// leave coordinate values unchanged, but they are named and reported
    /// rather than applied silently.
    pub const fn operation_to_wgs84(self) -> Option<DatumOperation> {
        match self {
            Self::Wgs84 => None,
            Self::Rgf93v1 => Some(DatumOperation {
                epsg_code: 1671,
                name: "RGF93 v1 to WGS 84 (1)",
                datum: self,
                method: "Geocentric translations (geog2D domain)",
                method_epsg_code: 9603,
                translation_m: [0.0, 0.0, 0.0],
                accuracy_m: 1.0,
            }),
            Self::Nad83 => Some(DatumOperation {
                epsg_code: 1188,
                name: "NAD83 to WGS 84 (1)",
                datum: self,
                method: "Geocentric translations (geog2D domain)",
                method_epsg_code: 9603,
                translation_m: [0.0, 0.0, 0.0],
                accuracy_m: 4.0,
            }),
        }
    }
}

/// A published EPSG coordinate operation between `datum` and WGS 84.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DatumOperation {
    pub epsg_code: u32,
    pub name: &'static str,
    /// The non-WGS 84 datum; the operation's target is WGS 84.
    pub datum: GeodeticDatum,
    pub method: &'static str,
    pub method_epsg_code: u32,
    /// Geocentric X/Y/Z translation parameters in metres.
    pub translation_m: [f64; 3],
    /// EPSG-published operation accuracy in metres.
    pub accuracy_m: f64,
}

/// One datum operation applied while routing a transform through WGS 84.
/// `reverse` is true when the operation runs WGS 84 → `operation.datum`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DatumStep {
    pub operation: DatumOperation,
    pub reverse: bool,
}

impl DatumStep {
    pub const fn source_datum(&self) -> GeodeticDatum {
        if self.reverse {
            GeodeticDatum::Wgs84
        } else {
            self.operation.datum
        }
    }

    pub const fn target_datum(&self) -> GeodeticDatum {
        if self.reverse {
            self.operation.datum
        } else {
            GeodeticDatum::Wgs84
        }
    }
}

/// Geodetic datum of an EPSG code the built-in engine can transform: the WGS 84
/// geographic 2D/3D and geocentric codes (4326, 4979, 4978) plus every code in
/// [`epsg_projection_definition`]. Kept beside that table so the two cannot
/// drift; `None` means the engine has no path for the code.
pub fn epsg_geodetic_datum(code: u32) -> Option<GeodeticDatum> {
    match code {
        2154 => Some(GeodeticDatum::Rgf93v1),
        5070 => Some(GeodeticDatum::Nad83),
        4326 | 4978 | 4979 => Some(GeodeticDatum::Wgs84),
        code if epsg_projection_definition(code).is_some() => Some(GeodeticDatum::Wgs84),
        _ => None,
    }
}

/// Forward-project WGS84 lon/lat degrees into a supported EPSG projected CRS.
/// Delegates to the authoritative [`epsg_projection_definition`] table.
pub fn epsg_forward(code: u32, lon_deg: f64, lat_deg: f64) -> Option<ProjResult<(f64, f64)>> {
    Some(epsg_projection_definition(code)?.forward(lon_deg, lat_deg))
}

/// Inverse-project a supported EPSG projected CRS back to WGS84 lon/lat degrees.
/// Delegates to the authoritative [`epsg_projection_definition`] table.
pub fn epsg_inverse(code: u32, easting: f64, northing: f64) -> Option<ProjResult<(f64, f64)>> {
    Some(epsg_projection_definition(code)?.inverse(easting, northing))
}

#[cfg(test)]
pub(crate) fn dms(d: f64, m: f64, s: f64) -> f64 {
    d.signum() * (d.abs() + m / 60.0 + s / 3600.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::body::MARS;

    #[test]
    fn mars_ocentric_projections_match_proj_iau_2015() {
        let eqc = iau_projection(&MARS, 49912).unwrap();
        let polar = iau_projection(&MARS, 49932).unwrap();
        let (x, y) = eqc.forward(10.0, 20.0).unwrap();
        assert!((x - 592_746.975_233_062_2).abs() < 1e-6);
        assert!((y - 1_198_439.570_168_155_2).abs() < 1e-6);
        let (x, y) = polar.forward(10.0, 80.0).unwrap();
        assert!((x - 102_584.234_327_933_29).abs() < 1e-6);
        assert!((y + 581_784.103_123_411_1).abs() < 1e-6);
    }
}

#[cfg(test)]
mod datum_tests {
    use super::*;

    #[test]
    fn every_projection_table_code_has_a_datum_and_only_2154_5070_leave_wgs84() {
        let mut non_wgs84 = Vec::new();
        for code in 1..100_000u32 {
            if epsg_projection_definition(code).is_some() {
                let datum = epsg_geodetic_datum(code).expect("table code without a datum");
                if datum != GeodeticDatum::Wgs84 {
                    non_wgs84.push((code, datum));
                }
            }
        }
        assert_eq!(
            non_wgs84,
            vec![(2154, GeodeticDatum::Rgf93v1), (5070, GeodeticDatum::Nad83)]
        );
        assert_eq!(epsg_geodetic_datum(4269), None);
    }

    #[test]
    fn shipped_datum_operations_are_the_published_epsg_null_transformations() {
        assert_eq!(GeodeticDatum::Wgs84.operation_to_wgs84(), None);
        for (datum, code, accuracy) in [
            (GeodeticDatum::Rgf93v1, 1671, 1.0),
            (GeodeticDatum::Nad83, 1188, 4.0),
        ] {
            let op = datum.operation_to_wgs84().unwrap();
            assert_eq!((op.epsg_code, op.method_epsg_code), (code, 9603));
            assert_eq!(op.translation_m, [0.0; 3]);
            assert_eq!(op.accuracy_m, accuracy);
            assert_eq!(op.datum, datum);
        }
    }
}

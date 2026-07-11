// src/geo/geoid.rs
// MENSURA: EGM96 geoid undulation via spherical-harmonic synthesis to
// degree/order 120, following NGA's F477 reference program: potential
// coefficients relative to the WGS84(G873) normal field, the NGA
// height-anomaly→geoid correction model, and the −0.53 m zero-degree term.
// Coefficients ship as a compact committed binary (assets/geoid/egm96_n120.bin,
// ~231 KiB) and are synthesized on demand — never expanded into a dense grid.
// RELEVANT FILES: assets/geoid/README.md, src/geo/units.rs, src/gis/terrarium.rs

use once_cell::sync::Lazy;

use super::units::{Angle, Degree, Egm96, Ellipsoidal, Height, Length, Metre, Orthometric};

const NMAX: usize = 120;
/// WGS84(G873) constants exactly as in NGA F477.
const GM: f64 = 3.986_004_418e14;
const AE: f64 = 6_378_137.0;
const E2: f64 = 0.006_694_379_990_13;
const GEQT: f64 = 9.780_325_335_9;
const SOMIGLIANA_K: f64 = 0.001_931_852_652_46;
/// WGS84(G873) even-degree zonal harmonics of the normal field (F477/DHCSIN).
const J2: f64 = 0.108_262_982_131e-2;
const J4: f64 = -0.237_091_120_053e-5;
const J6: f64 = 0.608_346_498_882e-8;
const J8: f64 = -0.142_681_087_920e-10;
const J10: f64 = 0.121_439_275_882e-13;
/// Zero-degree term referring EGM96 undulations to the WGS84 ellipsoid.
const ZERO_DEGREE_M: f64 = -0.53;

const EGM96_BIN: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/geoid/egm96_n120.bin"
));

struct Egm96Model {
    /// (C̄, S̄) potential pairs, n = 2..=120, m = 0..=n, minus the WGS84
    /// normal-field even zonals (applied at load).
    pot: Vec<(f64, f64)>,
    /// NGA correction-model pairs in centimetres, n = 0..=120, m = 0..=n.
    corr: Vec<(f64, f64)>,
}

fn read_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("bounds checked"))
}

static MODEL: Lazy<Egm96Model> = Lazy::new(|| {
    let b = EGM96_BIN;
    assert_eq!(&b[..8], b"F3DEGM96", "geoid asset magic mismatch");
    assert_eq!(read_u32(b, 8), 1, "geoid asset version mismatch");
    let nmax = read_u32(b, 12) as usize;
    assert_eq!(nmax, NMAX, "geoid asset degree mismatch");
    let npot = read_u32(b, 16) as usize;
    let ncorr = read_u32(b, 20) as usize;
    assert_eq!(npot, tri_count(2, NMAX), "potential coefficient count");
    assert_eq!(ncorr, tri_count(0, NMAX), "correction coefficient count");
    let mut off = 24usize;
    let mut read_pairs = |count: usize| {
        let mut v = Vec::with_capacity(count);
        for _ in 0..count {
            let c = f64::from_le_bytes(b[off..off + 8].try_into().expect("bounds"));
            let s = f64::from_le_bytes(b[off + 8..off + 16].try_into().expect("bounds"));
            off += 16;
            v.push((c, s));
        }
        v
    };
    let mut pot = read_pairs(npot);
    let corr = read_pairs(ncorr);
    // Subtract the normal field's even zonals (stored positively as +Jn/√(2n+1),
    // matching F477's DHCSIN which ADDS them to the negative C̄n0).
    for (n, j) in [(2usize, J2), (4, J4), (6, J6), (8, J8), (10, J10)] {
        pot[pot_index(n, 0)].0 += j / ((2 * n + 1) as f64).sqrt();
    }
    Egm96Model { pot, corr }
});

fn tri_count(nmin: usize, nmax: usize) -> usize {
    (nmin..=nmax).map(|n| n + 1).sum()
}

/// Index of (n, m) within the potential table (n starts at 2).
fn pot_index(n: usize, m: usize) -> usize {
    (n * (n + 1)) / 2 - 3 + m
}

/// Index of (n, m) within the correction table (n starts at 0).
fn corr_index(n: usize, m: usize) -> usize {
    (n * (n + 1)) / 2 + m
}

/// Fully-normalized associated Legendre functions P̄nm(cos θ) for all
/// n ≤ NMAX, m ≤ n, by the standard forward-column recursion on the
/// FULLY-NORMALIZED functions (Holmes & Featherstone 2002, eqs. 11-13;
/// identical to Colombo's LEGFDN used by NGA F477). The naive recursion on
/// unnormalized Pnm overflows f64 above degree ~150 and loses all precision
/// well before that (~degree 60 for the associated terms) — normalization
/// inside the recursion is what keeps every intermediate O(1). At degree 120
/// the Holmes–Featherstone global (10⁻²⁸⁰-scaled) variant is not yet needed;
/// it only becomes necessary beyond degree ~1900 where sinᵐθ underflows.
fn legendre_all(cos_theta: f64, sin_theta: f64) -> Vec<f64> {
    let size = ((NMAX + 1) * (NMAX + 2)) / 2;
    let mut p = vec![0.0f64; size];
    let idx = |n: usize, m: usize| (n * (n + 1)) / 2 + m;

    p[idx(0, 0)] = 1.0;
    if NMAX == 0 {
        return p;
    }
    p[idx(1, 0)] = 3.0f64.sqrt() * cos_theta;
    p[idx(1, 1)] = 3.0f64.sqrt() * sin_theta;

    // Sectorals: P̄mm = √((2m+1)/(2m)) · sinθ · P̄(m−1)(m−1).
    for m in 2..=NMAX {
        let f = ((2 * m + 1) as f64 / (2 * m) as f64).sqrt();
        p[idx(m, m)] = f * sin_theta * p[idx(m - 1, m - 1)];
    }
    // First off-sectoral: P̄(m+1)m = √(2m+3) · cosθ · P̄mm.
    for m in 0..NMAX {
        p[idx(m + 1, m)] = ((2 * m + 3) as f64).sqrt() * cos_theta * p[idx(m, m)];
    }
    // General forward-column recursion.
    for m in 0..=NMAX {
        for n in (m + 2)..=NMAX {
            let nf = n as f64;
            let mf = m as f64;
            let a = ((2.0 * nf + 1.0) / ((nf + mf) * (nf - mf))).sqrt();
            let b = (2.0 * nf - 1.0).sqrt();
            let c = ((nf + mf - 1.0) * (nf - mf - 1.0) / (2.0 * nf - 3.0)).sqrt();
            p[idx(n, m)] = a * (b * cos_theta * p[idx(n - 1, m)] - c * p[idx(n - 2, m)]);
        }
    }
    p
}

/// EGM96 geoid undulation N(φ, λ) in metres. `lat_deg` is geodetic latitude,
/// `lon_deg` longitude (either ±180 or 0..360 convention).
pub fn undulation_deg(lat_deg: f64, lon_deg: f64) -> f64 {
    let model = &*MODEL;
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();

    // Geocentric latitude and radius of the point on the ellipsoid (F477 RADGRA).
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let t1 = sin_lat * sin_lat;
    let nu = AE / (1.0 - E2 * t1).sqrt();
    let p = nu * cos_lat;
    let z = nu * (1.0 - E2) * sin_lat;
    let r = (p * p + z * z).sqrt();
    let lat_gc = z.atan2(p);
    // Somigliana normal gravity on the ellipsoid.
    let gamma = GEQT * (1.0 + SOMIGLIANA_K * t1) / (1.0 - E2 * t1).sqrt();

    let theta = core::f64::consts::FRAC_PI_2 - lat_gc;
    let pnm = legendre_all(theta.cos(), theta.sin());
    let idx = |n: usize, m: usize| (n * (n + 1)) / 2 + m;

    // cos mλ / sin mλ by recurrence.
    let mut cosml = [0.0f64; NMAX + 1];
    let mut sinml = [0.0f64; NMAX + 1];
    cosml[0] = 1.0;
    sinml[0] = 0.0;
    if NMAX >= 1 {
        cosml[1] = lon.cos();
        sinml[1] = lon.sin();
        for m in 2..=NMAX {
            cosml[m] = 2.0 * cosml[1] * cosml[m - 1] - cosml[m - 2];
            sinml[m] = 2.0 * cosml[1] * sinml[m - 1] - sinml[m - 2];
        }
    }

    // Height anomaly on the ellipsoid from the disturbing potential.
    let ar = AE / r;
    let mut arn = ar;
    let mut a_sum = 0.0;
    for n in 2..=NMAX {
        arn *= ar;
        let mut sum = 0.0;
        for m in 0..=n {
            let (c, s) = model.pot[pot_index(n, m)];
            sum += pnm[idx(n, m)] * (c * cosml[m] + s * sinml[m]);
        }
        a_sum += sum * arn;
    }
    let zeta = a_sum * GM / (gamma * r);

    // NGA correction model (centimetres), degrees 0..=NMAX.
    let mut corr_sum = 0.0;
    for n in 0..=NMAX {
        for m in 0..=n {
            let (c, s) = model.corr[corr_index(n, m)];
            corr_sum += pnm[idx(n, m)] * (c * cosml[m] + s * sinml[m]);
        }
    }
    zeta + corr_sum / 100.0 + ZERO_DEGREE_M
}

/// Typed API: geoid undulation as a metric length.
pub fn geoid_undulation(lat: Angle<Degree>, lon: Angle<Degree>) -> Length<Metre> {
    Length::new(undulation_deg(lat.value(), lon.value()))
}

/// The ONLY sanctioned bridge between the orthometric and ellipsoidal height
/// systems: h = H + N(φ, λ).
pub fn orthometric_to_ellipsoidal(
    h: Height<Orthometric<Egm96>>,
    lat: Angle<Degree>,
    lon: Angle<Degree>,
) -> Height<Ellipsoidal> {
    Height::new(h.metres() + undulation_deg(lat.value(), lon.value()))
}

/// Inverse bridge: H = h − N(φ, λ).
pub fn ellipsoidal_to_orthometric(
    h: Height<Ellipsoidal>,
    lat: Angle<Degree>,
    lon: Angle<Degree>,
) -> Height<Orthometric<Egm96>> {
    Height::new(h.metres() - undulation_deg(lat.value(), lon.value()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn load_reference() -> Vec<(f64, f64, f64, String)> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/egm96_test_values.txt");
        std::fs::read_to_string(path)
            .expect("committed EGM96 reference values")
            .lines()
            .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
            .map(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                (
                    f[0].parse().unwrap(),
                    f[1].parse().unwrap(),
                    f[2].parse().unwrap(),
                    f[3].to_string(),
                )
            })
            .collect()
    }

    #[test]
    fn degree_120_matches_nga_published_values_below_half_metre() {
        let mut worst = 0.0f64;
        let mut worst_at = String::new();
        let mut failures = Vec::new();
        for (lat, lon, n_ref, src) in load_reference() {
            let n = undulation_deg(lat, lon);
            let err = (n - n_ref).abs();
            println!("EGM96 ({lat:>11.5}, {lon:>11.5}) [{src:7}]: got {n:8.3}, want {n_ref:8.3}, |Δ| = {err:.3} m");
            if err > worst {
                worst = err;
                worst_at = format!("({lat}, {lon}) [{src}]");
            }
            if err >= 0.5 {
                failures.push(format!("{err:.3} m at ({lat}, {lon}) [{src}]"));
            }
        }
        println!("EGM96 degree-120 worst residual vs published degree-360 values: {worst:.4} m at {worst_at}");
        assert!(failures.is_empty(), "residuals over 0.5 m: {failures:?}");
    }

    #[test]
    fn height_conversion_roundtrips_exactly() {
        let lat = Angle::<Degree>::new(46.87);
        let lon = Angle::<Degree>::new(102.45);
        let ortho = Height::<Orthometric<Egm96>>::new(812.5);
        let ell = orthometric_to_ellipsoidal(ortho, lat, lon);
        let back = ellipsoidal_to_orthometric(ell, lat, lon);
        assert!((back.metres() - 812.5).abs() < 1e-12);
        let n = geoid_undulation(lat, lon).value();
        assert!((ell.metres() - (812.5 + n)).abs() < 1e-12);
    }
}

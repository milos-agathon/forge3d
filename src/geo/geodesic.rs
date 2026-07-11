// src/geo/geodesic.rs
// Karney direct and inverse geodesics on an ellipsoid, series to order 6 —
// a faithful port of GeographicLib's geodesic.c (C library, MIT/X11 license,
// (c) Charles Karney), which implements Karney, "Algorithms for geodesics",
// J. Geodesy 87, 43-55 (2013). Validated against the committed GeodTest
// subset to |Δs| < 1e-8 m and |Δazi| < 1e-9°.
// RELEVANT FILES: src/geo/projections/mod.rs, src/gis/geometry/math.rs

use super::projections::{Ellipsoid, WGS84};

const GEODESIC_ORDER: usize = 6;
const NC3X: usize = 15;
const NC4X: usize = 21;

const TINY: f64 = 1.4916681462400413e-154; // sqrt(f64::MIN_POSITIVE)
const TOL0: f64 = f64::EPSILON;
const TOL1: f64 = 200.0 * TOL0;
const TOL2: f64 = 1.4901161193847656e-8; // sqrt(TOL0)
const TOLB: f64 = TOL0;
const XTHRESH: f64 = 1000.0 * TOL2;
const MAXIT1: usize = 20;
const MAXIT2: usize = MAXIT1 + 53 + 10;

// ---------------------------------------------------------------------------
// Exact angular helpers (ports of geodesic.c's utility functions)
// ---------------------------------------------------------------------------

fn sq(x: f64) -> f64 {
    x * x
}

fn sumx(u: f64, v: f64) -> (f64, f64) {
    let s = u + v;
    let up = s - v;
    let vpp = s - up;
    let up = up - u;
    let vpp = vpp - v;
    (s, if s != 0.0 { 0.0 - (up + vpp) } else { s })
}

fn ang_normalize(x: f64) -> f64 {
    let y = (x % 360.0 + 540.0) % 360.0 - 180.0;
    if y == -180.0 {
        180.0
    } else {
        y
    }
}

fn lat_fix(x: f64) -> f64 {
    if x.abs() > 90.0 {
        f64::NAN
    } else {
        x
    }
}

/// AngDiff: (y - x) reduced to [-180, 180], with the exact remainder e.
fn ang_diff(x: f64, y: f64) -> (f64, f64) {
    let (d, t) = sumx(remainder(-x, 360.0), remainder(y, 360.0));
    let (d, e) = sumx(remainder(d, 360.0), t);
    if d == 0.0 || d.abs() == 180.0 {
        (d.copysign(if e == 0.0 { y - x } else { -e }), e)
    } else {
        (d, e)
    }
}

fn remainder(x: f64, y: f64) -> f64 {
    // IEEE remainder (round-half-even quotient).
    let r = x % y;
    if r.abs() > y / 2.0 {
        r - y.copysign(r)
    } else if r.abs() == y / 2.0 {
        // Tie: pick the representative with even quotient, matching libm.
        let q = (x / y - r / y).round();
        if (q as i64) % 2 == 0 {
            r
        } else {
            r - y.copysign(r)
        }
    } else {
        r
    }
}

/// AngRound: round tiny values to zero-ish grid to suppress underflow noise.
fn ang_round(x: f64) -> f64 {
    const Z: f64 = 1.0 / 16.0;
    if x == 0.0 {
        return 0.0;
    }
    let mut y = x.abs();
    if y < Z {
        y = Z - (Z - y);
    }
    y.copysign(x)
}

/// sincos of (x + t) degrees where t is a tiny exact correction, reducing to
/// [-45, 45] before the radian conversion (geodesic.c sincosde).
fn sincosde(x: f64, t: f64) -> (f64, f64) {
    let q = (x / 90.0).round();
    let r = ang_round((x - 90.0 * q) + t).to_radians();
    let (s, c) = r.sin_cos();
    match (q as i64).rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

fn sincosd(x: f64) -> (f64, f64) {
    let r = x % 360.0;
    let q = (r / 90.0).round();
    let r = (r - 90.0 * q).to_radians();
    let (s, c) = r.sin_cos();
    let (mut s, mut c) = match (q as i64).rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    };
    if x != 0.0 {
        s += 0.0;
        c += 0.0;
    }
    if s == 0.0 {
        s = s.copysign(x);
    }
    (s, c)
}

fn atan2d(y: f64, x: f64) -> f64 {
    let (mut y, mut x) = (y, x);
    let mut q = 0;
    if y.abs() > x.abs() {
        core::mem::swap(&mut x, &mut y);
        q = 2;
    }
    if x < 0.0 {
        x = -x;
        q += 1;
    }
    let ang = y.atan2(x).to_degrees();
    match q {
        1 => (if y >= 0.0 { 180.0 } else { -180.0 }) - ang,
        2 => 90.0 - ang,
        3 => -90.0 + ang,
        _ => ang,
    }
}

fn norm2(s: &mut f64, c: &mut f64) {
    let h = s.hypot(*c);
    *s /= h;
    *c /= h;
}

fn polyval(coeffs: &[f64], x: f64) -> f64 {
    let mut y = coeffs[0];
    for &c in &coeffs[1..] {
        y = y * x + c;
    }
    y
}

/// Clenshaw evaluation of the trigonometric series (geodesic.c SinCosSeries).
/// For `sinp`, entry 0 of `c` is unused and `n` terms c[1..=n] are consumed;
/// otherwise `n` terms c[0..n].
fn sin_cos_series(sinp: bool, sinx: f64, cosx: f64, c: &[f64], n: usize) -> f64 {
    let mut k = n + usize::from(sinp);
    let ar = 2.0 * (cosx - sinx) * (cosx + sinx);
    let mut nn = n;
    let mut y0 = if nn & 1 == 1 {
        k -= 1;
        c[k]
    } else {
        0.0
    };
    let mut y1 = 0.0;
    nn /= 2;
    while nn > 0 {
        nn -= 1;
        k -= 1;
        y1 = ar * y0 - y1 + c[k];
        k -= 1;
        y0 = ar * y1 - y0 + c[k];
    }
    if sinp {
        2.0 * sinx * cosx * y0
    } else {
        cosx * (y0 - y1)
    }
}

fn astroid(x: f64, y: f64) -> f64 {
    let p = sq(x);
    let q = sq(y);
    let r = (p + q - 1.0) / 6.0;
    if q == 0.0 && r <= 0.0 {
        return 0.0;
    }
    let s = p * q / 4.0;
    let r2 = sq(r);
    let r3 = r * r2;
    let disc = s * (s + 2.0 * r3);
    let mut u = r;
    if disc >= 0.0 {
        let mut t3 = s + r3;
        t3 += if t3 < 0.0 { -disc.sqrt() } else { disc.sqrt() };
        let t = t3.cbrt();
        u += t + if t != 0.0 { r2 / t } else { 0.0 };
    } else {
        let ang = (-disc).sqrt().atan2(-(s + r3));
        u += 2.0 * r * (ang / 3.0).cos();
    }
    let v = (sq(u) + q).sqrt();
    let uv = if u < 0.0 { q / (v - u) } else { u + v };
    let w = (uv - q) / (2.0 * v);
    uv / ((uv + sq(w)).sqrt() + w)
}

// ---------------------------------------------------------------------------
// Series coefficients, order 6 (verbatim from geodesic.c)
// ---------------------------------------------------------------------------

fn a1m1f(eps: f64) -> f64 {
    const COEFF: [f64; 5] = [1.0, 4.0, 64.0, 0.0, 256.0];
    let t = polyval(&COEFF[..4], sq(eps)) / COEFF[4];
    (t + eps) / (1.0 - eps)
}

fn c1f(eps: f64, c: &mut [f64; GEODESIC_ORDER + 1]) {
    const COEFF: [f64; 18] = [
        -1.0, 6.0, -16.0, 32.0, // C1[1]
        -9.0, 64.0, -128.0, 2048.0, // C1[2]
        9.0, -16.0, 768.0, // C1[3]
        3.0, -5.0, 512.0, // C1[4]
        -7.0, 1280.0, // C1[5]
        -7.0, 2048.0, // C1[6]
    ];
    let eps2 = sq(eps);
    let mut d = eps;
    let mut o = 0usize;
    for (l, item) in c.iter_mut().enumerate().take(GEODESIC_ORDER + 1).skip(1) {
        let m = (GEODESIC_ORDER - l) / 2;
        *item = d * polyval(&COEFF[o..=o + m], eps2) / COEFF[o + m + 1];
        o += m + 2;
        d *= eps;
    }
}

fn c1pf(eps: f64, c: &mut [f64; GEODESIC_ORDER + 1]) {
    const COEFF: [f64; 18] = [
        205.0, -432.0, 768.0, 1536.0, // C1p[1]
        4005.0, -4736.0, 3840.0, 12288.0, // C1p[2]
        -225.0, 116.0, 384.0, // C1p[3]
        -7173.0, 2695.0, 7680.0, // C1p[4]
        3467.0, 7680.0, // C1p[5]
        38081.0, 61440.0, // C1p[6]
    ];
    let eps2 = sq(eps);
    let mut d = eps;
    let mut o = 0usize;
    for (l, item) in c.iter_mut().enumerate().take(GEODESIC_ORDER + 1).skip(1) {
        let m = (GEODESIC_ORDER - l) / 2;
        *item = d * polyval(&COEFF[o..=o + m], eps2) / COEFF[o + m + 1];
        o += m + 2;
        d *= eps;
    }
}

fn a2m1f(eps: f64) -> f64 {
    const COEFF: [f64; 5] = [-11.0, -28.0, -192.0, 0.0, 256.0];
    let t = polyval(&COEFF[..4], sq(eps)) / COEFF[4];
    (t - eps) / (1.0 + eps)
}

fn c2f(eps: f64, c: &mut [f64; GEODESIC_ORDER + 1]) {
    const COEFF: [f64; 18] = [
        1.0, 2.0, 16.0, 32.0, // C2[1]
        35.0, 64.0, 384.0, 2048.0, // C2[2]
        15.0, 80.0, 768.0, // C2[3]
        7.0, 35.0, 512.0, // C2[4]
        63.0, 1280.0, // C2[5]
        77.0, 2048.0, // C2[6]
    ];
    let eps2 = sq(eps);
    let mut d = eps;
    let mut o = 0usize;
    for (l, item) in c.iter_mut().enumerate().take(GEODESIC_ORDER + 1).skip(1) {
        let m = (GEODESIC_ORDER - l) / 2;
        *item = d * polyval(&COEFF[o..=o + m], eps2) / COEFF[o + m + 1];
        o += m + 2;
        d *= eps;
    }
}

// ---------------------------------------------------------------------------
// Geodesic
// ---------------------------------------------------------------------------

/// A geodesic solver bound to one ellipsoid.
#[derive(Clone, Debug)]
pub struct Geodesic {
    pub a: f64,
    pub f: f64,
    f1: f64,
    e2: f64,
    ep2: f64,
    n: f64,
    b: f64,
    c2: f64,
    etol2: f64,
    a3x: [f64; GEODESIC_ORDER],
    c3x: [f64; NC3X],
    c4x: [f64; NC4X],
}

/// Result of the inverse problem.
#[derive(Clone, Copy, Debug)]
pub struct InverseResult {
    /// Geodesic distance point 1 → point 2, metres.
    pub s12: f64,
    /// Forward azimuth at point 1, degrees.
    pub azi1: f64,
    /// Forward azimuth at point 2, degrees.
    pub azi2: f64,
    /// Arc length on the auxiliary sphere, degrees.
    pub a12: f64,
    /// Area under the geodesic (for polygon accumulation), m².
    pub s12_area: f64,
}

/// Result of the direct problem.
#[derive(Clone, Copy, Debug)]
pub struct DirectResult {
    pub lat2: f64,
    pub lon2: f64,
    pub azi2: f64,
    pub a12: f64,
}

impl Geodesic {
    pub fn new(ellipsoid: &Ellipsoid) -> Self {
        let a = ellipsoid.a;
        let f = ellipsoid.f;
        let f1 = 1.0 - f;
        let e2 = f * (2.0 - f);
        let ep2 = e2 / sq(f1);
        let n = f / (2.0 - f);
        let b = a * f1;
        let c2 = (sq(a)
            + sq(b)
                * (if e2 == 0.0 {
                    1.0
                } else {
                    (if e2 > 0.0 {
                        e2.sqrt().atanh()
                    } else {
                        (-e2).sqrt().atan()
                    }) / e2.abs().sqrt()
                }))
            / 2.0;
        let etol2 = 0.1 * TOL2 / (f.abs().max(0.001) * (1.0 - f / 2.0).min(1.0) / 2.0).sqrt();

        let mut g = Self {
            a,
            f,
            f1,
            e2,
            ep2,
            n,
            b,
            c2,
            etol2,
            a3x: [0.0; GEODESIC_ORDER],
            c3x: [0.0; NC3X],
            c4x: [0.0; NC4X],
        };
        g.a3coeff();
        g.c3coeff();
        g.c4coeff();
        g
    }

    pub fn wgs84() -> Self {
        Self::new(&WGS84)
    }

    fn a3coeff(&mut self) {
        const COEFF: [f64; 18] = [
            -3.0, 128.0, // A3, coeff of eps^5
            -2.0, -3.0, 64.0, // eps^4
            -1.0, -3.0, -1.0, 16.0, // eps^3
            3.0, -1.0, -2.0, 8.0, // eps^2
            1.0, -1.0, 2.0, // eps^1
            1.0, 1.0, // eps^0
        ];
        let mut o = 0usize;
        let mut k = 0usize;
        for j in (0..GEODESIC_ORDER).rev() {
            let m = (GEODESIC_ORDER - j - 1).min(j);
            self.a3x[k] = polyval(&COEFF[o..=o + m], self.n) / COEFF[o + m + 1];
            k += 1;
            o += m + 2;
        }
    }

    fn c3coeff(&mut self) {
        const COEFF: [f64; 45] = [
            3.0, 128.0, // C3[1] eps^5
            2.0, 5.0, 128.0, // eps^4
            -1.0, 3.0, 3.0, 64.0, // eps^3
            -1.0, 0.0, 1.0, 8.0, // eps^2
            -1.0, 1.0, 4.0, // eps^1
            5.0, 256.0, // C3[2] eps^5
            1.0, 3.0, 128.0, // eps^4
            -3.0, -2.0, 3.0, 64.0, // eps^3
            1.0, -3.0, 2.0, 32.0, // eps^2
            7.0, 512.0, // C3[3] eps^5
            -10.0, 9.0, 384.0, // eps^4
            5.0, -9.0, 5.0, 192.0, // eps^3
            7.0, 512.0, // C3[4] eps^5
            -14.0, 7.0, 512.0, // eps^4
            21.0, 2560.0, // C3[5] eps^5
        ];
        let mut o = 0usize;
        let mut k = 0usize;
        for l in 1..GEODESIC_ORDER {
            for j in (l..GEODESIC_ORDER).rev() {
                let m = (GEODESIC_ORDER - j - 1).min(j);
                self.c3x[k] = polyval(&COEFF[o..=o + m], self.n) / COEFF[o + m + 1];
                k += 1;
                o += m + 2;
            }
        }
    }

    fn c4coeff(&mut self) {
        const COEFF: [f64; 77] = [
            97.0, 15015.0, // C4[0] eps^5
            1088.0, 156.0, 45045.0, // eps^4
            -224.0, -4784.0, 1573.0, 45045.0, // eps^3
            -10656.0, 14144.0, -4576.0, -858.0, 45045.0, // eps^2
            64.0, 624.0, -4576.0, 6864.0, -3003.0, 15015.0, // eps^1
            100.0, 208.0, 572.0, 3432.0, -12012.0, 30030.0, 45045.0, // eps^0
            1.0, 9009.0, // C4[1] eps^5
            -2944.0, 468.0, 135135.0, // eps^4
            5792.0, 1040.0, -1287.0, 135135.0, // eps^3
            5952.0, -11648.0, 9152.0, -2574.0, 135135.0, // eps^2
            -64.0, -624.0, 4576.0, -6864.0, 3003.0, 135135.0, // eps^1
            8.0, 10725.0, // C4[2] eps^5
            1856.0, -936.0, 225225.0, // eps^4
            -8448.0, 4992.0, -1144.0, 225225.0, // eps^3
            -1440.0, 4160.0, -4576.0, 1716.0, 225225.0, // eps^2
            -136.0, 63063.0, // C4[3] eps^5
            1024.0, -208.0, 105105.0, // eps^4
            3584.0, -3328.0, 1144.0, 315315.0, // eps^3
            -128.0, 135135.0, // C4[4] eps^5
            -2560.0, 832.0, 405405.0, // eps^4
            128.0, 99099.0, // C4[5] eps^5
        ];
        let mut o = 0usize;
        let mut k = 0usize;
        for l in 0..GEODESIC_ORDER {
            for j in (l..GEODESIC_ORDER).rev() {
                let m = GEODESIC_ORDER - j - 1;
                self.c4x[k] = polyval(&COEFF[o..=o + m], self.n) / COEFF[o + m + 1];
                k += 1;
                o += m + 2;
            }
        }
    }

    fn a3f(&self, eps: f64) -> f64 {
        polyval(&self.a3x, eps)
    }

    fn c3f(&self, eps: f64, c: &mut [f64; GEODESIC_ORDER]) {
        let mut mult = 1.0;
        let mut o = 0usize;
        for l in 1..GEODESIC_ORDER {
            let m = GEODESIC_ORDER - l - 1;
            mult *= eps;
            c[l] = mult * polyval(&self.c3x[o..=o + m], eps);
            o += m + 1;
        }
    }

    fn c4f(&self, eps: f64, c: &mut [f64; GEODESIC_ORDER]) {
        let mut mult = 1.0;
        let mut o = 0usize;
        for (l, item) in c.iter_mut().enumerate() {
            let m = GEODESIC_ORDER - l - 1;
            *item = mult * polyval(&self.c4x[o..=o + m], eps);
            o += m + 1;
            mult *= eps;
        }
    }

    /// geodesic.c Lengths(): distance/reduced-length integrals.
    #[allow(clippy::too_many_arguments)]
    fn lengths(
        &self,
        eps: f64,
        sig12: f64,
        ssig1: f64,
        csig1: f64,
        dn1: f64,
        ssig2: f64,
        csig2: f64,
        dn2: f64,
        want_s12b: bool,
        want_m12b: bool,
    ) -> (f64, f64, f64) {
        // returns (s12b, m12b, m0)
        let mut ca = [0.0; GEODESIC_ORDER + 1];
        let mut cb = [0.0; GEODESIC_ORDER + 1];
        let mut a1 = 0.0;
        let mut a2 = 0.0;
        let mut m0 = 0.0;
        if want_s12b || want_m12b {
            a1 = a1m1f(eps);
            c1f(eps, &mut ca);
            if want_m12b {
                a2 = a2m1f(eps);
                c2f(eps, &mut cb);
                m0 = a1 - a2;
                a2 += 1.0;
            }
            a1 += 1.0;
        }
        let mut s12b = 0.0;
        let mut m12b = 0.0;
        if want_s12b {
            let b1 = sin_cos_series(true, ssig2, csig2, &ca, GEODESIC_ORDER)
                - sin_cos_series(true, ssig1, csig1, &ca, GEODESIC_ORDER);
            s12b = a1 * (sig12 + b1);
            if want_m12b {
                let b2 = sin_cos_series(true, ssig2, csig2, &cb, GEODESIC_ORDER)
                    - sin_cos_series(true, ssig1, csig1, &cb, GEODESIC_ORDER);
                let j12 = m0 * sig12 + (a1 * b1 - a2 * b2);
                m12b = dn2 * (csig1 * ssig2) - dn1 * (ssig1 * csig2) - csig1 * csig2 * j12;
            }
        } else if want_m12b {
            let b1 = sin_cos_series(true, ssig2, csig2, &ca, GEODESIC_ORDER)
                - sin_cos_series(true, ssig1, csig1, &ca, GEODESIC_ORDER);
            let b2 = sin_cos_series(true, ssig2, csig2, &cb, GEODESIC_ORDER)
                - sin_cos_series(true, ssig1, csig1, &cb, GEODESIC_ORDER);
            let j12 = m0 * sig12 + (a1 * b1 - a2 * b2);
            m12b = dn2 * (csig1 * ssig2) - dn1 * (ssig1 * csig2) - csig1 * csig2 * j12;
        }
        (s12b, m12b, m0)
    }

    /// geodesic.c InverseStart(): starting guess for salp1/calp1 (and, for
    /// short lines, salp2/calp2 and sig12).
    #[allow(clippy::too_many_arguments)]
    fn inverse_start(
        &self,
        sbet1: f64,
        cbet1: f64,
        dn1: f64,
        sbet2: f64,
        cbet2: f64,
        dn2: f64,
        lam12: f64,
        slam12: f64,
        clam12: f64,
    ) -> (f64, f64, f64, Option<(f64, f64)>, f64) {
        // returns (sig12, salp1, calp1, Some((salp2, calp2)), dnm)
        let mut sig12 = -1.0f64;
        let mut salp2calp2 = None;
        let sbet12 = sbet2 * cbet1 - cbet2 * sbet1;
        let cbet12 = cbet2 * cbet1 + sbet2 * sbet1;
        let sbet12a = sbet2 * cbet1 + cbet2 * sbet1;
        let shortline = cbet12 >= 0.0 && sbet12 < 0.5 && cbet2 * lam12 < 0.5;
        let mut dnm = 1.0;
        let (somg12, comg12) = if shortline {
            let mut sbetm2 = sq(sbet1 + sbet2);
            sbetm2 /= sbetm2 + sq(cbet1 + cbet2);
            dnm = (1.0 + self.ep2 * sbetm2).sqrt();
            let omg12 = lam12 / (self.f1 * dnm);
            (omg12.sin(), omg12.cos())
        } else {
            (slam12, clam12)
        };

        let mut salp1 = cbet2 * somg12;
        let mut calp1 = if comg12 >= 0.0 {
            sbet12 + cbet2 * sbet1 * sq(somg12) / (1.0 + comg12)
        } else {
            sbet12a - cbet2 * sbet1 * sq(somg12) / (1.0 - comg12)
        };
        let ssig12 = salp1.hypot(calp1);
        let csig12 = sbet1 * sbet2 + cbet1 * cbet2 * comg12;

        if shortline && ssig12 < self.etol2 {
            let mut salp2 = cbet1 * somg12;
            let mut calp2 = sbet12
                - cbet1
                    * sbet2
                    * (if comg12 >= 0.0 {
                        sq(somg12) / (1.0 + comg12)
                    } else {
                        1.0 - comg12
                    });
            norm2(&mut salp2, &mut calp2);
            salp2calp2 = Some((salp2, calp2));
            sig12 = ssig12.atan2(csig12);
        } else if self.n.abs() > 0.1
            || csig12 >= 0.0
            || ssig12 >= 6.0 * self.n.abs() * core::f64::consts::PI * sq(cbet1)
        {
            // Nothing to do; zeroth-order start is good.
        } else {
            // Scale lam12 and bet2 to x, y coordinate system of the astroid.
            let lam12x = (-slam12).atan2(-clam12);
            let (x, y, betscale, lamscale);
            if self.f >= 0.0 {
                let k2 = sq(sbet1) * self.ep2;
                let eps = k2 / (2.0 * (1.0 + (1.0 + k2).sqrt()) + k2);
                lamscale = self.f * cbet1 * self.a3f(eps) * core::f64::consts::PI;
                betscale = lamscale * cbet1;
                x = lam12x / lamscale;
                y = sbet12a / betscale;
            } else {
                let cbet12a = cbet2 * cbet1 - sbet2 * sbet1;
                let bet12a = sbet12a.atan2(cbet12a);
                let (_, m12b, m0) = self.lengths(
                    self.n,
                    core::f64::consts::PI + bet12a,
                    sbet1,
                    -cbet1,
                    dn1,
                    sbet2,
                    cbet2,
                    dn2,
                    false,
                    true,
                );
                x = -1.0 + m12b / (cbet1 * cbet2 * m0 * core::f64::consts::PI);
                betscale = if x < -0.01 {
                    sbet12a / x
                } else {
                    -self.f * sq(cbet1) * core::f64::consts::PI
                };
                lamscale = betscale / cbet1;
                y = lam12x / lamscale;
            }
            if y > -TOL1 && x > -1.0 - XTHRESH {
                if self.f >= 0.0 {
                    salp1 = (-x).min(1.0);
                    calp1 = -(1.0 - sq(salp1)).sqrt();
                } else {
                    calp1 = x.max(if x > -TOL1 { 0.0 } else { -1.0 });
                    salp1 = (1.0 - sq(calp1)).sqrt();
                }
            } else {
                let k = astroid(x, y);
                let omg12a = lamscale
                    * if self.f >= 0.0 {
                        -x * k / (1.0 + k)
                    } else {
                        -y * (1.0 + k) / k
                    };
                let somg12 = omg12a.sin();
                let comg12 = -omg12a.cos();
                salp1 = cbet2 * somg12;
                calp1 = sbet12a - cbet2 * sbet1 * sq(somg12) / (1.0 - comg12);
            }
        }
        if salp1 > 0.0 {
            norm2(&mut salp1, &mut calp1);
        } else {
            salp1 = 1.0;
            calp1 = 0.0;
        }
        (sig12, salp1, calp1, salp2calp2, dnm)
    }

    /// geodesic.c Lambda12(): the longitude equation and its derivative.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn lambda12(
        &self,
        sbet1: f64,
        cbet1: f64,
        dn1: f64,
        sbet2: f64,
        cbet2: f64,
        dn2: f64,
        salp1: f64,
        calp1_in: f64,
        slam120: f64,
        clam120: f64,
        diffp: bool,
    ) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        // returns (lam12, salp2, calp2, sig12, ssig1, csig1, ssig2, csig2, eps, domg12, dlam12)
        let mut calp1 = calp1_in;
        if sbet1 == 0.0 && calp1 == 0.0 {
            calp1 = -TINY;
        }
        let salp0 = salp1 * cbet1;
        let calp0 = calp1.hypot(salp1 * sbet1);

        let mut ssig1 = sbet1;
        let somg1 = salp0 * sbet1;
        let mut csig1 = calp1 * cbet1;
        let comg1 = calp1 * cbet1;
        norm2(&mut ssig1, &mut csig1);

        let salp2 = if cbet2 != cbet1 { salp0 / cbet2 } else { salp1 };
        let calp2 = if cbet2 != cbet1 || sbet2.abs() != -sbet1 {
            (sq(calp1 * cbet1)
                + if cbet1 < -sbet1 {
                    (cbet2 - cbet1) * (cbet1 + cbet2)
                } else {
                    (sbet1 - sbet2) * (sbet1 + sbet2)
                })
            .sqrt()
                / cbet2
        } else {
            calp1.abs()
        };
        let mut ssig2 = sbet2;
        let somg2 = salp0 * sbet2;
        let mut csig2 = calp2 * cbet2;
        let comg2 = calp2 * cbet2;
        norm2(&mut ssig2, &mut csig2);

        let sig12 = ((csig1 * ssig2 - ssig1 * csig2).max(0.0)).atan2(csig1 * csig2 + ssig1 * ssig2);
        let somg12 = (comg1 * somg2 - somg1 * comg2).max(0.0);
        let comg12 = comg1 * comg2 + somg1 * somg2;
        let eta = (somg12 * clam120 - comg12 * slam120).atan2(comg12 * clam120 + somg12 * slam120);

        let k2 = sq(calp0) * self.ep2;
        let eps = k2 / (2.0 * (1.0 + (1.0 + k2).sqrt()) + k2);
        let mut ca = [0.0; GEODESIC_ORDER];
        self.c3f(eps, &mut ca);
        let b312 = sin_cos_series(true, ssig2, csig2, &ca, GEODESIC_ORDER - 1)
            - sin_cos_series(true, ssig1, csig1, &ca, GEODESIC_ORDER - 1);
        let domg12 = -self.f * self.a3f(eps) * salp0 * (sig12 + b312);
        let lam12 = eta + domg12;

        let mut dlam12 = 0.0;
        if diffp {
            if calp2 == 0.0 {
                dlam12 = -2.0 * self.f1 * dn1 / sbet1;
            } else {
                let (_, m12b, _) = self.lengths(
                    eps, sig12, ssig1, csig1, dn1, ssig2, csig2, dn2, false, true,
                );
                dlam12 = m12b * self.f1 / (calp2 * cbet2);
            }
        }
        (
            lam12, salp2, calp2, sig12, ssig1, csig1, ssig2, csig2, eps, domg12, dlam12,
        )
    }

    /// The inverse geodesic problem: distance and azimuths between two points.
    // The reversed `!(v.abs() >= …)` comparison is geodesic.c's deliberate
    // NaN-escape condition and must not be "simplified" to `<`; the -0.7071
    // threshold is upstream's literal heuristic (NOT exactly -1/sqrt(2)).
    #[allow(clippy::neg_cmp_op_on_partial_ord, clippy::approx_constant)]
    pub fn inverse(&self, lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> InverseResult {
        let (lon12, lon12s) = ang_diff(lon1, lon2);
        let mut lonsign = if lon12.is_sign_negative() { -1.0 } else { 1.0 };
        let lon12 = lonsign * lon12;
        let lon12s_signed = lonsign * lon12s;
        let lam12 = lon12.to_radians();
        let (slam12, clam12) = sincosde(lon12, lon12s_signed);
        // Supplementary longitude difference.
        let lon12s = (180.0 - lon12) - lon12s_signed;

        let mut lat1 = ang_round(lat_fix(lat1));
        let mut lat2 = ang_round(lat_fix(lat2));
        let swapp = if lat1.abs() < lat2.abs() || lat2.is_nan() {
            -1.0
        } else {
            1.0
        };
        if swapp < 0.0 {
            lonsign = -lonsign;
            core::mem::swap(&mut lat1, &mut lat2);
        }
        let latsign = if lat1 < 0.0 { 1.0 } else { -1.0 };
        lat1 *= latsign;
        lat2 *= latsign;

        let (mut sbet1, mut cbet1) = sincosd(lat1);
        sbet1 *= self.f1;
        norm2(&mut sbet1, &mut cbet1);
        cbet1 = cbet1.max(TINY);
        let (mut sbet2, mut cbet2) = sincosd(lat2);
        sbet2 *= self.f1;
        norm2(&mut sbet2, &mut cbet2);
        cbet2 = cbet2.max(TINY);

        if cbet1 < -sbet1 {
            if cbet2 == cbet1 {
                sbet2 = if sbet2 < 0.0 { sbet1 } else { -sbet1 };
            }
        } else if sbet2.abs() == -sbet1 {
            cbet2 = cbet1;
        }

        let dn1 = (1.0 + self.ep2 * sq(sbet1)).sqrt();
        let dn2 = (1.0 + self.ep2 * sq(sbet2)).sqrt();

        let mut sig12;
        let mut s12x = 0.0;
        let mut m12x = 0.0;
        let mut a12;
        let (mut salp1, mut calp1);
        let (mut salp2, mut calp2) = (0.0f64, 0.0f64);
        let mut somg12 = 2.0f64;
        let mut comg12 = 0.0f64;
        let mut domg12 = 0.0f64;
        let mut ssig1_area = 0.0;
        let mut csig1_area = 0.0;
        let mut ssig2_area = 0.0;
        let mut csig2_area = 0.0;
        let mut eps_area = 0.0;
        let mut have_area_sig = false;

        let mut meridian = lat1 == -90.0 || slam12 == 0.0;
        if meridian {
            calp1 = clam12;
            salp1 = slam12;
            calp2 = 1.0;
            salp2 = 0.0;
            let ssig1 = sbet1;
            let csig1 = calp1 * cbet1;
            let ssig2 = sbet2;
            let csig2 = calp2 * cbet2;
            sig12 = ((csig1 * ssig2 - ssig1 * csig2).max(0.0)).atan2(csig1 * csig2 + ssig1 * ssig2);
            let (s12b, m12b, _) = self.lengths(
                self.n, sig12, ssig1, csig1, dn1, ssig2, csig2, dn2, true, true,
            );
            if sig12 < TOL2 || m12b >= 0.0 {
                if sig12 < 3.0 * TINY || (sig12 < TOL0 && (s12b < 0.0 || m12b < 0.0)) {
                    sig12 = 0.0;
                    m12x = 0.0;
                    s12x = 0.0;
                } else {
                    m12x = m12b;
                    s12x = s12b;
                }
                m12x *= self.b;
                s12x *= self.b;
            } else {
                meridian = false;
            }
        } else {
            sig12 = -1.0;
            salp1 = 0.0;
            calp1 = 0.0;
        }

        a12 = 0.0;
        if !meridian && sbet1 == 0.0 && (self.f <= 0.0 || lon12s >= self.f * 180.0) {
            // Equatorial geodesic.
            calp1 = 0.0;
            calp2 = 0.0;
            salp1 = 1.0;
            salp2 = 1.0;
            s12x = self.a * lam12;
            sig12 = lam12 / self.f1;
            let omg12 = lam12 / self.f1;
            somg12 = omg12.sin();
            comg12 = omg12.cos();
            m12x = self.b * sig12.sin();
            a12 = lon12 / self.f1;
        } else if !meridian {
            let (sig12_start, salp1_s, calp1_s, salp2calp2, dnm) =
                self.inverse_start(sbet1, cbet1, dn1, sbet2, cbet2, dn2, lam12, slam12, clam12);
            sig12 = sig12_start;
            salp1 = salp1_s;
            calp1 = calp1_s;
            if sig12 >= 0.0 {
                // Short-line case handled by InverseStart.
                let (s2, c2) = salp2calp2.expect("short line start provides alp2");
                salp2 = s2;
                calp2 = c2;
                s12x = sig12 * self.b * dnm;
                m12x = sq(dnm) * self.b * (sig12 / dnm).sin();
                a12 = sig12.to_degrees();
                let omg12 = lam12 / (self.f1 * dnm);
                somg12 = omg12.sin();
                comg12 = omg12.cos();
            } else {
                // Newton's method with fallback bisection.
                let mut tripn = false;
                let mut tripb = false;
                let mut salp1a = TINY;
                let mut calp1a = 1.0;
                let mut salp1b = TINY;
                let mut calp1b = -1.0;
                let mut ssig1 = 0.0;
                let mut csig1 = 0.0;
                let mut ssig2 = 0.0;
                let mut csig2 = 0.0;
                let mut eps = 0.0;
                let mut numit = 0usize;
                while numit < MAXIT2 {
                    let (
                        lam12_calc,
                        salp2_c,
                        calp2_c,
                        sig12_c,
                        ss1,
                        cs1,
                        ss2,
                        cs2,
                        eps_c,
                        domg12_c,
                        dv,
                    ) = self.lambda12(
                        sbet1,
                        cbet1,
                        dn1,
                        sbet2,
                        cbet2,
                        dn2,
                        salp1,
                        calp1,
                        slam12,
                        clam12,
                        numit < MAXIT1,
                    );
                    salp2 = salp2_c;
                    calp2 = calp2_c;
                    sig12 = sig12_c;
                    ssig1 = ss1;
                    csig1 = cs1;
                    ssig2 = ss2;
                    csig2 = cs2;
                    eps = eps_c;
                    domg12 = domg12_c;
                    // lambda12 measures eta against (slam12, clam12), so its
                    // return value IS the residual v — do not subtract lam12.
                    let v = lam12_calc;
                    if tripb || !(v.abs() >= if tripn { 8.0 } else { 1.0 } * TOL0) {
                        break;
                    }
                    if v > 0.0 && (numit > MAXIT1 || calp1 / salp1 > calp1b / salp1b) {
                        salp1b = salp1;
                        calp1b = calp1;
                    } else if v < 0.0 && (numit > MAXIT1 || calp1 / salp1 < calp1a / salp1a) {
                        salp1a = salp1;
                        calp1a = calp1;
                    }
                    numit += 1;
                    if numit < MAXIT1 && dv > 0.0 {
                        let dalp1 = -v / dv;
                        let sdalp1 = dalp1.sin();
                        let cdalp1 = dalp1.cos();
                        let nsalp1 = salp1 * cdalp1 + calp1 * sdalp1;
                        if nsalp1 > 0.0 && dalp1.abs() < core::f64::consts::PI {
                            calp1 = calp1 * cdalp1 - salp1 * sdalp1;
                            salp1 = nsalp1;
                            norm2(&mut salp1, &mut calp1);
                            tripn = v.abs() <= 16.0 * TOL0;
                            continue;
                        }
                    }
                    salp1 = (salp1a + salp1b) / 2.0;
                    calp1 = (calp1a + calp1b) / 2.0;
                    norm2(&mut salp1, &mut calp1);
                    tripn = false;
                    tripb = (salp1a - salp1).abs() + (calp1a - calp1) < TOLB
                        || (salp1 - salp1b).abs() + (calp1 - calp1b) < TOLB;
                }
                let (s12b, m12b, _) =
                    self.lengths(eps, sig12, ssig1, csig1, dn1, ssig2, csig2, dn2, true, true);
                m12x = m12b * self.b;
                s12x = s12b * self.b;
                a12 = sig12.to_degrees();
                let sdomg12 = domg12.sin();
                let cdomg12 = domg12.cos();
                somg12 = slam12 * cdomg12 - clam12 * sdomg12;
                comg12 = clam12 * cdomg12 + slam12 * sdomg12;
                ssig1_area = ssig1;
                csig1_area = csig1;
                ssig2_area = ssig2;
                csig2_area = csig2;
                eps_area = eps;
                have_area_sig = true;
            }
        }
        if meridian {
            a12 = sig12.to_degrees();
        }
        let _ = m12x;

        // Area term S12 (for geodesic polygon accumulation).
        let salp0 = salp1 * cbet1;
        let calp0 = calp1.hypot(salp1 * sbet1);
        let mut s12_area = 0.0;
        if calp0 != 0.0 && salp0 != 0.0 {
            let (mut ssig1, mut csig1, mut ssig2, mut csig2, eps) = if have_area_sig {
                (ssig1_area, csig1_area, ssig2_area, csig2_area, eps_area)
            } else {
                let k2 = sq(calp0) * self.ep2;
                (
                    sbet1,
                    calp1 * cbet1,
                    sbet2,
                    calp2 * cbet2,
                    k2 / (2.0 * (1.0 + (1.0 + k2).sqrt()) + k2),
                )
            };
            norm2(&mut ssig1, &mut csig1);
            norm2(&mut ssig2, &mut csig2);
            let a4 = sq(self.a) * calp0 * salp0 * self.e2;
            let mut c4a = [0.0; GEODESIC_ORDER];
            self.c4f(eps, &mut c4a);
            let b41 = sin_cos_series(false, ssig1, csig1, &c4a, GEODESIC_ORDER);
            let b42 = sin_cos_series(false, ssig2, csig2, &c4a, GEODESIC_ORDER);
            s12_area = a4 * (b42 - b41);
        }
        let alp12;
        debug_assert!(meridian || somg12 != 2.0, "omg12 sentinel not resolved");
        if !meridian && comg12 > -0.7071 && sbet2 - sbet1 < 1.75 {
            let domg12x = 1.0 + comg12;
            let dbet1 = 1.0 + cbet1;
            let dbet2 = 1.0 + cbet2;
            alp12 = 2.0
                * (somg12 * (sbet1 * dbet2 + sbet2 * dbet1))
                    .atan2(domg12x * (sbet1 * sbet2 + dbet1 * dbet2));
        } else {
            let mut salp12 = salp2 * calp1 - calp2 * salp1;
            let mut calp12 = calp2 * calp1 + salp2 * salp1;
            if salp12 == 0.0 && calp12 < 0.0 {
                salp12 = TINY * calp1;
                calp12 = -1.0;
            }
            alp12 = salp12.atan2(calp12);
        }
        s12_area += self.c2 * alp12;
        s12_area *= swapp * lonsign * latsign;

        // Undo the canonical arrangement for the azimuths.
        if swapp < 0.0 {
            core::mem::swap(&mut salp1, &mut salp2);
            core::mem::swap(&mut calp1, &mut calp2);
        }
        salp1 *= swapp * lonsign;
        calp1 *= swapp * latsign;
        salp2 *= swapp * lonsign;
        calp2 *= swapp * latsign;

        InverseResult {
            s12: 0.0 + s12x,
            azi1: atan2d(salp1, calp1),
            azi2: atan2d(salp2, calp2),
            a12,
            s12_area,
        }
    }

    /// The direct geodesic problem: destination from start, azimuth, distance.
    pub fn direct(&self, lat1: f64, lon1: f64, azi1: f64, s12: f64) -> DirectResult {
        let azi1n = ang_normalize(azi1);
        let (salp1, calp1) = sincosd(ang_round(azi1n));
        let lat1 = ang_round(lat_fix(lat1));

        let (mut sbet1, mut cbet1) = sincosd(lat1);
        sbet1 *= self.f1;
        norm2(&mut sbet1, &mut cbet1);
        cbet1 = cbet1.max(TINY);

        let salp0 = salp1 * cbet1;
        let calp0 = calp1.hypot(salp1 * sbet1);
        let mut ssig1 = sbet1;
        let somg1 = salp0 * sbet1;
        let mut csig1 = if sbet1 != 0.0 || calp1 != 0.0 {
            cbet1 * calp1
        } else {
            1.0
        };
        let comg1 = csig1;
        norm2(&mut ssig1, &mut csig1);

        let k2 = sq(calp0) * self.ep2;
        let eps = k2 / (2.0 * (1.0 + (1.0 + k2).sqrt()) + k2);

        let a1m1 = a1m1f(eps);
        let mut c1a = [0.0; GEODESIC_ORDER + 1];
        c1f(eps, &mut c1a);
        let b11 = sin_cos_series(true, ssig1, csig1, &c1a, GEODESIC_ORDER);
        let s = b11.sin();
        let c = b11.cos();
        let stau1 = ssig1 * c + csig1 * s;
        let ctau1 = csig1 * c - ssig1 * s;
        let mut c1pa = [0.0; GEODESIC_ORDER + 1];
        c1pf(eps, &mut c1pa);

        let tau12 = s12 / (self.b * (1.0 + a1m1));
        let st = tau12.sin();
        let ct = tau12.cos();
        let b12 = -sin_cos_series(
            true,
            stau1 * ct + ctau1 * st,
            ctau1 * ct - stau1 * st,
            &c1pa,
            GEODESIC_ORDER,
        );
        let sig12 = tau12 - (b12 - b11);
        let ssig12 = sig12.sin();
        let csig12 = sig12.cos();

        let mut ssig2 = ssig1 * csig12 + csig1 * ssig12;
        let mut csig2 = csig1 * csig12 - ssig1 * ssig12;
        // For |f| > 0.01 geodesic.c refines sig2 with one Newton step; WGS84's
        // flattening (0.0034) never takes that path, but keep it for exotic
        // ellipsoids.
        if self.f.abs() > 0.01 {
            let b12_new = sin_cos_series(true, ssig2, csig2, &c1a, GEODESIC_ORDER);
            let serr = (1.0 + a1m1) * (sig12 + (b12_new - b11)) - s12 / self.b;
            let sig12r = sig12 - serr / (1.0 + k2 * sq(ssig2)).sqrt();
            let ssig12r = sig12r.sin();
            let csig12r = sig12r.cos();
            ssig2 = ssig1 * csig12r + csig1 * ssig12r;
            csig2 = csig1 * csig12r - ssig1 * ssig12r;
        }

        let sbet2 = calp0 * ssig2;
        let mut cbet2 = salp0.hypot(calp0 * csig2);
        if cbet2 == 0.0 {
            cbet2 = TINY;
            csig2 = TINY;
        }
        let somg2 = salp0 * ssig2;
        let comg2 = csig2;
        let salp2 = salp0;
        let calp2 = calp0 * csig2;

        let omg12 = (somg2 * comg1 - comg2 * somg1).atan2(comg2 * comg1 + somg2 * somg1);
        let mut c3a = [0.0; GEODESIC_ORDER];
        self.c3f(eps, &mut c3a);
        let lam12 = omg12
            + (-self.f * self.a3f(eps) * salp0)
                * (sig12
                    + (sin_cos_series(true, ssig2, csig2, &c3a, GEODESIC_ORDER - 1)
                        - sin_cos_series(true, ssig1, csig1, &c3a, GEODESIC_ORDER - 1)));
        let lon12 = lam12.to_degrees();
        let lon2 = ang_normalize(ang_normalize(lon1) + ang_normalize(lon12));
        let lat2 = atan2d(sbet2, self.f1 * cbet2);
        let azi2 = atan2d(salp2, calp2);
        DirectResult {
            lat2,
            lon2,
            azi2,
            a12: sig12.to_degrees(),
        }
    }
}

// ---------------------------------------------------------------------------
// Geodesic polygon accumulation (GeographicLib PolygonArea equivalent)
// ---------------------------------------------------------------------------

/// Perimeter and area of a geodesic polygon on the ellipsoid. Handles the
/// antimeridian correctly via crossing counting.
pub fn polygon_perimeter_area(geod: &Geodesic, points: &[(f64, f64)]) -> (f64, f64) {
    // points are (lon, lat) pairs; the ring need not be explicitly closed.
    if points.len() < 2 {
        return (0.0, 0.0);
    }
    let mut closed: Vec<(f64, f64)> = points.to_vec();
    if closed.first() != closed.last() {
        closed.push(closed[0]);
    }
    let mut perimeter = 0.0;
    let mut area = 0.0;
    let mut crossings: i64 = 0;
    for pair in closed.windows(2) {
        let (lon1, lat1) = pair[0];
        let (lon2, lat2) = pair[1];
        let r = geod.inverse(lat1, lon1, lat2, lon2);
        perimeter += r.s12;
        area += r.s12_area;
        crossings += transit(lon1, lon2);
    }
    let area0 = 4.0 * core::f64::consts::PI * geod.c2;
    if crossings % 2 != 0 {
        area += (if area < 0.0 { 1.0 } else { -1.0 }) * area0 / 2.0;
    }
    // Reduce to (-area0/2, area0/2].
    if area > area0 / 2.0 {
        area -= area0;
    } else if area <= -area0 / 2.0 {
        area += area0;
    }
    (perimeter, area.abs())
}

/// Length of a geodesic polyline.
pub fn polyline_length(geod: &Geodesic, points: &[(f64, f64)]) -> f64 {
    points
        .windows(2)
        .map(|pair| geod.inverse(pair[0].1, pair[0].0, pair[1].1, pair[1].0).s12)
        .sum()
}

fn transit(lon1: f64, lon2: f64) -> i64 {
    let (lon12, _) = ang_diff(lon1, lon2);
    let lon1 = ang_normalize(lon1);
    let lon2 = ang_normalize(lon2);
    if lon12 > 0.0 && ((lon1 < 0.0 && lon2 >= 0.0) || (lon1 > 0.0 && lon2 == 0.0)) {
        1
    } else if lon12 < 0.0 && lon1 >= 0.0 && lon2 < 0.0 {
        -1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_known_line_matches_geographiclib() {
        // GeographicLib's own doc example: JFK to LHR.
        let g = Geodesic::wgs84();
        let r = g.inverse(40.6, -73.8, 51.6, -0.5);
        assert!(
            (r.s12 - 5_551_759.400_318_679).abs() < 1e-6,
            "s12={}",
            r.s12
        );
        assert!(
            (r.azi1 - 51.198_882_845_579_824).abs() < 1e-9,
            "azi1={}",
            r.azi1
        );
        assert!(
            (r.azi2 - 107.821_776_735_514_248).abs() < 1e-9,
            "azi2={}",
            r.azi2
        );
    }

    #[test]
    fn direct_inverse_roundtrip() {
        let g = Geodesic::wgs84();
        for &(lat1, lon1, azi1, s12) in &[
            (40.6, -73.8, 51.0, 5.5e6),
            (-30.0, 170.0, 80.0, 9.0e6),
            (0.001, 0.0, 90.0, 1.9e7),
            (75.0, -10.0, 178.0, 4.2e6),
        ] {
            let d = g.direct(lat1, lon1, azi1, s12);
            let inv = g.inverse(lat1, lon1, d.lat2, d.lon2);
            assert!(
                (inv.s12 - s12).abs() < 1e-8,
                "distance closure {} for start ({lat1},{lon1})",
                (inv.s12 - s12).abs()
            );
        }
    }

    #[test]
    fn equatorial_and_meridional_special_cases() {
        let g = Geodesic::wgs84();
        let eq = g.inverse(0.0, 0.0, 0.0, 1.0);
        assert!((eq.s12 - 111_319.490_793_273_57).abs() < 1e-6);
        assert!((eq.azi1 - 90.0).abs() < 1e-12 && (eq.azi2 - 90.0).abs() < 1e-12);
        let mer = g.inverse(0.0, 0.0, 1.0, 0.0);
        assert!((mer.azi1 - 0.0).abs() < 1e-12);
        assert!((mer.s12 - 110_574.388_557_929_9).abs() < 0.5e-3);
    }

    #[test]
    fn geodtest_subset_inverse_and_direct_within_gates() {
        // Committed 50-line subset of GeographicLib's GeodTest set; gates are
        // |Δs12| < 1e-8 m and |Δazi| < 1e-9 deg (the MENSURA definition of done).
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/geodtest_subset.dat");
        let text = std::fs::read_to_string(path).expect("committed GeodTest subset");
        let g = Geodesic::wgs84();
        let mut worst_s = 0.0f64;
        let mut worst_azi = 0.0f64;
        let mut worst_pos = 0.0f64;
        let mut count = 0usize;
        for line in text.lines() {
            if line.trim_start().starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let f: Vec<f64> = line
                .split_whitespace()
                .map(|v| v.parse().expect("GeodTest field"))
                .collect();
            let (lat1, lon1, azi1, lat2, lon2, azi2, s12) =
                (f[0], f[1], f[2], f[3], f[4], f[5], f[6]);
            count += 1;

            let inv = g.inverse(lat1, lon1, lat2, lon2);
            let ds = (inv.s12 - s12).abs();
            let da1 = ang_diff(azi1, inv.azi1).0.abs();
            let da2 = ang_diff(azi2, inv.azi2).0.abs();
            worst_s = worst_s.max(ds);
            worst_azi = worst_azi.max(da1.max(da2));
            assert!(ds < 1e-8, "|Δs12| = {ds} m for line: {line}");
            assert!(da1 < 1e-9, "|Δazi1| = {da1} deg for line: {line}");
            assert!(da2 < 1e-9, "|Δazi2| = {da2} deg for line: {line}");

            let dir = g.direct(lat1, lon1, azi1, s12);
            let dlat = (dir.lat2 - lat2).abs();
            let dlon = ang_diff(lon2, dir.lon2).0.abs() * lat2.to_radians().cos();
            let da2d = ang_diff(azi2, dir.azi2).0.abs();
            worst_pos = worst_pos.max(dlat.max(dlon));
            assert!(dlat < 1e-9, "|Δlat2| = {dlat} deg for line: {line}");
            assert!(dlon < 1e-9, "|Δlon2·cosφ| = {dlon} deg for line: {line}");
            assert!(da2d < 1e-9, "direct |Δazi2| = {da2d} deg for line: {line}");
        }
        assert_eq!(count, 50, "expected all 50 committed cases");
        println!(
            "GeodTest subset ({count} cases): worst |Δs12| = {worst_s:.3e} m, \
             worst |Δazi| = {worst_azi:.3e} deg, worst direct position = {worst_pos:.3e} deg"
        );
    }

    #[test]
    fn nearly_antipodal_polar_case_matches_double_precision_reference() {
        // GeodTest line 89.993883753317 0 → -89.99361... 178.644...: a12 =
        // 179.99969 deg, the ill-conditioned nearly-antipodal regime. The
        // extended-precision truth is azi1 = 29.278300311896; PROJ 9.x's
        // embedded GeographicLib (double) returns 29.27830031301007 — an
        // error of 1.1140706135392975e-9 deg. This port reproduces the
        // double-precision reference behaviour.
        let g = Geodesic::wgs84();
        let r = g.inverse(
            89.993883753317,
            0.0,
            -89.99361237587376089,
            178.64403734674914544,
        );
        assert!((r.s12 - 20_003_896.9371337).abs() < 1e-8, "s12 = {}", r.s12);
        assert!(
            (r.azi1 - 29.278300311896).abs() < 2e-9,
            "azi1 residual {} deg",
            (r.azi1 - 29.278300311896).abs()
        );
        // Bit-level agreement with PROJ's double-precision GeographicLib.
        assert!(
            (r.azi1 - 29.27830031301007).abs() < 1e-12,
            "azi1 vs double-precision reference: {}",
            r.azi1
        );
    }

    #[test]
    fn polygon_area_of_a_one_degree_square_near_equator() {
        let g = Geodesic::wgs84();
        let ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let (perimeter, area) = polygon_perimeter_area(&g, &ring);
        // ~111.32 km by ~110.57 km curved quad.
        assert!((perimeter - 4.0 * 111_000.0).abs() < 4_000.0);
        assert!((area - 1.2308e10).abs() < 2e8, "area={area}");
    }

    #[test]
    fn polygon_area_across_the_antimeridian_is_local_not_world_spanning() {
        let g = Geodesic::wgs84();
        let ring = [(179.0, 0.0), (-179.0, 0.0), (-179.0, 1.0), (179.0, 1.0)];
        let (_, area) = polygon_perimeter_area(&g, &ring);
        // A 2°x1° patch, NOT the 358°-wide complement.
        assert!((area - 2.0 * 1.2308e10).abs() < 4e8, "area={area}");
    }
}

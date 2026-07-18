//! Adaptive exact planar predicates.
//!
//! The algorithms follow J. R. Shewchuk, *Adaptive Precision Floating-Point
//! Arithmetic and Fast Robust Geometric Predicates* (1997).  All expansions
//! are stored least-significant component first.  IEEE-754 binary64 operations
//! in Rust use correctly rounded basic arithmetic, which is the precondition
//! for the error-free transforms below.

use core::cmp::Ordering;

const EPSILON: f64 = 1.110_223_024_625_156_5e-16; // 2^-53
const SPLITTER: f64 = 134_217_729.0; // 2^27 + 1 for binary64

// A subtraction/multiplication contributing to orient2d incurs at most one
// unit roundoff.  Summing the two products adds one more; retaining first-order
// terms gives 3u and bounding the discarded products geometrically gives
// (3 + 16u)u.  This is Shewchuk's derived static A-filter, expressed from u
// rather than copied as an opaque decimal.
const CCW_ERRBOUND_A: f64 = (3.0 + 16.0 * EPSILON) * EPSILON;

// incircle has ten rounded first-order contributions (six products and four
// accumulation steps on the lifted 3x3 determinant).  The same geometric
// summation of higher-order terms yields (10 + 96u)u times the permanent.
const INCIRCLE_ERRBOUND_A: f64 = (10.0 + 96.0 * EPSILON) * EPSILON;

/// The last stage needed to establish a predicate sign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredicateStage {
    /// The static error filter proved the ordinary binary64 determinant.
    Filter,
    /// A head-only expansion proved the sign.
    Adaptive,
    /// A complete expansion of the exact input differences was required.
    Full,
}

/// Error-free `a + b`: `(rounded_sum, exact_roundoff)`.
#[inline]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let bv = x - a;
    let av = x - bv;
    let br = b - bv;
    let ar = a - av;
    (x, ar + br)
}

/// Error-free `a - b`: `(rounded_difference, exact_roundoff)`.
#[inline]
pub fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let x = a - b;
    let bv = a - x;
    let av = x + bv;
    let br = bv - b;
    let ar = a - av;
    (x, ar + br)
}

#[inline]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    (x, b - (x - a))
}

#[inline]
fn split(a: f64) -> (f64, f64) {
    let c = SPLITTER * a;
    let abig = c - a;
    let hi = c - abig;
    (hi, a - hi)
}

/// Error-free `a * b`: `(rounded_product, exact_roundoff)`.
#[inline]
pub fn two_product(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    let (ahi, alo) = split(a);
    let (bhi, blo) = split(b);
    let err1 = x - ahi * bhi;
    let err2 = err1 - alo * bhi;
    let err3 = err2 - ahi * blo;
    (x, alo * blo - err3)
}

fn grow_expansion(expansion: &[f64], value: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(expansion.len() + 1);
    let mut q = value;
    for &component in expansion {
        let (sum, error) = two_sum(q, component);
        if error != 0.0 {
            out.push(error);
        }
        q = sum;
    }
    if q != 0.0 || out.is_empty() {
        out.push(q);
    }
    out
}

/// Sum two non-overlapping expansions, eliminating zero components.
pub fn expansion_sum(left: &[f64], right: &[f64]) -> Vec<f64> {
    let mut out = if left.is_empty() {
        vec![0.0]
    } else {
        left.to_vec()
    };
    for &component in right {
        out = grow_expansion(&out, component);
    }
    out
}

/// Multiply an expansion by one scalar exactly.
pub fn scale_expansion(expansion: &[f64], scalar: f64) -> Vec<f64> {
    if expansion.is_empty() || scalar == 0.0 {
        return vec![0.0];
    }
    let mut out = Vec::with_capacity(expansion.len() * 2);
    let (product, error) = two_product(expansion[0], scalar);
    if error != 0.0 {
        out.push(error);
    }
    let mut q = product;
    for &component in &expansion[1..] {
        let (product, product_error) = two_product(component, scalar);
        let (sum, sum_error) = two_sum(q, product_error);
        if sum_error != 0.0 {
            out.push(sum_error);
        }
        let (next_q, carry) = fast_two_sum(product, sum);
        if carry != 0.0 {
            out.push(carry);
        }
        q = next_q;
    }
    if q != 0.0 || out.is_empty() {
        out.push(q);
    }
    out
}

pub(crate) fn expansion_product(left: &[f64], right: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0];
    for &component in right {
        out = expansion_sum(&out, &scale_expansion(left, component));
    }
    out
}

fn negate(expansion: &[f64]) -> Vec<f64> {
    expansion.iter().map(|value| -*value).collect()
}

pub(crate) fn expansion_diff(left: &[f64], right: &[f64]) -> Vec<f64> {
    expansion_sum(left, &negate(right))
}

pub(crate) fn expansion_estimate(expansion: &[f64]) -> f64 {
    expansion.iter().copied().sum()
}

pub(crate) fn expansion_value_with_exact_sign(expansion: &[f64]) -> f64 {
    match expansion.iter().rev().find(|value| **value != 0.0) {
        Some(value) => *value,
        None => 0.0,
    }
}

pub(crate) fn difference_expansion(a: f64, b: f64) -> Vec<f64> {
    let (head, tail) = two_diff(a, b);
    if tail == 0.0 {
        vec![head]
    } else {
        vec![tail, head]
    }
}

fn cross_expansion(ax: &[f64], ay: &[f64], bx: &[f64], by: &[f64]) -> Vec<f64> {
    expansion_diff(&expansion_product(ax, by), &expansion_product(ay, bx))
}

fn lift_expansion(x: &[f64], y: &[f64]) -> Vec<f64> {
    expansion_sum(&expansion_product(x, x), &expansion_product(y, y))
}

fn finite_points(points: &[[f64; 2]]) -> bool {
    points
        .iter()
        .all(|point| point[0].is_finite() && point[1].is_finite())
}

/// Ordinary binary64 orientation determinant, exposed for ablation tests only.
#[doc(hidden)]
pub fn orient2d_fast(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])
}

/// Exact-sign `orient2d` with the stage used to prove the answer.
pub fn orient2d_with_stage(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> (f64, PredicateStage) {
    if !finite_points(&[a, b, c]) {
        return (f64::NAN, PredicateStage::Filter);
    }
    let acx = a[0] - c[0];
    let bcx = b[0] - c[0];
    let acy = a[1] - c[1];
    let bcy = b[1] - c[1];
    let detleft = acx * bcy;
    let detright = acy * bcx;
    let det = detleft - detright;
    let detsum = if detleft > 0.0 {
        if detright <= 0.0 {
            return (det, PredicateStage::Filter);
        }
        detleft + detright
    } else if detleft < 0.0 {
        if detright >= 0.0 {
            return (det, PredicateStage::Filter);
        }
        -detleft - detright
    } else {
        return (det, PredicateStage::Filter);
    };
    if det.abs() >= CCW_ERRBOUND_A * detsum {
        return (det, PredicateStage::Filter);
    }

    let (_, acxtail) = two_diff(a[0], c[0]);
    let (_, bcxtail) = two_diff(b[0], c[0]);
    let (_, acytail) = two_diff(a[1], c[1]);
    let (_, bcytail) = two_diff(b[1], c[1]);
    if acxtail == 0.0 && bcxtail == 0.0 && acytail == 0.0 && bcytail == 0.0 {
        let exact = cross_expansion(&[acx], &[acy], &[bcx], &[bcy]);
        return (
            expansion_value_with_exact_sign(&exact),
            PredicateStage::Adaptive,
        );
    }
    // Non-zero subtraction tails make the short correction proof subtle.  We
    // deliberately choose the conservative side of Shewchuk's ladder and
    // evaluate the complete expansion instead of accepting an under-bounded
    // correction.  The zero-tail path above is already an exact adaptive
    // expansion, so all three stages remain observable.
    let acx = difference_expansion(a[0], c[0]);
    let acy = difference_expansion(a[1], c[1]);
    let bcx = difference_expansion(b[0], c[0]);
    let bcy = difference_expansion(b[1], c[1]);
    let exact = cross_expansion(&acx, &acy, &bcx, &bcy);
    (
        expansion_value_with_exact_sign(&exact),
        PredicateStage::Full,
    )
}

/// Exact-sign orientation predicate.  Positive is counter-clockwise.
#[inline]
pub fn orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    orient2d_with_stage(a, b, c).0
}

/// Ordinary binary64 incircle determinant, exposed for ablation tests only.
#[doc(hidden)]
pub fn incircle_fast(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> f64 {
    let adx = a[0] - d[0];
    let ady = a[1] - d[1];
    let bdx = b[0] - d[0];
    let bdy = b[1] - d[1];
    let cdx = c[0] - d[0];
    let cdy = c[1] - d[1];
    let abdet = adx * bdy - bdx * ady;
    let bcdet = bdx * cdy - cdx * bdy;
    let cadet = cdx * ady - adx * cdy;
    let alift = adx * adx + ady * ady;
    let blift = bdx * bdx + bdy * bdy;
    let clift = cdx * cdx + cdy * cdy;
    alift * bcdet + blift * cadet + clift * abdet
}

fn incircle_full(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> Vec<f64> {
    let adx = difference_expansion(a[0], d[0]);
    let ady = difference_expansion(a[1], d[1]);
    let bdx = difference_expansion(b[0], d[0]);
    let bdy = difference_expansion(b[1], d[1]);
    let cdx = difference_expansion(c[0], d[0]);
    let cdy = difference_expansion(c[1], d[1]);

    let abdet = cross_expansion(&adx, &ady, &bdx, &bdy);
    let bcdet = cross_expansion(&bdx, &bdy, &cdx, &cdy);
    let cadet = cross_expansion(&cdx, &cdy, &adx, &ady);
    let adet = expansion_product(&lift_expansion(&adx, &ady), &bcdet);
    let bdet = expansion_product(&lift_expansion(&bdx, &bdy), &cadet);
    let cdet = expansion_product(&lift_expansion(&cdx, &cdy), &abdet);
    expansion_sum(&expansion_sum(&adet, &bdet), &cdet)
}

/// Exact-sign `incircle` with the stage used to prove the answer.
///
/// For counter-clockwise `a,b,c`, a positive result places `d` inside.
pub fn incircle_with_stage(
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    d: [f64; 2],
) -> (f64, PredicateStage) {
    if !finite_points(&[a, b, c, d]) {
        return (f64::NAN, PredicateStage::Filter);
    }
    let adx = a[0] - d[0];
    let ady = a[1] - d[1];
    let bdx = b[0] - d[0];
    let bdy = b[1] - d[1];
    let cdx = c[0] - d[0];
    let cdy = c[1] - d[1];
    let bdxcdy = bdx * cdy;
    let cdxbdy = cdx * bdy;
    let cdxady = cdx * ady;
    let adxcdy = adx * cdy;
    let adxbdy = adx * bdy;
    let bdxady = bdx * ady;
    let alift = adx * adx + ady * ady;
    let blift = bdx * bdx + bdy * bdy;
    let clift = cdx * cdx + cdy * cdy;
    let det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady);
    let permanent = (bdxcdy.abs() + cdxbdy.abs()) * alift
        + (cdxady.abs() + adxcdy.abs()) * blift
        + (adxbdy.abs() + bdxady.abs()) * clift;
    if det.abs() > INCIRCLE_ERRBOUND_A * permanent {
        return (det, PredicateStage::Filter);
    }

    let head_exact = {
        let abdet = cross_expansion(&[adx], &[ady], &[bdx], &[bdy]);
        let bcdet = cross_expansion(&[bdx], &[bdy], &[cdx], &[cdy]);
        let cadet = cross_expansion(&[cdx], &[cdy], &[adx], &[ady]);
        let adet = expansion_product(&lift_expansion(&[adx], &[ady]), &bcdet);
        let bdet = expansion_product(&lift_expansion(&[bdx], &[bdy]), &cadet);
        let cdet = expansion_product(&lift_expansion(&[cdx], &[cdy]), &abdet);
        expansion_sum(&expansion_sum(&adet, &bdet), &cdet)
    };
    let all_tails_zero = [
        two_diff(a[0], d[0]).1,
        two_diff(a[1], d[1]).1,
        two_diff(b[0], d[0]).1,
        two_diff(b[1], d[1]).1,
        two_diff(c[0], d[0]).1,
        two_diff(c[1], d[1]).1,
    ]
    .iter()
    .all(|tail| *tail == 0.0);
    if all_tails_zero {
        return (
            expansion_value_with_exact_sign(&head_exact),
            PredicateStage::Adaptive,
        );
    }

    let exact = incircle_full(a, b, c, d);
    let estimate = expansion_estimate(&exact);
    let signed = expansion_value_with_exact_sign(&exact);
    (
        if estimate.signum() == signed.signum() {
            estimate
        } else {
            signed
        },
        PredicateStage::Full,
    )
}

/// Exact-sign incircle predicate.
#[inline]
pub fn incircle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> f64 {
    incircle_with_stage(a, b, c, d).0
}

/// Exact-sign doubled signed area of a closed or open polygon ring.
pub fn signed_area2(points: &[[f64; 2]]) -> f64 {
    if points.len() < 3 || !finite_points(points) {
        return if finite_points(points) { 0.0 } else { f64::NAN };
    }
    let edge_count = if points.first() == points.last() {
        points.len() - 1
    } else {
        points.len()
    };
    let mut area = vec![0.0];
    for index in 0..edge_count {
        let next = (index + 1) % edge_count;
        let (left, left_tail) = two_product(points[index][0], points[next][1]);
        let (right, right_tail) = two_product(points[index][1], points[next][0]);
        let left = if left_tail == 0.0 {
            vec![left]
        } else {
            vec![left_tail, left]
        };
        let right = if right_tail == 0.0 {
            vec![right]
        } else {
            vec![right_tail, right]
        };
        area = expansion_sum(&area, &expansion_diff(&left, &right));
    }
    expansion_value_with_exact_sign(&area)
}

/// Convert a predicate value to a total sign ordering.
#[inline]
pub fn sign_ordering(value: f64) -> Ordering {
    if value > 0.0 {
        Ordering::Greater
    } else if value < 0.0 {
        Ordering::Less
    } else {
        Ordering::Equal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::exact::oracle;

    #[test]
    fn error_free_primitives_reconstruct_textbook_identities() {
        let samples = [
            (1.0, f64::EPSILON),
            (1.0e100, 1.0),
            (-1.0e-100, 3.0e-116),
            (std::f64::consts::PI, std::f64::consts::E),
        ];
        for (a, b) in samples {
            let (sum, error) = two_sum(a, b);
            assert_eq!(sum + error, a + b);
            let (product, product_error) = two_product(a, b);
            assert_eq!(product + product_error, a * b);
        }
        let expansion = expansion_sum(&[f64::EPSILON, 1.0], &[-1.0]);
        assert_eq!(expansion_value_with_exact_sign(&expansion).signum(), 1.0);
    }

    #[test]
    fn near_degenerate_orientation_matches_dyadic_oracle() {
        let cases = [
            ([0.0, 0.0], [1.0, 1.0], [2.0, 2.0]),
            (
                [1.0e20, 1.0e20],
                [1.0e20 + 32768.0, 1.0e20],
                [1.0e20, 1.0e20 + 32768.0],
            ),
            (
                [0.0, 0.0],
                [1.0, f64::from_bits(1.0f64.to_bits() + 1)],
                [2.0, 2.0],
            ),
        ];
        for (a, b, c) in cases {
            assert_eq!(sign_ordering(orient2d(a, b, c)), oracle::orient2d(a, b, c));
        }
    }

    #[test]
    fn incircle_matches_oracle_on_cocircular_ulp_offsets() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let c = [-1.0, 0.0];
        for bits in (1.0f64.to_bits() - 4)..=(1.0f64.to_bits() + 4) {
            let d = [0.0, -f64::from_bits(bits)];
            assert_eq!(
                sign_ordering(incircle(a, b, c, d)),
                oracle::incircle(a, b, c, d)
            );
        }
    }

    #[test]
    fn filter_bound_forces_adaptivity_near_degeneracy() {
        let (_, stage) = orient2d_with_stage([0.0, 0.0], [1.0, 1.0], [2.0, 2.0]);
        assert_ne!(stage, PredicateStage::Filter);
        let (_, stage) = incircle_with_stage([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]);
        assert_ne!(stage, PredicateStage::Filter);
    }

    #[test]
    fn derived_filters_and_exact_fallback_match_oracle_on_100k_cases() {
        fn next(state: &mut u64) -> u64 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        }
        fn coordinate(state: &mut u64) -> f64 {
            let signed = (next(state) >> 12) as i64 - (1i64 << 51);
            signed as f64 * (1.0 / (1u64 << 28) as f64)
        }
        fn point(state: &mut u64) -> [f64; 2] {
            [coordinate(state), coordinate(state)]
        }

        let mut state = 0x4555_434c_4944_4541;
        let mut filter_accepts = 0usize;
        for _ in 0..50_000 {
            let (a, b, c) = (point(&mut state), point(&mut state), point(&mut state));
            let (value, stage) = orient2d_with_stage(a, b, c);
            assert_eq!(sign_ordering(value), oracle::orient2d(a, b, c));
            filter_accepts += usize::from(stage == PredicateStage::Filter);
        }
        for _ in 0..50_000 {
            let (a, b, c, d) = (
                point(&mut state),
                point(&mut state),
                point(&mut state),
                point(&mut state),
            );
            let (value, stage) = incircle_with_stage(a, b, c, d);
            assert_eq!(sign_ordering(value), oracle::incircle(a, b, c, d));
            filter_accepts += usize::from(stage == PredicateStage::Filter);
        }
        assert!(filter_accepts > 99_000, "ordinary cases should hit filters");
    }
}

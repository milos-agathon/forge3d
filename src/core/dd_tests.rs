use super::dd::*;
use glam::DVec3;

fn assert_bits_equal(actual: DD, expected: DD) {
    assert_eq!(actual.hi.to_bits(), expected.hi.to_bits());
    assert_eq!(actual.lo.to_bits(), expected.lo.to_bits());
}

fn error_u2(actual: f64, exact: f64) -> f64 {
    (actual - exact).abs() / exact.abs() / DD_U.powi(2)
}

#[test]
fn encode_decode_preserves_submillimetres_at_ecef_scale() {
    let value = 6_378_137.000_25_f64;
    let encoded = DD::from_f64(value);
    assert!((encoded.to_f64() - value).abs() < 2.0e-8);
    assert!(encoded.lo != 0.0);
}

#[test]
fn two_sum_recovers_a_rounded_away_addend() {
    let a = 1.0_f32;
    let b = f32::from_bits((127 - 24) << 23);
    assert_eq!(two_sum(a, b).to_f64(), a as f64 + b as f64);
}

#[test]
fn scaled_split_covers_extreme_branches_and_residuals() {
    let cases = [
        (f32::from_bits(0x7e80_1234), f32::from_bits(0x0080_4321)),
        (f32::from_bits(0x0001_2345), f32::from_bits(0x5d12_3456)),
        (-f32::from_bits(0x7d55_4321), f32::from_bits(0x0180_0123)),
        (f32::from_bits(0x3f80_0001), f32::from_bits(0x3f7f_ffff)),
    ];
    let mut saw_residual = false;
    for (a, b) in cases {
        let result = two_prod_split(a, b);
        assert_eq!(result.to_f64(), a as f64 * b as f64, "{a:e} * {b:e}");
        saw_residual |= result.lo != 0.0;
    }
    assert!(saw_residual);
}

#[test]
fn focused_corpus_meets_published_relative_error_bounds() {
    for index in 1..=512 {
        let wobble = (index as f64 * 0.754_877_666_246_692_7).fract();
        let av = 0.5 + wobble * 3.0 + (index as f64).sin() * 2.0_f64.powi(-35);
        let bv = 0.75 + (1.0 - wobble) * 2.0 + (index as f64).cos() * 2.0_f64.powi(-36);
        let a = DD::from_f64(av);
        let b = DD::from_f64(bv);
        assert!(error_u2(dd_add(a, b).to_f64(), av + bv) <= DD_ADD_BOUND_U2);
        assert!(error_u2(dd_mul(a, b).to_f64(), av * bv) <= DD_MUL_BOUND_U2);
        assert!(error_u2(dd_div(a, b).to_f64(), av / bv) <= DD_DIV_BOUND_U2);
        assert!(error_u2(dd_sqrt(a).to_f64(), av.sqrt()) <= DD_SQRT_BOUND_U2);
    }
}

#[test]
fn vector_subtraction_keeps_millimetres_at_planet_scale() {
    let p = DDVec3::from_dvec3(DVec3::new(6_378_137.002, 20.0, -5.0));
    let camera = DDVec3::from_dvec3(DVec3::new(6_378_137.001, 19.0, -7.0));
    let actual = dd_sub_vec3(p, camera);
    assert!((actual.to_dvec3() - DVec3::new(0.001, 1.0, 2.0)).length() < 1e-9);
}

#[test]
fn normalization_is_idempotent() {
    let value = DD {
        hi: 1.0,
        lo: f32::EPSILON,
    };
    let once = dd_renorm(value);
    assert_bits_equal(dd_renorm(once), once);
}

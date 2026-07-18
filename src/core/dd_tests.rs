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

#[test]
fn gpu_canary_reports_backend_variant_and_shader_hash() {
    let report = selftest().expect("DD GPU exactness canary");
    assert!(report.passed);
    assert_eq!(report.mismatch_count, 0);
    assert_eq!(report.canary_count, 10);
    assert!(!report.backend.is_empty());
    assert!(report.shader_label.starts_with("dupla.dd.two_prod."));
    assert_eq!(report.shader_hash.len(), 64);
}

#[test]
fn gpu_harness_small_proof_is_bit_locked_and_leak_free() {
    let before = crate::core::resource_tracker::ledger_snapshot();
    let generated = std::env::var("FORGE3D_DD_TEST_N")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4096);
    let adversarial = std::env::var("FORGE3D_DD_TEST_ADVERSARIAL")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4096);
    let requested = std::env::var("FORGE3D_DD_TEST_OP").ok();
    for operation in [
        DdOperation::Add,
        DdOperation::Mul,
        DdOperation::Div,
        DdOperation::Sqrt,
    ] {
        if requested
            .as_deref()
            .is_some_and(|name| DdOperation::parse(name).ok() != Some(operation))
        {
            continue;
        }
        let report = harness_for_test(operation, generated, adversarial).expect("DD GPU proof");
        eprintln!("DUPLA {:?}: max={}u² bound={}u² backend={} variant={:?} generated={} adversarial={} mismatches={}", operation, report.max_err_u2, report.cited_bound_u2, report.backend, report.two_prod_variant, report.generated_count, report.adversarial_count, report.mismatch_count);
        assert_eq!(report.mismatch_count, 0);
        assert!(report.max_err_u2 <= report.cited_bound_u2);
        let certificate: serde_json::Value =
            serde_json::from_str(&report.certificate_json).expect("DD certificate parses");
        let pass = certificate["passes"]
            .as_array()
            .expect("certificate passes")
            .iter()
            .find(|pass| pass["label"] == "dupla.dd_harness")
            .expect("certified DD timing pass");
        let gpu_ms = pass["gpu_ms"].as_f64().expect("DD pass gpu_ms");
        assert!(gpu_ms.is_finite() && gpu_ms >= 0.0);
        assert_eq!(
            certificate["engine"]["wgsl_module_hashes"][&report.shader_label],
            report.shader_hash
        );
        let precision = &certificate["precision"];
        assert_eq!(precision["backend"], report.backend);
        assert_eq!(precision["adapter"], report.adapter);
        assert_eq!(precision["operation"], report.operation.as_str());
        assert_eq!(precision["shader_label"], report.shader_label);
        assert_eq!(precision["shader_hash"], report.shader_hash);
        assert_eq!(precision["generated_count"], report.generated_count);
        assert_eq!(precision["adversarial_count"], report.adversarial_count);
        assert_eq!(precision["mismatch_count"], report.mismatch_count);
        assert_eq!(
            precision["two_prod_variant"],
            report.two_prod_variant.as_str()
        );
        assert_eq!(precision["max_err_u2"], report.max_err_u2);
        assert_eq!(precision["cited_bound_u2"], report.cited_bound_u2);
    }
    let after = crate::core::resource_tracker::ledger_snapshot();
    assert_eq!(
        after.current_host_visible_bytes,
        before.current_host_visible_bytes
    );
    assert_eq!(
        after.current_device_local_bytes,
        before.current_device_local_bytes
    );
}

#[test]
fn forced_canary_failure_records_degradation_refuses_and_releases_resources() {
    const CHILD: &str = "FORGE3D_DD_FORCE_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .arg("--exact")
            .arg("core::dd_tests::forced_canary_failure_records_degradation_refuses_and_releases_resources")
            .arg("--test-threads=1")
            .env(CHILD, "1")
            .env("FORGE3D_DD_FORCE_SELFTEST_FAIL", "1")
            .env_remove("FORGE3D_DETERMINISTIC")
            .status()
            .expect("forced-refusal subprocess starts");
        assert!(status.success(), "forced-refusal subprocess failed");
        return;
    }

    crate::core::degradation::clear_degradations();
    crate::core::gpu::try_ctx().expect("GPU context initializes before forced DD preflight");
    let before = crate::core::resource_tracker::ledger_snapshot();
    let error = selftest().expect_err("forced canary must refuse selftest");
    assert!(matches!(
        error,
        crate::core::error::RenderError::DegradedCapability(_)
    ));
    let error = harness("add", 100_000_000).expect_err("poisoned DD harness must refuse");
    assert!(matches!(
        error,
        crate::core::error::RenderError::DegradedCapability(_)
    ));
    let degradations = crate::core::degradation::degradations_snapshot();
    assert!(degradations
        .iter()
        .any(|item| { item.kind == "precision_selftest_failed" && item.name == "double_float" }));
    let after = crate::core::resource_tracker::ledger_snapshot();
    assert_eq!(
        after.current_host_visible_bytes,
        before.current_host_visible_bytes
    );
    assert_eq!(
        after.current_device_local_bytes,
        before.current_device_local_bytes
    );
}

#[test]
fn harness_rejects_invalid_operation_and_short_proof() {
    assert!(DdOperation::parse("pow").is_err());
    let error = harness("add", 99_999_999).expect_err("short proof must fail");
    assert!(error.to_string().contains("at least 100000000"));
    let error = harness("add", u32::MAX as u64 + 1).expect_err("oversized proof must fail");
    assert!(error.to_string().contains("must not exceed"));
}

#[test]
fn sqrt_half_ulp_tie_is_canonical_across_gpu_and_rust() {
    let (mismatches, max_err_u2) = harness_window_for_test(DdOperation::Sqrt, 35_896_984, 1)
        .expect("sqrt tie regression window");
    assert_eq!(mismatches, 0);
    assert!(max_err_u2 <= DD_SQRT_BOUND_U2);
}

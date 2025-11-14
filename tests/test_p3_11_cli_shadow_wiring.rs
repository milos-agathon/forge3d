// tests/test_p3_11_cli_shadow_wiring.rs
// P3-11: CLI wiring for shadow controls
// Exit criteria: CLI flags map to ShadowManagerConfig correctly

use forge3d::lighting::types::ShadowTechnique as LightingShadowTechnique;
use forge3d::render::params::{ShadowParams, ShadowTechnique as ParamsShadowTechnique};

#[test]
fn test_shadow_params_to_manager_config_default() {
    // Test default conversion
    let params = ShadowParams::default();
    let config = params.to_shadow_manager_config();

    // Verify CSM config
    assert_eq!(config.csm.cascade_count, 4);
    assert_eq!(config.csm.shadow_map_size, 2048);
    assert_eq!(config.csm.pcf_kernel_size, 3);
    assert!(config.csm.stabilize_cascades);
    assert_eq!(config.csm.cascade_blend_range, 0.1);

    // Verify technique conversion (default is PCF)
    assert_eq!(config.technique, LightingShadowTechnique::PCF);

    // Verify memory budget
    assert_eq!(config.max_memory_bytes, 256 * 1024 * 1024);
}

#[test]
fn test_shadow_params_to_manager_config_hard() {
    // Test hard shadows
    let mut params = ShadowParams::default();
    params.technique = ParamsShadowTechnique::Hard;
    params.map_size = 1024;
    params.cascades = 2;

    let config = params.to_shadow_manager_config();

    assert_eq!(config.csm.cascade_count, 2);
    assert_eq!(config.csm.shadow_map_size, 1024);
    assert_eq!(config.technique, LightingShadowTechnique::Hard);
}

#[test]
fn test_shadow_params_to_manager_config_pcf() {
    // Test PCF shadows
    let mut params = ShadowParams::default();
    params.technique = ParamsShadowTechnique::Pcf;
    params.map_size = 2048;
    params.cascades = 3;

    let config = params.to_shadow_manager_config();

    assert_eq!(config.csm.cascade_count, 3);
    assert_eq!(config.csm.shadow_map_size, 2048);
    assert_eq!(config.technique, LightingShadowTechnique::PCF);
}

#[test]
fn test_shadow_params_to_manager_config_pcss() {
    // Test PCSS shadows with blocker/filter radii
    let mut params = ShadowParams::default();
    params.technique = ParamsShadowTechnique::Pcss;
    params.map_size = 4096;
    params.cascades = 4;
    params.pcss_blocker_radius = 0.05;
    params.pcss_filter_radius = 0.08;
    params.light_size = 0.5;

    let config = params.to_shadow_manager_config();

    assert_eq!(config.csm.cascade_count, 4);
    assert_eq!(config.csm.shadow_map_size, 4096);
    assert_eq!(config.technique, LightingShadowTechnique::PCSS);
    assert_eq!(config.pcss_blocker_radius, 0.05);
    assert_eq!(config.pcss_filter_radius, 0.08);
    assert_eq!(config.light_size, 0.5);
}

#[test]
fn test_shadow_params_to_manager_config_vsm() {
    // Test VSM (Variance Shadow Maps)
    let mut params = ShadowParams::default();
    params.technique = ParamsShadowTechnique::Vsm;
    params.moment_bias = 0.001;

    let config = params.to_shadow_manager_config();

    assert_eq!(config.technique, LightingShadowTechnique::VSM);
    assert_eq!(config.moment_bias, 0.001);
    assert!(!config.csm.enable_evsm); // VSM, not EVSM
}

#[test]
fn test_shadow_params_to_manager_config_evsm() {
    // Test EVSM (Exponential Variance Shadow Maps)
    let mut params = ShadowParams::default();
    params.technique = ParamsShadowTechnique::Evsm;
    params.moment_bias = 0.0005;

    let config = params.to_shadow_manager_config();

    assert_eq!(config.technique, LightingShadowTechnique::EVSM);
    assert_eq!(config.moment_bias, 0.0005);
    assert!(config.csm.enable_evsm); // EVSM enabled
    assert_eq!(config.csm.evsm_positive_exp, 40.0);
    assert_eq!(config.csm.evsm_negative_exp, 40.0);
}

#[test]
fn test_shadow_params_to_manager_config_msm() {
    // Test MSM (Moment Shadow Maps)
    let mut params = ShadowParams::default();
    params.technique = ParamsShadowTechnique::Msm;
    params.moment_bias = 0.0002;

    let config = params.to_shadow_manager_config();

    assert_eq!(config.technique, LightingShadowTechnique::MSM);
    assert_eq!(config.moment_bias, 0.0002);
}

#[test]
fn test_shadow_params_to_manager_config_csm_maps_to_pcf() {
    // Test that CSM technique (which is a layout) maps to PCF filtering
    let mut params = ShadowParams::default();
    params.technique = ParamsShadowTechnique::Csm;

    let config = params.to_shadow_manager_config();

    // CSM is a layout, should use PCF for filtering
    assert_eq!(config.technique, LightingShadowTechnique::PCF);
}

#[test]
fn test_shadow_params_cascade_splits_auto_calculated() {
    // Test that cascade splits are auto-calculated (empty vec)
    let params = ShadowParams::default();
    let config = params.to_shadow_manager_config();

    assert!(
        config.csm.cascade_splits.is_empty(),
        "Cascade splits should be empty for auto-calculation"
    );
}

#[test]
fn test_shadow_params_bias_values() {
    // Test that bias values are correctly set
    let params = ShadowParams::default();
    let config = params.to_shadow_manager_config();

    assert_eq!(config.csm.depth_bias, 0.0005);
    assert_eq!(config.csm.slope_bias, 0.001);
    assert_eq!(config.csm.peter_panning_offset, 0.002);
}

#[test]
fn test_shadow_params_stabilization_enabled() {
    // Test that cascade stabilization is enabled by default
    let params = ShadowParams::default();
    let config = params.to_shadow_manager_config();

    assert!(
        config.csm.stabilize_cascades,
        "Cascade stabilization should be enabled"
    );
}

#[test]
fn test_shadow_params_blend_range() {
    // Test cascade blend range
    let params = ShadowParams::default();
    let config = params.to_shadow_manager_config();

    assert_eq!(
        config.csm.cascade_blend_range, 0.1,
        "Blend range should be 0.1 (10%)"
    );
}

#[test]
fn test_shadow_params_memory_budget() {
    // Test memory budget is set correctly
    let params = ShadowParams::default();
    let config = params.to_shadow_manager_config();

    assert_eq!(
        config.max_memory_bytes,
        256 * 1024 * 1024,
        "Memory budget should be 256 MiB"
    );
}

#[test]
fn test_shadow_params_cli_typical_high_quality() {
    // Test typical high-quality CLI args: --shadows=pcss --shadow-map-res=4096 --cascades=4
    let mut params = ShadowParams::default();
    params.technique = ParamsShadowTechnique::Pcss;
    params.map_size = 4096;
    params.cascades = 4;

    let config = params.to_shadow_manager_config();

    assert_eq!(config.technique, LightingShadowTechnique::PCSS);
    assert_eq!(config.csm.shadow_map_size, 4096);
    assert_eq!(config.csm.cascade_count, 4);
}

#[test]
fn test_shadow_params_cli_typical_performance() {
    // Test typical performance CLI args: --shadows=hard --shadow-map-res=1024 --cascades=2
    let mut params = ShadowParams::default();
    params.technique = ParamsShadowTechnique::Hard;
    params.map_size = 1024;
    params.cascades = 2;

    let config = params.to_shadow_manager_config();

    assert_eq!(config.technique, LightingShadowTechnique::Hard);
    assert_eq!(config.csm.shadow_map_size, 1024);
    assert_eq!(config.csm.cascade_count, 2);
}

#[test]
fn test_shadow_params_cli_typical_balanced() {
    // Test typical balanced CLI args: --shadows=pcf --shadow-map-res=2048 --cascades=3
    let mut params = ShadowParams::default();
    params.technique = ParamsShadowTechnique::Pcf;
    params.map_size = 2048;
    params.cascades = 3;

    let config = params.to_shadow_manager_config();

    assert_eq!(config.technique, LightingShadowTechnique::PCF);
    assert_eq!(config.csm.shadow_map_size, 2048);
    assert_eq!(config.csm.cascade_count, 3);
}

#[test]
fn test_shadow_params_all_techniques_convert() {
    // Test that all shadow techniques convert correctly
    let techniques = vec![
        (ParamsShadowTechnique::Hard, LightingShadowTechnique::Hard),
        (ParamsShadowTechnique::Pcf, LightingShadowTechnique::PCF),
        (ParamsShadowTechnique::Pcss, LightingShadowTechnique::PCSS),
        (ParamsShadowTechnique::Vsm, LightingShadowTechnique::VSM),
        (ParamsShadowTechnique::Evsm, LightingShadowTechnique::EVSM),
        (ParamsShadowTechnique::Msm, LightingShadowTechnique::MSM),
        (ParamsShadowTechnique::Csm, LightingShadowTechnique::PCF), // CSM -> PCF
    ];

    for (params_tech, expected_tech) in techniques {
        let mut params = ShadowParams::default();
        params.technique = params_tech;

        let config = params.to_shadow_manager_config();

        assert_eq!(
            config.technique, expected_tech,
            "Technique {:?} should map to {:?}",
            params_tech, expected_tech
        );
    }
}

#[test]
fn test_shadow_params_pcss_parameters_preserved() {
    // Test that PCSS-specific parameters are preserved
    let mut params = ShadowParams::default();
    params.technique = ParamsShadowTechnique::Pcss;
    params.pcss_blocker_radius = 0.1;
    params.pcss_filter_radius = 0.15;
    params.light_size = 1.0;

    let config = params.to_shadow_manager_config();

    assert_eq!(config.pcss_blocker_radius, 0.1);
    assert_eq!(config.pcss_filter_radius, 0.15);
    assert_eq!(config.light_size, 1.0);
}

#[test]
fn test_shadow_params_moment_parameters_preserved() {
    // Test that moment-based technique parameters are preserved
    let mut params = ShadowParams::default();
    params.technique = ParamsShadowTechnique::Evsm;
    params.moment_bias = 0.0008;

    let config = params.to_shadow_manager_config();

    assert_eq!(config.moment_bias, 0.0008);
}

#[test]
fn test_shadow_params_cascade_count_range() {
    // Test various cascade counts
    for count in 1..=4 {
        let mut params = ShadowParams::default();
        params.cascades = count;

        let config = params.to_shadow_manager_config();

        assert_eq!(
            config.csm.cascade_count, count,
            "Cascade count should be preserved"
        );
    }
}

#[test]
fn test_shadow_params_map_size_powers_of_two() {
    // Test various shadow map resolutions (powers of two)
    for power in 8..=13 {
        let size = 1u32 << power; // 256, 512, 1024, 2048, 4096, 8192
        let mut params = ShadowParams::default();
        params.map_size = size;

        let config = params.to_shadow_manager_config();

        assert_eq!(
            config.csm.shadow_map_size, size,
            "Shadow map size should be preserved"
        );
    }
}

#[test]
fn test_shadow_params_debug_mode_disabled() {
    // Test that debug mode is disabled by default
    let params = ShadowParams::default();
    let config = params.to_shadow_manager_config();

    assert_eq!(config.csm.debug_mode, 0, "Debug mode should be disabled");
}

#[test]
fn test_shadow_params_unclipped_depth_disabled() {
    // Test that unclipped depth is disabled by default
    let params = ShadowParams::default();
    let config = params.to_shadow_manager_config();

    assert!(
        !config.csm.enable_unclipped_depth,
        "Unclipped depth should be disabled"
    );
}

#[test]
fn test_shadow_params_max_shadow_distance_default() {
    // Test that max shadow distance has a reasonable default
    let params = ShadowParams::default();
    let config = params.to_shadow_manager_config();

    assert_eq!(
        config.csm.max_shadow_distance, 1000.0,
        "Max shadow distance should be 1000.0"
    );
}

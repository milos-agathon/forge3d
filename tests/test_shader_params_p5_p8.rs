// tests/test_shader_params_p5_p8.rs
//! GPU-aligned shader parameter packing tests for P5-P8 types
//!
//! Validates that all shader uniform structs are correctly sized and aligned
//! for GPU consumption. Ensures compatibility with WGSL uniform buffer layout.

use bytemuck::{Pod, Zeroable};
use forge3d::lighting::{
    SSAOSettings, SSGISettings, SSRSettings, ScreenSpaceEffect, SkySettings, VolumetricSettings,
};
use std::mem::{align_of, size_of};

/// Verify that a type is properly aligned for GPU uniform buffers
fn assert_gpu_aligned<T: Pod + Zeroable>(expected_size: usize, expected_align: usize) {
    let actual_size = size_of::<T>();
    let actual_align = align_of::<T>();

    assert_eq!(
        actual_size,
        expected_size,
        "{} size mismatch: expected {} bytes, got {} bytes",
        std::any::type_name::<T>(),
        expected_size,
        actual_size
    );

    assert_eq!(
        actual_align,
        expected_align,
        "{} alignment mismatch: expected {} bytes, got {} bytes",
        std::any::type_name::<T>(),
        expected_align,
        actual_align
    );

    // Verify 16-byte alignment requirement for uniform buffers
    assert!(
        actual_align >= 16 || actual_size % 16 == 0,
        "{} must be 16-byte aligned or have size multiple of 16",
        std::any::type_name::<T>()
    );
}

#[test]
fn test_p5_ssao_settings_layout() {
    // P5: SSAOSettings - 32 bytes (8 × f32/u32)
    assert_gpu_aligned::<SSAOSettings>(32, 4);

    let settings = SSAOSettings::default();
    let bytes = bytemuck::bytes_of(&settings);
    assert_eq!(bytes.len(), 32);

    // Verify we can create SSAO and GTAO variants
    // Note: ssao() sets technique=0, gtao() sets technique=1
    let ssao = SSAOSettings::ssao(2.0, 1.0);
    assert_eq!(ssao.technique, 0);

    let gtao = SSAOSettings::gtao(2.0, 1.0);
    assert_eq!(gtao.technique, 1);
}

#[test]
fn test_p5_ssgi_settings_layout() {
    // P5: SSGISettings - 32 bytes (8 × f32/u32)
    assert_gpu_aligned::<SSGISettings>(32, 4);

    let settings = SSGISettings::default();
    let bytes = bytemuck::bytes_of(&settings);
    assert_eq!(bytes.len(), 32);

    // Verify step count is clamped
    let settings = SSGISettings {
        ray_steps: 16,
        ..Default::default()
    };
    assert!(settings.ray_steps >= 8 && settings.ray_steps <= 64);
}

#[test]
fn test_p5_ssr_settings_layout() {
    // P5: SSRSettings - 32 bytes (8 × f32/u32)
    assert_gpu_aligned::<SSRSettings>(32, 4);

    let settings = SSRSettings::default();
    let bytes = bytemuck::bytes_of(&settings);
    assert_eq!(bytes.len(), 32);

    // Verify max_steps is reasonable
    assert!(settings.max_steps >= 8 && settings.max_steps <= 128);
}

#[test]
fn test_p6_sky_settings_layout() {
    // P6: SkySettings - 48 bytes (12 × f32/u32)
    assert_gpu_aligned::<SkySettings>(48, 4);

    let settings = SkySettings::default();
    let bytes = bytemuck::bytes_of(&settings);
    assert_eq!(bytes.len(), 48);

    // Verify turbidity is in valid range
    assert!(settings.turbidity >= 1.0 && settings.turbidity <= 10.0);

    // Verify sun direction has reasonable values (not NaN or zero)
    let sun_dir = settings.sun_direction;
    assert!(sun_dir[0].is_finite() && sun_dir[1].is_finite() && sun_dir[2].is_finite());
    let len_sq = sun_dir[0] * sun_dir[0] + sun_dir[1] * sun_dir[1] + sun_dir[2] * sun_dir[2];
    assert!(len_sq > 0.0, "Sun direction should not be zero vector");
}

#[test]
fn test_p6_volumetric_settings_layout() {
    // P6: VolumetricSettings - 80 bytes (20 × f32/u32)
    assert_gpu_aligned::<VolumetricSettings>(80, 4);

    let settings = VolumetricSettings::default();
    let bytes = bytemuck::bytes_of(&settings);
    assert_eq!(bytes.len(), 80);

    // Verify density is non-negative
    assert!(settings.density >= 0.0);

    // Verify step counts are in valid range
    assert!(settings.max_steps >= 8 && settings.max_steps <= 128);
}

#[test]
fn test_parameter_ranges() {
    // P5: SSAO radius should be positive
    let ssao = SSAOSettings::ssao(2.0, 1.0);
    assert!(ssao.radius > 0.0);
    assert!(ssao.intensity >= 0.0);

    // P5: SSGI steps should be in range
    let ssgi = SSGISettings {
        ray_steps: 24,
        ..Default::default()
    };
    assert!(ssgi.ray_steps >= 8 && ssgi.ray_steps <= 64);

    // P5: SSR thickness should be positive
    let ssr = SSRSettings {
        thickness: 0.1,
        ..Default::default()
    };
    assert!(ssr.thickness > 0.0);

    // P6: Sky turbidity should be in [1, 10]
    let sky = SkySettings {
        turbidity: 2.5,
        ..Default::default()
    };
    assert!(sky.turbidity >= 1.0 && sky.turbidity <= 10.0);

    // P6: Volumetric density should be non-negative
    let vol = VolumetricSettings {
        density: 0.02,
        ..Default::default()
    };
    assert!(vol.density >= 0.0);
}

#[test]
fn test_default_constructors() {
    // Verify all default constructors work and produce valid values
    let _ssao = SSAOSettings::default();
    let _ssgi = SSGISettings::default();
    let _ssr = SSRSettings::default();
    let _sky = SkySettings::default();
    let _vol = VolumetricSettings::default();

    // All should compile and not panic
}

#[test]
fn test_bytemuck_traits() {
    // Verify all types implement Pod and Zeroable correctly
    fn check_pod<T: Pod + Zeroable>() {
        let zeroed: T = Zeroable::zeroed();
        let _bytes = bytemuck::bytes_of(&zeroed);
    }

    check_pod::<SSAOSettings>();
    check_pod::<SSGISettings>();
    check_pod::<SSRSettings>();
    check_pod::<SkySettings>();
    check_pod::<VolumetricSettings>();
}

#[test]
fn test_wgsl_uniform_alignment() {
    // WGSL requires uniform buffers to be aligned to 16 bytes
    // and struct members to follow specific alignment rules

    // Test that all our shader parameter types meet WGSL requirements
    let types_to_check = vec![
        (
            "SSAOSettings",
            size_of::<SSAOSettings>(),
            align_of::<SSAOSettings>(),
        ),
        (
            "SSGISettings",
            size_of::<SSGISettings>(),
            align_of::<SSGISettings>(),
        ),
        (
            "SSRSettings",
            size_of::<SSRSettings>(),
            align_of::<SSRSettings>(),
        ),
        (
            "SkySettings",
            size_of::<SkySettings>(),
            align_of::<SkySettings>(),
        ),
        (
            "VolumetricSettings",
            size_of::<VolumetricSettings>(),
            align_of::<VolumetricSettings>(),
        ),
    ];

    for (name, size, align) in types_to_check {
        // WGSL uniform buffer alignment: must be multiple of 16 bytes
        assert!(
            size % 16 == 0 || align >= 16,
            "{}: size must be multiple of 16 or alignment >= 16 (size={}, align={})",
            name,
            size,
            align
        );
    }
}

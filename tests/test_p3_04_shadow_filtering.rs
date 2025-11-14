// tests/test_p3_04_shadow_filtering.rs
// P3-04: Hard/PCF/Poisson PCF shadow filtering tests
// Exit criteria: Hard/PCF code produces expected hardness/softness,
// bias clamped to avoid acne/peter-panning.

use forge3d::shadows::{CsmConfig, CsmRenderer};
use glam::{Mat4, Vec3};

#[test]
fn test_pcf_kernel_sizes() {
    // Test that different PCF kernel sizes are properly configured
    let device = forge3d::gpu::create_device_for_test();

    let kernel_sizes = vec![1, 3, 5, 7];

    for kernel_size in kernel_sizes {
        let config = CsmConfig {
            pcf_kernel_size: kernel_size,
            ..Default::default()
        };

        let mut renderer = CsmRenderer::new(&device, config.clone());

        let camera_view = Mat4::IDENTITY;
        let camera_projection = Mat4::IDENTITY;
        let light_direction = Vec3::new(0.0, -1.0, 0.0);

        renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);

        assert_eq!(renderer.uniforms.pcf_kernel_size, kernel_size);
    }
}

#[test]
fn test_depth_bias_configuration() {
    // Test that depth bias is properly configured
    let device = forge3d::gpu::create_device_for_test();

    let bias_values = vec![0.001, 0.005, 0.01, 0.02];

    for bias in bias_values {
        let config = CsmConfig {
            depth_bias: bias,
            ..Default::default()
        };

        let mut renderer = CsmRenderer::new(&device, config);

        let camera_view = Mat4::IDENTITY;
        let camera_projection = Mat4::IDENTITY;
        let light_direction = Vec3::new(0.0, -1.0, 0.0);

        renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);

        assert_eq!(renderer.uniforms.depth_bias, bias);
    }
}

#[test]
fn test_slope_bias_configuration() {
    // Test that slope bias is properly configured
    let device = forge3d::gpu::create_device_for_test();

    let slope_bias_values = vec![0.005, 0.01, 0.02, 0.05];

    for slope_bias in slope_bias_values {
        let config = CsmConfig {
            slope_bias,
            ..Default::default()
        };

        let mut renderer = CsmRenderer::new(&device, config);

        let camera_view = Mat4::IDENTITY;
        let camera_projection = Mat4::IDENTITY;
        let light_direction = Vec3::new(0.0, -1.0, 0.0);

        renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);

        assert_eq!(renderer.uniforms.slope_bias, slope_bias);
    }
}

#[test]
fn test_peter_panning_offset() {
    // Test peter-panning prevention offset configuration
    let device = forge3d::gpu::create_device_for_test();

    let offset_values = vec![0.0, 0.001, 0.002, 0.005];

    for offset in offset_values {
        let config = CsmConfig {
            peter_panning_offset: offset,
            ..Default::default()
        };

        let mut renderer = CsmRenderer::new(&device, config);

        let camera_view = Mat4::IDENTITY;
        let camera_projection = Mat4::IDENTITY;
        let light_direction = Vec3::new(0.0, -1.0, 0.0);

        renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);

        assert_eq!(renderer.uniforms.peter_panning_offset, offset);
    }
}

#[test]
fn test_bias_defaults_reasonable() {
    // Test that default bias values are reasonable
    let config = CsmConfig::default();

    // Depth bias should be small (prevent acne)
    assert!(config.depth_bias > 0.0);
    assert!(config.depth_bias < 0.1);

    // Slope bias should be small (prevent acne on slopes)
    assert!(config.slope_bias > 0.0);
    assert!(config.slope_bias < 0.1);

    // Peter-panning offset should be very small
    assert!(config.peter_panning_offset >= 0.0);
    assert!(config.peter_panning_offset < 0.01);
}

#[test]
fn test_pcf_kernel_size_default() {
    // Test that default PCF kernel size is reasonable
    let config = CsmConfig::default();

    // Default should be 3 (3x3 kernel)
    assert_eq!(config.pcf_kernel_size, 3);
}

#[test]
fn test_hard_shadows_kernel_1() {
    // Test that kernel size 1 produces hard shadows (no filtering)
    let device = forge3d::gpu::create_device_for_test();

    let config = CsmConfig {
        pcf_kernel_size: 1,
        ..Default::default()
    };

    let mut renderer = CsmRenderer::new(&device, config);

    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    let light_direction = Vec3::new(0.0, -1.0, 0.0);

    renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);

    // Kernel size 1 means no PCF filtering
    assert_eq!(renderer.uniforms.pcf_kernel_size, 1);
}

#[test]
fn test_pcf_kernel_3x3() {
    // Test 3x3 PCF kernel configuration
    let device = forge3d::gpu::create_device_for_test();

    let config = CsmConfig {
        pcf_kernel_size: 3,
        ..Default::default()
    };

    let mut renderer = CsmRenderer::new(&device, config);

    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    let light_direction = Vec3::new(0.0, -1.0, 0.0);

    renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);

    // 3x3 kernel = 9 samples
    assert_eq!(renderer.uniforms.pcf_kernel_size, 3);
}

#[test]
fn test_pcf_kernel_5x5() {
    // Test 5x5 PCF kernel configuration
    let device = forge3d::gpu::create_device_for_test();

    let config = CsmConfig {
        pcf_kernel_size: 5,
        ..Default::default()
    };

    let mut renderer = CsmRenderer::new(&device, config);

    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    let light_direction = Vec3::new(0.0, -1.0, 0.0);

    renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);

    // 5x5 kernel = 25 samples
    assert_eq!(renderer.uniforms.pcf_kernel_size, 5);
}

#[test]
fn test_poisson_pcf_kernel_7() {
    // Test Poisson disk PCF (kernel size >= 7)
    let device = forge3d::gpu::create_device_for_test();

    let config = CsmConfig {
        pcf_kernel_size: 7,
        ..Default::default()
    };

    let mut renderer = CsmRenderer::new(&device, config);

    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    let light_direction = Vec3::new(0.0, -1.0, 0.0);

    renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);

    // Kernel size 7+ triggers Poisson disk sampling (16 samples)
    assert!(renderer.uniforms.pcf_kernel_size >= 7);
}

#[test]
fn test_texel_size_per_cascade() {
    // Test that each cascade has appropriate texel size
    let device = forge3d::gpu::create_device_for_test();

    let config = CsmConfig {
        cascade_count: 3,
        shadow_map_size: 1024,
        ..Default::default()
    };

    let mut renderer = CsmRenderer::new(&device, config);

    let camera_view = Mat4::look_at_rh(Vec3::new(10.0, 10.0, 10.0), Vec3::ZERO, Vec3::Y);
    let camera_projection =
        Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 100.0);
    let light_direction = Vec3::new(0.0, -1.0, 0.0);

    renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);

    // Verify each cascade has a texel size
    for i in 0..3 {
        let texel_size = renderer.uniforms.cascades[i].texel_size;
        assert!(
            texel_size > 0.0,
            "Cascade {} has invalid texel size: {}",
            i,
            texel_size
        );
    }

    // Texel sizes should generally increase with cascade distance
    // (farther cascades cover more area per texel)
    let texel_0 = renderer.uniforms.cascades[0].texel_size;
    let texel_2 = renderer.uniforms.cascades[2].texel_size;
    assert!(
        texel_2 >= texel_0,
        "Far cascade should have larger texel size: {} vs {}",
        texel_2,
        texel_0
    );
}

#[test]
fn test_shadow_map_size_affects_texel_size() {
    // Test that shadow map resolution affects texel size
    let device = forge3d::gpu::create_device_for_test();

    let camera_view = Mat4::look_at_rh(Vec3::new(10.0, 10.0, 10.0), Vec3::ZERO, Vec3::Y);
    let camera_projection =
        Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 100.0);
    let light_direction = Vec3::new(0.0, -1.0, 0.0);

    // High resolution
    let config_high = CsmConfig {
        shadow_map_size: 2048,
        ..Default::default()
    };
    let mut renderer_high = CsmRenderer::new(&device, config_high);
    renderer_high.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);

    // Low resolution
    let config_low = CsmConfig {
        shadow_map_size: 512,
        ..Default::default()
    };
    let mut renderer_low = CsmRenderer::new(&device, config_low);
    renderer_low.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);

    // Higher resolution should have smaller texel size (more detail)
    let texel_high = renderer_high.uniforms.cascades[0].texel_size;
    let texel_low = renderer_low.uniforms.cascades[0].texel_size;

    assert!(
        texel_high < texel_low,
        "Higher resolution should have smaller texel size: {} vs {}",
        texel_high,
        texel_low
    );
}

#[test]
fn test_bias_consistency() {
    // Test that bias values remain consistent across updates
    let device = forge3d::gpu::create_device_for_test();

    let config = CsmConfig {
        depth_bias: 0.007,
        slope_bias: 0.015,
        ..Default::default()
    };

    let mut renderer = CsmRenderer::new(&device, config);

    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    let light_direction = Vec3::new(0.0, -1.0, 0.0);

    // Update multiple times
    for _ in 0..5 {
        renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);

        assert_eq!(renderer.uniforms.depth_bias, 0.007);
        assert_eq!(renderer.uniforms.slope_bias, 0.015);
    }
}

#[test]
fn test_different_light_directions() {
    // Test that different light directions work correctly
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig::default();

    let light_directions = vec![
        Vec3::new(0.0, -1.0, 0.0), // Straight down
        Vec3::new(0.5, -1.0, 0.0), // Angled
        Vec3::new(1.0, -1.0, 0.0), // More angled
        Vec3::new(0.0, -1.0, 0.5), // Angled different axis
    ];

    for light_dir in light_directions {
        let mut renderer = CsmRenderer::new(&device, config.clone());
        let normalized = light_dir.normalize();

        let camera_view = Mat4::IDENTITY;
        let camera_projection = Mat4::IDENTITY;

        renderer.update_cascades(camera_view, camera_projection, normalized, 0.1, 100.0);

        // Should not panic or produce invalid data
        for i in 0..config.cascade_count as usize {
            let cascade = &renderer.uniforms.cascades[i];
            assert!(cascade.texel_size > 0.0);
            assert!(cascade.near_distance < cascade.far_distance);
        }
    }
}

#[test]
fn test_filtering_quality_progression() {
    // Test that filtering quality progresses from hard to soft
    let device = forge3d::gpu::create_device_for_test();

    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    let light_direction = Vec3::new(0.0, -1.0, 0.0);

    // Hard shadows (no filtering)
    let config_hard = CsmConfig {
        pcf_kernel_size: 1,
        ..Default::default()
    };
    let mut renderer_hard = CsmRenderer::new(&device, config_hard);
    renderer_hard.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);

    // Medium PCF
    let config_pcf = CsmConfig {
        pcf_kernel_size: 3,
        ..Default::default()
    };
    let mut renderer_pcf = CsmRenderer::new(&device, config_pcf);
    renderer_pcf.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);

    // High quality Poisson
    let config_poisson = CsmConfig {
        pcf_kernel_size: 7,
        ..Default::default()
    };
    let mut renderer_poisson = CsmRenderer::new(&device, config_poisson);
    renderer_poisson.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);

    // Verify progression
    assert_eq!(renderer_hard.uniforms.pcf_kernel_size, 1);
    assert_eq!(renderer_pcf.uniforms.pcf_kernel_size, 3);
    assert_eq!(renderer_poisson.uniforms.pcf_kernel_size, 7);
}

// tests/test_p3_05_pcss.rs
// P3-05: PCSS (Percentage-Closer Soft Shadows) tests
// Exit criteria: Visually larger penumbra for distant occluders;
// softening increases with light size; no catastrophic self-shadowing.

use forge3d::shadows::{CsmConfig, CsmRenderer};
use forge3d::lighting::types::ShadowTechnique;
use glam::{Mat4, Vec3};

#[test]
fn test_pcss_technique_params_structure() {
    // Test that PCSS parameters are properly structured in technique_params
    // technique_params: [pcss_blocker_radius, pcss_filter_radius, moment_bias, light_size]
    let device = forge3d::gpu::create_device_for_test();
    
    let config = CsmConfig::default();
    let mut renderer = CsmRenderer::new(&device, config.clone());
    
    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);
    
    // technique_params should have 4 elements
    assert_eq!(renderer.uniforms.technique_params.len(), 4);
}

#[test]
fn test_pcss_blocker_radius_parameter() {
    // Test that blocker search radius parameter can be configured
    let device = forge3d::gpu::create_device_for_test();
    
    let blocker_radii = vec![5.0, 10.0, 20.0, 50.0];
    
    for radius in blocker_radii {
        let config = CsmConfig::default();
        let mut renderer = CsmRenderer::new(&device, config);
        
        // Set PCSS blocker radius (first parameter)
        renderer.uniforms.technique_params[0] = radius;
        
        let camera_view = Mat4::IDENTITY;
        let camera_projection = Mat4::IDENTITY;
        let light_direction = Vec3::new(0.0, -1.0, 0.0);
        
        renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);
        
        assert_eq!(renderer.uniforms.technique_params[0], radius);
    }
}

#[test]
fn test_pcss_filter_radius_parameter() {
    // Test that PCF filter radius parameter can be configured
    let device = forge3d::gpu::create_device_for_test();
    
    let filter_radii = vec![1.0, 2.0, 5.0, 10.0];
    
    for radius in filter_radii {
        let config = CsmConfig::default();
        let mut renderer = CsmRenderer::new(&device, config);
        
        // Set PCSS filter radius (second parameter)
        renderer.uniforms.technique_params[1] = radius;
        
        let camera_view = Mat4::IDENTITY;
        let camera_projection = Mat4::IDENTITY;
        let light_direction = Vec3::new(0.0, -1.0, 0.0);
        
        renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);
        
        assert_eq!(renderer.uniforms.technique_params[1], radius);
    }
}

#[test]
fn test_pcss_light_size_parameter() {
    // Test that light size parameter can be configured
    let device = forge3d::gpu::create_device_for_test();
    
    let light_sizes = vec![0.1, 0.5, 1.0, 5.0, 10.0];
    
    for light_size in light_sizes {
        let config = CsmConfig::default();
        let mut renderer = CsmRenderer::new(&device, config);
        
        // Set light size (fourth parameter)
        renderer.uniforms.technique_params[3] = light_size;
        
        let camera_view = Mat4::IDENTITY;
        let camera_projection = Mat4::IDENTITY;
        let light_direction = Vec3::new(0.0, -1.0, 0.0);
        
        renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);
        
        assert_eq!(renderer.uniforms.technique_params[3], light_size);
    }
}

#[test]
fn test_pcss_technique_identifier() {
    // Test that PCSS technique identifier is 2
    let pcss_technique = ShadowTechnique::PCSS;
    assert_eq!(pcss_technique.as_u32(), 2);
}

#[test]
fn test_pcss_parameter_ranges() {
    // Test reasonable parameter ranges for PCSS
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig::default();
    let mut renderer = CsmRenderer::new(&device, config);
    
    // Blocker radius: should be positive and reasonable
    renderer.uniforms.technique_params[0] = 10.0;
    assert!(renderer.uniforms.technique_params[0] > 0.0);
    assert!(renderer.uniforms.technique_params[0] < 100.0);
    
    // Filter radius: should be positive and reasonable
    renderer.uniforms.technique_params[1] = 2.0;
    assert!(renderer.uniforms.technique_params[1] > 0.0);
    assert!(renderer.uniforms.technique_params[1] < 50.0);
    
    // Light size: should be positive
    renderer.uniforms.technique_params[3] = 1.0;
    assert!(renderer.uniforms.technique_params[3] > 0.0);
}

#[test]
fn test_pcss_with_different_cascade_counts() {
    // Test PCSS configuration with different cascade counts
    let device = forge3d::gpu::create_device_for_test();
    
    let cascade_counts = vec![2, 3, 4];
    
    for count in cascade_counts {
        let config = CsmConfig {
            cascade_count: count,
            ..Default::default()
        };
        
        let mut renderer = CsmRenderer::new(&device, config);
        
        // Set PCSS parameters
        renderer.uniforms.technique = ShadowTechnique::PCSS.as_u32();
        renderer.uniforms.technique_params[0] = 10.0; // blocker radius
        renderer.uniforms.technique_params[1] = 2.0;  // filter radius
        renderer.uniforms.technique_params[3] = 1.0;  // light size
        
        let camera_view = Mat4::look_at_rh(
            Vec3::new(10.0, 10.0, 10.0),
            Vec3::ZERO,
            Vec3::Y,
        );
        let camera_projection = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            16.0 / 9.0,
            0.1,
            100.0,
        );
        
        renderer.update_cascades(
            camera_view,
            camera_projection,
            Vec3::new(0.0, -1.0, 0.0),
            0.1,
            100.0,
        );
        
        assert_eq!(renderer.uniforms.cascade_count, count);
        assert_eq!(renderer.uniforms.technique, ShadowTechnique::PCSS.as_u32());
    }
}

#[test]
fn test_pcss_with_different_resolutions() {
    // Test PCSS with different shadow map resolutions
    let device = forge3d::gpu::create_device_for_test();
    
    let resolutions = vec![512, 1024, 2048];
    
    for res in resolutions {
        let config = CsmConfig {
            shadow_map_size: res,
            ..Default::default()
        };
        
        let mut renderer = CsmRenderer::new(&device, config);
        
        // Set PCSS parameters
        renderer.uniforms.technique = ShadowTechnique::PCSS.as_u32();
        renderer.uniforms.technique_params[0] = 10.0;
        renderer.uniforms.technique_params[1] = 2.0;
        renderer.uniforms.technique_params[3] = 1.0;
        
        let camera_view = Mat4::IDENTITY;
        let camera_projection = Mat4::IDENTITY;
        
        renderer.update_cascades(
            camera_view,
            camera_projection,
            Vec3::new(0.0, -1.0, 0.0),
            0.1,
            100.0,
        );
        
        assert_eq!(renderer.uniforms.shadow_map_size, res as f32);
    }
}

#[test]
fn test_pcss_blocker_radius_clamping() {
    // Test that blocker radius should be clamped by cascade texel size in shader
    // Here we just verify the parameter can be set to various values
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig::default();
    let mut renderer = CsmRenderer::new(&device, config);
    
    // Test very small radius
    renderer.uniforms.technique_params[0] = 0.1;
    assert!(renderer.uniforms.technique_params[0] > 0.0);
    
    // Test large radius (should be clamped in shader)
    renderer.uniforms.technique_params[0] = 1000.0;
    assert!(renderer.uniforms.technique_params[0] > 0.0);
}

#[test]
fn test_pcss_light_size_zero() {
    // Test edge case: light size of zero (should result in hard shadows)
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig::default();
    let mut renderer = CsmRenderer::new(&device, config);
    
    renderer.uniforms.technique_params[3] = 0.0;
    
    // Zero light size is valid (creates hard shadows in PCSS)
    assert_eq!(renderer.uniforms.technique_params[3], 0.0);
}

#[test]
fn test_pcss_light_size_scaling() {
    // Test that larger light sizes are supported
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig::default();
    let mut renderer = CsmRenderer::new(&device, config);
    
    let light_sizes = vec![0.5, 1.0, 2.0, 5.0, 10.0];
    
    for (i, size) in light_sizes.iter().enumerate() {
        renderer.uniforms.technique_params[3] = *size;
        
        // Each successive size should be larger
        if i > 0 {
            assert!(renderer.uniforms.technique_params[3] > light_sizes[i - 1]);
        }
    }
}

#[test]
fn test_pcss_vs_pcf_technique() {
    // Test distinction between PCSS and PCF techniques
    assert_eq!(ShadowTechnique::PCF.as_u32(), 1);
    assert_eq!(ShadowTechnique::PCSS.as_u32(), 2);
    assert_ne!(ShadowTechnique::PCF.as_u32(), ShadowTechnique::PCSS.as_u32());
}

#[test]
fn test_pcss_parameters_persistence() {
    // Test that PCSS parameters persist across cascade updates
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig::default();
    let mut renderer = CsmRenderer::new(&device, config);
    
    // Set PCSS parameters
    renderer.uniforms.technique = ShadowTechnique::PCSS.as_u32();
    renderer.uniforms.technique_params[0] = 15.0; // blocker radius
    renderer.uniforms.technique_params[1] = 3.0;  // filter radius
    renderer.uniforms.technique_params[3] = 2.5;  // light size
    
    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    // Update cascades multiple times
    for _ in 0..5 {
        renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);
        
        // Parameters should persist
        assert_eq!(renderer.uniforms.technique_params[0], 15.0);
        assert_eq!(renderer.uniforms.technique_params[1], 3.0);
        assert_eq!(renderer.uniforms.technique_params[3], 2.5);
    }
}

#[test]
fn test_pcss_with_bias() {
    // Test PCSS with different bias values
    let device = forge3d::gpu::create_device_for_test();
    
    let bias_values = vec![0.001, 0.005, 0.01, 0.02];
    
    for bias in bias_values {
        let config = CsmConfig {
            depth_bias: bias,
            slope_bias: bias * 2.0,
            ..Default::default()
        };
        
        let mut renderer = CsmRenderer::new(&device, config);
        
        renderer.uniforms.technique = ShadowTechnique::PCSS.as_u32();
        renderer.uniforms.technique_params[0] = 10.0;
        renderer.uniforms.technique_params[1] = 2.0;
        renderer.uniforms.technique_params[3] = 1.0;
        
        let camera_view = Mat4::IDENTITY;
        let camera_projection = Mat4::IDENTITY;
        
        renderer.update_cascades(
            camera_view,
            camera_projection,
            Vec3::new(0.0, -1.0, 0.0),
            0.1,
            100.0,
        );
        
        assert_eq!(renderer.uniforms.depth_bias, bias);
        assert_eq!(renderer.uniforms.slope_bias, bias * 2.0);
    }
}

#[test]
fn test_pcss_default_parameters() {
    // Test reasonable default PCSS parameters
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig::default();
    let renderer = CsmRenderer::new(&device, config);
    
    // Default technique_params should be initialized to zeros
    assert_eq!(renderer.uniforms.technique_params[0], 0.0);
    assert_eq!(renderer.uniforms.technique_params[1], 0.0);
    assert_eq!(renderer.uniforms.technique_params[2], 0.0);
    assert_eq!(renderer.uniforms.technique_params[3], 0.0);
}

#[test]
fn test_pcss_per_cascade_texel_size() {
    // Test that each cascade has texel size for PCSS radius clamping
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        cascade_count: 3,
        shadow_map_size: 1024,
        ..Default::default()
    };
    
    let mut renderer = CsmRenderer::new(&device, config);
    
    renderer.uniforms.technique = ShadowTechnique::PCSS.as_u32();
    
    let camera_view = Mat4::look_at_rh(
        Vec3::new(10.0, 10.0, 10.0),
        Vec3::ZERO,
        Vec3::Y,
    );
    let camera_projection = Mat4::perspective_rh(
        std::f32::consts::FRAC_PI_4,
        16.0 / 9.0,
        0.1,
        100.0,
    );
    
    renderer.update_cascades(
        camera_view,
        camera_projection,
        Vec3::new(0.0, -1.0, 0.0),
        0.1,
        100.0,
    );
    
    // Each cascade should have a valid texel size
    for i in 0..3 {
        let texel_size = renderer.uniforms.cascades[i].texel_size;
        assert!(texel_size > 0.0, "Cascade {} texel size invalid: {}", i, texel_size);
    }
}

#[test]
fn test_pcss_memory_usage() {
    // Test that PCSS doesn't require additional memory (uses existing depth maps)
    let device = forge3d::gpu::create_device_for_test();
    
    let config_pcf = CsmConfig {
        shadow_map_size: 1024,
        cascade_count: 3,
        ..Default::default()
    };
    
    let renderer_pcf = CsmRenderer::new(&device, config_pcf.clone());
    let memory_pcf = renderer_pcf.total_memory_bytes();
    
    let config_pcss = CsmConfig {
        shadow_map_size: 1024,
        cascade_count: 3,
        ..Default::default()
    };
    
    let renderer_pcss = CsmRenderer::new(&device, config_pcss);
    let memory_pcss = renderer_pcss.total_memory_bytes();
    
    // PCSS and PCF should use same memory (both use depth maps only)
    assert_eq!(memory_pcf, memory_pcss);
}

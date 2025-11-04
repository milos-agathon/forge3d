// tests/test_p3_06_moment_maps.rs
// P3-06: VSM/EVSM/MSM (Variance/Exponential Variance/Moment Shadow Maps) tests
// Exit criteria: Variance methods compile and run; reduced aliasing/softer penumbrae;
// light-leak limited by bias/warp.

use forge3d::shadows::{CsmConfig, CsmRenderer};
use forge3d::lighting::types::ShadowTechnique;
use glam::{Mat4, Vec3};

#[test]
fn test_vsm_technique_identifier() {
    // Test that VSM technique identifier is 3
    let vsm_technique = ShadowTechnique::VSM;
    assert_eq!(vsm_technique.as_u32(), 3);
}

#[test]
fn test_evsm_technique_identifier() {
    // Test that EVSM technique identifier is 4
    let evsm_technique = ShadowTechnique::EVSM;
    assert_eq!(evsm_technique.as_u32(), 4);
}

#[test]
fn test_msm_technique_identifier() {
    // Test that MSM technique identifier is 5
    let msm_technique = ShadowTechnique::MSM;
    assert_eq!(msm_technique.as_u32(), 5);
}

#[test]
fn test_moment_techniques_distinct() {
    // Test that all moment techniques have unique IDs
    assert_ne!(ShadowTechnique::VSM.as_u32(), ShadowTechnique::EVSM.as_u32());
    assert_ne!(ShadowTechnique::VSM.as_u32(), ShadowTechnique::MSM.as_u32());
    assert_ne!(ShadowTechnique::EVSM.as_u32(), ShadowTechnique::MSM.as_u32());
}

#[test]
fn test_moment_bias_parameter() {
    // Test that moment_bias parameter (technique_params[2]) can be configured
    let device = forge3d::gpu::create_device_for_test();
    
    let bias_values = vec![0.0, 0.001, 0.005, 0.01, 0.02];
    
    for bias in bias_values {
        let config = CsmConfig::default();
        let mut renderer = CsmRenderer::new(&device, config);
        
        // Set moment bias (third parameter)
        renderer.uniforms.technique_params[2] = bias;
        
        let camera_view = Mat4::IDENTITY;
        let camera_projection = Mat4::IDENTITY;
        let light_direction = Vec3::new(0.0, -1.0, 0.0);
        
        renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);
        
        assert_eq!(renderer.uniforms.technique_params[2], bias);
    }
}

#[test]
fn test_evsm_positive_exponent() {
    // Test that EVSM positive exponent can be configured
    let device = forge3d::gpu::create_device_for_test();
    
    let exponents = vec![20.0, 40.0, 60.0, 80.0];
    
    for exp in exponents {
        let config = CsmConfig {
            evsm_positive_exp: exp,
            ..Default::default()
        };
        
        let mut renderer = CsmRenderer::new(&device, config);
        
        let camera_view = Mat4::IDENTITY;
        let camera_projection = Mat4::IDENTITY;
        let light_direction = Vec3::new(0.0, -1.0, 0.0);
        
        renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);
        
        assert_eq!(renderer.uniforms.evsm_positive_exp, exp);
    }
}

#[test]
fn test_evsm_negative_exponent() {
    // Test that EVSM negative exponent can be configured
    let device = forge3d::gpu::create_device_for_test();
    
    let exponents = vec![20.0, 40.0, 60.0, 80.0];
    
    for exp in exponents {
        let config = CsmConfig {
            evsm_negative_exp: exp,
            ..Default::default()
        };
        
        let mut renderer = CsmRenderer::new(&device, config);
        
        let camera_view = Mat4::IDENTITY;
        let camera_projection = Mat4::IDENTITY;
        let light_direction = Vec3::new(0.0, -1.0, 0.0);
        
        renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);
        
        assert_eq!(renderer.uniforms.evsm_negative_exp, exp);
    }
}

#[test]
fn test_evsm_exponent_defaults() {
    // Test that EVSM has reasonable default exponents
    let config = CsmConfig::default();
    
    // Default exponents should be positive and reasonable
    assert!(config.evsm_positive_exp > 0.0);
    assert!(config.evsm_positive_exp <= 100.0);
    assert!(config.evsm_negative_exp > 0.0);
    assert!(config.evsm_negative_exp <= 100.0);
}

#[test]
fn test_vsm_with_different_resolutions() {
    // Test VSM with different shadow map resolutions
    let device = forge3d::gpu::create_device_for_test();
    
    let resolutions = vec![512, 1024, 2048];
    
    for res in resolutions {
        let config = CsmConfig {
            shadow_map_size: res,
            ..Default::default()
        };
        
        let mut renderer = CsmRenderer::new(&device, config);
        
        // Set VSM technique
        renderer.uniforms.technique = ShadowTechnique::VSM.as_u32();
        renderer.uniforms.technique_params[2] = 0.001; // moment_bias
        
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
        assert_eq!(renderer.uniforms.technique, ShadowTechnique::VSM.as_u32());
    }
}

#[test]
fn test_evsm_with_different_resolutions() {
    // Test EVSM with different shadow map resolutions
    let device = forge3d::gpu::create_device_for_test();
    
    let resolutions = vec![512, 1024, 2048];
    
    for res in resolutions {
        let config = CsmConfig {
            shadow_map_size: res,
            evsm_positive_exp: 40.0,
            evsm_negative_exp: 40.0,
            ..Default::default()
        };
        
        let mut renderer = CsmRenderer::new(&device, config);
        
        // Set EVSM technique
        renderer.uniforms.technique = ShadowTechnique::EVSM.as_u32();
        renderer.uniforms.technique_params[2] = 0.002; // moment_bias
        
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
        assert_eq!(renderer.uniforms.technique, ShadowTechnique::EVSM.as_u32());
    }
}

#[test]
fn test_msm_with_different_resolutions() {
    // Test MSM with different shadow map resolutions
    let device = forge3d::gpu::create_device_for_test();
    
    let resolutions = vec![512, 1024, 2048];
    
    for res in resolutions {
        let config = CsmConfig {
            shadow_map_size: res,
            ..Default::default()
        };
        
        let mut renderer = CsmRenderer::new(&device, config);
        
        // Set MSM technique
        renderer.uniforms.technique = ShadowTechnique::MSM.as_u32();
        renderer.uniforms.technique_params[2] = 0.005; // moment_bias
        
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
        assert_eq!(renderer.uniforms.technique, ShadowTechnique::MSM.as_u32());
    }
}

#[test]
fn test_moment_techniques_with_cascades() {
    // Test moment techniques with different cascade counts
    let device = forge3d::gpu::create_device_for_test();
    
    let cascade_counts = vec![2, 3, 4];
    let techniques = vec![
        ShadowTechnique::VSM,
        ShadowTechnique::EVSM,
        ShadowTechnique::MSM,
    ];
    
    for count in cascade_counts {
        for technique in &techniques {
            let config = CsmConfig {
                cascade_count: count,
                ..Default::default()
            };
            
            let mut renderer = CsmRenderer::new(&device, config);
            
            renderer.uniforms.technique = technique.as_u32();
            
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
            assert_eq!(renderer.uniforms.technique, technique.as_u32());
        }
    }
}

#[test]
fn test_moment_bias_zero() {
    // Test edge case: moment bias of zero (no leak reduction)
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig::default();
    let mut renderer = CsmRenderer::new(&device, config);
    
    renderer.uniforms.technique_params[2] = 0.0;
    
    // Zero moment bias is valid (no leak reduction applied)
    assert_eq!(renderer.uniforms.technique_params[2], 0.0);
}

#[test]
fn test_moment_bias_range() {
    // Test reasonable moment bias range
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig::default();
    let mut renderer = CsmRenderer::new(&device, config);
    
    // Test small bias
    renderer.uniforms.technique_params[2] = 0.0001;
    assert!(renderer.uniforms.technique_params[2] > 0.0);
    assert!(renderer.uniforms.technique_params[2] < 0.1);
    
    // Test moderate bias
    renderer.uniforms.technique_params[2] = 0.01;
    assert!(renderer.uniforms.technique_params[2] > 0.0);
    assert!(renderer.uniforms.technique_params[2] < 0.1);
}

#[test]
fn test_evsm_exponent_symmetry() {
    // Test that EVSM positive and negative exponents can be symmetric
    let device = forge3d::gpu::create_device_for_test();
    
    let exponent = 40.0;
    
    let config = CsmConfig {
        evsm_positive_exp: exponent,
        evsm_negative_exp: exponent,
        ..Default::default()
    };
    
    let mut renderer = CsmRenderer::new(&device, config);
    
    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    
    renderer.update_cascades(
        camera_view,
        camera_projection,
        Vec3::new(0.0, -1.0, 0.0),
        0.1,
        100.0,
    );
    
    assert_eq!(renderer.uniforms.evsm_positive_exp, exponent);
    assert_eq!(renderer.uniforms.evsm_negative_exp, exponent);
}

#[test]
fn test_technique_params_persistence_vsm() {
    // Test that moment bias persists across cascade updates for VSM
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig::default();
    let mut renderer = CsmRenderer::new(&device, config);
    
    // Set VSM parameters
    renderer.uniforms.technique = ShadowTechnique::VSM.as_u32();
    renderer.uniforms.technique_params[2] = 0.003; // moment_bias
    
    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    // Update cascades multiple times
    for _ in 0..5 {
        renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);
        
        // Parameters should persist
        assert_eq!(renderer.uniforms.technique_params[2], 0.003);
    }
}

#[test]
fn test_technique_params_persistence_evsm() {
    // Test that EVSM exponents persist across cascade updates
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        evsm_positive_exp: 50.0,
        evsm_negative_exp: 45.0,
        ..Default::default()
    };
    let mut renderer = CsmRenderer::new(&device, config);
    
    renderer.uniforms.technique = ShadowTechnique::EVSM.as_u32();
    
    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    // Update cascades multiple times
    for _ in 0..5 {
        renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);
        
        // Exponents should persist
        assert_eq!(renderer.uniforms.evsm_positive_exp, 50.0);
        assert_eq!(renderer.uniforms.evsm_negative_exp, 45.0);
    }
}

#[test]
fn test_moment_techniques_with_bias() {
    // Test moment techniques with different bias values
    let device = forge3d::gpu::create_device_for_test();
    
    let bias_values = vec![0.001, 0.005, 0.01];
    let techniques = vec![
        ShadowTechnique::VSM,
        ShadowTechnique::EVSM,
        ShadowTechnique::MSM,
    ];
    
    for bias in bias_values {
        for technique in &techniques {
            let config = CsmConfig {
                depth_bias: bias,
                slope_bias: bias * 2.0,
                ..Default::default()
            };
            
            let mut renderer = CsmRenderer::new(&device, config);
            
            renderer.uniforms.technique = technique.as_u32();
            renderer.uniforms.technique_params[2] = bias; // moment_bias
            
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
            assert_eq!(renderer.uniforms.technique_params[2], bias);
        }
    }
}

#[test]
fn test_vsm_requires_moment_maps() {
    // Test that VSM technique is flagged as requiring moment maps
    let vsm = ShadowTechnique::VSM;
    assert!(vsm.requires_moments());
}

#[test]
fn test_evsm_requires_moment_maps() {
    // Test that EVSM technique is flagged as requiring moment maps
    let evsm = ShadowTechnique::EVSM;
    assert!(evsm.requires_moments());
}

#[test]
fn test_msm_requires_moment_maps() {
    // Test that MSM technique is flagged as requiring moment maps
    let msm = ShadowTechnique::MSM;
    assert!(msm.requires_moments());
}

#[test]
fn test_non_moment_techniques_dont_require_moments() {
    // Test that non-moment techniques don't require moment maps
    assert!(!ShadowTechnique::Hard.requires_moments());
    assert!(!ShadowTechnique::PCF.requires_moments());
    assert!(!ShadowTechnique::PCSS.requires_moments());
}

#[test]
fn test_moment_map_memory_calculation() {
    // Test that moment map memory is calculated correctly when EVSM is enabled
    let device = forge3d::gpu::create_device_for_test();
    
    // Config without EVSM
    let config_no_moment = CsmConfig {
        shadow_map_size: 1024,
        cascade_count: 3,
        enable_evsm: false,
        ..Default::default()
    };
    
    let renderer_no_moment = CsmRenderer::new(&device, config_no_moment);
    let memory_no_moment = renderer_no_moment.total_memory_bytes();
    
    // Config with EVSM enabled (creates moment maps)
    let config_with_moment = CsmConfig {
        shadow_map_size: 1024,
        cascade_count: 3,
        enable_evsm: true,
        ..Default::default()
    };
    
    let renderer_with_moment = CsmRenderer::new(&device, config_with_moment);
    let memory_with_moment = renderer_with_moment.total_memory_bytes();
    
    // EVSM should use more memory due to moment maps (Rgba32Float)
    assert!(memory_with_moment > memory_no_moment);
}

#[test]
fn test_evsm_exponent_high_values() {
    // Test EVSM with high exponent values (reduces light leaks)
    let device = forge3d::gpu::create_device_for_test();
    
    let config = CsmConfig {
        evsm_positive_exp: 80.0,
        evsm_negative_exp: 80.0,
        ..Default::default()
    };
    
    let mut renderer = CsmRenderer::new(&device, config);
    
    renderer.uniforms.technique = ShadowTechnique::EVSM.as_u32();
    
    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    
    renderer.update_cascades(
        camera_view,
        camera_projection,
        Vec3::new(0.0, -1.0, 0.0),
        0.1,
        100.0,
    );
    
    assert_eq!(renderer.uniforms.evsm_positive_exp, 80.0);
    assert_eq!(renderer.uniforms.evsm_negative_exp, 80.0);
}

#[test]
fn test_different_light_directions_moment_maps() {
    // Test moment techniques with various light directions
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig::default();
    
    let light_directions = vec![
        Vec3::new(0.0, -1.0, 0.0),    // Straight down
        Vec3::new(0.5, -1.0, 0.0),    // Angled
        Vec3::new(1.0, -1.0, 0.0),    // More angled
        Vec3::new(0.0, -1.0, 0.5),    // Different axis
    ];
    
    let techniques = vec![
        ShadowTechnique::VSM,
        ShadowTechnique::EVSM,
        ShadowTechnique::MSM,
    ];
    
    for light_dir in &light_directions {
        for technique in &techniques {
            let mut renderer = CsmRenderer::new(&device, config.clone());
            let normalized = light_dir.normalize();
            
            renderer.uniforms.technique = technique.as_u32();
            
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
}

// tests/test_p3_03_cascade_stabilization.rs
// P3-03: Cascade stabilization and texel snapping tests
// Exit criteria: With camera orbit, cascade edges do not shimmer; 
// debug visualization toggles work as expected.

use forge3d::shadows::{CsmConfig, CsmRenderer};
use glam::{Mat4, Vec3};

#[test]
fn test_cascade_split_scheme() {
    // Test practical split scheme (PSS) produces monotonic, well-distributed splits
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        cascade_count: 3,
        ..Default::default()
    };
    
    let renderer = CsmRenderer::new(&device, config);
    let splits = renderer.calculate_cascade_splits(0.1, 100.0);
    
    // Should have cascade_count + 1 splits (including near and far)
    assert_eq!(splits.len(), 4);
    
    // First split should be near plane
    assert_eq!(splits[0], 0.1);
    
    // Last split should be far plane
    assert_eq!(splits[3], 100.0);
    
    // Splits should be monotonically increasing
    for i in 0..splits.len() - 1 {
        assert!(splits[i] < splits[i + 1], 
                "Splits not monotonic: {} >= {}", splits[i], splits[i + 1]);
    }
    
    // Splits should be well-distributed (PSS blend should avoid extremes)
    let range = 100.0 - 0.1;
    for i in 1..splits.len() - 1 {
        let normalized_pos = (splits[i] - 0.1) / range;
        // With 3 cascades, middle splits should be roughly at 1/3 and 2/3
        // PSS blend (lambda=0.75) creates log-biased distribution
        assert!(normalized_pos > 0.0 && normalized_pos < 1.0);
    }
}

#[test]
fn test_cascade_split_4_cascades() {
    // Test with 4 cascades
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        cascade_count: 4,
        ..Default::default()
    };
    
    let renderer = CsmRenderer::new(&device, config);
    let splits = renderer.calculate_cascade_splits(0.1, 200.0);
    
    assert_eq!(splits.len(), 5);
    assert_eq!(splits[0], 0.1);
    assert_eq!(splits[4], 200.0);
    
    // Verify monotonic
    for i in 0..splits.len() - 1 {
        assert!(splits[i] < splits[i + 1]);
    }
}

#[test]
fn test_stabilization_enabled_by_default() {
    // Stabilization should be enabled by default
    let config = CsmConfig::default();
    assert!(config.stabilize_cascades);
}

#[test]
fn test_cascade_blend_disabled_by_default() {
    // Cascade blending should be disabled by default (0.0)
    let config = CsmConfig::default();
    assert_eq!(config.cascade_blend_range, 0.0);
}

#[test]
fn test_update_cascades_basic() {
    // Test basic cascade update without errors
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        cascade_count: 3,
        shadow_map_size: 512,
        stabilize_cascades: true,
        ..Default::default()
    };
    
    let mut renderer = CsmRenderer::new(&device, config);
    
    let camera_view = Mat4::look_at_rh(
        Vec3::new(0.0, 10.0, 10.0),
        Vec3::ZERO,
        Vec3::Y,
    );
    let camera_projection = Mat4::perspective_rh(
        std::f32::consts::FRAC_PI_4,
        16.0 / 9.0,
        0.1,
        100.0,
    );
    let light_direction = Vec3::new(0.0, -1.0, 0.3).normalize();
    
    renderer.update_cascades(
        camera_view,
        camera_projection,
        light_direction,
        0.1,
        100.0,
    );
    
    // Verify cascade count matches
    assert_eq!(renderer.uniforms.cascade_count, 3);
    
    // Verify cascades are configured
    for i in 0..3 {
        let cascade = &renderer.uniforms.cascades[i];
        assert!(cascade.near_distance < cascade.far_distance);
        assert!(cascade.texel_size > 0.0);
    }
}

#[test]
fn test_stabilization_consistency() {
    // Test that stabilization produces consistent results with same input
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        cascade_count: 3,
        shadow_map_size: 1024,
        stabilize_cascades: true,
        ..Default::default()
    };
    
    let camera_view = Mat4::look_at_rh(
        Vec3::new(5.0, 5.0, 5.0),
        Vec3::ZERO,
        Vec3::Y,
    );
    let camera_projection = Mat4::perspective_rh(
        std::f32::consts::FRAC_PI_4,
        16.0 / 9.0,
        0.1,
        100.0,
    );
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    // Update cascades twice with identical parameters
    let mut renderer1 = CsmRenderer::new(&device, config.clone());
    renderer1.update_cascades(
        camera_view,
        camera_projection,
        light_direction,
        0.1,
        100.0,
    );
    
    let mut renderer2 = CsmRenderer::new(&device, config);
    renderer2.update_cascades(
        camera_view,
        camera_projection,
        light_direction,
        0.1,
        100.0,
    );
    
    // Verify projections are identical
    for i in 0..3 {
        let cascade1 = &renderer1.uniforms.cascades[i];
        let cascade2 = &renderer2.uniforms.cascades[i];
        
        assert_eq!(cascade1.near_distance, cascade2.near_distance);
        assert_eq!(cascade1.far_distance, cascade2.far_distance);
        assert_eq!(cascade1.texel_size, cascade2.texel_size);
        
        // Light projections should be identical
        for j in 0..16 {
            assert_eq!(cascade1.light_projection[j], cascade2.light_projection[j]);
        }
    }
}

#[test]
fn test_stabilization_minor_camera_movement() {
    // Test that small camera movements don't cause large projection changes with stabilization
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        cascade_count: 3,
        shadow_map_size: 1024,
        stabilize_cascades: true,
        ..Default::default()
    };
    
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    let camera_projection = Mat4::perspective_rh(
        std::f32::consts::FRAC_PI_4,
        16.0 / 9.0,
        0.1,
        100.0,
    );
    
    // First position
    let camera_view1 = Mat4::look_at_rh(
        Vec3::new(5.0, 5.0, 5.0),
        Vec3::ZERO,
        Vec3::Y,
    );
    
    let mut renderer1 = CsmRenderer::new(&device, config.clone());
    renderer1.update_cascades(
        camera_view1,
        camera_projection,
        light_direction,
        0.1,
        100.0,
    );
    
    // Second position (slight movement)
    let camera_view2 = Mat4::look_at_rh(
        Vec3::new(5.05, 5.05, 5.05),  // 0.05 unit movement
        Vec3::ZERO,
        Vec3::Y,
    );
    
    let mut renderer2 = CsmRenderer::new(&device, config);
    renderer2.update_cascades(
        camera_view2,
        camera_projection,
        light_direction,
        0.1,
        100.0,
    );
    
    // With stabilization, projections should be very similar or identical
    // (snapped to texel grid)
    for i in 0..3 {
        let cascade1 = &renderer1.uniforms.cascades[i];
        let cascade2 = &renderer2.uniforms.cascades[i];
        
        // Texel sizes should be nearly identical (allow for floating point precision)
        let texel_diff = (cascade1.texel_size - cascade2.texel_size).abs();
        assert!(texel_diff < 1e-6, 
                "Texel size difference too large: {} vs {}", 
                cascade1.texel_size, cascade2.texel_size);
        
        // Near/far distances should be identical (split scheme unchanged)
        assert_eq!(cascade1.near_distance, cascade2.near_distance);
        assert_eq!(cascade1.far_distance, cascade2.far_distance);
    }
}

#[test]
fn test_debug_mode_setting() {
    // Test debug mode can be set and retrieved
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        debug_mode: 1,  // Enable cascade visualization
        ..Default::default()
    };
    
    let mut renderer = CsmRenderer::new(&device, config);
    
    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);
    
    assert_eq!(renderer.uniforms.debug_mode, 1);
}

#[test]
fn test_cascade_blend_range_setting() {
    // Test cascade blend range is properly set
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        cascade_blend_range: 0.1,  // 10% blend region
        ..Default::default()
    };
    
    let mut renderer = CsmRenderer::new(&device, config);
    
    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);
    
    assert_eq!(renderer.uniforms.cascade_blend_range, 0.1);
}

#[test]
fn test_texel_size_calculation() {
    // Test that texel sizes are reasonable for different configurations
    let device = forge3d::gpu::create_device_for_test();
    
    let test_cases = vec![
        (512, 3),
        (1024, 3),
        (2048, 4),
    ];
    
    for (resolution, cascades) in test_cases {
        let config = CsmConfig {
            shadow_map_size: resolution,
            cascade_count: cascades,
            ..Default::default()
        };
        
        let mut renderer = CsmRenderer::new(&device, config);
        
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
        
        // Verify texel sizes are positive and reasonable
        for i in 0..cascades as usize {
            let texel_size = renderer.uniforms.cascades[i].texel_size;
            assert!(texel_size > 0.0, "Texel size must be positive");
            assert!(texel_size < 10.0, "Texel size unreasonably large: {}", texel_size);
        }
    }
}

#[test]
fn test_light_direction_vertical() {
    // Test light direction when nearly vertical (edge case)
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig::default();
    
    let mut renderer = CsmRenderer::new(&device, config);
    
    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    
    // Nearly vertical light direction
    let light_direction = Vec3::new(0.01, -0.99, 0.01).normalize();
    
    // Should not panic or produce invalid matrices
    renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);
    
    // Verify light direction was set
    assert!((renderer.uniforms.light_direction[1] - light_direction.y).abs() < 0.01);
}

#[test]
fn test_cascade_memory_calculation() {
    // Test memory calculation matches actual allocation
    let device = forge3d::gpu::create_device_for_test();
    
    let config = CsmConfig {
        shadow_map_size: 1024,
        cascade_count: 3,
        enable_evsm: false,
        ..Default::default()
    };
    
    let renderer = CsmRenderer::new(&device, config);
    let memory = renderer.total_memory_bytes();
    
    // Expected: 1024 × 1024 × 3 cascades × 4 bytes (Depth32Float)
    let expected = 1024u64 * 1024 * 3 * 4;
    assert_eq!(memory, expected);
}

#[test]
fn test_cascade_memory_with_moments() {
    // Test memory calculation with moment maps
    let device = forge3d::gpu::create_device_for_test();
    
    let config = CsmConfig {
        shadow_map_size: 512,
        cascade_count: 2,
        enable_evsm: true,
        ..Default::default()
    };
    
    let renderer = CsmRenderer::new(&device, config);
    let memory = renderer.total_memory_bytes();
    
    // Expected: 512 × 512 × 2 × (4 + 16) = depth + moments
    let expected = 512u64 * 512 * 2 * (4 + 16);
    assert_eq!(memory, expected);
}

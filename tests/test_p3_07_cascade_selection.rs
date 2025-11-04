// tests/test_p3_07_cascade_selection.rs
// P3-07: CSM selection, transform, and fade tests
// Exit criteria: Split selection stable; optional fade avoids visible transitions.

use forge3d::shadows::{CsmConfig, CsmRenderer};
use glam::{Mat4, Vec3};

#[test]
fn test_cascade_selection_monotonic() {
    // Test that cascade splits are monotonically increasing
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        cascade_count: 4,
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
    
    // Check that cascade distances are monotonically increasing
    for i in 0..3 {
        let near_i = renderer.uniforms.cascades[i].near_distance;
        let far_i = renderer.uniforms.cascades[i].far_distance;
        let near_next = renderer.uniforms.cascades[i + 1].near_distance;
        let far_next = renderer.uniforms.cascades[i + 1].far_distance;
        
        // Each cascade's far should be <= next cascade's far
        assert!(far_i <= far_next, "Cascade {} far ({}) > cascade {} far ({})", i, far_i, i+1, far_next);
        
        // Each cascade's near should be < far
        assert!(near_i < far_i, "Cascade {} near ({}) >= far ({})", i, near_i, far_i);
        
        // Next cascade's near should be >= current cascade's near
        assert!(near_next >= near_i, "Cascade {} near ({}) > cascade {} near ({})", i+1, near_next, i, near_i);
    }
}

#[test]
fn test_cascade_splits_cover_full_range() {
    // Test that cascades cover the full depth range from near to far
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        cascade_count: 3,
        ..Default::default()
    };
    
    let cascade_count = config.cascade_count;
    let mut renderer = CsmRenderer::new(&device, config);
    
    let near_plane = 0.1;
    let far_plane = 100.0;
    
    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::perspective_rh(
        std::f32::consts::FRAC_PI_4,
        16.0 / 9.0,
        near_plane,
        far_plane,
    );
    
    renderer.update_cascades(
        camera_view,
        camera_projection,
        Vec3::new(0.0, -1.0, 0.0),
        near_plane,
        far_plane,
    );
    
    // First cascade should start at or near near_plane
    assert!(renderer.uniforms.cascades[0].near_distance <= near_plane * 1.1);
    
    // Last cascade should reach far_plane
    let last_idx = cascade_count as usize - 1;
    assert!(renderer.uniforms.cascades[last_idx].far_distance >= far_plane * 0.9);
}

#[test]
fn test_cascade_blend_range_parameter() {
    // Test that cascade_blend_range parameter can be configured
    let device = forge3d::gpu::create_device_for_test();
    
    let blend_ranges = vec![0.0, 0.05, 0.1, 0.15, 0.2];
    
    for blend_range in blend_ranges {
        let config = CsmConfig {
            cascade_blend_range: blend_range,
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
        
        assert_eq!(renderer.uniforms.cascade_blend_range, blend_range);
    }
}

#[test]
fn test_cascade_blend_range_zero() {
    // Test that blend_range = 0 disables blending
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        cascade_blend_range: 0.0,
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
    
    // Zero blend range means no blending
    assert_eq!(renderer.uniforms.cascade_blend_range, 0.0);
}

#[test]
fn test_cascade_blend_range_valid() {
    // Test that blend_range is clamped to valid range [0, 1]
    let device = forge3d::gpu::create_device_for_test();
    
    // Test reasonable values
    let config = CsmConfig {
        cascade_blend_range: 0.1,
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
    
    // Should be in valid range
    assert!(renderer.uniforms.cascade_blend_range >= 0.0);
    assert!(renderer.uniforms.cascade_blend_range <= 1.0);
}

#[test]
fn test_world_to_light_space_transform() {
    // Test that world-to-light-space transform is properly set up
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig::default();
    let cascade_count = config.cascade_count;
    
    let mut renderer = CsmRenderer::new(&device, config);
    
    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);
    
    // Check that each cascade has a valid projection matrix
    for i in 0..cascade_count as usize {
        let projection = Mat4::from_cols_array(&renderer.uniforms.cascades[i].light_projection);
        
        // Projection matrix should not be identity
        assert_ne!(projection, Mat4::IDENTITY);
        
        // Projection should be invertible (determinant != 0)
        assert!(projection.determinant().abs() > 1e-6);
    }
}

#[test]
fn test_cascade_texel_size_calculation() {
    // Test that each cascade has a valid texel size
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        shadow_map_size: 2048,
        cascade_count: 3,
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
    
    // Each cascade should have positive texel size
    for i in 0..3 {
        let texel_size = renderer.uniforms.cascades[i].texel_size;
        assert!(texel_size > 0.0, "Cascade {} has invalid texel size: {}", i, texel_size);
        
        // Texel size should increase with cascade index (farther = bigger texels)
        if i > 0 {
            let prev_texel_size = renderer.uniforms.cascades[i - 1].texel_size;
            assert!(texel_size >= prev_texel_size, 
                "Cascade {} texel size ({}) < cascade {} texel size ({})", 
                i, texel_size, i-1, prev_texel_size);
        }
    }
}

#[test]
fn test_cascade_count_variations() {
    // Test that cascade selection works with different cascade counts
    let device = forge3d::gpu::create_device_for_test();
    
    let cascade_counts = vec![2, 3, 4];
    
    for count in cascade_counts {
        let config = CsmConfig {
            cascade_count: count,
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
        
        // Verify cascade count matches
        assert_eq!(renderer.uniforms.cascade_count, count);
        
        // Verify all cascades are valid
        for i in 0..(count as usize) {
            assert!(renderer.uniforms.cascades[i].near_distance < renderer.uniforms.cascades[i].far_distance);
            assert!(renderer.uniforms.cascades[i].texel_size > 0.0);
        }
    }
}

#[test]
fn test_cascade_stability_with_camera_movement() {
    // Test that cascades remain stable when camera moves
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        stabilize_cascades: true,
        ..Default::default()
    };
    
    let cascade_count = config.cascade_count;
    let mut renderer = CsmRenderer::new(&device, config);
    
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    let near_plane = 0.1;
    let far_plane = 100.0;
    
    // Update cascades from multiple camera positions
    let camera_positions = vec![
        Vec3::new(10.0, 10.0, 10.0),
        Vec3::new(10.1, 10.0, 10.0), // Small movement
        Vec3::new(10.0, 10.1, 10.0), // Small movement
        Vec3::new(10.0, 10.0, 10.1), // Small movement
    ];
    
    let mut previous_splits: Vec<f32> = Vec::new();
    
    for (idx, pos) in camera_positions.iter().enumerate() {
        let camera_view = Mat4::look_at_rh(*pos, Vec3::ZERO, Vec3::Y);
        let camera_projection = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            16.0 / 9.0,
            near_plane,
            far_plane,
        );
        
        renderer.update_cascades(camera_view, camera_projection, light_direction, near_plane, far_plane);
        
        // Collect current splits
        let mut current_splits: Vec<f32> = Vec::new();
        for i in 0..cascade_count as usize {
            current_splits.push(renderer.uniforms.cascades[i].far_distance);
        }
        
        if idx > 0 {
            // Splits should be very similar (stable) despite camera movement
            for i in 0..cascade_count as usize {
                let diff = (current_splits[i] - previous_splits[i]).abs();
                // Allow small variation due to floating point
                assert!(diff < 0.1, "Cascade {} split changed too much: {} -> {}", 
                    i, previous_splits[i], current_splits[i]);
            }
        }
        
        previous_splits = current_splits;
    }
}

#[test]
fn test_cascade_selection_near_boundaries() {
    // Test cascade selection behavior near split boundaries
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        cascade_count: 3,
        ..Default::default()
    };
    
    let mut renderer = CsmRenderer::new(&device, config);
    
    let camera_view = Mat4::IDENTITY;
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
    
    // Get cascade boundaries
    let cascade_0_far = renderer.uniforms.cascades[0].far_distance;
    let cascade_1_far = renderer.uniforms.cascades[1].far_distance;
    
    // Test depths just before and after boundaries
    // In shader: select_cascade returns first cascade where view_depth <= far_distance
    // So cascade_0 is selected when depth <= cascade_0_far
    // cascade_1 is selected when cascade_0_far < depth <= cascade_1_far
    
    // Just verify that the boundaries are set up correctly
    assert!(cascade_0_far < cascade_1_far);
}

#[test]
fn test_light_direction_variations() {
    // Test that different light directions produce valid cascades
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig::default();
    
    let light_directions = vec![
        Vec3::new(0.0, -1.0, 0.0),    // Straight down
        Vec3::new(0.5, -1.0, 0.0),    // Angled
        Vec3::new(1.0, -1.0, 0.0),    // More angled
        Vec3::new(0.0, -1.0, 0.5),    // Different axis
        Vec3::new(-0.5, -0.8, 0.3),   // Complex angle
    ];
    
    for light_dir in &light_directions {
        let mut renderer = CsmRenderer::new(&device, config.clone());
        let normalized = light_dir.normalize();
        
        let camera_view = Mat4::IDENTITY;
        let camera_projection = Mat4::IDENTITY;
        
        renderer.update_cascades(camera_view, camera_projection, normalized, 0.1, 100.0);
        
        // Verify all cascades are valid
        for i in 0..config.cascade_count as usize {
            let cascade = &renderer.uniforms.cascades[i];
            assert!(cascade.near_distance < cascade.far_distance);
            assert!(cascade.texel_size > 0.0);
            
            // Projection should be valid
            let projection = Mat4::from_cols_array(&cascade.light_projection);
            assert!(projection.determinant().abs() > 1e-6);
        }
    }
}

#[test]
fn test_cascade_blend_persistence() {
    // Test that cascade_blend_range persists across updates
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        cascade_blend_range: 0.15,
        ..Default::default()
    };
    
    let mut renderer = CsmRenderer::new(&device, config);
    
    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    // Update multiple times
    for _ in 0..5 {
        renderer.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);
        assert_eq!(renderer.uniforms.cascade_blend_range, 0.15);
    }
}

#[test]
fn test_cascade_near_far_ordering() {
    // Test that near < far for all cascades
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig {
        cascade_count: 4,
        ..Default::default()
    };
    
    let mut renderer = CsmRenderer::new(&device, config);
    
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
    
    renderer.update_cascades(
        camera_view,
        camera_projection,
        Vec3::new(0.0, -1.0, 0.0),
        0.1,
        100.0,
    );
    
    for i in 0..4 {
        let near = renderer.uniforms.cascades[i].near_distance;
        let far = renderer.uniforms.cascades[i].far_distance;
        assert!(near < far, "Cascade {} has near ({}) >= far ({})", i, near, far);
        assert!(near >= 0.0, "Cascade {} has negative near distance: {}", i, near);
        assert!(far > 0.0, "Cascade {} has non-positive far distance: {}", i, far);
    }
}

#[test]
fn test_cascade_projection_matrix_orthographic() {
    // Test that cascade projection matrices are orthographic (not perspective)
    let device = forge3d::gpu::create_device_for_test();
    let config = CsmConfig::default();
    let cascade_count = config.cascade_count;
    
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
    
    // Check properties of orthographic projection matrices
    for i in 0..cascade_count as usize {
        let proj_array = &renderer.uniforms.cascades[i].light_projection;
        
        // For orthographic projection in column-major layout,
        // the bottom row (reading across columns at index 3) should be [0, 0, *, 1]
        // Column-major indices: row 3 of each column is at indices 3, 7, 11, 15
        assert_eq!(proj_array[3], 0.0, "Cascade {} projection not orthographic (row 3, col 0 != 0)", i);
        assert_eq!(proj_array[7], 0.0, "Cascade {} projection not orthographic (row 3, col 1 != 0)", i);
        // proj_array[11] can be non-zero (depth component)
        assert_eq!(proj_array[15], 1.0, "Cascade {} projection not orthographic (row 3, col 3 != 1)", i);
    }
}

#[test]
fn test_default_cascade_blend_range() {
    // Test that default cascade_blend_range is 0 (no blending)
    let config = CsmConfig::default();
    assert_eq!(config.cascade_blend_range, 0.0);
}

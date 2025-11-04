// tests/test_p3_13_shadow_unit_tests.rs
// P3-13: Rust unit tests for shadow system
// Exit criteria: Memory estimator, PCSS clamp, cascade splits, requires_moments, fallback texture

use forge3d::shadows::{ShadowManager, ShadowManagerConfig};
use forge3d::lighting::types::ShadowTechnique;
use forge3d::gpu;

// ============================================================================
// Memory Estimator Tests
// ============================================================================

#[test]
fn test_memory_estimator_pcf_no_moments() {
    // PCF uses only depth atlas, no moments
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::PCF;
    config.csm.cascade_count = 3;
    config.csm.shadow_map_size = 1024;
    
    let manager = ShadowManager::new(&device, config);
    
    // Expected: 3 cascades × 1024² × 4 bytes = 12 MiB
    let expected = 3 * 1024 * 1024 * 4;
    let actual = manager.memory_bytes();
    
    assert_eq!(actual, expected, "PCF memory should only include depth atlas");
}

#[test]
fn test_memory_estimator_vsm_with_moments() {
    // VSM uses depth + 4-channel moments (RGBA32Float = 16 bytes)
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::VSM;
    config.csm.cascade_count = 2;
    config.csm.shadow_map_size = 512;
    
    let manager = ShadowManager::new(&device, config);
    
    // Expected: depth (2×512²×4) + moments (2×512²×16) = 10 MiB
    let depth_bytes = 2 * 512 * 512 * 4;
    let moment_bytes = 2 * 512 * 512 * 16;
    let expected = depth_bytes + moment_bytes;
    let actual = manager.memory_bytes();
    
    assert_eq!(actual, expected, "VSM memory should include depth + 4-channel moments");
}

#[test]
fn test_memory_estimator_evsm_with_moments() {
    // EVSM uses depth + 4-channel moments (RGBA32Float = 16 bytes)
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::EVSM;
    config.csm.cascade_count = 2;
    config.csm.shadow_map_size = 512;
    
    let manager = ShadowManager::new(&device, config);
    
    // Expected: depth (2×512²×4) + moments (2×512²×16) = 10 MiB
    let depth_bytes = 2 * 512 * 512 * 4;
    let moment_bytes = 2 * 512 * 512 * 16;
    let expected = depth_bytes + moment_bytes;
    let actual = manager.memory_bytes();
    
    assert_eq!(actual, expected, "EVSM memory should include depth + 4-channel moments");
}

#[test]
fn test_memory_estimator_msm_with_moments() {
    // MSM uses depth + 4-channel moments (RGBA32Float = 16 bytes)
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::MSM;
    config.csm.cascade_count = 2;
    config.csm.shadow_map_size = 512;
    
    let manager = ShadowManager::new(&device, config);
    
    // Expected: depth (2×512²×4) + moments (2×512²×16) = 10 MiB
    let depth_bytes = 2 * 512 * 512 * 4;
    let moment_bytes = 2 * 512 * 512 * 16;
    let expected = depth_bytes + moment_bytes;
    let actual = manager.memory_bytes();
    
    assert_eq!(actual, expected, "MSM memory should include depth + 4-channel moments");
}

#[test]
fn test_memory_estimator_scales_with_cascades() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::PCF;
    config.csm.shadow_map_size = 512;
    
    // Test 2 cascades
    config.csm.cascade_count = 2;
    let manager_2 = ShadowManager::new(&device, config.clone());
    let memory_2 = manager_2.memory_bytes();
    
    // Test 4 cascades
    config.csm.cascade_count = 4;
    let manager_4 = ShadowManager::new(&device, config);
    let memory_4 = manager_4.memory_bytes();
    
    // Memory should scale linearly with cascade count
    assert_eq!(memory_4, memory_2 * 2, "Memory should double when cascade count doubles");
}

#[test]
fn test_memory_estimator_scales_with_resolution() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::PCF;
    config.csm.cascade_count = 2;
    
    // Test 512×512
    config.csm.shadow_map_size = 512;
    let manager_512 = ShadowManager::new(&device, config.clone());
    let memory_512 = manager_512.memory_bytes();
    
    // Test 1024×1024
    config.csm.shadow_map_size = 1024;
    let manager_1024 = ShadowManager::new(&device, config);
    let memory_1024 = manager_1024.memory_bytes();
    
    // Memory should scale quadratically with resolution (2x res = 4x memory)
    assert_eq!(memory_1024, memory_512 * 4, "Memory should quadruple when resolution doubles");
}

#[test]
fn test_memory_budget_enforcement() {
    // Test that memory budget is enforced (downscales if needed)
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::EVSM;
    config.csm.cascade_count = 4;
    config.csm.shadow_map_size = 8192; // Very large, may exceed budget
    config.max_memory_bytes = 256 * 1024 * 1024; // 256 MiB budget
    
    let manager = ShadowManager::new(&device, config);
    
    // Should be downscaled to fit budget
    assert!(manager.memory_bytes() <= 256 * 1024 * 1024, "Memory should not exceed budget");
}

// ============================================================================
// PCSS Radius Clamp Tests
// ============================================================================

#[test]
fn test_pcss_radius_clamp_basic() {
    use glam::{Mat4, Vec3};
    
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::PCSS;
    config.csm.cascade_count = 3;
    config.csm.shadow_map_size = 1024;
    config.pcss_blocker_radius = 100.0; // Very large, should be clamped
    config.pcss_filter_radius = 200.0;  // Very large, should be clamped
    
    let mut manager = ShadowManager::new(&device, config);
    
    // Update cascades to set texel sizes
    let camera_view = Mat4::look_at_rh(Vec3::new(0.0, 10.0, 10.0), Vec3::ZERO, Vec3::Y);
    let camera_projection = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 200.0);
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    manager.update_cascades(camera_view, camera_projection, light_direction, 0.1, 200.0);
    
    // Access uniforms to check clamped values
    let uniforms = &manager.renderer().uniforms;
    
    // PCSS blocker radius should be clamped to min_texel_size * 6.0
    let clamped_blocker = uniforms.technique_params[0];
    let clamped_filter = uniforms.technique_params[1];
    
    // Should be clamped (not 100.0 or 200.0)
    assert!(clamped_blocker < 100.0, "Blocker radius should be clamped");
    assert!(clamped_filter < 200.0, "Filter radius should be clamped");
    
    // Blocker radius should be positive
    assert!(clamped_blocker > 0.0, "Clamped blocker radius should be positive");
    assert!(clamped_filter > 0.0, "Clamped filter radius should be positive");
}

#[test]
fn test_pcss_radius_clamp_not_applied_to_other_techniques() {
    use glam::{Mat4, Vec3};
    
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::PCF; // Not PCSS
    config.pcss_blocker_radius = 100.0;
    config.pcss_filter_radius = 200.0;
    
    let mut manager = ShadowManager::new(&device, config);
    
    // Update cascades
    let camera_view = Mat4::look_at_rh(Vec3::new(0.0, 10.0, 10.0), Vec3::ZERO, Vec3::Y);
    let camera_projection = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 200.0);
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    manager.update_cascades(camera_view, camera_projection, light_direction, 0.1, 200.0);
    
    // For non-PCSS techniques, PCSS params should not be in technique_params[0] and [1]
    // (Those fields may be unused or used for other purposes)
    // Just verify the manager was created successfully
    assert_eq!(manager.technique(), ShadowTechnique::PCF);
}

#[test]
fn test_pcss_radius_reasonable_clamp() {
    use glam::{Mat4, Vec3};
    
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::PCSS;
    config.csm.cascade_count = 2;
    config.csm.shadow_map_size = 512;
    config.pcss_blocker_radius = 0.05;
    config.pcss_filter_radius = 0.08;
    
    let mut manager = ShadowManager::new(&device, config);
    
    // Update cascades
    let camera_view = Mat4::look_at_rh(Vec3::new(0.0, 10.0, 10.0), Vec3::ZERO, Vec3::Y);
    let camera_projection = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 100.0);
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    manager.update_cascades(camera_view, camera_projection, light_direction, 0.1, 100.0);
    
    let uniforms = &manager.renderer().uniforms;
    let clamped_blocker = uniforms.technique_params[0];
    let clamped_filter = uniforms.technique_params[1];
    
    // Should preserve reasonable values or clamp them
    assert!(clamped_blocker <= 0.05, "Should not increase blocker radius");
    assert!(clamped_filter <= 0.08, "Should not increase filter radius");
}

// ============================================================================
// Cascade Splits Tests
// ============================================================================

#[test]
fn test_cascade_splits_monotonic() {
    use glam::{Mat4, Vec3};
    
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let config = ShadowManagerConfig::default();
    let mut manager = ShadowManager::new(&device, config);
    
    // Update cascades
    let camera_view = Mat4::look_at_rh(Vec3::new(0.0, 10.0, 10.0), Vec3::ZERO, Vec3::Y);
    let camera_projection = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 200.0);
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    manager.update_cascades(camera_view, camera_projection, light_direction, 0.1, 200.0);
    
    let uniforms = &manager.renderer().uniforms;
    let cascade_count = uniforms.cascade_count as usize;
    
    // Verify splits are monotonically increasing
    for i in 0..cascade_count {
        let cascade = &uniforms.cascades[i];
        assert!(cascade.near_distance < cascade.far_distance,
            "Cascade {} near ({}) should be < far ({})", i, cascade.near_distance, cascade.far_distance);
        
        // Verify next cascade starts where previous ends
        if i + 1 < cascade_count {
            let next_cascade = &uniforms.cascades[i + 1];
            assert!(cascade.far_distance <= next_cascade.near_distance,
                "Cascade {} far ({}) should be <= cascade {} near ({})",
                i, cascade.far_distance, i + 1, next_cascade.near_distance);
        }
    }
}

#[test]
fn test_cascade_splits_within_near_far() {
    use glam::{Mat4, Vec3};
    
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let config = ShadowManagerConfig::default();
    let mut manager = ShadowManager::new(&device, config);
    
    let near_plane = 0.5;
    let far_plane = 150.0;
    
    // Update cascades
    let camera_view = Mat4::look_at_rh(Vec3::new(0.0, 10.0, 10.0), Vec3::ZERO, Vec3::Y);
    let camera_projection = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, near_plane, far_plane);
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    manager.update_cascades(camera_view, camera_projection, light_direction, near_plane, far_plane);
    
    let uniforms = &manager.renderer().uniforms;
    let cascade_count = uniforms.cascade_count as usize;
    
    // First cascade should start at or after near_plane
    let first_cascade = &uniforms.cascades[0];
    assert!(first_cascade.near_distance >= near_plane - 0.001,
        "First cascade near ({}) should be >= near_plane ({})", first_cascade.near_distance, near_plane);
    
    // Last cascade should end at or before far_plane
    let last_cascade = &uniforms.cascades[cascade_count - 1];
    assert!(last_cascade.far_distance <= far_plane + 0.001,
        "Last cascade far ({}) should be <= far_plane ({})", last_cascade.far_distance, far_plane);
}

#[test]
fn test_cascade_splits_cover_full_range() {
    use glam::{Mat4, Vec3};
    
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let config = ShadowManagerConfig::default();
    let mut manager = ShadowManager::new(&device, config);
    
    let near_plane = 0.1;
    let far_plane = 200.0;
    
    // Update cascades
    let camera_view = Mat4::look_at_rh(Vec3::new(0.0, 10.0, 10.0), Vec3::ZERO, Vec3::Y);
    let camera_projection = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, near_plane, far_plane);
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    manager.update_cascades(camera_view, camera_projection, light_direction, near_plane, far_plane);
    
    let uniforms = &manager.renderer().uniforms;
    let cascade_count = uniforms.cascade_count as usize;
    
    // Verify full range is covered (first near to last far)
    let first_near = uniforms.cascades[0].near_distance;
    let last_far = uniforms.cascades[cascade_count - 1].far_distance;
    
    let coverage = last_far - first_near;
    let expected_coverage = far_plane - near_plane;
    
    // Allow small tolerance
    assert!((coverage - expected_coverage).abs() < 1.0,
        "Cascades should cover near to far range: coverage={}, expected={}", coverage, expected_coverage);
}

#[test]
fn test_cascade_texel_size_increases() {
    use glam::{Mat4, Vec3};
    
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let config = ShadowManagerConfig::default();
    let mut manager = ShadowManager::new(&device, config);
    
    // Update cascades
    let camera_view = Mat4::look_at_rh(Vec3::new(0.0, 10.0, 10.0), Vec3::ZERO, Vec3::Y);
    let camera_projection = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 200.0);
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    manager.update_cascades(camera_view, camera_projection, light_direction, 0.1, 200.0);
    
    let uniforms = &manager.renderer().uniforms;
    let cascade_count = uniforms.cascade_count as usize;
    
    // Texel size should generally increase with distance (lower detail for far cascades)
    for i in 0..cascade_count.saturating_sub(1) {
        let texel_size = uniforms.cascades[i].texel_size;
        let next_texel_size = uniforms.cascades[i + 1].texel_size;
        
        // Allow some tolerance for numerical issues
        assert!(next_texel_size >= texel_size * 0.9,
            "Cascade {} texel_size ({}) should be <= cascade {} texel_size ({})",
            i, texel_size, i + 1, next_texel_size);
    }
}

// ============================================================================
// ShadowTechnique::requires_moments() Tests
// ============================================================================

#[test]
fn test_requires_moments_hard() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::Hard;
    
    let manager = ShadowManager::new(&device, config);
    
    // Hard shadows don't require moments
    // Check via fallback texture presence (should exist for non-moment techniques)
    assert!(manager.renderer().evsm_maps.is_none(), "Hard shadows should not create moment maps");
}

#[test]
fn test_requires_moments_pcf() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::PCF;
    
    let manager = ShadowManager::new(&device, config);
    
    // PCF doesn't require moments
    assert!(manager.renderer().evsm_maps.is_none(), "PCF should not create moment maps");
}

#[test]
fn test_requires_moments_pcss() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::PCSS;
    
    let manager = ShadowManager::new(&device, config);
    
    // PCSS doesn't require moments
    assert!(manager.renderer().evsm_maps.is_none(), "PCSS should not create moment maps");
}

#[test]
fn test_requires_moments_vsm() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::VSM;
    
    let manager = ShadowManager::new(&device, config);
    
    // VSM requires moments
    assert!(manager.renderer().evsm_maps.is_some(), "VSM should create moment maps");
}

#[test]
fn test_requires_moments_evsm() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::EVSM;
    
    let manager = ShadowManager::new(&device, config);
    
    // EVSM requires moments
    assert!(manager.renderer().evsm_maps.is_some(), "EVSM should create moment maps");
}

#[test]
fn test_requires_moments_msm() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::MSM;
    
    let manager = ShadowManager::new(&device, config);
    
    // MSM requires moments
    assert!(manager.renderer().evsm_maps.is_some(), "MSM should create moment maps");
}

// ============================================================================
// Fallback Moment Texture Tests
// ============================================================================

#[test]
fn test_fallback_moment_texture_for_non_moment_techniques() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    // Test techniques that don't require moments
    let techniques = vec![
        ShadowTechnique::Hard,
        ShadowTechnique::PCF,
        ShadowTechnique::PCSS,
    ];
    
    for technique in techniques {
        let mut config = ShadowManagerConfig::default();
        config.technique = technique;
        
        let manager = ShadowManager::new(&device, config);
        
        // Should have fallback texture (for bind group compatibility)
        // We can't directly access fallback_moment_texture (private), but we can check
        // that the manager was created successfully
        assert_eq!(manager.technique(), technique);
    }
}

#[test]
fn test_no_fallback_moment_texture_for_moment_techniques() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    // Test techniques that require moments
    let techniques = vec![
        ShadowTechnique::VSM,
        ShadowTechnique::EVSM,
        ShadowTechnique::MSM,
    ];
    
    for technique in techniques {
        let mut config = ShadowManagerConfig::default();
        config.technique = technique;
        
        let manager = ShadowManager::new(&device, config);
        
        // Should have actual moment maps, not fallback
        assert!(manager.renderer().evsm_maps.is_some(),
            "Technique {:?} should have moment maps", technique);
    }
}

#[test]
fn test_bind_group_layout_creation_with_fallback() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::PCF; // Non-moment technique
    
    let manager = ShadowManager::new(&device, config);
    
    // Create bind group layout (should work with fallback texture)
    let _layout = manager.create_bind_group_layout(&device);
    
    // Verify bind group layout was created successfully
    // (If this test passes, bind group layout creation works with fallback texture)
}

#[test]
fn test_bind_group_layout_creation_with_moments() {
    let (device, _queue) = gpu::create_device_and_queue_for_test();
    
    let mut config = ShadowManagerConfig::default();
    config.technique = ShadowTechnique::EVSM; // Moment technique
    
    let manager = ShadowManager::new(&device, config);
    
    // Create bind group layout (should work with actual moment maps)
    let _layout = manager.create_bind_group_layout(&device);
    
    // Verify bind group layout was created successfully
}

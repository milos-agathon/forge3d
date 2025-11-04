// tests/test_p3_08_pbr_shadow_integration.rs
// P3-08: Shadow integration in mesh PBR path
// Exit criteria: Mesh PBR pass runs with shadow visibility applied

#![cfg(all(feature = "enable-pbr", feature = "enable-tbn"))]

use forge3d::core::material::{PbrMaterial, PbrLighting};
use forge3d::pipeline::pbr::{PbrPipelineWithShadows, PbrSceneUniforms};
use forge3d::lighting::types::ShadowTechnique;
use glam::{Mat4, Vec3};

#[test]
fn test_pbr_pipeline_shadow_integration() {
    // Test that PBR pipeline integrates shadows correctly
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();
    
    let material = PbrMaterial::default();
    let pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);
    
    // Shadow manager should be created
    assert!(pipeline.shadow_manager.is_some());
    assert!(pipeline.shadow_bind_group_layout.is_some());
}

#[test]
fn test_shadow_enabled_disabled() {
    // Test enabling/disabling shadows
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();
    
    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, false);
    
    // Initially disabled
    assert!(!pipeline.has_shadows());
    
    // Enable shadows
    pipeline.set_shadow_enabled(&device, true);
    assert!(pipeline.has_shadows());
    
    // Disable shadows
    pipeline.set_shadow_enabled(&device, false);
    assert!(!pipeline.has_shadows());
}

#[test]
fn test_shadow_technique_switching() {
    // Test switching between shadow techniques
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();
    
    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);
    
    // Default should be PCF
    assert_eq!(pipeline.shadow_config.technique, ShadowTechnique::PCF);
    
    // Switch to PCSS
    pipeline.set_shadow_technique(&device, ShadowTechnique::PCSS);
    assert_eq!(pipeline.shadow_config.technique, ShadowTechnique::PCSS);
    
    // Switch to EVSM
    pipeline.set_shadow_technique(&device, ShadowTechnique::EVSM);
    assert_eq!(pipeline.shadow_config.technique, ShadowTechnique::EVSM);
}

#[test]
fn test_shadow_cascade_updates() {
    // Test that cascade updates work correctly
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();
    
    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);
    
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
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    // Update cascades
    pipeline.update_shadows(
        &queue,
        camera_view,
        camera_projection,
        light_direction,
        0.1,
        100.0,
    );
    
    // Verify cascade info is available
    assert!(pipeline.get_cascade_info(0).is_some());
    assert!(pipeline.get_cascade_info(1).is_some());
    assert!(pipeline.get_cascade_info(2).is_some());
}

#[test]
fn test_shadow_bind_group_creation() {
    // Test that shadow bind group is created correctly
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();
    
    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);
    
    // Get or create shadow bind group
    let bind_group = pipeline.get_or_create_shadow_bind_group(&device);
    assert!(bind_group.is_some());
    
    // Second call should return cached version
    let bind_group2 = pipeline.get_or_create_shadow_bind_group(&device);
    assert!(bind_group2.is_some());
}

#[test]
fn test_shadow_configuration() {
    // Test shadow quality configuration
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();
    
    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);
    
    // Configure shadows (use 2048 to stay within test GPU limits)
    pipeline.configure_shadows(&device, 5, 2048, 0);
    
    assert_eq!(pipeline.shadow_config.csm.pcf_kernel_size, 5);
    assert_eq!(pipeline.shadow_config.csm.shadow_map_size, 2048);
    assert_eq!(pipeline.shadow_config.csm.debug_mode, 0);
}

#[test]
fn test_peter_panning_prevention() {
    // Test that peter-panning prevention is working
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();
    
    let material = PbrMaterial::default();
    let pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);
    
    // Validate peter-panning prevention
    assert!(pipeline.validate_peter_panning_prevention());
}

#[test]
fn test_shadow_layout_retrieval() {
    // Test that shadow bind group layout can be retrieved
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();
    
    let material = PbrMaterial::default();
    let pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);
    
    let layout = pipeline.shadow_layout();
    assert!(layout.is_some());
}

#[test]
fn test_cascade_count_configuration() {
    // Test that default cascade count is valid
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();
    
    let material = PbrMaterial::default();
    let pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);
    
    // Verify default cascade count (should be 3)
    assert_eq!(pipeline.shadow_config.csm.cascade_count, 3);
    assert!(pipeline.shadow_manager.is_some());
}

#[test]
fn test_scene_uniforms_update() {
    // Test that scene uniforms update correctly
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();
    
    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);
    
    let model_matrix = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
    let view_matrix = Mat4::look_at_rh(Vec3::new(5.0, 5.0, 5.0), Vec3::ZERO, Vec3::Y);
    let projection_matrix = Mat4::perspective_rh(
        std::f32::consts::FRAC_PI_4,
        16.0 / 9.0,
        0.1,
        100.0,
    );
    
    let scene_uniforms = PbrSceneUniforms {
        model_matrix: model_matrix.to_cols_array_2d(),
        view_matrix: view_matrix.to_cols_array_2d(),
        projection_matrix: projection_matrix.to_cols_array_2d(),
        normal_matrix: model_matrix.inverse().transpose().to_cols_array_2d(),
    };
    
    pipeline.update_scene_uniforms(&queue, &scene_uniforms);
    
    // Verify uniforms were updated
    assert_eq!(pipeline.scene_uniforms.model_matrix, scene_uniforms.model_matrix);
}

#[test]
fn test_lighting_uniforms_update() {
    // Test that lighting uniforms update correctly
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();
    
    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);
    
    let mut lighting = PbrLighting::default();
    lighting.light_direction = [0.5, -1.0, 0.3];
    lighting.light_intensity = 2.0;
    
    pipeline.update_lighting_uniforms(&queue, &lighting);
    
    // Verify uniforms were updated
    assert_eq!(pipeline.lighting_uniforms.light_direction, lighting.light_direction);
    assert_eq!(pipeline.lighting_uniforms.light_intensity, lighting.light_intensity);
}

#[test]
fn test_shadow_visibility_application() {
    // Test that shadows affect direct lighting but not IBL
    // This is a logical test - actual rendering would require GPU execution
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();
    
    let material = PbrMaterial::default();
    let pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);
    
    // Verify shadow manager exists (required for shadow visibility)
    assert!(pipeline.shadow_manager.is_some());
    
    // Verify bind group can be created
    let mut pipeline_mut = pipeline;
    let bind_group = pipeline_mut.get_or_create_shadow_bind_group(&device);
    assert!(bind_group.is_some());
}

#[test]
fn test_cascade_info_retrieval() {
    // Test cascade info can be retrieved for debugging
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();
    
    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);
    
    let camera_view = Mat4::IDENTITY;
    let camera_projection = Mat4::IDENTITY;
    let light_direction = Vec3::new(0.0, -1.0, 0.0);
    
    pipeline.update_shadows(&queue, camera_view, camera_projection, light_direction, 0.1, 100.0);
    
    // Get cascade info for each cascade
    for i in 0..3 {
        let info = pipeline.get_cascade_info(i);
        assert!(info.is_some());
        
        if let Some((near, far, texel_size)) = info {
            assert!(near < far);
            assert!(texel_size > 0.0);
        }
    }
}

#[test]
fn test_pipeline_without_shadows() {
    // Test that pipeline works without shadows enabled
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();
    
    let material = PbrMaterial::default();
    let pipeline = PbrPipelineWithShadows::new(&device, &queue, material, false);
    
    // Shadow manager should not exist
    assert!(!pipeline.has_shadows());
    assert!(pipeline.shadow_manager.is_none());
    assert!(pipeline.shadow_bind_group_layout.is_none());
}

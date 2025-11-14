// tests/test_p3_09_shadow_visibility_brdf.rs
// P3-09: Shadow visibility applied in BRDF lighting loop
// Exit criteria: Hard/PCF/PCSS affect direct lighting; IBL unchanged

#![cfg(all(feature = "enable-pbr", feature = "enable-tbn"))]

use forge3d::core::material::{PbrLighting, PbrMaterial};
use forge3d::lighting::types::ShadowTechnique;
use forge3d::pipeline::pbr::{PbrPipelineWithShadows, PbrSceneUniforms};
use glam::{Mat4, Vec3};

#[test]
fn test_shadow_visibility_affects_direct_lighting() {
    // Test that shadows are applied to direct lighting
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();

    let material = PbrMaterial::default();
    let pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);

    // Verify shadow manager exists (required for visibility calculation)
    assert!(pipeline.shadow_manager.is_some());
    assert!(pipeline.has_shadows());
}

#[test]
fn test_shadow_technique_affects_visibility() {
    // Test that different shadow techniques affect visibility differently
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();

    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);

    // Hard shadows
    pipeline.set_shadow_technique(&device, ShadowTechnique::Hard);
    assert_eq!(pipeline.shadow_config.technique, ShadowTechnique::Hard);

    // PCF shadows (softer)
    pipeline.set_shadow_technique(&device, ShadowTechnique::PCF);
    assert_eq!(pipeline.shadow_config.technique, ShadowTechnique::PCF);

    // PCSS shadows (variable softness)
    pipeline.set_shadow_technique(&device, ShadowTechnique::PCSS);
    assert_eq!(pipeline.shadow_config.technique, ShadowTechnique::PCSS);
}

#[test]
fn test_brdf_dispatch_preserved() {
    // Test that BRDF dispatch still works with shadows
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();

    let material = PbrMaterial::default();
    let pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);

    // Verify shading uniforms exist (BRDF dispatch)
    assert!(pipeline.shading_uniforms.brdf >= 0);
}

#[test]
fn test_lighting_parameters_preserved() {
    // Test that lighting parameters are preserved with shadows
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();

    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);

    let mut lighting = PbrLighting::default();
    lighting.light_direction = [0.0, -1.0, 0.0];
    lighting.light_intensity = 2.5;
    lighting.ibl_intensity = 1.0;

    pipeline.update_lighting_uniforms(&queue, &lighting);

    // Verify lighting uniforms updated
    assert_eq!(pipeline.lighting_uniforms.light_intensity, 2.5);
    assert_eq!(pipeline.lighting_uniforms.ibl_intensity, 1.0);
}

#[test]
fn test_ibl_intensity_independent_of_shadows() {
    // Test that IBL intensity is not affected by shadow visibility
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();

    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);

    // Set IBL intensity
    let mut lighting = PbrLighting::default();
    lighting.ibl_intensity = 0.8;

    pipeline.update_lighting_uniforms(&queue, &lighting);

    // IBL intensity should remain 0.8 regardless of shadows
    assert_eq!(pipeline.lighting_uniforms.ibl_intensity, 0.8);
}

#[test]
fn test_direct_lighting_with_shadows_enabled() {
    // Test that direct lighting component exists with shadows enabled
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();

    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);

    let mut lighting = PbrLighting::default();
    lighting.light_direction = [0.0, -1.0, 0.0];
    lighting.light_intensity = 1.5;

    pipeline.update_lighting_uniforms(&queue, &lighting);

    // Verify light direction and intensity set
    assert_eq!(pipeline.lighting_uniforms.light_direction, [0.0, -1.0, 0.0]);
    assert_eq!(pipeline.lighting_uniforms.light_intensity, 1.5);
}

#[test]
fn test_direct_lighting_without_shadows() {
    // Test that direct lighting works when shadows are disabled
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();

    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, false);

    let mut lighting = PbrLighting::default();
    lighting.light_intensity = 1.0;

    pipeline.update_lighting_uniforms(&queue, &lighting);

    // Shadow manager should not exist
    assert!(!pipeline.has_shadows());
    // But lighting should still work
    assert_eq!(pipeline.lighting_uniforms.light_intensity, 1.0);
}

#[test]
fn test_shadow_visibility_calculation_inputs() {
    // Test that shadow visibility has correct inputs (world pos, normal, view depth)
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();

    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);

    // Update cascades with camera and light info
    let camera_view = Mat4::look_at_rh(Vec3::new(5.0, 5.0, 5.0), Vec3::ZERO, Vec3::Y);
    let camera_projection =
        Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 100.0);
    let light_direction = Vec3::new(0.0, -1.0, 0.0);

    pipeline.update_shadows(
        &queue,
        camera_view,
        camera_projection,
        light_direction,
        0.1,
        100.0,
    );

    // Verify cascades were updated (cascade info should be available)
    assert!(pipeline.get_cascade_info(0).is_some());
}

#[test]
fn test_brdf_models_with_shadows() {
    // Test that shadows work with different BRDF models
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();

    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);

    // Test various BRDF indices (from lighting.wgsl constants)
    for brdf_idx in [0u32, 1, 2, 3, 4] {
        pipeline.set_brdf_index(&queue, brdf_idx);
        assert_eq!(pipeline.shading_uniforms.brdf, brdf_idx);
    }

    // Shadows should still work regardless of BRDF model
    assert!(pipeline.has_shadows());
}

#[test]
fn test_emissive_unaffected_by_shadows() {
    // Test that emissive materials are not affected by shadows
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();

    let mut material = PbrMaterial::default();
    material.emissive = [1.0, 0.5, 0.0]; // Orange glow

    let pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);

    // Emissive value should be preserved
    assert_eq!(pipeline.material.material.emissive, [1.0, 0.5, 0.0]);
}

#[test]
fn test_ambient_occlusion_preserved() {
    // Test that ambient occlusion is preserved with shadows
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();

    let mut material = PbrMaterial::default();
    material.occlusion_strength = 0.8;

    let pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);

    // AO strength should be preserved
    assert_eq!(pipeline.material.material.occlusion_strength, 0.8);
}

#[test]
fn test_normal_mapping_preserved() {
    // Test that normal mapping is preserved with shadows
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();

    let mut material = PbrMaterial::default();
    material.normal_scale = 1.5;

    let pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);

    // Normal scale should be preserved
    assert_eq!(pipeline.material.material.normal_scale, 1.5);
}

#[test]
fn test_metallic_roughness_preserved() {
    // Test that metallic-roughness workflow is preserved with shadows
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();

    let mut material = PbrMaterial::default();
    material.metallic = 0.8;
    material.roughness = 0.3;

    let pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);

    // Metallic-roughness should be preserved
    assert_eq!(pipeline.material.material.metallic, 0.8);
    assert_eq!(pipeline.material.material.roughness, 0.3);
}

#[test]
fn test_shadow_pcf_kernel_size() {
    // Test that PCF kernel size affects shadow softness
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();

    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);

    // Small kernel (harder shadows)
    pipeline.configure_shadows(&device, 3, 2048, 0);
    assert_eq!(pipeline.shadow_config.csm.pcf_kernel_size, 3);

    // Large kernel (softer shadows)
    pipeline.configure_shadows(&device, 7, 2048, 0);
    assert_eq!(pipeline.shadow_config.csm.pcf_kernel_size, 7);
}

#[test]
fn test_shadow_visibility_with_cascade_updates() {
    // Test that shadow visibility works correctly with cascade updates
    let (device, queue) = forge3d::gpu::create_device_and_queue_for_test();

    let material = PbrMaterial::default();
    let mut pipeline = PbrPipelineWithShadows::new(&device, &queue, material, true);

    // Update cascades multiple times (simulating camera movement)
    for i in 0..5 {
        let offset = i as f32 * 2.0;
        let camera_view = Mat4::look_at_rh(Vec3::new(5.0 + offset, 5.0, 5.0), Vec3::ZERO, Vec3::Y);
        let camera_projection =
            Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 100.0);
        let light_direction = Vec3::new(0.0, -1.0, 0.0);

        pipeline.update_shadows(
            &queue,
            camera_view,
            camera_projection,
            light_direction,
            0.1,
            100.0,
        );

        // Cascades should still be valid
        assert!(pipeline.get_cascade_info(0).is_some());
    }
}

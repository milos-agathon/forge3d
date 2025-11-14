// tests/test_p1_light_buffer.rs
// P1-11: GPU integration smoke tests for LightBuffer
// Validates LightBuffer GPU upload, triple-buffering, and debug inspection

use forge3d::lighting::types::Light;
use forge3d::lighting::LightBuffer;
use wgpu::{Device, DeviceDescriptor, Instance, InstanceDescriptor, Queue, RequestAdapterOptions};

/// Create device and queue for testing (gracefully fails if no GPU)
fn create_device_queue() -> Option<(Device, Queue)> {
    let instance = Instance::new(InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    let limits = wgpu::Limits::downlevel_defaults();
    let desc = DeviceDescriptor {
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        label: Some("p1_light_buffer_test_device"),
    };

    let (device, queue) = pollster::block_on(adapter.request_device(&desc, None)).ok()?;
    Some((device, queue))
}

#[test]
fn test_light_buffer_heterogeneous_upload() {
    // P1-11: Upload 4 heterogeneous lights and validate via debug API
    let Some((device, queue)) = create_device_queue() else {
        eprintln!("Skipping test_light_buffer_heterogeneous_upload (no GPU)");
        return;
    };

    let mut light_buffer = LightBuffer::new(&device);

    // Create 4 heterogeneous lights
    let lights = vec![
        Light::directional(45.0, 30.0, 3.0, [1.0, 0.9, 0.8]), // Type 0
        Light::point([10.0, 5.0, -3.0], 50.0, 10.0, [1.0, 1.0, 1.0]), // Type 1
        Light::spot(
            [0.0, 10.0, 0.0],
            [0.0, -1.0, 0.0],
            100.0,
            15.0, // inner_deg
            30.0, // outer_deg
            5.0,
            [1.0, 0.95, 0.85],
        ), // Type 2
        Light::area_rect(
            [5.0, 8.0, 5.0],
            [0.0, -1.0, 0.0],
            2.0, // half_width
            1.5, // half_height
            8.0,
            [1.0, 1.0, 0.9],
        ), // Type 4
    ];

    // Upload to GPU
    light_buffer
        .update(&device, &queue, &lights)
        .expect("Failed to upload lights");

    // Validate via debug API (P1-07)
    let uploaded = light_buffer.last_uploaded_lights();
    assert_eq!(uploaded.len(), 4, "Should have 4 lights");

    // Check types
    assert_eq!(uploaded[0].kind, 0, "Light 0 should be Directional");
    assert_eq!(uploaded[1].kind, 1, "Light 1 should be Point");
    assert_eq!(uploaded[2].kind, 2, "Light 2 should be Spot");
    assert_eq!(uploaded[3].kind, 4, "Light 3 should be AreaRect");

    // Check intensities
    assert!((uploaded[0].intensity - 3.0).abs() < 0.01);
    assert!((uploaded[1].intensity - 10.0).abs() < 0.01);
    assert!((uploaded[2].intensity - 5.0).abs() < 0.01);
    assert!((uploaded[3].intensity - 8.0).abs() < 0.01);

    // Check type-specific fields
    // Point light position
    assert!((uploaded[1].pos_ws[0] - 10.0).abs() < 0.01);
    assert!((uploaded[1].pos_ws[1] - 5.0).abs() < 0.01);
    assert!((uploaded[1].pos_ws[2] - (-3.0)).abs() < 0.01);

    // Spot light cone (should be cosines, not degrees)
    assert!(
        uploaded[2].cone_cos[0] > 0.9,
        "Spot inner cone should be cos(15°) ≈ 0.97"
    );
    assert!(
        uploaded[2].cone_cos[1] > 0.8 && uploaded[2].cone_cos[1] < 0.9,
        "Spot outer cone should be cos(30°) ≈ 0.87"
    );

    // Area light dimensions
    assert!(
        (uploaded[3].area_half[0] - 2.0).abs() < 0.01,
        "Rect half_width should be 2.0"
    );
    assert!(
        (uploaded[3].area_half[1] - 1.5).abs() < 0.01,
        "Rect half_height should be 1.5"
    );

    println!("✓ P1-11: Heterogeneous light upload validation passed");
}

#[test]
fn test_light_buffer_max_lights_enforcement() {
    // P1-11: Verify MAX_LIGHTS enforcement in GPU path
    let Some((device, queue)) = create_device_queue() else {
        eprintln!("Skipping test_light_buffer_max_lights_enforcement (no GPU)");
        return;
    };

    let mut light_buffer = LightBuffer::new(&device);

    // Create MAX_LIGHTS + 1 lights (should fail)
    let mut lights = Vec::new();
    for i in 0..17 {
        // MAX_LIGHTS is 16
        lights.push(Light::point(
            [i as f32, 0.0, 0.0],
            10.0,
            1.0,
            [1.0, 1.0, 1.0],
        ));
    }

    // Should return error
    let result = light_buffer.update(&device, &queue, &lights);
    assert!(result.is_err(), "Should fail with > MAX_LIGHTS");

    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("Too many lights"),
        "Error should mention 'Too many lights'"
    );
    assert!(
        err_msg.contains("17"),
        "Error should mention actual count (17)"
    );
    assert!(
        err_msg.contains("16"),
        "Error should mention MAX_LIGHTS (16)"
    );

    println!("✓ P1-11: MAX_LIGHTS enforcement passed");
}

#[test]
fn test_light_buffer_triple_buffering() {
    // P1-11: Verify triple-buffering with frame cycling
    let Some((device, queue)) = create_device_queue() else {
        eprintln!("Skipping test_light_buffer_triple_buffering (no GPU)");
        return;
    };

    let mut light_buffer = LightBuffer::new(&device);

    // Upload lights and cycle through 3 frames
    let lights = vec![
        Light::directional(0.0, 45.0, 2.0, [1.0, 1.0, 1.0]),
        Light::point([0.0, 5.0, 0.0], 20.0, 5.0, [1.0, 0.9, 0.8]),
    ];

    for frame in 0..3 {
        light_buffer
            .update(&device, &queue, &lights)
            .expect("Failed to upload lights");

        // Verify debug API shows correct lights
        let uploaded = light_buffer.last_uploaded_lights();
        assert_eq!(uploaded.len(), 2, "Frame {}: Should have 2 lights", frame);
        assert_eq!(
            uploaded[0].kind, 0,
            "Frame {}: Light 0 should be Directional",
            frame
        );
        assert_eq!(
            uploaded[1].kind, 1,
            "Frame {}: Light 1 should be Point",
            frame
        );

        // Advance to next frame
        light_buffer.next_frame();
    }

    // Verify frame counter advanced
    assert_eq!(
        light_buffer.frame_counter(),
        3,
        "Frame counter should be 3 after 3 next_frame() calls"
    );

    println!("✓ P1-11: Triple-buffering frame cycling passed");
}

#[test]
fn test_light_buffer_debug_info_format() {
    // P1-11: Verify debug_info() produces valid output
    let Some((device, queue)) = create_device_queue() else {
        eprintln!("Skipping test_light_buffer_debug_info_format (no GPU)");
        return;
    };

    let mut light_buffer = LightBuffer::new(&device);

    let lights = vec![
        Light::directional(135.0, 35.0, 3.0, [1.0, 0.9, 0.8]),
        Light::point([0.0, 10.0, 0.0], 50.0, 10.0, [1.0, 1.0, 1.0]),
    ];

    light_buffer
        .update(&device, &queue, &lights)
        .expect("Failed to upload lights");

    // Get debug output
    let debug_output = light_buffer.debug_info();

    // Verify output structure
    assert!(
        debug_output.contains("LightBuffer Debug Info"),
        "Should have header"
    );
    assert!(
        debug_output.contains("Count: 2 lights"),
        "Should show light count"
    );
    assert!(
        debug_output.contains("Frame: 0"),
        "Should show frame counter"
    );
    assert!(
        debug_output.contains("Light 0: Directional"),
        "Should show first light type"
    );
    assert!(
        debug_output.contains("Light 1: Point"),
        "Should show second light type"
    );
    assert!(
        debug_output.contains("Intensity: 3.00"),
        "Should show directional intensity"
    );
    assert!(
        debug_output.contains("Intensity: 10.00"),
        "Should show point intensity"
    );
    assert!(
        debug_output.contains("Position: [0.00, 10.00, 0.00]"),
        "Should show point position"
    );
    assert!(
        debug_output.contains("Range: 50.00"),
        "Should show point range"
    );

    println!("✓ P1-11: Debug info format validation passed");
    println!("\nDebug output sample:\n{}", debug_output);
}

#[test]
fn test_light_buffer_empty_upload() {
    // P1-11: Verify empty light array is handled correctly
    let Some((device, queue)) = create_device_queue() else {
        eprintln!("Skipping test_light_buffer_empty_upload (no GPU)");
        return;
    };

    let mut light_buffer = LightBuffer::new(&device);

    let lights: Vec<Light> = vec![];

    // Should succeed with empty array
    light_buffer
        .update(&device, &queue, &lights)
        .expect("Empty light array should be valid");

    let uploaded = light_buffer.last_uploaded_lights();
    assert_eq!(uploaded.len(), 0, "Should have 0 lights");

    let debug_output = light_buffer.debug_info();
    assert!(
        debug_output.contains("Count: 0 lights"),
        "Debug output should show 0 lights"
    );

    println!("✓ P1-11: Empty light array handled correctly");
}

#[test]
fn test_light_buffer_bind_group_creation() {
    // P1-11: Verify bind group is created after update
    let Some((device, queue)) = create_device_queue() else {
        eprintln!("Skipping test_light_buffer_bind_group_creation (no GPU)");
        return;
    };

    let mut light_buffer = LightBuffer::new(&device);

    // Before update, bind_group should be None (implicit via Option)
    // After update, bind_group should exist

    let lights = vec![Light::directional(0.0, 45.0, 1.0, [1.0, 1.0, 1.0])];

    light_buffer
        .update(&device, &queue, &lights)
        .expect("Failed to upload lights");

    // Verify bind group exists (indirectly via successful update)
    let bg = light_buffer.bind_group();
    assert!(bg.is_some(), "Bind group should exist after update");

    println!("✓ P1-11: Bind group creation validated");
}

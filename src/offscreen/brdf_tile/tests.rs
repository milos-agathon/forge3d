use super::*;

#[test]
fn test_brdf_tile_scaffold_returns_tight_buffer() {
    // Skip if no GPU
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }));

    let Some(adapter) = adapter else {
        eprintln!("No GPU adapter available, skipping test");
        return;
    };

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to create device");

    // Test GGX at roughness 0.5
    let result = render_brdf_tile_offscreen(
        &device,
        &queue,
        4,               // GGX
        0.5,             // roughness
        256,             // width
        256,             // height
        false,           // ndf_only
        false,           // g_only
        false,           // dfg_only
        false,           // spec_only
        false,           // roughness_visualize
        1.0,             // exposure
        0.8,             // light_intensity
        [0.5, 0.5, 0.5], // base_color
        0.0,             // clearcoat
        0.0,             // clearcoat_roughness
        0.0,             // sheen
        0.0,             // sheen_tint
        0.0,             // specular_tint
        false,           // debug_dot_products
        // M2 defaults
        false, // debug_lambert_only
        false, // debug_diffuse_only
        false, // debug_d
        false, // debug_spec_no_nl
        false, // debug_energy
        false, // debug_angle_sweep
        2,     // debug_angle_component
        false, // debug_no_srgb
        1,     // output_mode (srgb)
        0.0,   // metallic_override
        0,     // wi3_debug_mode
        0.5,   // wi3_debug_roughness
        64,    // sphere_sectors
        32,    // sphere_stacks
    );

    assert!(
        result.is_ok(),
        "render_brdf_tile_offscreen failed: {:?}",
        result.err()
    );
    let buffer = result.unwrap();

    // Verify tight buffer shape: (H, W, 4)
    assert_eq!(buffer.len(), 256 * 256 * 4, "buffer size mismatch");

    // Scaffold encodes model/roughness in clear color; verify it's non-zero
    assert!(buffer.iter().any(|&b| b > 0), "buffer is all zeros");
}

#[test]
fn test_brdf_tile_validates_inputs() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));
    let Some(adapter) = adapter else {
        return;
    };
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .expect("Failed to create device");

    // Invalid BRDF model
    let result = render_brdf_tile_offscreen(
        &device,
        &queue,
        99,
        0.5,
        256,
        256,
        false,
        false,
        false,
        false,
        false,
        1.0,
        0.8,
        [0.5, 0.5, 0.5],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        false, // debug_dot_products
        false, // debug_lambert_only
        false, // debug_diffuse_only
        false, // debug_d
        false, // debug_spec_no_nl
        false, // debug_energy
        false, // debug_angle_sweep
        2,     // debug_angle_component
        false, // debug_no_srgb
        1,     // output_mode
        0.0,
        0,
        0.5,
        64,
        32,
    );
    assert!(result.is_err(), "should reject invalid BRDF model");

    // Zero dimensions
    let result = render_brdf_tile_offscreen(
        &device,
        &queue,
        4,
        0.5,
        0,
        256,
        false,
        false,
        false,
        false,
        false,
        1.0,
        0.8,
        [0.5, 0.5, 0.5],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        false, // debug_dot_products
        false, // debug_lambert_only
        false, // debug_diffuse_only
        false, // debug_d
        false, // debug_spec_no_nl
        false, // debug_energy
        false, // debug_angle_sweep
        2,     // debug_angle_component
        false, // debug_no_srgb
        1,     // output_mode
        0.0,
        0,
        0.5,
        64,
        32,
    );
    assert!(result.is_err(), "should reject zero width");
}

/// P7-04: CPU readback validation
/// This test demonstrates that the returned buffer is suitable for PNG export
#[test]
fn test_brdf_tile_readback_png_compatible() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }));

    let Some(adapter) = adapter else {
        eprintln!("No GPU adapter available, skipping test");
        return;
    };

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to create device");

    let width = 256u32;
    let height = 256u32;

    let result = render_brdf_tile_offscreen(
        &device,
        &queue,
        4,   // GGX
        0.5, // roughness
        width,
        height,
        false,           // ndf_only
        false,           // g_only
        false,           // dfg_only
        false,           // spec_only
        false,           // roughness_visualize
        1.0,             // exposure
        0.8,             // light_intensity
        [0.5, 0.5, 0.5], // base_color
        0.0,             // clearcoat
        0.0,             // clearcoat_roughness
        0.0,             // sheen
        0.0,             // sheen_tint
        0.0,             // specular_tint
        false,           // debug_dot_products
        false,           // debug_lambert_only
        false,           // debug_diffuse_only
        false,           // debug_d
        false,           // debug_spec_no_nl
        false,           // debug_energy
        false,           // debug_angle_sweep
        2,               // debug_angle_component
        false,           // debug_no_srgb
        1,               // output_mode
        0.0,             // metallic_override
        0,               // wi3_debug_mode
        0.3,             // wi3_debug_roughness
        64,
        32,
    );

    assert!(
        result.is_ok(),
        "render_brdf_tile_offscreen failed: {:?}",
        result.err()
    );
    let buffer = result.unwrap();

    // P7-04 Exit Criteria: Verify buffer is PNG-compatible
    // 1. Tight layout: H × W × 4 bytes (no padding)
    assert_eq!(
        buffer.len(),
        (height * width * 4) as usize,
        "buffer must be tight row-major RGBA8"
    );

    // 2. Non-trivial content: should have some variation (not all black or all white)
    let unique_pixels = buffer
        .chunks_exact(4)
        .map(|rgba| (rgba[0], rgba[1], rgba[2], rgba[3]))
        .collect::<std::collections::HashSet<_>>();
    assert!(
        unique_pixels.len() > 10,
        "rendered output should have variation, got {} unique pixels",
        unique_pixels.len()
    );

    // 3. Valid RGBA range: all bytes in [0, 255] (automatically true for u8)
    // 4. Alpha channel: should be 255 (opaque)
    let all_opaque = buffer.chunks_exact(4).all(|rgba| rgba[3] == 255);
    assert!(all_opaque, "all pixels should be opaque (alpha=255)");

    // 5. Some pixels should be non-black (sphere should be lit)
    let has_lighting = buffer
        .chunks_exact(4)
        .any(|rgba| rgba[0] > 10 || rgba[1] > 10 || rgba[2] > 10);
    assert!(has_lighting, "rendered sphere should have visible lighting");

    // Buffer is now suitable for PNG export via python/forge3d/__init__.py:numpy_to_png()
    // In Python: numpy_to_png("output.png", buffer.reshape(height, width, 4))
}

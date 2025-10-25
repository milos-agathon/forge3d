// tests/readback_depad.rs
// Byte-accurate depadding validation for GPU readback with row padding
// Tests acceptance criteria: 100% unique row hashes at odd and 4K sizes

use sha2::{Digest, Sha256};
use std::collections::HashSet;

fn create_test_device() -> (wgpu::Device, wgpu::Queue) {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("Failed to find GPU adapter");

        adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("Failed to create device")
    })
}

fn validate_pattern_readback(width: u32, height: u32) {
    let (device, queue) = create_test_device();

    let texture =
        forge3d::util::debug_pattern::render_debug_pattern(&device, &queue, width, height)
            .expect("Failed to render debug pattern");

    let pixels = forge3d::renderer::readback::read_texture_tight(
        &device,
        &queue,
        &texture,
        (width, height),
        wgpu::TextureFormat::Rgba8UnormSrgb,
    )
    .expect("Readback failed");

    let expected_size = (width as usize) * (height as usize) * 4;
    assert_eq!(
        pixels.len(),
        expected_size,
        "Buffer size mismatch at {}x{}: expected {}, got {}",
        width,
        height,
        expected_size,
        pixels.len()
    );

    let mut row_hashes = HashSet::new();
    let bytes_per_row = (width as usize) * 4;

    for y in 0..height as usize {
        let row_start = y * bytes_per_row;
        let row_end = row_start + bytes_per_row;
        let row_data = &pixels[row_start..row_end];

        let mut hasher = Sha256::new();
        hasher.update(row_data);
        let hash_hex = format!("{:x}", hasher.finalize());

        let inserted = row_hashes.insert(hash_hex.clone());
        assert!(
            inserted,
            "Duplicate row hash at {}x{}, row {}: {}",
            width, height, y, hash_hex
        );
    }

    assert_eq!(
        row_hashes.len(),
        height as usize,
        "Not all rows unique at {}x{}: {} unique out of {}",
        width,
        height,
        row_hashes.len(),
        height
    );

    println!(
        "{}x{}: {} unique row hashes (100%)",
        width,
        height,
        row_hashes.len()
    );
}

#[test]
fn test_readback_depad_odd_size() {
    // Odd width forces non-256-aligned bytes_per_row
    validate_pattern_readback(1023, 17);
}

#[test]
fn test_readback_depad_4k() {
    // 4K resolution (real-world size)
    validate_pattern_readback(3840, 2160);
}

#[test]
fn test_readback_depad_various() {
    // Additional sizes that force padding
    validate_pattern_readback(1919, 5);
    validate_pattern_readback(2560, 1440);
    validate_pattern_readback(1024, 768);
}

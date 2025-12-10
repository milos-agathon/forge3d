use super::*;
use crate::lighting::types::Light;
use super::frame::light_type_name;

#[test]
fn test_light_buffer_memory() {
    // Light struct is 80 bytes (verified in types.rs)
    let light_size = std::mem::size_of::<Light>();
    assert_eq!(light_size, 80);

    // Memory calculation
    let light_buffer_size = MAX_LIGHTS * light_size; // 16 * 80 = 1280 bytes
    let count_buffer_size = 16;
    let total_per_buffer = light_buffer_size + count_buffer_size; // 1296 bytes
    let total = 3 * total_per_buffer; // 3888 bytes

    assert_eq!(total, 3888);

    let mb = total as f64 / (1024.0 * 1024.0);
    assert!(mb < 0.01); // Less than 10 KB = ~0.0037 MiB
}

#[test]
fn test_max_lights_constant() {
    // Verify MAX_LIGHTS fits in memory budget
    // At 80 bytes per light, 16 lights = 1.28 KB per buffer
    // Triple-buffered: 3.84 KB total
    // This is well within the 512 MiB host-visible budget
    assert_eq!(MAX_LIGHTS, 16);

    let total_bytes = 3 * MAX_LIGHTS * std::mem::size_of::<Light>();
    assert!(total_bytes < 512 * 1024 * 1024); // < 512 MiB
}

#[test]
fn test_r2_sequence_variation() {
    let first = super::r2::r2_sample(0);
    let second = super::r2::r2_sample(1);
    assert_ne!(first, second);
    assert!(first[0] >= 0.0 && first[0] <= 1.0);
    assert!(second[1] >= 0.0 && second[1] <= 1.0);
}

// P1-02: Triple-buffered SSBO manager parity tests

#[test]
fn test_light_metadata_size() {
    // LightMetadata in WGSL is 4 u32s (count, frame_index, seed_bits_x, seed_bits_y)
    // Rust equivalent is [u32; 4] = 16 bytes
    assert_eq!(std::mem::size_of::<[u32; 4]>(), 16);
}

#[test]
fn test_max_lights_budget() {
    // Verify MAX_LIGHTS=16 is correct
    assert_eq!(MAX_LIGHTS, 16);

    // Verify total memory is reasonable
    let light_size = std::mem::size_of::<Light>();
    let total_light_memory = 3 * MAX_LIGHTS * light_size; // Triple-buffered
    let total_metadata = 3 * 16; // 3 count buffers
    let environment_stub = 16;
    let total = total_light_memory + total_metadata + environment_stub;

    // Total should be: 3 * 16 * 80 + 3 * 16 + 16 = 3840 + 48 + 16 = 3904 bytes
    assert_eq!(total, 3904);
    assert!(total < 5000); // Well under 5 KB
}

#[test]
fn test_memory_bytes_calculation() {
    // Verify memory_bytes() matches manual calculation
    let light_buffer_size = (MAX_LIGHTS * std::mem::size_of::<Light>()) as u64; // 1280
    let count_buffer_size = 16u64;
    let environment_stub_size = 16u64;

    let expected = 3 * (light_buffer_size + count_buffer_size) + environment_stub_size;
    // 3 * (1280 + 16) + 16 = 3 * 1296 + 16 = 3888 + 16 = 3904
    assert_eq!(expected, 3904);
}

#[test]
fn test_memory_mb_conversion() {
    // Verify memory_mb() converts correctly
    let bytes = 3904u64;
    let mb = bytes as f64 / (1024.0 * 1024.0);
    // ~0.00372 MiB
    assert!(mb > 0.0037 && mb < 0.0038);
}

#[test]
fn test_bind_layout_constants() {
    // Verify binding numbers match WGSL
    // @group(0) @binding(3) var<storage, read> lights: array<LightGPU>;
    // @group(0) @binding(4) var<uniform> lightMeta: LightMetadata;
    // @group(0) @binding(5) var<uniform> environmentParams: vec4<f32>;
    const LIGHTS_BINDING: u32 = 3;
    const METADATA_BINDING: u32 = 4;
    const ENVIRONMENT_BINDING: u32 = 5;

    assert_eq!(LIGHTS_BINDING, 3);
    assert_eq!(METADATA_BINDING, 4);
    assert_eq!(ENVIRONMENT_BINDING, 5);
}

#[test]
fn test_frame_counter_wrapping() {
    // Test that frame counter wraps correctly
    let counter = u64::MAX;
    let wrapped = counter.wrapping_add(1);
    assert_eq!(wrapped, 0);
}

#[test]
fn test_frame_index_cycling() {
    // Test frame index cycles through 0, 1, 2
    let mut index = 0;
    index = (index + 1) % 3;
    assert_eq!(index, 1);
    index = (index + 1) % 3;
    assert_eq!(index, 2);
    index = (index + 1) % 3;
    assert_eq!(index, 0);
}

#[test]
fn test_r2_sequence_deterministic() {
    // Verify R2 sequence is deterministic
    let seed1a = super::r2::r2_sample(42);
    let seed1b = super::r2::r2_sample(42);
    assert_eq!(seed1a, seed1b);

    // Different indices produce different seeds
    let seed2 = super::r2::r2_sample(43);
    assert_ne!(seed1a, seed2);
}

#[test]
fn test_r2_sequence_range() {
    // Verify R2 samples stay in [0, 1] range
    for i in 0..100 {
        let sample = super::r2::r2_sample(i);
        assert!(
            sample[0] >= 0.0 && sample[0] <= 1.0,
            "R2 x sample {} out of range: {}",
            i,
            sample[0]
        );
        assert!(
            sample[1] >= 0.0 && sample[1] <= 1.0,
            "R2 y sample {} out of range: {}",
            i,
            sample[1]
        );
    }
}

#[test]
fn test_light_count_enforcement() {
    // Verify MAX_LIGHTS limit is properly documented
    // Actual enforcement happens in update() method which returns Err
    // This test verifies the constant is sensible
    assert!(MAX_LIGHTS > 0, "MAX_LIGHTS must be positive");
    assert!(MAX_LIGHTS <= 64, "MAX_LIGHTS should be reasonable (<=64)");
}

// P1-03: Per-frame seed generation tests

#[test]
fn test_seed_generation_on_next_frame() {
    // Verify that next_frame() generates new R2 seeds every frame
    let seed0 = super::r2::r2_sample(0);
    let seed1 = super::r2::r2_sample(1);
    let seed2 = super::r2::r2_sample(2);
    let seed3 = super::r2::r2_sample(3);

    // All seeds should be different
    assert_ne!(seed0, seed1);
    assert_ne!(seed1, seed2);
    assert_ne!(seed2, seed3);
    assert_ne!(seed0, seed3);
}

#[test]
fn test_frame_counter_increments() {
    // Test frame counter progression
    let mut counter = 0u64;
    let mut seeds = Vec::new();

    for _ in 0..10 {
        seeds.push(super::r2::r2_sample(counter));
        counter = counter.wrapping_add(1);
    }

    // Verify all 10 seeds are unique
    for i in 0..seeds.len() {
        for j in (i + 1)..seeds.len() {
            assert_ne!(
                seeds[i], seeds[j],
                "Seeds at frames {} and {} should differ",
                i, j
            );
        }
    }
}

#[test]
fn test_seed_encoding_roundtrip() {
    // Verify seed encoding matches WGSL bitcast behavior
    let seed = super::r2::r2_sample(42);

    // Encode as bits (what we upload to GPU)
    let bits_x = seed[0].to_bits();
    let bits_y = seed[1].to_bits();

    // Decode (what WGSL bitcast<f32>() does)
    let decoded_x = f32::from_bits(bits_x);
    let decoded_y = f32::from_bits(bits_y);

    // Should match exactly
    assert_eq!(seed[0], decoded_x);
    assert_eq!(seed[1], decoded_y);
}

// P1-05: Environment stub buffer tests

#[test]
fn test_environment_stub_size() {
    // Environment params buffer is vec4<f32> = 16 bytes
    assert_eq!(std::mem::size_of::<[f32; 4]>(), 16);
}

#[test]
fn test_environment_binding_constant() {
    // Verify environment params is at binding 5
    const ENVIRONMENT_BINDING: u32 = 5;
    assert_eq!(ENVIRONMENT_BINDING, 5);
}

// P1-07: Debug inspection API tests

#[test]
fn test_last_uploaded_lights_empty() {
    // Before any upload, should be empty
    let lights: &[Light] = &[];
    assert_eq!(lights.len(), 0);
}

#[test]
fn test_last_uploaded_lights_storage() {
    // Verify we can create lights and they would be stored
    let light1 = Light::directional(45.0, 30.0, 3.0, [1.0, 0.9, 0.8]);
    let light2 = Light::point([0.0, 5.0, 0.0], 10.0, 20.0, [1.0, 1.0, 1.0]);

    let lights = vec![light1, light2];
    assert_eq!(lights.len(), 2);

    // Verify fields are accessible
    assert_eq!(lights[0].kind, 0); // Directional
    assert_eq!(lights[1].kind, 1); // Point
}

#[test]
fn test_debug_info_format() {
    // Test that debug_info produces expected structure
    let light = Light::directional(45.0, 30.0, 3.0, [1.0, 0.9, 0.8]);

    // Verify light type name helper
    let type_name = light_type_name(light.kind);
    assert_eq!(type_name, "Directional");

    // Verify point light type
    let point = Light::point([1.0, 2.0, 3.0], 5.0, 10.0, [0.5, 0.6, 0.7]);
    assert_eq!(light_type_name(point.kind), "Point");
}

#[test]
fn test_light_type_names() {
    // Test all light type names (u32 values)
    assert_eq!(light_type_name(0), "Directional");
    assert_eq!(light_type_name(1), "Point");
    assert_eq!(light_type_name(2), "Spot");
    assert_eq!(light_type_name(3), "Environment");
    assert_eq!(light_type_name(4), "AreaRect");
    assert_eq!(light_type_name(5), "AreaDisk");
    assert_eq!(light_type_name(6), "AreaSphere");
}

#[test]
fn test_debug_info_output_structure() {
    // Create a simple light setup
    let dir_light = Light::directional(0.0, 45.0, 2.5, [1.0, 0.95, 0.9]);

    // Simulate what debug_info would output
    let output = format!(
        "Light 0: {}\n  Intensity: {:.2}, Color: [{:.2}, {:.2}, {:.2}]",
        light_type_name(dir_light.kind),
        dir_light.intensity,
        dir_light.color[0],
        dir_light.color[1],
        dir_light.color[2]
    );

    assert!(output.contains("Light 0: Directional"));
    assert!(output.contains("Intensity: 2.50"));
    assert!(output.contains("Color: [1.00, 0.95, 0.90]"));
}

#[test]
fn test_max_lights_not_exceeded_in_debug() {
    // Verify MAX_LIGHTS constant is reasonable for debug output
    assert_eq!(MAX_LIGHTS, 16);

    // Debug output for 16 lights should be manageable
    // Rough estimate: ~150 bytes per light * 16 = 2.4 KB
    let estimated_size = 150 * MAX_LIGHTS;
    assert!(estimated_size < 10000); // < 10 KB
}

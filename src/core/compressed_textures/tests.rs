use super::bc4::decode_channel;
use super::compression::{calculate_mip_levels, compress_rgba_to_format, rgb8_to_rgb565};
use super::load::generate_rgba_mip_chain;
use super::*;

#[test]
fn test_mip_level_calculation() {
    assert_eq!(calculate_mip_levels(256, 256), 9);
    assert_eq!(calculate_mip_levels(512, 256), 10);
    assert_eq!(calculate_mip_levels(1, 1), 1);
}

#[test]
fn rgba_mip_chain_matches_webgpu_extents() {
    // 5x3 floors per axis: 5x3 -> 2x1 -> 1x1 (three levels, not div_ceil).
    let base: Vec<u8> = (0..(5 * 3 * 4)).map(|i| (i % 251) as u8).collect();
    let chain = generate_rgba_mip_chain(&base, 5, 3, calculate_mip_levels(5, 3));

    assert_eq!(calculate_mip_levels(5, 3), 3);
    assert_eq!(chain.len(), 3);
    assert_eq!(chain[0].len(), 5 * 3 * 4);
    assert_eq!(chain[1].len(), 2 * 1 * 4);
    assert_eq!(chain[2].len(), 1 * 1 * 4);
}

#[test]
fn rgba_mip_chain_box_filters_to_channel_average() {
    let base = vec![
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
    ];
    let chain = generate_rgba_mip_chain(&base, 2, 2, 2);

    // Each 1x1 channel is the mean of its four source pixels.
    assert_eq!(chain[1], vec![60, 70, 80, 90]);
}

#[test]
fn packed_level_layouts_match_compressed_chain_sizes() {
    // 4x4 BC1 chain: three levels, one 4x4 block each (8 bytes per level).
    let image = CompressedImage {
        data: vec![0u8; 24],
        width: 4,
        height: 4,
        mip_levels: 3,
        format: wgpu::TextureFormat::Bc1RgbaUnorm,
        is_srgb: false,
        source_format: "Test".to_string(),
    };

    let levels = image.packed_level_layouts().expect("BC1 is registered");
    let sizes: Vec<u64> = levels.iter().map(|(_, layout)| layout.payload_bytes).collect();
    assert_eq!(sizes, vec![8, 8, 8]);
    assert_eq!(sizes.iter().sum::<u64>(), image.data.len() as u64);
}

#[test]
fn test_compression_stats() {
    let image = CompressedImage {
        data: vec![0u8; 1024],
        width: 64,
        height: 64,
        mip_levels: 1,
        format: wgpu::TextureFormat::Bc1RgbaUnorm,
        is_srgb: false,
        source_format: "Test".to_string(),
    };

    let stats = image.get_compression_stats();
    assert_eq!(stats.uncompressed_size, 64 * 64 * 4);
    assert_eq!(stats.compressed_size, 1024);
    assert!((stats.compression_ratio - 16.0).abs() < 0.1);
}

#[test]
fn test_rgb565_conversion() {
    assert_eq!(rgb8_to_rgb565(255, 255, 255), 0xFFFF);
    assert_eq!(rgb8_to_rgb565(0, 0, 0), 0x0000);
    assert_eq!(rgb8_to_rgb565(255, 0, 0), 0xF800);
}

#[test]
fn bc4_compression_uses_real_bc4_encoder_on_red_channel() {
    let mut rgba = Vec::with_capacity(4 * 4 * 4);
    for _ in 0..16 {
        rgba.extend_from_slice(&[200, 10, 20, 255]);
    }
    let blocks =
        compress_rgba_to_format(&rgba, 4, 4, wgpu::TextureFormat::Bc4RUnorm).expect("BC4 encode");
    assert_eq!(blocks.len(), 8);
    assert_eq!(decode_channel(&blocks, 4, 4).unwrap(), vec![200u8; 16]);
}

#[test]
fn etc2_and_unimplemented_compression_fail_loudly_without_zero_blocks() {
    use crate::core::degradation::{begin_degradation_capture, finish_degradation_capture};

    let rgba = vec![128u8; 4 * 4 * 4];
    for format in [
        wgpu::TextureFormat::Etc2Rgb8Unorm,
        wgpu::TextureFormat::Etc2Rgb8UnormSrgb,
        wgpu::TextureFormat::Etc2Rgba8Unorm,
        wgpu::TextureFormat::Etc2Rgba8UnormSrgb,
    ] {
        begin_degradation_capture();
        let result = compress_rgba_to_format(&rgba, 4, 4, format);
        let captured = finish_degradation_capture();
        assert!(result.is_err(), "{format:?} must not fabricate zero blocks");
        assert!(
            captured
                .iter()
                .any(|d| d.kind == "format_unsupported" && d.name == "compressed_textures.etc2"),
            "{format:?} must record the compressed_textures.etc2 degradation"
        );
    }

    begin_degradation_capture();
    let result = compress_rgba_to_format(&rgba, 4, 4, wgpu::TextureFormat::Bc6hRgbUfloat);
    let captured = finish_degradation_capture();
    assert!(result.is_err());
    assert!(captured
        .iter()
        .any(|d| d.kind == "format_unsupported" && d.name == "compressed_textures.encode"));
}

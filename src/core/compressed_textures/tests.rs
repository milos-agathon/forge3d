use super::bc4::decode_channel;
use super::compression::{calculate_mip_levels, compress_rgba_to_format, rgb8_to_rgb565};
use super::*;

#[test]
fn test_mip_level_calculation() {
    assert_eq!(calculate_mip_levels(256, 256), 9);
    assert_eq!(calculate_mip_levels(512, 256), 10);
    assert_eq!(calculate_mip_levels(1, 1), 1);
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

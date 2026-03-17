use super::compression::{calculate_mip_levels, rgb8_to_rgb565};
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

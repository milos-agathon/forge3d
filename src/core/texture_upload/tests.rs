use super::hdr::{align_copy_bytes_per_row, validate_config};
use super::*;

#[test]
fn test_hdr_format_bytes_per_pixel() {
    assert_eq!(HdrFormat::Rgba16Float.bytes_per_pixel(), 8);
    assert_eq!(HdrFormat::Rgba32Float.bytes_per_pixel(), 16);
}

#[test]
fn test_align_copy_bytes_per_row() {
    assert_eq!(align_copy_bytes_per_row(100), 256);
    assert_eq!(align_copy_bytes_per_row(256), 256);
    assert_eq!(align_copy_bytes_per_row(300), 512);
}

#[test]
fn test_validate_config() {
    let mut config = HdrTextureConfig::default();

    config.width = 512;
    config.height = 512;
    assert!(validate_config(&config).is_ok());

    config.width = 0;
    assert!(validate_config(&config).is_err());

    config.width = 512;
    config.height = 0;
    assert!(validate_config(&config).is_err());

    config.width = 20000;
    config.height = 20000;
    assert!(validate_config(&config).is_err());
}

#[test]
fn test_rgb_to_rgba_conversion() {
    let rgb_data = [1.0, 0.5, 0.25, 0.75, 1.0, 0.5];
    let alpha = 0.8;
    let expected_rgba = [1.0, 0.5, 0.25, 0.8, 0.75, 1.0, 0.5, 0.8];

    let pixel_count = 2;
    let mut rgba_data = Vec::with_capacity(pixel_count * 4);
    for i in 0..pixel_count {
        let base = i * 3;
        rgba_data.push(rgb_data[base]);
        rgba_data.push(rgb_data[base + 1]);
        rgba_data.push(rgb_data[base + 2]);
        rgba_data.push(alpha);
    }

    assert_eq!(rgba_data.as_slice(), expected_rgba);
}

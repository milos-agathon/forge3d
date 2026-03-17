use super::*;
use wgpu::{BufferUsages, TextureFormat};

#[test]
fn registry_basic_operations() {
    let registry = ResourceRegistry::new();

    let metrics = registry.get_metrics();
    assert_eq!(metrics.buffer_count, 0);
    assert_eq!(metrics.buffer_bytes, 0);
    assert!(metrics.within_budget);

    registry.track_buffer_allocation(1024, true);
    let metrics = registry.get_metrics();
    assert_eq!(metrics.buffer_count, 1);
    assert_eq!(metrics.buffer_bytes, 1024);
    assert_eq!(metrics.host_visible_bytes, 1024);

    registry.free_buffer_allocation(1024, true);
    let metrics = registry.get_metrics();
    assert_eq!(metrics.buffer_count, 0);
    assert_eq!(metrics.buffer_bytes, 0);
    assert_eq!(metrics.host_visible_bytes, 0);
}

#[test]
fn budget_checking() {
    let registry = ResourceRegistry::new();
    assert!(registry.check_budget(100 * 1024 * 1024).is_ok());
    assert!(registry.check_budget(600 * 1024 * 1024).is_err());
}

#[test]
fn host_visible_detection() {
    assert!(is_host_visible_usage(BufferUsages::MAP_READ));
    assert!(is_host_visible_usage(BufferUsages::MAP_WRITE));
    assert!(is_host_visible_usage(
        BufferUsages::COPY_DST | BufferUsages::MAP_READ
    ));
    assert!(!is_host_visible_usage(BufferUsages::VERTEX));
    assert!(!is_host_visible_usage(BufferUsages::INDEX));
}

#[test]
fn texture_format_sizes() {
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::R8Unorm),
        16 * 16
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Rg8Unorm),
        16 * 16 * 2
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::R16Float),
        16 * 16 * 2
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Depth16Unorm),
        16 * 16 * 2
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Rgba8Unorm),
        16 * 16 * 4
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Rgba8UnormSrgb),
        16 * 16 * 4
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Bgra8Unorm),
        16 * 16 * 4
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::R32Float),
        16 * 16 * 4
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Depth32Float),
        16 * 16 * 4
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Depth24Plus),
        16 * 16 * 4
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Depth24PlusStencil8),
        16 * 16 * 4
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Rgba16Float),
        16 * 16 * 8
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Rg32Float),
        16 * 16 * 8
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Depth32FloatStencil8),
        16 * 16 * 8
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Rgba32Float),
        16 * 16 * 16
    );
}

#[test]
fn compressed_texture_sizes() {
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Bc1RgbaUnorm),
        4 * 4 * 8
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Bc3RgbaUnorm),
        4 * 4 * 16
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Bc5RgUnorm),
        4 * 4 * 16
    );
    assert_eq!(
        calculate_texture_size(17, 17, TextureFormat::Bc1RgbaUnorm),
        5 * 5 * 8
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Etc2Rgb8Unorm),
        4 * 4 * 8
    );
    assert_eq!(
        calculate_texture_size(16, 16, TextureFormat::Etc2Rgba8Unorm),
        4 * 4 * 16
    );
}

#[test]
fn compressed_texture_size_calculation() {
    assert_eq!(calculate_compressed_texture_size(16, 16, 8, 4), 4 * 4 * 8);
    assert_eq!(calculate_compressed_texture_size(15, 15, 8, 4), 4 * 4 * 8);
    assert_eq!(calculate_compressed_texture_size(17, 17, 8, 4), 5 * 5 * 8);
    assert_eq!(calculate_compressed_texture_size(16, 16, 16, 8), 2 * 2 * 16);
}

#[test]
fn memory_accounting_accuracy() {
    let registry = ResourceRegistry::new();
    let test_cases = [
        (TextureFormat::R8Unorm, 1024, 1024, 1024 * 1024 * 1),
        (TextureFormat::Rg8Unorm, 512, 512, 512 * 512 * 2),
        (TextureFormat::Rgba8Unorm, 256, 256, 256 * 256 * 4),
        (TextureFormat::R16Float, 512, 512, 512 * 512 * 2),
        (TextureFormat::Rgba16Float, 128, 128, 128 * 128 * 8),
        (TextureFormat::R32Float, 256, 256, 256 * 256 * 4),
        (TextureFormat::Rgba32Float, 64, 64, 64 * 64 * 16),
    ];

    let mut expected_total = 0;
    for (format, width, height, expected_size) in test_cases {
        registry.track_texture_allocation(width, height, format);
        expected_total += expected_size;

        let metrics = registry.get_metrics();
        assert_eq!(metrics.texture_bytes, expected_total);
    }
}

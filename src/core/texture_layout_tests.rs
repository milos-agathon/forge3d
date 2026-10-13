use super::*;
use crate::core::memory_tracker::calculate_texture_size;
use crate::core::texture_format::TextureFormatRegistry;
use wgpu::{AstcBlock, AstcChannel};

fn extent(width: u32, height: u32, slices: u32) -> Extent3d {
    Extent3d {
        width,
        height,
        depth_or_array_layers: slices,
    }
}

#[test]
fn format_families_and_odd_extents() {
    use TextureFormat::*;
    // Independent expected byte counts for 7x5, including partial blocks.
    let cases = [
        (R8Unorm, 35),
        (Stencil8, 35),
        (R16Unorm, 70),
        (Rg8Unorm, 70),
        (Rgba8Unorm, 140),
        (Bgra8UnormSrgb, 140),
        (Rgb9e5Ufloat, 140),
        (Rg16Unorm, 140),
        (Rgba16Unorm, 280),
        (Rgba32Float, 560),
        (Depth24Plus, 140),
        (Depth24PlusStencil8, 140),
        (Depth32FloatStencil8, 280),
        (Bc1RgbaUnorm, 32),
        (Bc2RgbaUnorm, 64),
        (Bc3RgbaUnorm, 64),
        (Bc4RSnorm, 32),
        (Bc5RgUnorm, 64),
        (Bc6hRgbFloat, 64),
        (Bc7RgbaUnormSrgb, 64),
        (Etc2Rgb8Unorm, 32),
        (Etc2Rgb8A1UnormSrgb, 32),
        (Etc2Rgba8Unorm, 64),
        (EacR11Snorm, 32),
        (EacRg11Unorm, 64),
    ];
    for (format, expected) in cases {
        assert_eq!(resident_2d_size(7, 5, format), expected, "{format:?}");
        assert_eq!(
            calculate_texture_size(7, 5, format),
            expected,
            "legacy helper: {format:?}"
        );
        assert_eq!(resident_2d_size(0, 5, format), 0);
    }
}

#[test]
fn every_astc_block_shape_uses_its_own_geometry() {
    use AstcBlock::*;
    // 13x11 texels: the expected number of occupied blocks varies by shape.
    let cases = [
        (B4x4, 192),
        (B5x4, 144),
        (B5x5, 144),
        (B6x5, 144),
        (B6x6, 96),
        (B8x5, 96),
        (B8x6, 64),
        (B8x8, 64),
        (B10x5, 96),
        (B10x6, 64),
        (B10x8, 64),
        (B10x10, 64),
        (B12x10, 64),
        (B12x12, 32),
    ];
    for channel in [AstcChannel::Unorm, AstcChannel::UnormSrgb, AstcChannel::Hdr] {
        for (block, expected) in cases {
            let format = TextureFormat::Astc { block, channel };
            assert_eq!(
                calculate_texture_size(13, 11, format),
                expected,
                "{format:?}"
            );
            let transfer =
                TransferLayout::queue_write(format, TextureAspect::All, extent(13, 11, 1)).unwrap();
            assert_eq!(transfer.payload_bytes, expected);
        }
    }
}

#[test]
fn registry_metadata_and_size_use_the_same_facts() {
    let registry = TextureFormatRegistry::new();
    for info in super::super::texture_format_defs::all_format_definitions() {
        let format = info.format;
        let stored = registry.get_format_info(format).unwrap();
        assert_eq!(
            stored.bytes_per_pixel,
            format.block_copy_size(None).unwrap()
        );
        assert_eq!(stored.block_size, format.block_dimensions().0);
        assert_eq!(
            stored.calculate_size(7, 5),
            calculate_texture_size(7, 5, format)
        );
        assert_eq!(stored.is_compressed, format.is_compressed());
        assert_eq!(stored.is_srgb, format.is_srgb());
    }
    // Caller-constructed records retain their field-based sizing contract.
    let mut info = *registry
        .get_format_info(TextureFormat::Bc1RgbaUnorm)
        .unwrap();
    info.bytes_per_pixel = 999;
    info.block_size = 1;
    assert_eq!(info.calculate_size(7, 5), 34_965);
    info.bytes_per_pixel = 32;
    info.block_size = 8;
    assert_eq!(info.calculate_size(15, 17), 192);
    info.is_compressed = false;
    info.bytes_per_pixel = 8;
    assert_eq!(info.calculate_size(7, 5), 280);
}

#[test]
fn public_copy_alignment_uses_checked_shared_geometry() {
    for (input, expected) in [(0, 0), (100, 256), (256, 256), (300, 512)] {
        assert_eq!(crate::core::gpu::align_copy_bpr(input), expected);
    }
    assert_eq!(
        crate::core::gpu::align_copy_bpr(u32::MAX - 255),
        u32::MAX - 255
    );
    assert!(std::panic::catch_unwind(|| crate::core::gpu::align_copy_bpr(u32::MAX)).is_err());
}

#[test]
fn mips_layers_volumes_and_samples_are_distinct() {
    assert_eq!(
        resident_size(
            TextureFormat::Bc1RgbaUnorm,
            extent(7, 5, 3),
            TextureDimension::D2,
            3,
            1
        ),
        144
    );
    assert_eq!(
        resident_size(
            TextureFormat::Rgba8Unorm,
            extent(8, 8, 2),
            TextureDimension::D2,
            3,
            4
        ),
        2688
    );
    assert_eq!(
        resident_size(
            TextureFormat::R8Unorm,
            extent(4, 4, 4),
            TextureDimension::D3,
            3,
            1
        ),
        73
    );
    assert_eq!(
        resident_size(
            TextureFormat::R16Float,
            extent(5, 3, 3),
            TextureDimension::D3,
            3,
            1
        ),
        96
    );
    assert_eq!(
        resident_size(
            TextureFormat::R16Float,
            extent(5, 1, 1),
            TextureDimension::D1,
            3,
            1
        ),
        16
    );
    assert_eq!(
        mip_extent(extent(7, 5, 3), TextureDimension::D2, 1),
        extent(3, 2, 3)
    );
    assert_eq!(
        mip_extent(extent(7, 5, 3), TextureDimension::D3, 1),
        extent(3, 2, 1)
    );
    assert_eq!(
        mip_extent(extent(1, 1, 1), TextureDimension::D3, u32::BITS),
        extent(1, 1, 1)
    );
}

#[test]
fn tight_payload_and_padded_buffer_sizes_do_not_enter_resident_accounting() {
    let size = extent(7, 5, 3);
    let tight =
        TransferLayout::queue_write(TextureFormat::Bc1RgbaUnorm, TextureAspect::All, size).unwrap();
    let padded =
        TransferLayout::buffer_copy(TextureFormat::Bc1RgbaUnorm, TextureAspect::All, size).unwrap();
    assert_eq!(
        (
            tight.row_bytes,
            tight.rows_per_image,
            tight.payload_bytes,
            tight.transfer_bytes
        ),
        (16, 2, 96, 96)
    );
    assert_eq!(
        (
            padded.bytes_per_row,
            padded.payload_bytes,
            padded.transfer_bytes
        ),
        (256, 96, 1536)
    );
    assert_eq!(padded.image_data_layout().rows_per_image, Some(2));
    assert_eq!(
        resident_size(
            TextureFormat::Bc1RgbaUnorm,
            size,
            TextureDimension::D2,
            1,
            1
        ),
        96
    );
    let rgba = TransferLayout::buffer_copy(
        TextureFormat::Rgba8Unorm,
        TextureAspect::All,
        extent(7, 5, 1),
    )
    .unwrap();
    assert_eq!((rgba.payload_bytes, rgba.transfer_bytes), (140, 1280));
}

#[test]
fn depth_stencil_accounting_does_not_invent_an_all_aspect_payload() {
    let size = extent(3, 2, 1);
    assert_eq!(
        FormatLayout::new(TextureFormat::Depth32FloatStencil8).resident_level_size(size),
        48
    );
    assert!(TransferLayout::queue_write(
        TextureFormat::Depth32FloatStencil8,
        TextureAspect::All,
        size
    )
    .is_err());
    assert!(TransferLayout::queue_write(
        TextureFormat::Depth24Plus,
        TextureAspect::DepthOnly,
        size
    )
    .is_err());
    let depth = TransferLayout::queue_write(
        TextureFormat::Depth32FloatStencil8,
        TextureAspect::DepthOnly,
        size,
    )
    .unwrap();
    let stencil = TransferLayout::queue_write(
        TextureFormat::Depth32FloatStencil8,
        TextureAspect::StencilOnly,
        size,
    )
    .unwrap();
    assert_eq!(depth.payload_bytes, 24);
    assert_eq!(stencil.payload_bytes, 6);
    assert_eq!(resident_2d_size(3, 2, TextureFormat::NV12), 24); // Existing conservative ledger policy.
    assert!(TransferLayout::queue_write(TextureFormat::NV12, TextureAspect::All, size).is_err());
    let plane0 =
        TransferLayout::queue_write(TextureFormat::NV12, TextureAspect::Plane0, size).unwrap();
    let plane1 =
        TransferLayout::queue_write(TextureFormat::NV12, TextureAspect::Plane1, size).unwrap();
    assert_eq!((plane0.payload_bytes, plane1.payload_bytes), (6, 12));
    assert!(
        TransferLayout::queue_write(TextureFormat::Rgba8Unorm, TextureAspect::DepthOnly, size)
            .is_err()
    );
}

#[test]
fn accounting_saturates_and_transfers_reject_overflow_or_empty_extents() {
    assert_eq!(
        resident_size(
            TextureFormat::Rgba32Float,
            extent(u32::MAX, u32::MAX, u32::MAX),
            TextureDimension::D2,
            1,
            4
        ),
        u64::MAX
    );
    assert!(TransferLayout::queue_write(
        TextureFormat::Rgba32Float,
        TextureAspect::All,
        extent(u32::MAX, 1, 1)
    )
    .is_err());
    assert!(TransferLayout::queue_write(
        TextureFormat::R8Unorm,
        TextureAspect::All,
        extent(u32::MAX, u32::MAX, u32::MAX)
    )
    .is_err());
    assert!(align_copy_row(u32::MAX).is_err());
    assert_eq!(align_copy_row(256), Ok(256));
    for size in [extent(0, 1, 1), extent(1, 0, 1), extent(1, 1, 0)] {
        assert!(
            TransferLayout::buffer_copy(TextureFormat::Rgba8Unorm, TextureAspect::All, size)
                .is_err()
        );
    }
}

#[test]
fn compressed_image_size_includes_all_mips_without_registry_fallback() {
    let image = crate::core::compressed_textures::CompressedImage {
        data: vec![],
        width: 13,
        height: 11,
        mip_levels: 3,
        format: TextureFormat::Astc {
            block: AstcBlock::B8x5,
            channel: AstcChannel::Unorm,
        },
        is_srgb: false,
        source_format: "layout-test".to_string(),
    };
    assert_eq!(image.calculate_gpu_size(), 128); // 96 + 16 + 16, despite absent ASTC registry preset.
}

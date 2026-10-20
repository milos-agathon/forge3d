//! `VkFormat` → `wgpu::TextureFormat` mapping for KTX2 containers.
//!
//! KTX2 stores its pixel format as a Vulkan `VkFormat` enum value. The codes
//! below are the core Vulkan 1.0 values (see `VkFormat` in the Vulkan spec).

use wgpu::{AstcBlock, AstcChannel, TextureFormat};

/// Map a KTX2 `vkFormat` code to the matching wgpu texture format.
///
/// `VK_FORMAT_BC1_RGB_*` (131/132) has no wgpu equivalent and maps to the BC1
/// RGBA variant: colour decodes identically, but blocks in three-colour mode
/// decode index 3 as transparent black instead of opaque black.
pub(super) fn vk_format_to_wgpu(vk_format: u32) -> Result<TextureFormat, String> {
    use TextureFormat::*;

    let format = match vk_format {
        // 8-bit RGBA / BGRA
        37 => Rgba8Unorm,
        38 => Rgba8Snorm,
        43 => Rgba8UnormSrgb,
        44 => Bgra8Unorm,
        50 => Bgra8UnormSrgb,

        // BC1–BC7
        131 | 133 => Bc1RgbaUnorm,
        132 | 134 => Bc1RgbaUnormSrgb,
        135 => Bc2RgbaUnorm,
        136 => Bc2RgbaUnormSrgb,
        137 => Bc3RgbaUnorm,
        138 => Bc3RgbaUnormSrgb,
        139 => Bc4RUnorm,
        140 => Bc4RSnorm,
        141 => Bc5RgUnorm,
        142 => Bc5RgSnorm,
        143 => Bc6hRgbUfloat,
        144 => Bc6hRgbFloat,
        145 => Bc7RgbaUnorm,
        146 => Bc7RgbaUnormSrgb,

        // ETC2 / EAC
        147 => Etc2Rgb8Unorm,
        148 => Etc2Rgb8UnormSrgb,
        149 => Etc2Rgb8A1Unorm,
        150 => Etc2Rgb8A1UnormSrgb,
        151 => Etc2Rgba8Unorm,
        152 => Etc2Rgba8UnormSrgb,
        153 => EacR11Unorm,
        154 => EacR11Snorm,
        155 => EacRg11Unorm,
        156 => EacRg11Snorm,

        // ASTC LDR: 157..=184, one UNORM/SRGB pair per block size.
        157..=184 => {
            const BLOCKS: [AstcBlock; 14] = [
                AstcBlock::B4x4,
                AstcBlock::B5x4,
                AstcBlock::B5x5,
                AstcBlock::B6x5,
                AstcBlock::B6x6,
                AstcBlock::B8x5,
                AstcBlock::B8x6,
                AstcBlock::B8x8,
                AstcBlock::B10x5,
                AstcBlock::B10x6,
                AstcBlock::B10x8,
                AstcBlock::B10x10,
                AstcBlock::B12x10,
                AstcBlock::B12x12,
            ];
            let offset = (vk_format - 157) as usize;
            let channel = if offset.is_multiple_of(2) {
                AstcChannel::Unorm
            } else {
                AstcChannel::UnormSrgb
            };
            Astc {
                block: BLOCKS[offset / 2],
                channel,
            }
        }

        _ => return Err(format!("Unsupported Vulkan format: {}", vk_format)),
    };
    Ok(format)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected mappings, written out against the `VkFormat` names in the
    /// Vulkan spec rather than derived from the implementation's arithmetic.
    const EXPECTED: &[(u32, &str, TextureFormat)] = &[
        (37, "R8G8B8A8_UNORM", TextureFormat::Rgba8Unorm),
        (38, "R8G8B8A8_SNORM", TextureFormat::Rgba8Snorm),
        (43, "R8G8B8A8_SRGB", TextureFormat::Rgba8UnormSrgb),
        (44, "B8G8R8A8_UNORM", TextureFormat::Bgra8Unorm),
        (50, "B8G8R8A8_SRGB", TextureFormat::Bgra8UnormSrgb),
        (131, "BC1_RGB_UNORM_BLOCK", TextureFormat::Bc1RgbaUnorm),
        (132, "BC1_RGB_SRGB_BLOCK", TextureFormat::Bc1RgbaUnormSrgb),
        (133, "BC1_RGBA_UNORM_BLOCK", TextureFormat::Bc1RgbaUnorm),
        (134, "BC1_RGBA_SRGB_BLOCK", TextureFormat::Bc1RgbaUnormSrgb),
        (135, "BC2_UNORM_BLOCK", TextureFormat::Bc2RgbaUnorm),
        (136, "BC2_SRGB_BLOCK", TextureFormat::Bc2RgbaUnormSrgb),
        (137, "BC3_UNORM_BLOCK", TextureFormat::Bc3RgbaUnorm),
        (138, "BC3_SRGB_BLOCK", TextureFormat::Bc3RgbaUnormSrgb),
        (139, "BC4_UNORM_BLOCK", TextureFormat::Bc4RUnorm),
        (140, "BC4_SNORM_BLOCK", TextureFormat::Bc4RSnorm),
        (141, "BC5_UNORM_BLOCK", TextureFormat::Bc5RgUnorm),
        (142, "BC5_SNORM_BLOCK", TextureFormat::Bc5RgSnorm),
        (143, "BC6H_UFLOAT_BLOCK", TextureFormat::Bc6hRgbUfloat),
        (144, "BC6H_SFLOAT_BLOCK", TextureFormat::Bc6hRgbFloat),
        (145, "BC7_UNORM_BLOCK", TextureFormat::Bc7RgbaUnorm),
        (146, "BC7_SRGB_BLOCK", TextureFormat::Bc7RgbaUnormSrgb),
        (147, "ETC2_R8G8B8_UNORM_BLOCK", TextureFormat::Etc2Rgb8Unorm),
        (148, "ETC2_R8G8B8_SRGB_BLOCK", TextureFormat::Etc2Rgb8UnormSrgb),
        (149, "ETC2_R8G8B8A1_UNORM_BLOCK", TextureFormat::Etc2Rgb8A1Unorm),
        (150, "ETC2_R8G8B8A1_SRGB_BLOCK", TextureFormat::Etc2Rgb8A1UnormSrgb),
        (151, "ETC2_R8G8B8A8_UNORM_BLOCK", TextureFormat::Etc2Rgba8Unorm),
        (152, "ETC2_R8G8B8A8_SRGB_BLOCK", TextureFormat::Etc2Rgba8UnormSrgb),
        (153, "EAC_R11_UNORM_BLOCK", TextureFormat::EacR11Unorm),
        (154, "EAC_R11_SNORM_BLOCK", TextureFormat::EacR11Snorm),
        (155, "EAC_R11G11_UNORM_BLOCK", TextureFormat::EacRg11Unorm),
        (156, "EAC_R11G11_SNORM_BLOCK", TextureFormat::EacRg11Snorm),
    ];

    const fn astc(block: AstcBlock, channel: AstcChannel) -> TextureFormat {
        TextureFormat::Astc { block, channel }
    }

    const EXPECTED_ASTC: &[(u32, &str, TextureFormat)] = &[
        (157, "ASTC_4x4_UNORM_BLOCK", astc(AstcBlock::B4x4, AstcChannel::Unorm)),
        (158, "ASTC_4x4_SRGB_BLOCK", astc(AstcBlock::B4x4, AstcChannel::UnormSrgb)),
        (163, "ASTC_6x5_UNORM_BLOCK", astc(AstcBlock::B6x5, AstcChannel::Unorm)),
        (164, "ASTC_6x5_SRGB_BLOCK", astc(AstcBlock::B6x5, AstcChannel::UnormSrgb)),
        (171, "ASTC_8x8_UNORM_BLOCK", astc(AstcBlock::B8x8, AstcChannel::Unorm)),
        (177, "ASTC_10x8_UNORM_BLOCK", astc(AstcBlock::B10x8, AstcChannel::Unorm)),
        (180, "ASTC_10x10_SRGB_BLOCK", astc(AstcBlock::B10x10, AstcChannel::UnormSrgb)),
        (181, "ASTC_12x10_UNORM_BLOCK", astc(AstcBlock::B12x10, AstcChannel::Unorm)),
        (184, "ASTC_12x12_SRGB_BLOCK", astc(AstcBlock::B12x12, AstcChannel::UnormSrgb)),
    ];

    #[test]
    fn vk_codes_map_to_their_spec_formats() {
        for &(code, name, expected) in EXPECTED.iter().chain(EXPECTED_ASTC) {
            assert_eq!(
                vk_format_to_wgpu(code),
                Ok(expected),
                "VK_FORMAT_{name} ({code})"
            );
        }
    }

    #[test]
    fn srgb_flag_follows_the_vk_srgb_codes() {
        for &(code, name, _) in EXPECTED.iter().chain(EXPECTED_ASTC) {
            let format = vk_format_to_wgpu(code).unwrap();
            assert_eq!(
                format.is_srgb(),
                name.contains("_SRGB"),
                "VK_FORMAT_{name} ({code})"
            );
        }
    }

    #[test]
    fn every_astc_code_maps_to_an_astc_format() {
        for code in 157..=184u32 {
            assert!(
                matches!(vk_format_to_wgpu(code), Ok(TextureFormat::Astc { .. })),
                "{code}"
            );
        }
    }

    #[test]
    fn unknown_codes_are_rejected() {
        for code in [0u32, 1, 36, 130, 185, 1_000_066_000] {
            assert!(vk_format_to_wgpu(code).is_err(), "{code}");
        }
    }
}

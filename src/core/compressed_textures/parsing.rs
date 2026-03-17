use wgpu::TextureFormat;

use super::CompressedImage;

pub(super) struct Ktx2Header {
    pub(super) pixel_width: u32,
    pub(super) pixel_height: u32,
    pub(super) level_count: u32,
    pub(super) vk_format: u32,
}

impl Ktx2Header {
    pub(super) fn vk_format_to_wgpu(&self) -> Result<TextureFormat, String> {
        match self.vk_format {
            _ => Err(format!("Unsupported Vulkan format: {}", self.vk_format)),
        }
    }

    pub(super) fn is_srgb(&self) -> bool {
        false
    }
}

pub(super) struct DdsHeader {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) mip_map_count: u32,
    pub(super) pixel_format: u32,
}

impl DdsHeader {
    pub(super) fn pixel_format_to_wgpu(&self) -> Result<TextureFormat, String> {
        match self.pixel_format {
            _ => Err(format!(
                "Unsupported DDS pixel format: {}",
                self.pixel_format
            )),
        }
    }

    pub(super) fn is_srgb(&self) -> bool {
        false
    }
}

/// Parse KTX2 header; currently unimplemented.
pub(super) fn parse_ktx2_header(_data: &[u8]) -> Result<Ktx2Header, String> {
    Err("KTX2 parsing not implemented".to_string())
}

/// Extract KTX2 texture data; currently unimplemented.
pub(super) fn extract_ktx2_texture_data(
    _data: &[u8],
    _header: &Ktx2Header,
) -> Result<CompressedImage, String> {
    Err("KTX2 extraction not implemented".to_string())
}

/// Parse DDS header; currently unimplemented.
pub(super) fn parse_dds_header(_data: &[u8]) -> Result<DdsHeader, String> {
    Err("DDS parsing not implemented".to_string())
}

/// Extract DDS texture data; currently unimplemented.
pub(super) fn extract_dds_texture_data(
    _data: &[u8],
    _header: &DdsHeader,
) -> Result<Vec<u8>, String> {
    Err("DDS extraction not implemented".to_string())
}

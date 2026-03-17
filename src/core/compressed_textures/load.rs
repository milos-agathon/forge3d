use std::path::Path;

use wgpu::{Device, TextureFormat};

use crate::core::texture_format::global_format_registry;

use super::compression::{calculate_mip_levels, compress_rgba_to_format};
use super::parsing::{
    extract_dds_texture_data, extract_ktx2_texture_data, parse_dds_header, parse_ktx2_header,
};
use super::{CompressedImage, CompressionOptions};

impl CompressedImage {
    /// Create from raw image data
    pub fn from_rgba_data(
        data: &[u8],
        width: u32,
        height: u32,
        device: &Device,
        options: &CompressionOptions,
    ) -> Result<Self, String> {
        let start_time = std::time::Instant::now();

        if data.len() != (width * height * 4) as usize {
            return Err(format!(
                "Data size {} doesn't match dimensions {}x{}x4",
                data.len(),
                width,
                height
            ));
        }

        let format = options.target_format.unwrap_or_else(|| {
            global_format_registry()
                .select_best_compressed_format(
                    options.use_case,
                    &device.features(),
                    options.quality,
                )
                .unwrap_or(TextureFormat::Bc7RgbaUnorm)
        });
        let format_info = global_format_registry()
            .get_format_info(format)
            .ok_or_else(|| format!("Unsupported format: {:?}", format))?;

        if !global_format_registry().is_format_supported(format, &device.features()) {
            return Err(format!("Format {:?} not supported by device", format));
        }

        let compressed_data = compress_rgba_to_format(data, width, height, format)?;
        let mip_levels = if options.generate_mipmaps {
            calculate_mip_levels(width, height)
        } else {
            1
        };
        let _compression_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(Self {
            data: compressed_data,
            width,
            height,
            mip_levels,
            format,
            is_srgb: format_info.is_srgb,
            source_format: "RGBA8".to_string(),
        })
    }

    /// Load from KTX2 file
    pub fn from_ktx2<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let data = std::fs::read(path)
            .map_err(|e| format!("Failed to read KTX2 file {}: {}", path.display(), e))?;
        Self::from_ktx2_data(&data)
    }

    /// Load from KTX2 data in memory
    pub fn from_ktx2_data(data: &[u8]) -> Result<Self, String> {
        let header = parse_ktx2_header(data)?;
        let texture_data = extract_ktx2_texture_data(data, &header)?;

        Ok(Self {
            data: texture_data.data,
            width: header.pixel_width,
            height: header.pixel_height,
            mip_levels: header.level_count.max(1),
            format: header.vk_format_to_wgpu()?,
            is_srgb: header.is_srgb(),
            source_format: "KTX2".to_string(),
        })
    }

    /// Load from DDS file (basic support)
    pub fn from_dds<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let data = std::fs::read(path)
            .map_err(|e| format!("Failed to read DDS file {}: {}", path.display(), e))?;
        Self::from_dds_data(&data)
    }

    /// Load from DDS data in memory
    pub fn from_dds_data(data: &[u8]) -> Result<Self, String> {
        let header = parse_dds_header(data)?;
        let texture_data = extract_dds_texture_data(data, &header)?;

        Ok(Self {
            data: texture_data,
            width: header.width,
            height: header.height,
            mip_levels: header.mip_map_count.max(1),
            format: header.pixel_format_to_wgpu()?,
            is_srgb: header.is_srgb(),
            source_format: "DDS".to_string(),
        })
    }
}

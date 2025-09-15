//! O3: Compressed texture pipeline
//!
//! This module provides compressed texture loading, decoding, and GPU upload
//! capabilities with format detection and device budget constraints.

use crate::core::memory_tracker::global_tracker;
use crate::core::texture_format::{global_format_registry, CompressionQuality, TextureUseCase};
use std::path::Path;
use wgpu::{
    Device, Extent3d, ImageCopyTexture, ImageDataLayout, Origin3d, Queue, Texture, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};

/// Compressed image data with metadata
#[derive(Debug, Clone)]
pub struct CompressedImage {
    /// Raw compressed data
    pub data: Vec<u8>,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Number of mip levels
    pub mip_levels: u32,
    /// Texture format
    pub format: TextureFormat,
    /// Whether this is sRGB format
    pub is_srgb: bool,
    /// Original file format
    pub source_format: String,
}

/// Compression statistics and metrics
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Original uncompressed size in bytes
    pub uncompressed_size: u64,
    /// Compressed size in bytes
    pub compressed_size: u64,
    /// Compression ratio (uncompressed / compressed)
    pub compression_ratio: f32,
    /// Time taken to compress in milliseconds
    pub compression_time_ms: f64,
    /// Quality metric (0.0-1.0, higher is better)
    pub quality_score: f32,
    /// Peak Signal-to-Noise Ratio in dB
    pub psnr_db: f32,
}

/// Texture compression options
#[derive(Debug, Clone)]
pub struct CompressionOptions {
    /// Target format (None = auto-select)
    pub target_format: Option<TextureFormat>,
    /// Quality level
    pub quality: CompressionQuality,
    /// Generate mip maps
    pub generate_mipmaps: bool,
    /// Use case for format selection
    pub use_case: TextureUseCase,
    /// Maximum texture size (power of 2)
    pub max_size: u32,
    /// Whether to enforce power-of-2 dimensions
    pub force_power_of_2: bool,
}

impl Default for CompressionOptions {
    fn default() -> Self {
        Self {
            target_format: None,
            quality: CompressionQuality::Normal,
            generate_mipmaps: true,
            use_case: TextureUseCase::Albedo,
            max_size: 4096,
            force_power_of_2: false,
        }
    }
}

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

        // Validate input
        if data.len() != (width * height * 4) as usize {
            return Err(format!(
                "Data size {} doesn't match dimensions {}x{}x4",
                data.len(),
                width,
                height
            ));
        }

        // Select target format
        let format = options.target_format.unwrap_or_else(|| {
            global_format_registry()
                .select_best_compressed_format(
                    options.use_case,
                    &device.features(),
                    options.quality,
                )
                .unwrap_or(TextureFormat::Bc7RgbaUnorm)
        });

        // Get format info
        let format_info = global_format_registry()
            .get_format_info(format)
            .ok_or_else(|| format!("Unsupported format: {:?}", format))?;

        // Check device support
        if !global_format_registry().is_format_supported(format, &device.features()) {
            return Err(format!("Format {:?} not supported by device", format));
        }

        // Compress the data
        let compressed_data = compress_rgba_to_format(data, width, height, format)?;

        // Calculate mip levels
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
        // Parse KTX2 header
        let header = parse_ktx2_header(data)?;

        // Extract texture data
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
        // Basic DDS support for common BC formats
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

    /// Create GPU texture from compressed image
    pub fn decode_to_gpu(
        &self,
        device: &Device,
        queue: &Queue,
        label: Option<&str>,
    ) -> Result<Texture, String> {
        // Check memory budget
        let texture_size = self.calculate_gpu_size();
        if let Err(e) = global_tracker().check_budget(texture_size) {
            return Err(e);
        }

        // Create texture descriptor
        let texture_desc = TextureDescriptor {
            label,
            size: Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: self.mip_levels,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: self.format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        };

        // Create texture
        let texture = device.create_texture(&texture_desc);

        // Upload compressed data
        self.upload_to_texture(&texture, queue)?;

        // Track texture allocation
        global_tracker().track_texture_allocation(self.width, self.height, self.format);

        Ok(texture)
    }

    /// Upload compressed data to existing texture
    pub fn upload_to_texture(&self, texture: &Texture, queue: &Queue) -> Result<(), String> {
        let format_info = global_format_registry()
            .get_format_info(self.format)
            .ok_or_else(|| format!("Unknown format: {:?}", self.format))?;

        // Calculate bytes per row for compressed format
        let bytes_per_row = if format_info.is_compressed {
            let blocks_per_row = (self.width + format_info.block_size - 1) / format_info.block_size;
            Some(
                std::num::NonZeroU32::new(blocks_per_row * format_info.bytes_per_pixel)
                    .ok_or("Invalid bytes per row calculation")?,
            )
        } else {
            Some(
                std::num::NonZeroU32::new(self.width * format_info.bytes_per_pixel)
                    .ok_or("Invalid bytes per row calculation")?,
            )
        };

        let rows_per_image = if format_info.is_compressed {
            let blocks_per_column =
                (self.height + format_info.block_size - 1) / format_info.block_size;
            Some(
                std::num::NonZeroU32::new(blocks_per_column)
                    .ok_or("Invalid rows per image calculation")?,
            )
        } else {
            Some(
                std::num::NonZeroU32::new(self.height)
                    .ok_or("Invalid rows per image calculation")?,
            )
        };

        // Upload texture data
        queue.write_texture(
            ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &self.data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: bytes_per_row.map(|v| v.get()),
                rows_per_image: rows_per_image.map(|v| v.get()),
            },
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        // Upload additional mip levels if present
        if self.mip_levels > 1 {
            self.upload_mip_levels(texture, queue)?;
        }

        Ok(())
    }

    /// Upload mip levels to texture
    fn upload_mip_levels(&self, texture: &Texture, queue: &Queue) -> Result<(), String> {
        // Generate or extract mip level data
        let mip_data = generate_mip_levels(&self.data, self.width, self.height, self.format)?;

        let format_info = global_format_registry()
            .get_format_info(self.format)
            .unwrap();

        for (mip_level, (mip_data, mip_width, mip_height)) in mip_data.iter().enumerate() {
            if mip_level == 0 {
                continue; // Skip base level, already uploaded
            }

            let bytes_per_row = if format_info.is_compressed {
                let blocks_per_row =
                    (*mip_width + format_info.block_size - 1) / format_info.block_size;
                Some(
                    std::num::NonZeroU32::new(blocks_per_row * format_info.bytes_per_pixel)
                        .unwrap(),
                )
            } else {
                Some(std::num::NonZeroU32::new(*mip_width * format_info.bytes_per_pixel).unwrap())
            };

            queue.write_texture(
                ImageCopyTexture {
                    texture,
                    mip_level: mip_level as u32,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                mip_data,
                ImageDataLayout {
                    offset: 0,
                    bytes_per_row: bytes_per_row.map(|v| v.get()),
                    rows_per_image: None,
                },
                Extent3d {
                    width: *mip_width,
                    height: *mip_height,
                    depth_or_array_layers: 1,
                },
            );
        }

        Ok(())
    }

    /// Calculate GPU memory size
    pub fn calculate_gpu_size(&self) -> u64 {
        let binding = crate::core::texture_format::TextureFormatInfo {
            format: self.format,
            is_compressed: false,
            bytes_per_pixel: 4,
            block_size: 1,
            channels: 4,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        };
        let format_info = global_format_registry()
            .get_format_info(self.format)
            .unwrap_or(&binding);

        // Calculate size including all mip levels
        let mut total_size = 0u64;
        let mut mip_width = self.width;
        let mut mip_height = self.height;

        for _ in 0..self.mip_levels {
            total_size += format_info.calculate_size(mip_width, mip_height);
            mip_width = (mip_width / 2).max(1);
            mip_height = (mip_height / 2).max(1);
        }

        total_size
    }

    /// Get compression statistics
    pub fn get_compression_stats(&self) -> CompressionStats {
        let uncompressed_size = (self.width * self.height * 4) as u64; // RGBA8 equivalent
        let compressed_size = self.data.len() as u64;
        let compression_ratio = uncompressed_size as f32 / compressed_size as f32;

        CompressionStats {
            uncompressed_size,
            compressed_size,
            compression_ratio,
            compression_time_ms: 0.0, // Would be set during compression
            quality_score: estimate_quality_score(self.format),
            psnr_db: estimate_psnr(self.format),
        }
    }
}

/// Compress RGBA data to specified format
fn compress_rgba_to_format(
    data: &[u8],
    width: u32,
    height: u32,
    target_format: TextureFormat,
) -> Result<Vec<u8>, String> {
    // This is a simplified implementation
    // In practice, you'd use libraries like:
    // - Intel ISPC Texture Compressor
    // - AMD Compressonator
    // - NVIDIA Texture Tools
    // - Basis Universal

    match target_format {
        TextureFormat::Bc1RgbaUnorm | TextureFormat::Bc1RgbaUnormSrgb => {
            compress_bc1(data, width, height)
        }
        TextureFormat::Bc3RgbaUnorm | TextureFormat::Bc3RgbaUnormSrgb => {
            compress_bc3(data, width, height)
        }
        TextureFormat::Bc7RgbaUnorm | TextureFormat::Bc7RgbaUnormSrgb => {
            compress_bc7(data, width, height)
        }
        TextureFormat::Etc2Rgb8Unorm | TextureFormat::Etc2Rgb8UnormSrgb => {
            compress_etc2_rgb(data, width, height)
        }
        TextureFormat::Etc2Rgba8Unorm | TextureFormat::Etc2Rgba8UnormSrgb => {
            compress_etc2_rgba(data, width, height)
        }
        _ => Err(format!(
            "Compression to {:?} not implemented",
            target_format
        )),
    }
}

/// Simplified BC1 compression (placeholder implementation)
fn compress_bc1(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let blocks_x = (width + 3) / 4;
    let blocks_y = (height + 3) / 4;
    let compressed_size = (blocks_x * blocks_y * 8) as usize;

    // Placeholder: In real implementation, perform actual BC1 compression
    let mut compressed = vec![0u8; compressed_size];

    // Simple color quantization placeholder
    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let block_offset = ((block_y * blocks_x + block_x) * 8) as usize;

            // Extract 4x4 block from source data
            let mut block_colors = Vec::new();
            for y in 0..4 {
                for x in 0..4 {
                    let src_x = (block_x * 4 + x).min(width - 1);
                    let src_y = (block_y * 4 + y).min(height - 1);
                    let pixel_offset = ((src_y * width + src_x) * 4) as usize;

                    if pixel_offset + 3 < data.len() {
                        block_colors.push([
                            data[pixel_offset],     // R
                            data[pixel_offset + 1], // G
                            data[pixel_offset + 2], // B
                            data[pixel_offset + 3], // A
                        ]);
                    } else {
                        block_colors.push([0, 0, 0, 0]);
                    }
                }
            }

            // Placeholder compression: store first and last colors as endpoints
            if !block_colors.is_empty() {
                let first_color = &block_colors[0];
                let last_color = &block_colors[block_colors.len() - 1];

                // Convert RGB8 to RGB565
                let color0 = rgb8_to_rgb565(first_color[0], first_color[1], first_color[2]);
                let color1 = rgb8_to_rgb565(last_color[0], last_color[1], last_color[2]);

                compressed[block_offset..block_offset + 2].copy_from_slice(&color0.to_le_bytes());
                compressed[block_offset + 2..block_offset + 4]
                    .copy_from_slice(&color1.to_le_bytes());

                // Placeholder indices (all pointing to color0)
                compressed[block_offset + 4..block_offset + 8]
                    .copy_from_slice(&[0x00, 0x00, 0x00, 0x00]);
            }
        }
    }

    Ok(compressed)
}

/// Convert RGB8 to RGB565
fn rgb8_to_rgb565(r: u8, g: u8, b: u8) -> u16 {
    let r5 = (r >> 3) as u16;
    let g6 = (g >> 2) as u16;
    let b5 = (b >> 3) as u16;
    (r5 << 11) | (g6 << 5) | b5
}

/// Placeholder BC3 compression
fn compress_bc3(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let blocks_x = (width + 3) / 4;
    let blocks_y = (height + 3) / 4;
    let compressed_size = (blocks_x * blocks_y * 16) as usize; // BC3 is 16 bytes per block

    // Placeholder: combine BC4 alpha + BC1 color
    let mut compressed = vec![0u8; compressed_size];

    // For now, just compress as BC1 and replicate to fill BC3 size
    let bc1_data = compress_bc1(data, width, height)?;

    for i in 0..blocks_x * blocks_y {
        let bc3_offset = (i * 16) as usize;
        let bc1_offset = (i * 8) as usize;

        // Alpha block (8 bytes) - placeholder
        compressed[bc3_offset..bc3_offset + 8].copy_from_slice(&[0xFF; 8]);

        // Color block (8 bytes) from BC1
        if bc1_offset + 8 <= bc1_data.len() && bc3_offset + 16 <= compressed.len() {
            compressed[bc3_offset + 8..bc3_offset + 16]
                .copy_from_slice(&bc1_data[bc1_offset..bc1_offset + 8]);
        }
    }

    Ok(compressed)
}

/// Placeholder BC7 compression
fn compress_bc7(_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let blocks_x = (width + 3) / 4;
    let blocks_y = (height + 3) / 4;
    let compressed_size = (blocks_x * blocks_y * 16) as usize; // BC7 is 16 bytes per block

    // Placeholder implementation - in practice, use Intel ISPC or similar
    Ok(vec![0u8; compressed_size])
}

/// Placeholder ETC2 RGB compression
fn compress_etc2_rgb(_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let blocks_x = (width + 3) / 4;
    let blocks_y = (height + 3) / 4;
    let compressed_size = (blocks_x * blocks_y * 8) as usize; // ETC2 RGB is 8 bytes per block

    // Placeholder implementation
    Ok(vec![0u8; compressed_size])
}

/// Placeholder ETC2 RGBA compression
fn compress_etc2_rgba(_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let blocks_x = (width + 3) / 4;
    let blocks_y = (height + 3) / 4;
    let compressed_size = (blocks_x * blocks_y * 16) as usize; // ETC2 RGBA is 16 bytes per block

    // Placeholder implementation
    Ok(vec![0u8; compressed_size])
}

/// Calculate number of mip levels for given dimensions
fn calculate_mip_levels(width: u32, height: u32) -> u32 {
    let max_dimension = width.max(height);
    (32 - max_dimension.leading_zeros()).max(1)
}

/// Generate mip level data (placeholder implementation)
fn generate_mip_levels(
    _base_data: &[u8],
    _base_width: u32,
    _base_height: u32,
    _format: TextureFormat,
) -> Result<Vec<(Vec<u8>, u32, u32)>, String> {
    // Placeholder: In practice, generate proper mip levels
    // This would involve decompressing, downsampling, and recompressing
    Ok(Vec::new())
}

/// Estimate quality score for format
fn estimate_quality_score(format: TextureFormat) -> f32 {
    match format {
        TextureFormat::Bc7RgbaUnorm | TextureFormat::Bc7RgbaUnormSrgb => 0.95,
        TextureFormat::Bc3RgbaUnorm | TextureFormat::Bc3RgbaUnormSrgb => 0.85,
        TextureFormat::Bc5RgUnorm | TextureFormat::Bc5RgSnorm => 0.90,
        TextureFormat::Bc1RgbaUnorm | TextureFormat::Bc1RgbaUnormSrgb => 0.75,
        TextureFormat::Etc2Rgba8Unorm | TextureFormat::Etc2Rgba8UnormSrgb => 0.85,
        TextureFormat::Etc2Rgb8Unorm | TextureFormat::Etc2Rgb8UnormSrgb => 0.80,
        _ => 0.5,
    }
}

/// Estimate PSNR for format
fn estimate_psnr(format: TextureFormat) -> f32 {
    match format {
        TextureFormat::Bc7RgbaUnorm | TextureFormat::Bc7RgbaUnormSrgb => 45.0,
        TextureFormat::Bc3RgbaUnorm | TextureFormat::Bc3RgbaUnormSrgb => 40.0,
        TextureFormat::Bc5RgUnorm | TextureFormat::Bc5RgSnorm => 42.0,
        TextureFormat::Bc1RgbaUnorm | TextureFormat::Bc1RgbaUnormSrgb => 35.0,
        TextureFormat::Etc2Rgba8Unorm | TextureFormat::Etc2Rgba8UnormSrgb => 38.0,
        TextureFormat::Etc2Rgb8Unorm | TextureFormat::Etc2Rgb8UnormSrgb => 36.0,
        _ => 30.0,
    }
}

// Placeholder structures for KTX2 and DDS parsing
struct Ktx2Header {
    pixel_width: u32,
    pixel_height: u32,
    level_count: u32,
    vk_format: u32,
}

impl Ktx2Header {
    fn vk_format_to_wgpu(&self) -> Result<TextureFormat, String> {
        // Map Vulkan format to WGPU format
        match self.vk_format {
            // Add Vulkan format mappings
            _ => Err(format!("Unsupported Vulkan format: {}", self.vk_format)),
        }
    }

    fn is_srgb(&self) -> bool {
        // Check if format is sRGB based on Vulkan format
        false // Placeholder
    }
}

struct DdsHeader {
    width: u32,
    height: u32,
    mip_map_count: u32,
    pixel_format: u32,
}

impl DdsHeader {
    fn pixel_format_to_wgpu(&self) -> Result<TextureFormat, String> {
        // Map DDS pixel format to WGPU
        match self.pixel_format {
            // Add DDS format mappings
            _ => Err(format!(
                "Unsupported DDS pixel format: {}",
                self.pixel_format
            )),
        }
    }

    fn is_srgb(&self) -> bool {
        false // Placeholder
    }
}

/// Parse KTX2 header (placeholder)
fn parse_ktx2_header(_data: &[u8]) -> Result<Ktx2Header, String> {
    Err("KTX2 parsing not implemented".to_string())
}

/// Extract KTX2 texture data (placeholder)
fn extract_ktx2_texture_data(
    _data: &[u8],
    _header: &Ktx2Header,
) -> Result<CompressedImage, String> {
    Err("KTX2 extraction not implemented".to_string())
}

/// Parse DDS header (placeholder)
fn parse_dds_header(_data: &[u8]) -> Result<DdsHeader, String> {
    Err("DDS parsing not implemented".to_string())
}

/// Extract DDS texture data (placeholder)
fn extract_dds_texture_data(_data: &[u8], _header: &DdsHeader) -> Result<Vec<u8>, String> {
    Err("DDS extraction not implemented".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mip_level_calculation() {
        assert_eq!(calculate_mip_levels(256, 256), 9); // log2(256) + 1
        assert_eq!(calculate_mip_levels(512, 256), 10); // log2(512) + 1
        assert_eq!(calculate_mip_levels(1, 1), 1);
    }

    #[test]
    fn test_compression_stats() {
        let image = CompressedImage {
            data: vec![0u8; 1024], // 1KB compressed
            width: 64,
            height: 64,
            mip_levels: 1,
            format: TextureFormat::Bc1RgbaUnorm,
            is_srgb: false,
            source_format: "Test".to_string(),
        };

        let stats = image.get_compression_stats();
        assert_eq!(stats.uncompressed_size, 64 * 64 * 4); // 16KB uncompressed
        assert_eq!(stats.compressed_size, 1024); // 1KB compressed
        assert!((stats.compression_ratio - 16.0).abs() < 0.1); // 16:1 ratio
    }

    #[test]
    fn test_rgb565_conversion() {
        // Test white
        assert_eq!(rgb8_to_rgb565(255, 255, 255), 0xFFFF);
        // Test black
        assert_eq!(rgb8_to_rgb565(0, 0, 0), 0x0000);
        // Test red
        assert_eq!(rgb8_to_rgb565(255, 0, 0), 0xF800);
    }
}

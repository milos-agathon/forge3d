use wgpu::{
    Device, Extent3d, ImageCopyTexture, ImageDataLayout, Origin3d, Queue, Texture, TextureAspect,
    TextureDescriptor, TextureDimension, TextureUsages,
};

use crate::core::memory_tracker::global_tracker;
use crate::core::texture_format::{global_format_registry, TextureFormatInfo};

use super::compression::{estimate_psnr, estimate_quality_score, generate_mip_levels};
use super::{CompressedImage, CompressionStats};

impl CompressedImage {
    /// Create GPU texture from compressed image
    pub fn decode_to_gpu(
        &self,
        device: &Device,
        queue: &Queue,
        label: Option<&str>,
    ) -> Result<Texture, String> {
        let texture_size = self.calculate_gpu_size();
        if let Err(e) = global_tracker().check_budget(texture_size) {
            return Err(e);
        }

        let texture = device.create_texture(&TextureDescriptor {
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
        });

        self.upload_to_texture(&texture, queue)?;
        global_tracker().track_texture_allocation(self.width, self.height, self.format);
        Ok(texture)
    }

    /// Upload compressed data to existing texture
    pub fn upload_to_texture(&self, texture: &Texture, queue: &Queue) -> Result<(), String> {
        let format_info = global_format_registry()
            .get_format_info(self.format)
            .ok_or_else(|| format!("Unknown format: {:?}", self.format))?;

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

        if self.mip_levels > 1 {
            self.upload_mip_levels(texture, queue)?;
        }

        Ok(())
    }

    /// Calculate GPU memory size
    pub fn calculate_gpu_size(&self) -> u64 {
        let fallback = TextureFormatInfo {
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
            .unwrap_or(&fallback);

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
        let uncompressed_size = (self.width * self.height * 4) as u64;
        let compressed_size = self.data.len() as u64;
        let compression_ratio = uncompressed_size as f32 / compressed_size as f32;

        CompressionStats {
            uncompressed_size,
            compressed_size,
            compression_ratio,
            compression_time_ms: 0.0,
            quality_score: estimate_quality_score(self.format),
            psnr_db: estimate_psnr(self.format),
        }
    }

    fn upload_mip_levels(&self, texture: &Texture, queue: &Queue) -> Result<(), String> {
        let mip_data = generate_mip_levels(&self.data, self.width, self.height, self.format)?;
        let format_info = global_format_registry()
            .get_format_info(self.format)
            .unwrap();

        for (mip_level, (mip_data, mip_width, mip_height)) in mip_data.iter().enumerate() {
            if mip_level == 0 {
                continue;
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
}

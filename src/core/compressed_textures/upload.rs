use wgpu::{
    Device, Extent3d, ImageCopyTexture, Origin3d, Queue, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureUsages,
};

use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use crate::core::texture_format::global_format_registry;

use crate::core::texture_layout::{resident_2d_size, resident_size, TransferLayout};

use super::compression::{estimate_psnr, estimate_quality_score, generate_mip_levels};
use super::{CompressedImage, CompressionStats};

impl CompressedImage {
    /// Create GPU texture from compressed image
    pub fn decode_to_gpu(
        &self,
        device: &Device,
        queue: &Queue,
        label: Option<&str>,
    ) -> Result<TrackedTexture, String> {
        // The tracked wrapper performs registry + ledger accounting itself, so
        // the previous manual budget check and `track_texture_allocation` call
        // are gone (they would double-count).
        let texture = tracked_create_texture(
            device,
            &TextureDescriptor {
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
            },
        )
        .map_err(|e| e.to_string())?;

        self.upload_to_texture(&texture, queue)?;
        Ok(texture)
    }

    /// Upload compressed data to existing texture
    pub fn upload_to_texture(&self, texture: &Texture, queue: &Queue) -> Result<(), String> {
        global_format_registry()
            .get_format_info(self.format)
            .ok_or_else(|| format!("Unknown format: {:?}", self.format))?;
        let layout = TransferLayout::queue_write(
            self.format,
            TextureAspect::All,
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        )
        .map_err(str::to_string)?;

        queue.write_texture(
            ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &self.data,
            layout.image_data_layout(),
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
        resident_size(
            self.format,
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            self.mip_levels,
            1,
        )
    }

    /// Get compression statistics
    pub fn get_compression_stats(&self) -> CompressionStats {
        let uncompressed_size =
            resident_2d_size(self.width, self.height, wgpu::TextureFormat::Rgba8Unorm);
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
        for (mip_level, (mip_data, mip_width, mip_height)) in mip_data.iter().enumerate() {
            if mip_level == 0 {
                continue;
            }

            let layout = TransferLayout::queue_write(
                self.format,
                TextureAspect::All,
                Extent3d {
                    width: *mip_width,
                    height: *mip_height,
                    depth_or_array_layers: 1,
                },
            )
            .map_err(str::to_string)?;

            queue.write_texture(
                ImageCopyTexture {
                    texture,
                    mip_level: mip_level as u32,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                mip_data,
                layout.image_data_layout(),
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

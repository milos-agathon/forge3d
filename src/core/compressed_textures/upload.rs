use wgpu::{
    Device, Extent3d, ImageCopyTexture, Origin3d, Queue, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureUsages,
};

use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use crate::core::texture_format::global_format_registry;

use crate::core::texture_layout::{mip_extent, resident_size, TransferLayout};

use super::compression::{estimate_psnr, estimate_quality_score};
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

    /// Tight extent/layout for every mip level in the packed payload.
    ///
    /// `data` holds one tightly packed payload per level, base level first.
    /// Level `n` occupies exactly the format footprint of that level's
    /// extent, the same rule `from_rgba_data` and the KTX2 loader follow.
    pub(crate) fn packed_level_layouts(&self) -> Result<Vec<(Extent3d, TransferLayout)>, String> {
        let base = Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        };
        (0..self.mip_levels)
            .map(|mip| {
                let extent = mip_extent(base, TextureDimension::D2, mip);
                let layout = TransferLayout::queue_write(self.format, TextureAspect::All, extent)
                    .map_err(str::to_string)?;
                Ok((extent, layout))
            })
            .collect()
    }

    /// Upload compressed data to existing texture
    pub fn upload_to_texture(&self, texture: &Texture, queue: &Queue) -> Result<(), String> {
        global_format_registry()
            .get_format_info(self.format)
            .ok_or_else(|| format!("Unknown format: {:?}", self.format))?;

        let levels = self.packed_level_layouts()?;
        let required = levels
            .iter()
            .map(|(_, layout)| layout.payload_bytes)
            .fold(0u64, u64::saturating_add);
        if required != self.data.len() as u64 {
            crate::core::degradation::record_degradation(
                "invalid_data",
                "compressed_textures.mip_payload",
                "Compressed image payload does not match its mip chain; refusing a partial upload",
            );
            return Err(format!(
                "Compressed image data has {} bytes but {} mip level(s) require {} bytes",
                self.data.len(),
                self.mip_levels,
                required
            ));
        }

        let mut offset = 0usize;
        for (mip, (extent, layout)) in levels.iter().enumerate() {
            let len = usize::try_from(layout.payload_bytes)
                .map_err(|_| "Compressed mip level exceeds addressable size".to_string())?;
            let end = offset
                .checked_add(len)
                .ok_or_else(|| "Compressed mip chain size overflows".to_string())?;
            queue.write_texture(
                ImageCopyTexture {
                    texture,
                    mip_level: mip as u32,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                &self.data[offset..end],
                layout.image_data_layout(),
                *extent,
            );
            offset = end;
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
        let uncompressed_size = resident_size(
            wgpu::TextureFormat::Rgba8Unorm,
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            self.mip_levels.max(1),
            1,
        );
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
}

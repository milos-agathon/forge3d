use super::*;
use wgpu::{Extent3d, ImageCopyTexture, ImageDataLayout, Origin3d, TextureAspect, TextureFormat};

impl VirtualTexture {
    /// Load tile data; current implementation generates a procedural tile.
    ///
    /// Generates slot_size × slot_size pixels (spec 4.1, 8.4):
    /// - Border region filled with procedural checkerboard
    /// - Content region (centered, tile_size × tile_size) filled with per-tile color
    pub(super) fn load_tile_data(&self, tile_id: TileId) -> Result<TileData, String> {
        let tile_size = self.config.tile_size as usize;
        let tile_border = self.config.tile_border as usize;
        let slot_size = tile_size + 2 * tile_border;
        let pixel_count = slot_size * slot_size;
        let bytes_per_pixel = match self.config.format {
            TextureFormat::Rgba8Unorm => 4,
            TextureFormat::Rgba8UnormSrgb => 4,
            TextureFormat::Rg8Unorm => 2,
            TextureFormat::R8Unorm => 1,
            _ => 4,
        };

        let mut data = vec![0u8; pixel_count * bytes_per_pixel];

        for y in 0..slot_size {
            for x in 0..slot_size {
                let pixel_index = (y * slot_size + x) * bytes_per_pixel;

                // Determine if this pixel is in border region or content region
                let in_border_x = x < tile_border || x >= tile_border + tile_size;
                let in_border_y = y < tile_border || y >= tile_border + tile_size;

                let (r, g, b, a) = if in_border_x || in_border_y {
                    // Border: checkerboard pattern
                    let checker = (((x / 2) + (y / 2)) & 1) as u8;
                    (
                        128 + checker * 64,
                        128 + checker * 64,
                        128 + checker * 64,
                        255,
                    )
                } else {
                    // Content region: per-tile color based on tile coordinates and mip
                    let content_x = x - tile_border;
                    let content_y = y - tile_border;
                    let r = ((tile_id.x * tile_size as u32 + content_x as u32) & 0xFF) as u8;
                    let g = ((tile_id.y * tile_size as u32 + content_y as u32) & 0xFF) as u8;
                    let b = (tile_id.mip_level * 32) as u8;
                    (r, g, b, 255)
                };

                if bytes_per_pixel >= 1 {
                    data[pixel_index] = r;
                }
                if bytes_per_pixel >= 2 {
                    data[pixel_index + 1] = g;
                }
                if bytes_per_pixel >= 3 {
                    data[pixel_index + 2] = b;
                }
                if bytes_per_pixel >= 4 {
                    data[pixel_index + 3] = a;
                }
            }
        }

        Ok(TileData {
            id: tile_id,
            data,
            width: slot_size as u32,
            height: slot_size as u32,
            format: self.config.format,
        })
    }

    pub(super) fn upload_tile_to_atlas(
        &self,
        device: &Device,
        queue: &Queue,
        tile_data: &TileData,
        atlas_slot: crate::core::tile_cache::AtlasSlot,
    ) -> Result<(), String> {
        let bytes_per_pixel = match tile_data.format {
            TextureFormat::Rgba8Unorm => 4,
            TextureFormat::Rgba8UnormSrgb => 4,
            TextureFormat::Rg8Unorm => 2,
            TextureFormat::R8Unorm => 1,
            _ => 4,
        };

        #[cfg(feature = "enable-staging-rings")]
        if let Some(ref staging_ring) = self.staging_ring {
            if let Ok(mut ring) = staging_ring.lock() {
                if let Some((buffer, offset)) = ring.allocate(tile_data.data.len() as u64) {
                    queue.write_buffer(buffer, offset, &tile_data.data);

                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("VirtualTexture_TileUpload"),
                        });

                    encoder.copy_buffer_to_texture(
                        wgpu::ImageCopyBuffer {
                            buffer,
                            layout: ImageDataLayout {
                                offset,
                                bytes_per_row: Some(tile_data.width * bytes_per_pixel),
                                rows_per_image: Some(tile_data.height),
                            },
                        },
                        ImageCopyTexture {
                            texture: &self.atlas_texture,
                            mip_level: 0,
                            origin: Origin3d {
                                x: atlas_slot.atlas_x,
                                y: atlas_slot.atlas_y,
                                z: 0,
                            },
                            aspect: TextureAspect::All,
                        },
                        Extent3d {
                            width: tile_data.width,
                            height: tile_data.height,
                            depth_or_array_layers: 1,
                        },
                    );

                    queue.submit([encoder.finish()]);
                    return Ok(());
                }
            }
        }

        let _ = device;
        queue.write_texture(
            ImageCopyTexture {
                texture: &self.atlas_texture,
                mip_level: 0,
                origin: Origin3d {
                    x: atlas_slot.atlas_x,
                    y: atlas_slot.atlas_y,
                    z: 0,
                },
                aspect: TextureAspect::All,
            },
            &tile_data.data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(tile_data.width * bytes_per_pixel),
                rows_per_image: Some(tile_data.height),
            },
            Extent3d {
                width: tile_data.width,
                height: tile_data.height,
                depth_or_array_layers: 1,
            },
        );

        Ok(())
    }

    pub(super) fn tile_id_to_page_index(&self, tile_id: TileId) -> usize {
        let pages_x = (self.config.width + self.config.tile_size - 1) / self.config.tile_size;
        (tile_id.y * pages_x + tile_id.x) as usize
    }

    pub(super) fn resident_tile_memory_bytes(&self) -> u64 {
        self.calculate_memory_usage()
    }

    pub(super) fn calculate_memory_usage(&self) -> u64 {
        let bytes_per_pixel = match self.config.format {
            TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => 4,
            TextureFormat::Rg8Unorm
            | TextureFormat::Rg8Snorm
            | TextureFormat::Rg8Uint
            | TextureFormat::Rg8Sint => 2,
            TextureFormat::R8Unorm
            | TextureFormat::R8Snorm
            | TextureFormat::R8Uint
            | TextureFormat::R8Sint => 1,
            _ => 4,
        };

        let tile_memory = (self.config.tile_size * self.config.tile_size * bytes_per_pixel) as u64;
        tile_memory * self.stats.resident_pages as u64
    }
}

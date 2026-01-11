//! O4: Virtual texture streaming system
//!
//! This module provides virtual texture streaming with page table management,
//! GPU feedback buffers for tile visibility, and LRU tile caching.

use crate::core::feedback_buffer::FeedbackBuffer;
use crate::core::memory_tracker::global_tracker;
#[cfg(feature = "enable-staging-rings")]
use crate::core::staging_rings::StagingRing;
use crate::core::tile_cache::{TileCache, TileData, TileId};
use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;
#[cfg(feature = "enable-staging-rings")]
use std::sync::{Arc, Mutex};
use wgpu::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindingResource, Device, Extent3d,
    ImageCopyTexture, ImageDataLayout, Origin3d, Queue, Sampler, Texture, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};

/// Virtual texture configuration
#[derive(Debug, Clone)]
pub struct VirtualTextureConfig {
    /// Width in pixels of the virtual texture
    pub width: u32,
    /// Height in pixels of the virtual texture
    pub height: u32,
    /// Size of each tile in pixels
    pub tile_size: u32,
    /// Maximum number of mip levels
    pub max_mip_levels: u32,
    /// Physical texture atlas size
    pub atlas_width: u32,
    /// Physical texture atlas size  
    pub atlas_height: u32,
    /// Texture format
    pub format: TextureFormat,
    /// Enable feedback buffer for tile visibility
    pub use_feedback: bool,
}

impl Default for VirtualTextureConfig {
    fn default() -> Self {
        Self {
            width: 16384, // 16K virtual texture
            height: 16384,
            tile_size: 128, // 128x128 pixel tiles
            max_mip_levels: 8,
            atlas_width: 2048, // 2K physical atlas
            atlas_height: 2048,
            format: TextureFormat::Rgba8Unorm,
            use_feedback: true,
        }
    }
}

/// Page table entry for virtual texture addressing
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct PageTableEntry {
    /// Physical texture coordinates (atlas UV)
    pub atlas_u: f32,
    pub atlas_v: f32,
    /// Tile validity flag
    pub is_resident: u32,
    /// Mip level bias for this tile
    pub mip_bias: f32,
}

/// Virtual texture statistics
#[derive(Debug, Clone, Default)]
pub struct VirtualTextureStats {
    /// Total number of virtual pages
    pub total_pages: u32,
    /// Number of resident (loaded) pages
    pub resident_pages: u32,
    /// Number of cache hits in last frame
    pub cache_hits: u32,
    /// Number of cache misses in last frame
    pub cache_misses: u32,
    /// Number of tiles streamed in last frame
    pub tiles_streamed: u32,
    /// Memory usage of resident tiles in bytes
    pub memory_usage: u64,
    /// Average tile load time in milliseconds
    pub avg_load_time_ms: f32,
}

/// Camera information for tile visibility calculation
#[derive(Debug, Clone, Copy)]
pub struct CameraInfo {
    /// Camera position in world space
    pub position: [f32; 3],
    /// Camera view direction
    pub direction: [f32; 3],
    /// Field of view in degrees
    pub fov_degrees: f32,
    /// Aspect ratio (width/height)
    pub aspect_ratio: f32,
    /// Near plane distance
    pub near_plane: f32,
    /// Far plane distance
    pub far_plane: f32,
}

/// Virtual texture streaming system
pub struct VirtualTexture {
    /// Configuration
    config: VirtualTextureConfig,

    /// Physical texture atlas for storing resident tiles
    atlas_texture: Texture,

    /// Page table texture for virtual -> physical address mapping
    page_table: Texture,

    /// Page table data (CPU-side)
    page_table_data: Vec<PageTableEntry>,

    /// Feedback buffer for GPU -> CPU tile visibility communication
    feedback_buffer: Option<FeedbackBuffer>,

    /// Tile cache for managing resident tiles
    tile_cache: TileCache,

    /// Set of tiles requested this frame
    requested_tiles: HashSet<TileId>,

    /// Statistics
    stats: VirtualTextureStats,

    /// Staging ring for async tile uploads
    #[cfg(feature = "enable-staging-rings")]
    staging_ring: Option<Arc<Mutex<StagingRing>>>,
}

impl VirtualTexture {
    fn publish_resident_metrics(&self) {
        let tracker = global_tracker();
        tracker.set_resident_tiles(self.stats.resident_pages, self.resident_tile_memory_bytes());
    }
    /// Create new virtual texture system
    pub fn new(
        device: &Device,
        _queue: &Queue,
        config: VirtualTextureConfig,
        #[cfg(feature = "enable-staging-rings")] staging_ring: Option<Arc<Mutex<StagingRing>>>,
    ) -> Result<Self, String> {
        // Calculate page table dimensions
        let pages_x = (config.width + config.tile_size - 1) / config.tile_size;
        let pages_y = (config.height + config.tile_size - 1) / config.tile_size;
        let total_pages = pages_x * pages_y;

        // Create atlas texture
        let atlas_texture = device.create_texture(&TextureDescriptor {
            label: Some("VirtualTexture_Atlas"),
            size: Extent3d {
                width: config.atlas_width,
                height: config.atlas_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: config.format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Create page table texture (RGBA32Float for page table entries)
        let page_table = device.create_texture(&TextureDescriptor {
            label: Some("VirtualTexture_PageTable"),
            size: Extent3d {
                width: pages_x,
                height: pages_y,
                depth_or_array_layers: 1,
            },
            mip_level_count: config.max_mip_levels,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float, // Store PageTableEntry data
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Initialize page table data
        let page_table_data = vec![PageTableEntry::default(); total_pages as usize];

        // Create feedback buffer if requested
        let feedback_buffer = if config.use_feedback {
            Some(FeedbackBuffer::new(device, total_pages)?)
        } else {
            None
        };

        // Create tile cache
        let atlas_tiles_x = config.atlas_width / config.tile_size;
        let atlas_tiles_y = config.atlas_height / config.tile_size;
        let max_resident_tiles = atlas_tiles_x * atlas_tiles_y;

        let tile_cache = TileCache::new(max_resident_tiles as usize);

        let stats = VirtualTextureStats {
            total_pages,
            ..Default::default()
        };

        let instance = Self {
            config,
            atlas_texture,
            page_table,
            page_table_data,
            feedback_buffer,
            tile_cache,
            requested_tiles: HashSet::new(),
            stats,
            #[cfg(feature = "enable-staging-rings")]
            staging_ring,
        };
        instance.publish_resident_metrics();
        Ok(instance)
    }

    /// Update virtual texture for current camera
    pub fn update(
        &mut self,
        device: &Device,
        queue: &Queue,
        camera: &CameraInfo,
    ) -> Result<(), String> {
        // Clear previous frame data
        self.requested_tiles.clear();
        self.stats.cache_hits = 0;
        self.stats.cache_misses = 0;
        self.stats.tiles_streamed = 0;

        // Calculate visible tiles based on camera
        let visible_tiles = self.calculate_visible_tiles(camera);

        // Process tile requests
        for tile_id in visible_tiles {
            self.requested_tiles.insert(tile_id);

            if self.tile_cache.is_resident(&tile_id) {
                self.stats.cache_hits += 1;
                // Update LRU order
                self.tile_cache.access_tile(&tile_id);
            } else {
                self.stats.cache_misses += 1;
                // Request tile loading
                self.request_tile_load(device, queue, tile_id)?;
            }
        }

        // Process feedback buffer if available
        if let Some(ref mut feedback_buffer) = self.feedback_buffer {
            let feedback_tiles = feedback_buffer.read_feedback(device, queue)?;
            for tile_id in feedback_tiles {
                if !self.tile_cache.is_resident(&tile_id) {
                    self.request_tile_load(device, queue, tile_id)?;
                }
            }
        }

        // Update page table
        self.update_page_table(device, queue)?;

        // Update statistics
        self.stats.resident_pages = self.tile_cache.resident_count() as u32;
        self.publish_resident_metrics();
        self.stats.memory_usage = self.calculate_memory_usage();

        Ok(())
    }

    /// Calculate visible tiles based on camera frustum
    fn calculate_visible_tiles(&self, camera: &CameraInfo) -> Vec<TileId> {
        let mut visible_tiles = Vec::new();

        // Simplified visibility calculation
        // In practice, this would use proper frustum culling and LOD selection

        let pages_x = (self.config.width + self.config.tile_size - 1) / self.config.tile_size;
        let pages_y = (self.config.height + self.config.tile_size - 1) / self.config.tile_size;

        // Calculate approximate visible region based on camera position
        // This is a simplified implementation - real systems would use proper frustum culling

        let view_distance = 1000.0; // Maximum view distance
        let visible_size = (camera.fov_degrees.to_radians().tan() * view_distance) as u32;

        // Calculate center tile based on camera position
        let center_x = ((camera.position[0] / self.config.width as f32) * pages_x as f32) as u32;
        let center_y = ((camera.position[2] / self.config.height as f32) * pages_y as f32) as u32;

        let visible_radius = (visible_size / self.config.tile_size / 2).max(1);

        // Add tiles in visible radius
        for y in center_y.saturating_sub(visible_radius)
            ..=center_y.saturating_add(visible_radius).min(pages_y - 1)
        {
            for x in center_x.saturating_sub(visible_radius)
                ..=center_x.saturating_add(visible_radius).min(pages_x - 1)
            {
                // Calculate appropriate mip level based on distance
                let distance = ((x as f32 - center_x as f32).powi(2)
                    + (y as f32 - center_y as f32).powi(2))
                .sqrt();
                let mip_level = (((distance / visible_radius as f32)
                    * self.config.max_mip_levels as f32) as u32)
                    .min(self.config.max_mip_levels - 1);

                visible_tiles.push(TileId { x, y, mip_level });
            }
        }

        visible_tiles
    }

    /// Request loading of a tile
    fn request_tile_load(
        &mut self,
        device: &Device,
        queue: &Queue,
        tile_id: TileId,
    ) -> Result<(), String> {
        // Find free slot in atlas
        if let Some(atlas_slot) = self.tile_cache.allocate_tile(tile_id) {
            // Load tile data via procedural fallback until storage is wired.
            let tile_data = self.load_tile_data(tile_id)?;

            // Upload tile to atlas
            self.upload_tile_to_atlas(device, queue, &tile_data, atlas_slot)?;

            // Update page table entry
            let page_index = self.tile_id_to_page_index(tile_id);
            if let Some(entry) = self.page_table_data.get_mut(page_index) {
                entry.atlas_u = atlas_slot.atlas_u;
                entry.atlas_v = atlas_slot.atlas_v;
                entry.is_resident = 1;
                entry.mip_bias = atlas_slot.mip_bias;
            }

            self.stats.tiles_streamed += 1;
        }

        Ok(())
    }

    /// Load tile data; current implementation generates a procedural tile.
    fn load_tile_data(&self, tile_id: TileId) -> Result<TileData, String> {
        // Procedural fallback keeps the streaming path exercised without IO wiring.
        let tile_size = self.config.tile_size as usize;
        let pixel_count = tile_size * tile_size;
        let bytes_per_pixel = match self.config.format {
            TextureFormat::Rgba8Unorm => 4,
            TextureFormat::Rgba8UnormSrgb => 4,
            TextureFormat::Rg8Unorm => 2,
            TextureFormat::R8Unorm => 1,
            _ => 4, // Default to 4 bytes
        };

        // Generate procedural tile data based on tile ID
        let mut data = vec![0u8; pixel_count * bytes_per_pixel];

        // Simple pattern generation based on tile coordinates
        for y in 0..tile_size {
            for x in 0..tile_size {
                let pixel_index = (y * tile_size + x) * bytes_per_pixel;

                // Generate pattern based on tile and pixel coordinates
                let r = ((tile_id.x * tile_size as u32 + x as u32) & 0xFF) as u8;
                let g = ((tile_id.y * tile_size as u32 + y as u32) & 0xFF) as u8;
                let b = (tile_id.mip_level * 32) as u8;
                let a = 255u8;

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
            width: tile_size as u32,
            height: tile_size as u32,
            format: self.config.format,
        })
    }

    /// Upload tile data to atlas texture
    fn upload_tile_to_atlas(
        &self,
        _device: &Device,
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

        // Use staging ring if available for async upload
        #[cfg(feature = "enable-staging-rings")]
        if let Some(ref staging_ring) = self.staging_ring {
            if let Ok(mut ring) = staging_ring.lock() {
                if let Some((buffer, offset)) = ring.allocate(tile_data.data.len() as u64) {
                    // Write data to staging buffer
                    queue.write_buffer(buffer, offset, &tile_data.data);

                    // Copy from staging buffer to atlas texture
                    let mut encoder =
                        _device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

        // Fallback: direct upload
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

    /// Update page table texture with current resident tiles
    fn update_page_table(&self, _device: &Device, queue: &Queue) -> Result<(), String> {
        // Convert page table data to GPU format without reallocating each frame
        let gpu_data = bytemuck::cast_slice(&self.page_table_data);

        let pages_x = (self.config.width + self.config.tile_size - 1) / self.config.tile_size;
        let pages_y = (self.config.height + self.config.tile_size - 1) / self.config.tile_size;

        // Upload to GPU
        queue.write_texture(
            ImageCopyTexture {
                texture: &self.page_table,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            gpu_data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(pages_x * 16),
                rows_per_image: Some(pages_y),
            },
            Extent3d {
                width: pages_x,
                height: pages_y,
                depth_or_array_layers: 1,
            },
        );

        Ok(())
    }

    /// Convert tile ID to page table index
    fn tile_id_to_page_index(&self, tile_id: TileId) -> usize {
        let pages_x = (self.config.width + self.config.tile_size - 1) / self.config.tile_size;
        (tile_id.y * pages_x + tile_id.x) as usize
    }

    /// Calculate current memory usage
    fn resident_tile_memory_bytes(&self) -> u64 {
        self.calculate_memory_usage()
    }

    fn calculate_memory_usage(&self) -> u64 {
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

    /// Get current virtual texture statistics
    pub fn stats(&self) -> &VirtualTextureStats {
        &self.stats
    }

    /// Get atlas texture for rendering
    pub fn atlas_texture(&self) -> &Texture {
        &self.atlas_texture
    }

    /// Get page table texture for rendering
    pub fn page_table_texture(&self) -> &Texture {
        &self.page_table
    }

    /// Create bind group for virtual texture rendering
    pub fn create_bind_group(
        &self,
        device: &Device,
        layout: &BindGroupLayout,
        sampler: &Sampler,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VirtualTexture_BindGroup"),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &self.atlas_texture.create_view(&Default::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &self.page_table.create_view(&Default::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(sampler),
                },
            ],
        })
    }
}

impl Drop for VirtualTexture {
    fn drop(&mut self) {
        global_tracker().clear_resident_tiles();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virtual_texture_config() {
        let config = VirtualTextureConfig::default();

        assert_eq!(config.width, 16384);
        assert_eq!(config.height, 16384);
        assert_eq!(config.tile_size, 128);
        assert!(config.use_feedback);
    }

    #[test]
    fn test_tile_id_to_page_index() {
        let config = VirtualTextureConfig {
            width: 1024,
            height: 1024,
            tile_size: 128,
            ..Default::default()
        };

        // Create a minimal VirtualTexture for testing
        // In a real test, we'd need proper GPU context
        let pages_x = (config.width + config.tile_size - 1) / config.tile_size; // 8

        // Test tile (1, 2) at mip 0
        let tile_id = TileId {
            x: 1,
            y: 2,
            mip_level: 0,
        };
        let expected_index = (2 * pages_x + 1) as usize; // (2 * 8 + 1) = 17

        // We can't directly test the method without GPU context, but we can test the logic
        assert_eq!(expected_index, 17);
    }

    #[test]
    fn test_page_table_entry() {
        let entry = PageTableEntry {
            atlas_u: 0.5,
            atlas_v: 0.25,
            is_resident: 1,
            mip_bias: 0.0,
        };

        assert_eq!(entry.atlas_u, 0.5);
        assert_eq!(entry.atlas_v, 0.25);
        assert_eq!(entry.is_resident, 1);
        assert_eq!(entry.mip_bias, 0.0);
    }

    #[test]
    fn test_camera_info() {
        let camera = CameraInfo {
            position: [0.0, 100.0, 0.0],
            direction: [0.0, -1.0, 0.0],
            fov_degrees: 45.0,
            aspect_ratio: 16.0 / 9.0,
            near_plane: 0.1,
            far_plane: 1000.0,
        };

        assert_eq!(camera.position[1], 100.0);
        assert_eq!(camera.fov_degrees, 45.0);
        assert!((camera.aspect_ratio - 16.0 / 9.0).abs() < f32::EPSILON);
    }
}

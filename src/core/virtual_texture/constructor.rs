use super::*;
use wgpu::{Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages};

impl VirtualTexture {
    pub(super) fn publish_resident_metrics(&self) {
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
        let pages_x = (config.width + config.tile_size - 1) / config.tile_size;
        let pages_y = (config.height + config.tile_size - 1) / config.tile_size;
        let total_pages = pages_x * pages_y;

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
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let page_table_data = vec![PageTableEntry::default(); total_pages as usize];
        let feedback_buffer = if config.use_feedback {
            Some(FeedbackBuffer::new(device, total_pages)?)
        } else {
            None
        };

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
}

impl Drop for VirtualTexture {
    fn drop(&mut self) {
        global_tracker().clear_resident_tiles();
    }
}

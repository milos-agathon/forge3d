use super::config::PostFxResourceDesc;
use crate::core::error::RenderResult;
use std::collections::HashMap;
use wgpu::*;

/// Resource pool for ping-pong and temporal textures
#[derive(Debug)]
pub struct PostFxResourcePool {
    /// Ping-pong texture pairs
    ping_pong_textures: Vec<Vec<Texture>>,
    /// Texture views for ping-pong resources
    ping_pong_views: Vec<Vec<TextureView>>,
    /// Temporal textures (for effects that need history)
    temporal_textures: HashMap<String, Vec<Texture>>,
    /// Temporal texture views
    temporal_views: HashMap<String, Vec<TextureView>>,
    /// Current ping-pong index
    ping_pong_index: usize,
    /// Resource dimensions
    width: u32,
    height: u32,
}

impl PostFxResourcePool {
    /// Create new resource pool
    pub fn new(_device: &Device, width: u32, height: u32, max_ping_pong_pairs: usize) -> Self {
        Self {
            ping_pong_textures: Vec::with_capacity(max_ping_pong_pairs),
            ping_pong_views: Vec::with_capacity(max_ping_pong_pairs),
            temporal_textures: HashMap::new(),
            temporal_views: HashMap::new(),
            ping_pong_index: 0,
            width,
            height,
        }
    }

    /// Get current ping-pong texture
    pub fn get_current_ping_pong(&self, pair_index: usize) -> Option<&TextureView> {
        self.ping_pong_views
            .get(pair_index)?
            .get(self.ping_pong_index)
    }

    /// Get previous ping-pong texture
    pub fn get_previous_ping_pong(&self, pair_index: usize) -> Option<&TextureView> {
        let prev_index = (self.ping_pong_index + 1) % 2;
        self.ping_pong_views.get(pair_index)?.get(prev_index)
    }

    /// Swap ping-pong buffers
    pub fn swap_ping_pong(&mut self) {
        self.ping_pong_index = (self.ping_pong_index + 1) % 2;
    }

    /// Allocate ping-pong texture pair
    pub fn allocate_ping_pong_pair(
        &mut self,
        device: &Device,
        desc: &PostFxResourceDesc,
    ) -> RenderResult<usize> {
        let actual_width = if desc.width == 0 {
            self.width
        } else {
            desc.width
        };
        let actual_height = if desc.height == 0 {
            self.height
        } else {
            desc.height
        };

        let mut textures = Vec::new();
        let mut views = Vec::new();

        // Create pair of textures
        for i in 0..2 {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some(&format!(
                    "postfx_ping_pong_{}_{}",
                    self.ping_pong_textures.len(),
                    i
                )),
                size: Extent3d {
                    width: actual_width,
                    height: actual_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: desc.mip_count,
                sample_count: desc.sample_count,
                dimension: TextureDimension::D2,
                format: desc.format,
                usage: desc.usage,
                view_formats: &[],
            });

            let view = texture.create_view(&TextureViewDescriptor::default());

            textures.push(texture);
            views.push(view);
        }

        let pair_index = self.ping_pong_textures.len();
        self.ping_pong_textures.push(textures);
        self.ping_pong_views.push(views);

        Ok(pair_index)
    }

    /// Allocate temporal texture
    pub fn allocate_temporal_texture(
        &mut self,
        device: &Device,
        name: &str,
        frame_count: usize,
        desc: &PostFxResourceDesc,
    ) -> RenderResult<()> {
        let actual_width = if desc.width == 0 {
            self.width
        } else {
            desc.width
        };
        let actual_height = if desc.height == 0 {
            self.height
        } else {
            desc.height
        };

        let mut textures = Vec::new();
        let mut views = Vec::new();

        for i in 0..frame_count {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some(&format!("postfx_temporal_{}_{}", name, i)),
                size: Extent3d {
                    width: actual_width,
                    height: actual_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: desc.mip_count,
                sample_count: desc.sample_count,
                dimension: TextureDimension::D2,
                format: desc.format,
                usage: desc.usage,
                view_formats: &[],
            });

            let view = texture.create_view(&TextureViewDescriptor::default());

            textures.push(texture);
            views.push(view);
        }

        self.temporal_textures.insert(name.to_string(), textures);
        self.temporal_views.insert(name.to_string(), views);

        Ok(())
    }

    /// Get temporal texture by name and frame index
    pub fn get_temporal_texture(&self, name: &str, frame_index: usize) -> Option<&TextureView> {
        self.temporal_views.get(name)?.get(frame_index)
    }
}

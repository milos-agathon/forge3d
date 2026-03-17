use super::OverlayStack;
use crate::viewer::terrain::overlay::{BlendMode, OverlayData, OverlayLayer, OverlayLayerGpu};

impl OverlayStack {
    /// Create a new overlay stack
    pub fn new(device: std::sync::Arc<wgpu::Device>, queue: std::sync::Arc<wgpu::Queue>) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("overlay_stack.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });

        Self {
            device,
            queue,
            layers: Vec::new(),
            next_id: 0,
            composite_texture: None,
            composite_view: None,
            composite_dimensions: (0, 0),
            dirty: true,
            sampler,
        }
    }

    /// Add an overlay layer from raw RGBA data. Returns layer ID.
    pub fn add_raster(
        &mut self,
        name: &str,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        extent: Option<[f32; 4]>,
        opacity: f32,
        blend_mode: BlendMode,
        z_order: i32,
    ) -> u32 {
        let config = OverlayLayer {
            name: name.to_string(),
            data: OverlayData::Raster {
                rgba: rgba.clone(),
                width,
                height,
            },
            extent,
            opacity,
            blend_mode,
            visible: true,
            z_order,
        };

        self.add_layer_internal(config, &rgba, width, height)
    }

    /// Add an overlay layer from an image file. Returns layer ID or error.
    pub fn add_image(
        &mut self,
        name: &str,
        path: &std::path::Path,
        extent: Option<[f32; 4]>,
        opacity: f32,
        blend_mode: BlendMode,
        z_order: i32,
    ) -> Result<u32, String> {
        let img = image::open(path)
            .map_err(|e| format!("Failed to load overlay image '{}': {}", path.display(), e))?;
        let rgba_img = img.to_rgba8();
        let (width, height) = rgba_img.dimensions();
        let rgba = rgba_img.into_raw();

        let config = OverlayLayer {
            name: name.to_string(),
            data: OverlayData::Raster {
                rgba: rgba.clone(),
                width,
                height,
            },
            extent,
            opacity,
            blend_mode,
            visible: true,
            z_order,
        };

        Ok(self.add_layer_internal(config, &rgba, width, height))
    }

    /// Remove an overlay by ID. Returns true if found and removed.
    pub fn remove(&mut self, id: u32) -> bool {
        if let Some(pos) = self.layers.iter().position(|l| l.id == id) {
            let removed = self.layers.remove(pos);
            println!(
                "[overlay] Removed layer '{}' (id={})",
                removed.config.name, id
            );
            self.dirty = true;
            true
        } else {
            false
        }
    }

    /// Set overlay visibility
    pub fn set_visible(&mut self, id: u32, visible: bool) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) {
            if layer.config.visible != visible {
                layer.config.visible = visible;
                self.dirty = true;
            }
        }
    }

    /// Set overlay opacity (0.0 - 1.0)
    pub fn set_opacity(&mut self, id: u32, opacity: f32) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) {
            let clamped = opacity.clamp(0.0, 1.0);
            if (layer.config.opacity - clamped).abs() > 0.001 {
                layer.config.opacity = clamped;
                self.dirty = true;
            }
        }
    }

    /// Reorder layers by ID. IDs not in the list keep their relative order.
    pub fn reorder(&mut self, order: &[u32]) {
        for (new_order, &id) in order.iter().enumerate() {
            if let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) {
                layer.config.z_order = new_order as i32;
            }
        }
        self.layers.sort_by_key(|l| l.config.z_order);
        self.dirty = true;
    }

    /// Get list of all layer IDs in z-order
    pub fn list_ids(&self) -> Vec<u32> {
        let mut sorted: Vec<_> = self.layers.iter().collect();
        sorted.sort_by_key(|l| l.config.z_order);
        sorted.iter().map(|l| l.id).collect()
    }

    /// Get number of layers
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Check if any visible layers exist
    pub fn has_visible_layers(&self) -> bool {
        self.layers
            .iter()
            .any(|l| l.config.visible && l.config.opacity > 0.001)
    }

    /// Check if composite needs rebuild
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Get the composite texture view (for binding to shader)
    /// Returns None if no layers or composite not built yet
    pub fn composite_view(&self) -> Option<&wgpu::TextureView> {
        self.composite_view.as_ref()
    }

    /// Get the overlay sampler
    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    fn add_layer_internal(
        &mut self,
        config: OverlayLayer,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("overlay_layer_{}", id)),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let layer_gpu = OverlayLayerGpu {
            id,
            config,
            texture,
            view,
            dimensions: (width, height),
        };

        self.layers.push(layer_gpu);
        self.dirty = true;

        println!(
            "[overlay] Added layer '{}' (id={}, {}x{})",
            self.layers.last().unwrap().config.name,
            id,
            width,
            height
        );

        id
    }
}

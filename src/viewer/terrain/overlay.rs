// src/viewer/terrain/overlay.rs
// Overlay layer stack for terrain viewer - Option A implementation
// Overlays are textures draped onto terrain, sampled in fragment shader and blended into albedo

use std::path::PathBuf;
use std::sync::Arc;

/// Bilinear interpolation sampling for high-quality overlay compositing
fn sample_bilinear(
    rgba: &[u8],
    width: u32,
    height: u32,
    u: f32,
    v: f32,
    opacity: f32,
) -> (f32, f32, f32, f32) {
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    
    let fx = u * (width.saturating_sub(1)) as f32;
    let fy = v * (height.saturating_sub(1)) as f32;
    
    let x0 = fx.floor() as u32;
    let y0 = fy.floor() as u32;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    
    let tx = fx.fract();
    let ty = fy.fract();
    
    let idx00 = ((y0 * width + x0) * 4) as usize;
    let idx10 = ((y0 * width + x1) * 4) as usize;
    let idx01 = ((y1 * width + x0) * 4) as usize;
    let idx11 = ((y1 * width + x1) * 4) as usize;
    
    // Safe sampling with bounds check
    let sample = |idx: usize, ch: usize| -> f32 {
        if idx + ch < rgba.len() {
            rgba[idx + ch] as f32 / 255.0
        } else {
            0.0
        }
    };
    
    // Bilinear interpolation per channel
    let bilerp = |ch: usize| -> f32 {
        let c00 = sample(idx00, ch);
        let c10 = sample(idx10, ch);
        let c01 = sample(idx01, ch);
        let c11 = sample(idx11, ch);
        let top = c00 * (1.0 - tx) + c10 * tx;
        let bot = c01 * (1.0 - tx) + c11 * tx;
        top * (1.0 - ty) + bot * ty
    };
    
    (bilerp(0), bilerp(1), bilerp(2), bilerp(3) * opacity)
}

/// Blend mode for overlay compositing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Standard alpha blend: mix(base, overlay, alpha)
    #[default]
    Normal,
    /// Multiply: base * overlay
    Multiply,
    /// Overlay: Photoshop-style overlay blend
    Overlay,
}

impl BlendMode {
    /// Convert to f32 for shader uniform
    pub fn to_shader_value(&self) -> f32 {
        match self {
            BlendMode::Normal => 0.0,
            BlendMode::Multiply => 1.0,
            BlendMode::Overlay => 2.0,
        }
    }
}

/// Source data for an overlay layer
#[derive(Debug, Clone)]
pub enum OverlayData {
    /// Raw RGBA pixels
    Raster { rgba: Vec<u8>, width: u32, height: u32 },
    /// Path to an image file (PNG, JPEG, etc.)
    Image { path: PathBuf },
}

/// Single overlay layer configuration
#[derive(Debug, Clone)]
pub struct OverlayLayer {
    /// Unique name for this layer
    pub name: String,
    /// Source data (raster pixels or image path)
    pub data: OverlayData,
    /// Extent in terrain UV space: [u_min, v_min, u_max, v_max]
    /// None means full terrain coverage [0, 0, 1, 1]
    pub extent: Option<[f32; 4]>,
    /// Opacity: 0.0 (transparent) to 1.0 (opaque)
    pub opacity: f32,
    /// Blend mode for compositing
    pub blend_mode: BlendMode,
    /// Whether this layer is visible
    pub visible: bool,
    /// Z-order for stacking (lower = behind, higher = in front)
    pub z_order: i32,
}

impl Default for OverlayLayer {
    fn default() -> Self {
        Self {
            name: String::new(),
            data: OverlayData::Raster { rgba: Vec::new(), width: 0, height: 0 },
            extent: None,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            visible: true,
            z_order: 0,
        }
    }
}

/// GPU resources for a single overlay layer
pub struct OverlayLayerGpu {
    /// Layer ID
    pub id: u32,
    /// Layer configuration (CPU side)
    pub config: OverlayLayer,
    /// GPU texture
    pub texture: wgpu::Texture,
    /// Texture view
    pub view: wgpu::TextureView,
    /// Texture dimensions
    pub dimensions: (u32, u32),
}

/// Overlay stack managing multiple layers and GPU resources
pub struct OverlayStack {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// All overlay layers
    layers: Vec<OverlayLayerGpu>,
    /// Next layer ID
    next_id: u32,
    /// Composite texture (result of flattening all layers)
    composite_texture: Option<wgpu::Texture>,
    composite_view: Option<wgpu::TextureView>,
    composite_dimensions: (u32, u32),
    /// Whether composite needs rebuild
    dirty: bool,
    /// Sampler for overlay textures (linear filtering)
    sampler: wgpu::Sampler,
}

impl OverlayStack {
    /// Create a new overlay stack
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
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
            data: OverlayData::Raster { rgba: rgba.clone(), width, height },
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
        // Load image using image crate
        let img = image::open(path)
            .map_err(|e| format!("Failed to load overlay image '{}': {}", path.display(), e))?;
        let rgba_img = img.to_rgba8();
        let (width, height) = rgba_img.dimensions();
        let rgba = rgba_img.into_raw();
        
        // Store RGBA data directly for CPU compositing (not just path reference)
        let config = OverlayLayer {
            name: name.to_string(),
            data: OverlayData::Raster { rgba: rgba.clone(), width, height },
            extent,
            opacity,
            blend_mode,
            visible: true,
            z_order,
        };

        Ok(self.add_layer_internal(config, &rgba, width, height))
    }

    /// Internal helper to add a layer with GPU resources
    fn add_layer_internal(
        &mut self,
        config: OverlayLayer,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        // Create GPU texture
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("overlay_layer_{}", id)),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload data
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
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
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

        println!("[overlay] Added layer '{}' (id={}, {}x{})", 
            self.layers.last().unwrap().config.name, id, width, height);

        id
    }

    /// Remove an overlay by ID. Returns true if found and removed.
    pub fn remove(&mut self, id: u32) -> bool {
        if let Some(pos) = self.layers.iter().position(|l| l.id == id) {
            let removed = self.layers.remove(pos);
            println!("[overlay] Removed layer '{}' (id={})", removed.config.name, id);
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
        // Assign new z_order based on position in order array
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
        self.layers.iter().any(|l| l.config.visible && l.config.opacity > 0.001)
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

    /// Build or rebuild the composite texture from all visible layers.
    /// This flattens the layer stack into a single RGBA texture.
    /// 
    /// For the initial implementation, we use a simple CPU compositing approach.
    /// A GPU compute pass could be added later for better performance.
    pub fn build_composite(&mut self, target_width: u32, target_height: u32) {
        if !self.dirty && self.composite_dimensions == (target_width, target_height) {
            return;
        }

        // Collect visible layers sorted by z-order
        let mut visible_layers: Vec<_> = self.layers.iter()
            .filter(|l| l.config.visible && l.config.opacity > 0.001)
            .collect();
        visible_layers.sort_by_key(|l| l.config.z_order);

        // Allocate CPU composite buffer
        let pixel_count = (target_width * target_height) as usize;
        let mut composite_rgba = vec![0u8; pixel_count * 4];

        // Composite each layer onto the buffer
        for layer in &visible_layers {
            let extent = layer.config.extent.unwrap_or([0.0, 0.0, 1.0, 1.0]);
            let opacity = layer.config.opacity;
            let blend_mode = layer.config.blend_mode;

            // Get layer RGBA data
            let (layer_rgba, layer_w, layer_h) = match &layer.config.data {
                OverlayData::Raster { rgba, width, height } => {
                    (rgba.as_slice(), *width, *height)
                },
                OverlayData::Image { path } => {
                    // Reload image (could cache this)
                    if let Ok(img) = image::open(path) {
                        let _rgba_img = img.to_rgba8();
                        // This is inefficient - should use cached GPU readback
                        continue; // Skip for now, use GPU-based compositing
                    } else {
                        continue;
                    }
                }
            };

            // Simple CPU compositing: for each pixel in composite
            for y in 0..target_height {
                for x in 0..target_width {
                    let u = x as f32 / target_width as f32;
                    let v = y as f32 / target_height as f32;

                    // Check if within layer extent
                    if u < extent[0] || u > extent[2] || v < extent[1] || v > extent[3] {
                        continue;
                    }

                    // Map to layer UV
                    let layer_u = (u - extent[0]) / (extent[2] - extent[0]);
                    let layer_v = (v - extent[1]) / (extent[3] - extent[1]);

                    // Bilinear interpolation for high-quality sampling
                    let (src_r, src_g, src_b, src_a) = sample_bilinear(
                        layer_rgba, layer_w, layer_h, layer_u, layer_v, opacity
                    );

                    if src_a < 0.001 {
                        continue;
                    }

                    let dst_idx = ((y * target_width + x) * 4) as usize;
                    let dst_r = composite_rgba[dst_idx] as f32 / 255.0;
                    let dst_g = composite_rgba[dst_idx + 1] as f32 / 255.0;
                    let dst_b = composite_rgba[dst_idx + 2] as f32 / 255.0;
                    let dst_a = composite_rgba[dst_idx + 3] as f32 / 255.0;

                    // Blend based on mode
                    let (out_r, out_g, out_b, out_a) = match blend_mode {
                        BlendMode::Normal => {
                            // Standard alpha blend
                            let r = dst_r * (1.0 - src_a) + src_r * src_a;
                            let g = dst_g * (1.0 - src_a) + src_g * src_a;
                            let b = dst_b * (1.0 - src_a) + src_b * src_a;
                            let a = dst_a + src_a * (1.0 - dst_a);
                            (r, g, b, a)
                        }
                        BlendMode::Multiply => {
                            // Multiply blend
                            let r = (dst_r * src_r) * src_a + dst_r * (1.0 - src_a);
                            let g = (dst_g * src_g) * src_a + dst_g * (1.0 - src_a);
                            let b = (dst_b * src_b) * src_a + dst_b * (1.0 - src_a);
                            let a = dst_a.max(src_a);
                            (r, g, b, a)
                        }
                        BlendMode::Overlay => {
                            // Overlay blend (soft light approximation)
                            fn overlay_channel(base: f32, blend: f32) -> f32 {
                                if base < 0.5 {
                                    2.0 * base * blend
                                } else {
                                    1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
                                }
                            }
                            let r = overlay_channel(dst_r, src_r) * src_a + dst_r * (1.0 - src_a);
                            let g = overlay_channel(dst_g, src_g) * src_a + dst_g * (1.0 - src_a);
                            let b = overlay_channel(dst_b, src_b) * src_a + dst_b * (1.0 - src_a);
                            let a = dst_a.max(src_a);
                            (r, g, b, a)
                        }
                    };

                    composite_rgba[dst_idx] = (out_r.clamp(0.0, 1.0) * 255.0) as u8;
                    composite_rgba[dst_idx + 1] = (out_g.clamp(0.0, 1.0) * 255.0) as u8;
                    composite_rgba[dst_idx + 2] = (out_b.clamp(0.0, 1.0) * 255.0) as u8;
                    composite_rgba[dst_idx + 3] = (out_a.clamp(0.0, 1.0) * 255.0) as u8;
                }
            }
        }

        // Create or recreate composite texture
        if self.composite_dimensions != (target_width, target_height) || self.composite_texture.is_none() {
            self.composite_texture = Some(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("overlay_composite"),
                size: wgpu::Extent3d { width: target_width, height: target_height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }));
            self.composite_view = Some(self.composite_texture.as_ref().unwrap()
                .create_view(&wgpu::TextureViewDescriptor::default()));
            self.composite_dimensions = (target_width, target_height);
        }

        // Upload composite data
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: self.composite_texture.as_ref().unwrap(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &composite_rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(target_width * 4),
                rows_per_image: Some(target_height),
            },
            wgpu::Extent3d { width: target_width, height: target_height, depth_or_array_layers: 1 },
        );

        self.dirty = false;
        
        println!("[overlay] Built composite texture {}x{} from {} visible layers",
            target_width, target_height, visible_layers.len());
    }

    /// Ensure a fallback 1x1 transparent texture exists for when no overlays are present
    pub fn ensure_fallback_texture(&mut self) -> (&wgpu::TextureView, &wgpu::Sampler) {
        if self.composite_texture.is_none() {
            // Create 1x1 transparent texture
            self.composite_texture = Some(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("overlay_fallback"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }));
            
            // Upload transparent pixel
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: self.composite_texture.as_ref().unwrap(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[0u8, 0, 0, 0],
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            
            self.composite_view = Some(self.composite_texture.as_ref().unwrap()
                .create_view(&wgpu::TextureViewDescriptor::default()));
            self.composite_dimensions = (1, 1);
        }
        
        (self.composite_view.as_ref().unwrap(), &self.sampler)
    }
}

/// Overlay configuration for IPC/API
#[derive(Debug, Clone, Default)]
pub struct OverlayConfig {
    /// Enable overlay system (default: false)
    pub enabled: bool,
    /// Global overlay opacity multiplier (0.0 - 1.0)
    pub global_opacity: f32,
    /// Show solid base surface (default: true)
    /// When false, fragments with overlay alpha=0 are discarded (like rayshader solid=FALSE)
    pub solid: bool,
}

impl OverlayConfig {
    pub fn new() -> Self {
        Self {
            enabled: false,
            global_opacity: 1.0,
            solid: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blend_mode_values() {
        assert_eq!(BlendMode::Normal.to_shader_value(), 0.0);
        assert_eq!(BlendMode::Multiply.to_shader_value(), 1.0);
        assert_eq!(BlendMode::Overlay.to_shader_value(), 2.0);
    }

    #[test]
    fn test_overlay_layer_default() {
        let layer = OverlayLayer::default();
        assert!(layer.name.is_empty());
        assert_eq!(layer.opacity, 1.0);
        assert!(layer.visible);
        assert_eq!(layer.z_order, 0);
        assert_eq!(layer.blend_mode, BlendMode::Normal);
    }
}

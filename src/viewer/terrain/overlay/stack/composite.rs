use super::OverlayStack;
use crate::viewer::terrain::overlay::sampling::{blend_pixel, sample_bilinear};
use crate::viewer::terrain::overlay::{OverlayData, OverlayLayerGpu};

impl OverlayStack {
    /// Build or rebuild the composite texture from all visible layers.
    /// This flattens the layer stack into a single RGBA texture.
    ///
    /// For the initial implementation, we use a simple CPU compositing approach.
    /// A GPU compute pass could be added later for better performance.
    pub fn build_composite(&mut self, target_width: u32, target_height: u32) {
        if !self.dirty && self.composite_dimensions == (target_width, target_height) {
            return;
        }

        let visible_layers = visible_layers(&self.layers);
        let visible_layer_count = visible_layers.len();
        let pixel_count = (target_width * target_height) as usize;
        let composite_rgba = composite_layers(&visible_layers, target_width, target_height);
        drop(visible_layers);

        self.ensure_composite_texture(target_width, target_height);
        upload_texture(
            &self.queue,
            self.composite_texture.as_ref().unwrap(),
            &composite_rgba,
            target_width,
            target_height,
        );

        let nonzero = composite_rgba.chunks(4).filter(|p| p[3] > 0).count();
        println!(
            "[overlay] Composite non-zero alpha pixels: {} / {}",
            nonzero, pixel_count
        );

        self.dirty = false;

        println!(
            "[overlay] Built composite texture {}x{} from {} visible layers",
            target_width, target_height, visible_layer_count
        );
    }

    /// Ensure a fallback 1x1 transparent texture exists for when no overlays are present
    pub fn ensure_fallback_texture(&mut self) -> (&wgpu::TextureView, &wgpu::Sampler) {
        if self.composite_texture.is_none() {
            let (texture, view) = create_texture_and_view(&self.device, 1, 1, "overlay_fallback");
            upload_texture(&self.queue, &texture, &[0u8, 0, 0, 0], 1, 1);

            self.composite_texture = Some(texture);
            self.composite_view = Some(view);
            self.composite_dimensions = (1, 1);
        }

        (self.composite_view.as_ref().unwrap(), &self.sampler)
    }

    fn ensure_composite_texture(&mut self, target_width: u32, target_height: u32) {
        if self.composite_dimensions != (target_width, target_height)
            || self.composite_texture.is_none()
        {
            let (texture, view) = create_texture_and_view(
                &self.device,
                target_width,
                target_height,
                "overlay_composite",
            );
            self.composite_texture = Some(texture);
            self.composite_view = Some(view);
            self.composite_dimensions = (target_width, target_height);
        }
    }
}

fn visible_layers(layers: &[OverlayLayerGpu]) -> Vec<&OverlayLayerGpu> {
    let mut visible_layers: Vec<_> = layers
        .iter()
        .filter(|layer| layer.config.visible && layer.config.opacity > 0.001)
        .collect();
    visible_layers.sort_by_key(|layer| layer.config.z_order);
    visible_layers
}

fn composite_layers(
    visible_layers: &[&OverlayLayerGpu],
    target_width: u32,
    target_height: u32,
) -> Vec<u8> {
    let pixel_count = (target_width * target_height) as usize;
    let mut composite_rgba = vec![0u8; pixel_count * 4];

    for layer in visible_layers {
        let Some((layer_rgba, layer_w, layer_h)) = layer_pixels(layer) else {
            continue;
        };

        let extent = layer.config.extent.unwrap_or([0.0, 0.0, 1.0, 1.0]);
        let opacity = layer.config.opacity;
        let blend_mode = layer.config.blend_mode;

        for y in 0..target_height {
            for x in 0..target_width {
                // Sample at pixel centers so a terrain-sized categorical overlay
                // round-trips without half-texel interpolation drift.
                let u = (x as f32 + 0.5) / target_width as f32;
                let v = (y as f32 + 0.5) / target_height as f32;
                if u < extent[0] || u > extent[2] || v < extent[1] || v > extent[3] {
                    continue;
                }

                let layer_u = (u - extent[0]) / (extent[2] - extent[0]);
                let layer_v = (v - extent[1]) / (extent[3] - extent[1]);
                let src = sample_bilinear(layer_rgba, layer_w, layer_h, layer_u, layer_v, opacity);
                if src[3] < 0.001 {
                    continue;
                }

                let dst_idx = ((y * target_width + x) * 4) as usize;
                let dst = [
                    composite_rgba[dst_idx] as f32 / 255.0,
                    composite_rgba[dst_idx + 1] as f32 / 255.0,
                    composite_rgba[dst_idx + 2] as f32 / 255.0,
                    composite_rgba[dst_idx + 3] as f32 / 255.0,
                ];
                let out = blend_pixel(blend_mode, dst, src);

                composite_rgba[dst_idx] = (out[0].clamp(0.0, 1.0) * 255.0) as u8;
                composite_rgba[dst_idx + 1] = (out[1].clamp(0.0, 1.0) * 255.0) as u8;
                composite_rgba[dst_idx + 2] = (out[2].clamp(0.0, 1.0) * 255.0) as u8;
                composite_rgba[dst_idx + 3] = (out[3].clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }

    composite_rgba
}

fn layer_pixels(layer: &OverlayLayerGpu) -> Option<(&[u8], u32, u32)> {
    match &layer.config.data {
        OverlayData::Raster {
            rgba,
            width,
            height,
        } => Some((rgba.as_slice(), *width, *height)),
        OverlayData::Image { path } => {
            if let Ok(img) = image::open(path) {
                let _rgba_img = img.to_rgba8();
            }
            None
        }
    }
}

fn create_texture_and_view(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
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
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn upload_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    rgba: &[u8],
    width: u32,
    height: u32,
) {
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture,
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
}

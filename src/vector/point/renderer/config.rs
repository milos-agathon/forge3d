use super::*;

impl PointRenderer {
    pub fn set_debug_flags(&mut self, debug_flags: DebugFlags) {
        self.debug_flags = debug_flags;
    }

    pub fn get_debug_flags(&self) -> DebugFlags {
        self.debug_flags
    }

    pub fn set_texture_atlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas: Option<TextureAtlas>,
    ) {
        self.texture_atlas = atlas;
        self.recreate_bind_group(device, queue);
    }

    fn recreate_bind_group(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (texture_view, sampler) = if let Some(atlas) = &self.texture_atlas {
            (&atlas.view, &atlas.sampler)
        } else {
            static DEFAULT_TEXTURE: std::sync::OnceLock<(wgpu::TextureView, wgpu::Sampler)> =
                std::sync::OnceLock::new();
            DEFAULT_TEXTURE.get_or_init(|| {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("vf.Vector.Point.DefaultTexture"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });

                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &[255, 255, 255, 255],
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4),
                        rows_per_image: Some(1),
                    },
                    wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );

                (
                    texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    device.create_sampler(&wgpu::SamplerDescriptor::default()),
                )
            });
            let (view, sampler) = DEFAULT_TEXTURE.get().unwrap();
            (view, sampler)
        };

        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vf.Vector.Point.BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
    }

    pub fn get_texture_atlas(&self) -> Option<&TextureAtlas> {
        self.texture_atlas.as_ref()
    }

    pub fn set_clip_w_scaling(&mut self, enabled: bool) {
        self.enable_clip_w_scaling = enabled;
    }

    pub fn set_depth_range(&mut self, near: f32, far: f32) {
        self.depth_range = (near, far);
    }

    pub fn is_clip_w_scaling_enabled(&self) -> bool {
        self.enable_clip_w_scaling
    }

    pub fn set_shape_mode(&mut self, mode: u32) {
        self.shape_mode = mode;
    }

    pub fn set_lod_threshold(&mut self, threshold: f32) {
        self.lod_threshold = threshold;
    }

    pub fn layer() -> Layer {
        Layer::Points
    }
}

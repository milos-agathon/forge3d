use wgpu::*;

pub struct ColormapGpu {
    pub texture: Texture,
    pub view: TextureView,
    pub sampler: Sampler,
    pub width: u32,
}

impl ColormapGpu {
    pub fn from_rgba_u8(device: &Device, queue: &Queue, data: &[u8], width: u32) -> Self {
        let size = Extent3d { width, height: 1, depth_or_array_layers: 1 };
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("forge3d_colormap"),
            size, mip_level_count: 1, sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm, usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            ImageCopyTexture { texture: &texture, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            data, ImageDataLayout { offset: 0, bytes_per_row: Some(width * 4), rows_per_image: Some(1) },
            size,
        );
        let view = texture.create_view(&TextureViewDescriptor::default());
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("forge3d_colormap_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear, min_filter: FilterMode::Linear, mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });
        Self { texture, view, sampler, width }
    }
}

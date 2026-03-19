use super::WaterReflectionUniforms;

pub(super) struct WaterReflectionInitResources {
    pub(super) water_reflection_uniform_buffer: wgpu::Buffer,
    pub(super) water_reflection_texture: wgpu::Texture,
    pub(super) water_reflection_view: wgpu::TextureView,
    pub(super) water_reflection_sampler: wgpu::Sampler,
    pub(super) water_reflection_depth_texture: wgpu::Texture,
    pub(super) water_reflection_depth_view: wgpu::TextureView,
    pub(super) water_reflection_size: (u32, u32),
    pub(super) water_reflection_fallback_view: wgpu::TextureView,
}

pub(super) fn create_water_reflection_init_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color_format: wgpu::TextureFormat,
) -> WaterReflectionInitResources {
    let water_reflection_uniform_buffer =
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain.water_reflection.uniform_buffer"),
            contents: bytemuck::bytes_of(&WaterReflectionUniforms::disabled()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let water_reflection_resolution = 512u32;
    let water_reflection_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.water_reflection.texture"),
        size: wgpu::Extent3d {
            width: water_reflection_resolution,
            height: water_reflection_resolution,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: color_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let water_reflection_view =
        water_reflection_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let water_reflection_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.water_reflection.depth"),
        size: wgpu::Extent3d {
            width: water_reflection_resolution,
            height: water_reflection_resolution,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let water_reflection_depth_view =
        water_reflection_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let water_reflection_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain.water_reflection.sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let water_reflection_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.water_reflection.fallback"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: color_format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &water_reflection_fallback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[0u8; 4],
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
    let water_reflection_fallback_view =
        water_reflection_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

    WaterReflectionInitResources {
        water_reflection_uniform_buffer,
        water_reflection_texture,
        water_reflection_view,
        water_reflection_sampler,
        water_reflection_depth_texture,
        water_reflection_depth_view,
        water_reflection_size: (water_reflection_resolution, water_reflection_resolution),
        water_reflection_fallback_view,
    }
}

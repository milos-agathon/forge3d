use super::pipelines::{create_height_ao_pipeline_resources, create_sun_vis_pipeline_resources};

pub(in crate::terrain::renderer) struct HeightfieldInitResources {
    pub(in crate::terrain::renderer) ao_debug_sampler: wgpu::Sampler,
    pub(in crate::terrain::renderer) ao_debug_fallback_texture: wgpu::Texture,
    pub(in crate::terrain::renderer) ao_debug_fallback_view: wgpu::TextureView,
    pub(in crate::terrain::renderer) height_ao_fallback_view: wgpu::TextureView,
    pub(in crate::terrain::renderer) height_ao_sampler: wgpu::Sampler,
    pub(in crate::terrain::renderer) height_ao_compute_pipeline: wgpu::ComputePipeline,
    pub(in crate::terrain::renderer) height_ao_bind_group_layout: wgpu::BindGroupLayout,
    pub(in crate::terrain::renderer) height_ao_uniform_buffer: wgpu::Buffer,
    pub(in crate::terrain::renderer) sun_vis_fallback_view: wgpu::TextureView,
    pub(in crate::terrain::renderer) sun_vis_sampler: wgpu::Sampler,
    pub(in crate::terrain::renderer) sun_vis_compute_pipeline: wgpu::ComputePipeline,
    pub(in crate::terrain::renderer) sun_vis_bind_group_layout: wgpu::BindGroupLayout,
    pub(in crate::terrain::renderer) sun_vis_uniform_buffer: wgpu::Buffer,
}

pub(in crate::terrain::renderer) fn create_heightfield_init_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> HeightfieldInitResources {
    let ao_debug_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain.ao_debug.sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let height_ao_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.height_ao_fallback"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &height_ao_fallback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&[1.0f32]),
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
    let height_ao_fallback_view =
        height_ao_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let height_ao_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain.height_ao.sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let sun_vis_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.sun_vis_fallback"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &sun_vis_fallback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&[1.0f32]),
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
    let sun_vis_fallback_view =
        sun_vis_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sun_vis_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain.sun_vis.sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let (height_ao_compute_pipeline, height_ao_bind_group_layout, height_ao_uniform_buffer) =
        create_height_ao_pipeline_resources(device);
    let (sun_vis_compute_pipeline, sun_vis_bind_group_layout, sun_vis_uniform_buffer) =
        create_sun_vis_pipeline_resources(device);

    let ao_debug_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.ao_debug_fallback"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &ao_debug_fallback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&[1.0f32]),
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
    let ao_debug_fallback_view =
        ao_debug_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

    HeightfieldInitResources {
        ao_debug_sampler,
        ao_debug_fallback_texture,
        ao_debug_fallback_view,
        height_ao_fallback_view,
        height_ao_sampler,
        height_ao_compute_pipeline,
        height_ao_bind_group_layout,
        height_ao_uniform_buffer,
        sun_vis_fallback_view,
        sun_vis_sampler,
        sun_vis_compute_pipeline,
        sun_vis_bind_group_layout,
        sun_vis_uniform_buffer,
    }
}

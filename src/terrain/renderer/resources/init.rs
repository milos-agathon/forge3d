use super::*;

pub(in crate::terrain::renderer) struct BaseInitResources {
    pub(in crate::terrain::renderer) sampler_linear: wgpu::Sampler,
    pub(in crate::terrain::renderer) height_curve_lut_sampler: wgpu::Sampler,
    pub(in crate::terrain::renderer) height_curve_identity_texture: wgpu::Texture,
    pub(in crate::terrain::renderer) height_curve_identity_view: wgpu::TextureView,
    pub(in crate::terrain::renderer) water_mask_fallback_texture: wgpu::Texture,
    pub(in crate::terrain::renderer) water_mask_fallback_view: wgpu::TextureView,
    pub(in crate::terrain::renderer) detail_normal_fallback_view: wgpu::TextureView,
    pub(in crate::terrain::renderer) detail_normal_sampler: wgpu::Sampler,
}

pub(in crate::terrain::renderer) fn create_base_init_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<BaseInitResources> {
    let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain.sampler.nearest"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let height_curve_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain.height_curve.lut_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let identity_lut_data: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
    let (height_curve_identity_texture, height_curve_identity_view) =
        TerrainScene::upload_height_curve_lut_internal(device, queue, &identity_lut_data)?;

    let water_mask_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.water_mask_fallback"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &water_mask_fallback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[0u8],
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(1),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let water_mask_fallback_view =
        water_mask_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let detail_normal_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.detail_normal_fallback"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &detail_normal_fallback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[128u8, 128u8, 255u8, 255u8],
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
    let detail_normal_fallback_view =
        detail_normal_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let detail_normal_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain.detail_normal.sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    Ok(BaseInitResources {
        sampler_linear,
        height_curve_lut_sampler,
        height_curve_identity_texture,
        height_curve_identity_view,
        water_mask_fallback_texture,
        water_mask_fallback_view,
        detail_normal_fallback_view,
        detail_normal_sampler,
    })
}

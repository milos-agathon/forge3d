use super::uniforms::WaterReflectionUniforms;
use crate::core::error::RenderResult;
use crate::core::resource_tracker::{
    tracked_create_buffer_init, tracked_create_texture, TrackedBuffer, TrackedTexture,
};

pub(in crate::terrain::renderer) struct WaterReflectionInitResources {
    pub(in crate::terrain::renderer) water_reflection_uniform_buffer: TrackedBuffer,
    pub(in crate::terrain::renderer) water_reflection_texture: TrackedTexture,
    pub(in crate::terrain::renderer) water_reflection_view: wgpu::TextureView,
    pub(in crate::terrain::renderer) water_reflection_sampler: wgpu::Sampler,
    pub(in crate::terrain::renderer) water_reflection_depth_texture: TrackedTexture,
    pub(in crate::terrain::renderer) water_reflection_depth_view: wgpu::TextureView,
    pub(in crate::terrain::renderer) water_reflection_size: (u32, u32),
    pub(in crate::terrain::renderer) water_reflection_fallback_view: wgpu::TextureView,
}

pub(in crate::terrain::renderer) fn create_water_reflection_init_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color_format: wgpu::TextureFormat,
) -> RenderResult<WaterReflectionInitResources> {
    let water_reflection_uniform_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("terrain.water_reflection.uniform_buffer"),
            contents: bytemuck::bytes_of(&WaterReflectionUniforms::disabled()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    )?;

    let water_reflection_resolution = 512u32;
    let water_reflection_texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
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
        },
    )?;
    let water_reflection_view =
        water_reflection_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let water_reflection_depth_texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
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
        },
    )?;
    let water_reflection_depth_view =
        water_reflection_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let water_reflection_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain.water_reflection.sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: crate::core::gpu::deterministic_filter_mode(wgpu::FilterMode::Linear),
        min_filter: crate::core::gpu::deterministic_filter_mode(wgpu::FilterMode::Linear),
        mipmap_filter: crate::core::gpu::deterministic_filter_mode(wgpu::FilterMode::Nearest),
        ..Default::default()
    });

    let water_reflection_fallback_texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
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
        },
    )?;
    let texel_bytes = color_format.block_copy_size(None).ok_or_else(|| {
        crate::core::error::RenderError::Render(
            "Unsupported water reflection fallback format".into(),
        )
    })?;
    let fallback_pixel = vec![0u8; texel_bytes as usize];
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &water_reflection_fallback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &fallback_pixel,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(texel_bytes),
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

    Ok(WaterReflectionInitResources {
        water_reflection_uniform_buffer,
        water_reflection_texture,
        water_reflection_view,
        water_reflection_sampler,
        water_reflection_depth_texture,
        water_reflection_depth_view,
        water_reflection_size: (water_reflection_resolution, water_reflection_resolution),
        water_reflection_fallback_view,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn rgba16float_fallback_upload_is_valid() {
        let context = crate::core::gpu::try_ctx().expect("physical GPU required");
        assert!(!context.software_fallback, "physical GPU required");
        context
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let _resources = super::create_water_reflection_init_resources(
            &context.device,
            &context.queue,
            wgpu::TextureFormat::Rgba16Float,
        )
        .unwrap();
        context.queue.submit(std::iter::empty());
        context.device.poll(wgpu::Maintain::Wait);
        assert!(pollster::block_on(context.device.pop_error_scope()).is_none());
    }
}

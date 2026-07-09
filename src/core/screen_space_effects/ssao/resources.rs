use super::*;
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, tracked_create_texture, TrackedBuffer,
    TrackedTexture,
};

pub(super) struct SsaoSharedBuffers {
    pub settings_buffer: TrackedBuffer,
    pub camera_buffer: TrackedBuffer,
    pub blur_settings: TrackedBuffer,
    pub temporal_params_buffer: TrackedBuffer,
    pub comp_uniform: TrackedBuffer,
}

pub(super) struct SsaoTextures {
    pub noise_texture: TrackedTexture,
    pub noise_view: TextureView,
    pub noise_sampler: Sampler,
    pub ssao_texture: TrackedTexture,
    pub ssao_view: TextureView,
    pub ssao_blurred: TrackedTexture,
    pub ssao_blurred_view: TextureView,
    pub ssao_history: TrackedTexture,
    pub ssao_history_view: TextureView,
    pub ssao_resolved: TrackedTexture,
    pub ssao_resolved_view: TextureView,
    pub ssao_tmp: TrackedTexture,
    pub ssao_tmp_view: TextureView,
    pub ssao_composited: TrackedTexture,
    pub ssao_composited_view: TextureView,
}

pub(super) fn create_shared_buffers(
    device: &Device,
    settings: &SsaoSettings,
) -> RenderResult<SsaoSharedBuffers> {
    let settings_buffer = tracked_create_buffer(
        device,
        &BufferDescriptor {
            label: Some("ssao_settings"),
            size: std::mem::size_of::<SsaoSettings>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )?;
    let camera_buffer = tracked_create_buffer(
        device,
        &BufferDescriptor {
            label: Some("ssao_camera"),
            size: std::mem::size_of::<CameraParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )?;

    let blur_params = BlurSettingsStd140 {
        blur_radius: 6,
        depth_sigma: 0.1,
        normal_sigma: 0.6,
        _pad: 0,
    };
    let blur_settings = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("ssao.blur.settings"),
            contents: bytemuck::bytes_of(&blur_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        },
    )?;

    let temporal_params = SsaoTemporalParamsUniform {
        temporal_alpha: 0.2,
        _pad: [0.0; 7],
    };
    let temporal_params_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("ssao.temporal.params"),
            contents: bytemuck::bytes_of(&temporal_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        },
    )?;

    let _ssr_composite_params = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("p5.ssr.composite.params"),
            contents: bytemuck::cast_slice(&[1.0, 0.0, 0.0, 0.0]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        },
    )?;
    let _ssao_composite_params = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("ssao.comp.uniform"),
            contents: bytemuck::cast_slice(&[1.0, 0.0, 0.0, 0.0]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        },
    )?;

    let comp_params: [f32; 4] = [1.0, 1.0, 0.0, 0.0];
    let comp_uniform = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("ssao.comp.uniform"),
            contents: bytemuck::cast_slice(&comp_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        },
    )?;

    let _ = settings;

    Ok(SsaoSharedBuffers {
        settings_buffer,
        camera_buffer,
        blur_settings,
        temporal_params_buffer,
        comp_uniform,
    })
}

pub(super) fn create_textures(
    device: &Device,
    width: u32,
    height: u32,
) -> RenderResult<SsaoTextures> {
    let (noise_texture, noise_view, noise_sampler) = create_noise_resources(device)?;
    let (ssao_texture, ssao_view) = create_viewed_texture(
        device,
        "ssao_texture",
        width,
        height,
        TextureFormat::R32Float,
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
    )?;
    let (ssao_blurred, ssao_blurred_view) = create_viewed_texture(
        device,
        "ssao_blurred",
        width,
        height,
        TextureFormat::R32Float,
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
    )?;
    let (ssao_tmp, ssao_tmp_view) = create_viewed_texture(
        device,
        "ssao_tmp",
        width,
        height,
        TextureFormat::R32Float,
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
    )?;
    let (ssao_history, ssao_history_view) = create_viewed_texture(
        device,
        "ssao_history",
        width,
        height,
        TextureFormat::R32Float,
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    )?;
    let (ssao_resolved, ssao_resolved_view) = create_viewed_texture(
        device,
        "ssao_resolved",
        width,
        height,
        TextureFormat::R32Float,
        TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC
            | TextureUsages::COPY_DST,
    )?;
    let (ssao_composited, ssao_composited_view) = create_viewed_texture(
        device,
        "ssao_composited",
        width,
        height,
        TextureFormat::Rgba8Unorm,
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
    )?;

    Ok(SsaoTextures {
        noise_texture,
        noise_view,
        noise_sampler,
        ssao_texture,
        ssao_view,
        ssao_blurred,
        ssao_blurred_view,
        ssao_history,
        ssao_history_view,
        ssao_resolved,
        ssao_resolved_view,
        ssao_tmp,
        ssao_tmp_view,
        ssao_composited,
        ssao_composited_view,
    })
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurSettingsStd140 {
    blur_radius: u32,
    depth_sigma: f32,
    normal_sigma: f32,
    _pad: u32,
}

fn create_noise_resources(device: &Device) -> RenderResult<(TrackedTexture, TextureView, Sampler)> {
    let noise_size = 4u32;
    let mut noise_data: Vec<f32> = vec![0.0; (noise_size * noise_size) as usize];
    for (i, sample) in noise_data.iter_mut().enumerate() {
        *sample = (i as f32 * 0.618033988749895) % 1.0;
    }
    let _ = noise_data;

    let noise_texture = tracked_create_texture(
        device,
        &TextureDescriptor {
            label: Some("ssao_noise"),
            size: Extent3d {
                width: noise_size,
                height: noise_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
    )?;
    let noise_view = noise_texture.create_view(&TextureViewDescriptor::default());
    let noise_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("ssao_noise_sampler"),
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        address_mode_w: AddressMode::Repeat,
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        ..Default::default()
    });

    Ok((noise_texture, noise_view, noise_sampler))
}

fn create_viewed_texture(
    device: &Device,
    label: &str,
    width: u32,
    height: u32,
    format: TextureFormat,
    usage: TextureUsages,
) -> RenderResult<(TrackedTexture, TextureView)> {
    let texture = tracked_create_texture(
        device,
        &TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        },
    )?;
    let view = texture.create_view(&TextureViewDescriptor::default());
    Ok((texture, view))
}

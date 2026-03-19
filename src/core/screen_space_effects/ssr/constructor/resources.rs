use super::*;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsrCompositeParamsStd140 {
    boost: f32,
    exposure: f32,
    gamma: f32,
    weight_floor: f32,
    tone_white: f32,
    tone_bias: f32,
    reinhard_k: f32,
    _pad0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsrTemporalParamsStd140 {
    temporal_alpha: f32,
    pad: [f32; 3],
}

pub(super) fn create_buffers(device: &Device) -> ConstructorBuffers {
    ConstructorBuffers {
        settings_buffer: uniform_buffer::<SsrSettings>(device, "ssr_settings"),
        camera_buffer: uniform_buffer::<CameraParams>(device, "ssr_camera"),
        counters_buffer: device.create_buffer(&BufferDescriptor {
            label: Some("p5.ssr.counters"),
            size: std::mem::size_of::<[u32; 5]>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        counters_readback: device.create_buffer(&BufferDescriptor {
            label: Some("p5.ssr.counters.readback"),
            size: std::mem::size_of::<[u32; 5]>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        composite_params: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("p5.ssr.composite.params"),
            contents: bytemuck::bytes_of(&SsrCompositeParamsStd140 {
                boost: 1.6,
                exposure: 1.1,
                gamma: 1.0,
                weight_floor: 0.2,
                tone_white: 1.0,
                tone_bias: 0.0,
                reinhard_k: 1.0,
                _pad0: 0.0,
            }),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }),
        temporal_params: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("p5.ssr.temporal.params"),
            contents: bytemuck::bytes_of(&SsrTemporalParamsStd140 {
                temporal_alpha: 0.85,
                pad: [0.0; 3],
            }),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }),
    }
}

pub(super) fn create_textures(device: &Device, width: u32, height: u32) -> ConstructorResources {
    let (ssr_spec_texture, ssr_spec_view) =
        rgba16_texture(device, "p5.ssr.spec", width, height, true);
    let (ssr_final_texture, ssr_final_view) =
        rgba16_texture(device, "p5.ssr.final", width, height, true);
    let (ssr_history_texture, ssr_history_view) =
        rgba16_texture(device, "p5.ssr.history", width, height, false);
    let (ssr_filtered_texture, ssr_filtered_view) =
        rgba16_texture(device, "p5.ssr.filtered", width, height, true);
    let (ssr_hit_texture, ssr_hit_view) = rgba16_texture(device, "p5.ssr.hit", width, height, true);
    let (ssr_composited_texture, ssr_composited_view) =
        rgba8_texture(device, "p5.ssr.composited", width, height);

    let env_texture = device.create_texture(&TextureDescriptor {
        label: Some("p5.ssr.env.placeholder"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let env_view = env_texture.create_view(&TextureViewDescriptor {
        label: Some("p5.ssr.env.view"),
        format: None,
        dimension: Some(TextureViewDimension::Cube),
        aspect: TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: None,
    });
    let env_sampler = linear_sampler(device, "p5.ssr.env.sampler");
    let linear_sampler = linear_sampler(device, "p5.ssr.linear");

    ConstructorResources {
        ssr_spec_texture,
        ssr_spec_view,
        ssr_final_texture,
        ssr_final_view,
        ssr_history_texture,
        ssr_history_view,
        ssr_filtered_texture,
        ssr_filtered_view,
        ssr_hit_texture,
        ssr_hit_view,
        ssr_composited_texture,
        ssr_composited_view,
        env_texture,
        env_view,
        env_sampler,
        linear_sampler,
    }
}

fn uniform_buffer<T>(device: &Device, label: &str) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn rgba16_texture(
    device: &Device,
    label: &str,
    width: u32,
    height: u32,
    storage_binding: bool,
) -> (Texture, TextureView) {
    let mut usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;
    if storage_binding {
        usage |= TextureUsages::STORAGE_BINDING;
    } else {
        usage |= TextureUsages::COPY_DST;
    }
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

fn rgba8_texture(device: &Device, label: &str, width: u32, height: u32) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

fn linear_sampler(device: &Device, label: &str) -> Sampler {
    device.create_sampler(&SamplerDescriptor {
        label: Some(label),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        ..Default::default()
    })
}

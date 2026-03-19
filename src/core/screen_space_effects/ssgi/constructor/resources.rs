use super::*;
use wgpu::util::DeviceExt;

pub(super) fn create_buffers(device: &Device) -> ConstructorBuffers {
    let comp_params: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
    ConstructorBuffers {
        settings_buffer: uniform_buffer::<SsgiSettings>(device, "ssgi_settings"),
        camera_buffer: uniform_buffer::<CameraParams>(device, "ssgi_camera"),
        composite_uniform: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssgi.composite.uniform"),
            contents: bytemuck::cast_slice(&comp_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }),
    }
}

pub(super) fn create_textures(
    device: &Device,
    width: u32,
    height: u32,
    material_format: TextureFormat,
) -> ConstructorResources {
    let (ssgi_hit, ssgi_hit_view) = rgba16_texture(
        device,
        "ssgi_hit",
        width,
        height,
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
    );
    let (ssgi_texture, ssgi_view) = rgba16_texture(
        device,
        "ssgi_texture",
        width,
        height,
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
    );
    let (ssgi_history, ssgi_history_view) = rgba16_texture(
        device,
        "ssgi_history",
        width,
        height,
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::COPY_SRC,
    );
    let (ssgi_filtered, ssgi_filtered_view) = rgba16_texture(
        device,
        "ssgi_filtered",
        width,
        height,
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
    );
    let (ssgi_upscaled, ssgi_upscaled_view) = rgba16_texture(
        device,
        "ssgi_upscaled",
        width,
        height,
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    );
    let (ssgi_composited, ssgi_composited_view) =
        rgba8_texture(device, "ssgi_composited", width, height);

    let history_usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::COPY_SRC;
    let scene_history = [
        scene_history_texture(
            device,
            "ssgi_scene_history_a",
            width,
            height,
            material_format,
            history_usage,
        ),
        scene_history_texture(
            device,
            "ssgi_scene_history_b",
            width,
            height,
            material_format,
            history_usage,
        ),
    ];
    let scene_history_views = [
        scene_history[0].create_view(&TextureViewDescriptor::default()),
        scene_history[1].create_view(&TextureViewDescriptor::default()),
    ];

    let env_texture = device.create_texture(&TextureDescriptor {
        label: Some("ssgi_env_cube"),
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
        label: Some("ssgi_env_cube_view"),
        format: Some(TextureFormat::Rgba8Unorm),
        dimension: Some(TextureViewDimension::Cube),
        aspect: TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: None,
    });

    ConstructorResources {
        ssgi_hit,
        ssgi_hit_view,
        ssgi_texture,
        ssgi_view,
        ssgi_history,
        ssgi_history_view,
        ssgi_filtered,
        ssgi_filtered_view,
        ssgi_upscaled,
        ssgi_upscaled_view,
        ssgi_composited,
        ssgi_composited_view,
        scene_history,
        scene_history_views,
        env_texture,
        env_view,
        env_sampler: device.create_sampler(&SamplerDescriptor::default()),
        linear_sampler: device.create_sampler(&SamplerDescriptor {
            label: Some("ssgi.linear.sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            ..Default::default()
        }),
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
    usage: TextureUsages,
) -> (Texture, TextureView) {
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
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

fn scene_history_texture(
    device: &Device,
    label: &str,
    width: u32,
    height: u32,
    format: TextureFormat,
    usage: TextureUsages,
) -> Texture {
    device.create_texture(&TextureDescriptor {
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
    })
}

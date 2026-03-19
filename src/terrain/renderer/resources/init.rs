use super::*;

pub(super) struct BaseInitResources {
    pub(super) sampler_linear: wgpu::Sampler,
    pub(super) height_curve_lut_sampler: wgpu::Sampler,
    pub(super) height_curve_identity_texture: wgpu::Texture,
    pub(super) height_curve_identity_view: wgpu::TextureView,
    pub(super) water_mask_fallback_texture: wgpu::Texture,
    pub(super) water_mask_fallback_view: wgpu::TextureView,
    pub(super) detail_normal_fallback_view: wgpu::TextureView,
    pub(super) detail_normal_sampler: wgpu::Sampler,
}

pub(super) struct AccumulationInitResources {
    pub(super) accumulation_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) accumulation_pipeline: wgpu::ComputePipeline,
    pub(super) accumulation_params_buffer: wgpu::Buffer,
}

pub(super) fn create_base_init_resources(
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

pub(super) fn create_accumulation_init_resources(
    device: &wgpu::Device,
) -> AccumulationInitResources {
    let accumulation_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.accumulation.bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

    let accumulation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("terrain.accumulation.shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "../../../shaders/accumulation_blend.wgsl"
        ))),
    });
    let accumulation_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.accumulation.pipeline_layout"),
            bind_group_layouts: &[&accumulation_bind_group_layout],
            push_constant_ranges: &[],
        });
    let accumulation_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("terrain.accumulation.pipeline"),
        layout: Some(&accumulation_pipeline_layout),
        module: &accumulation_shader,
        entry_point: "accumulate",
    });
    let accumulation_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain.accumulation.params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    AccumulationInitResources {
        accumulation_bind_group_layout,
        accumulation_pipeline,
        accumulation_params_buffer,
    }
}

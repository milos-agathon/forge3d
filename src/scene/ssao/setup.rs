struct SsaoLayouts {
    ssao_bind_group_layout: wgpu::BindGroupLayout,
    ssao_output_bind_group_layout: wgpu::BindGroupLayout,
    blur_bind_group_layout: wgpu::BindGroupLayout,
    composite_bind_group_layout: wgpu::BindGroupLayout,
}
struct SsaoPipelines {
    ssao_pipeline: wgpu::ComputePipeline,
    blur_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::ComputePipeline,
}
struct SsaoBuffers {
    sampler: wgpu::Sampler,
    blur_sampler: wgpu::Sampler,
    settings_buffer: wgpu::Buffer,
    blur_settings_buffer: wgpu::Buffer,
    view_buffer: wgpu::Buffer,
}
struct SsaoNoiseResources {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}
struct SsaoDepthResources {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

fn create_ssao_shader(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ssao-compute"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/ssao.wgsl").into()),
    })
}

fn create_ssao_layouts(device: &wgpu::Device) -> SsaoLayouts {
    let ssao_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_bind_group_layout"),
            entries: &[
                sampled_texture_layout_entry(0),
                sampled_texture_layout_entry(1),
                sampled_texture_layout_entry(2),
                sampled_texture_layout_entry(3),
                sampler_layout_entry(4, wgpu::SamplerBindingType::NonFiltering),
                storage_texture_layout_entry(5, wgpu::TextureFormat::R32Float),
                uniform_buffer_layout_entry(6),
                uniform_buffer_layout_entry(7),
            ],
        });

    let ssao_output_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_output_bind_group_layout_dummy"),
            entries: &[],
        });

    let blur_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_blur_bind_group_layout"),
            entries: &[
                sampled_texture_layout_entry(0),
                storage_texture_layout_entry(1, wgpu::TextureFormat::R32Float),
                uniform_buffer_layout_entry(2),
                sampled_texture_layout_entry(3),
            ],
        });

    let composite_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_composite_bind_group_layout"),
            entries: &[
                sampled_texture_layout_entry(0),
                storage_texture_layout_entry(1, wgpu::TextureFormat::Rgba8Unorm),
                sampled_texture_layout_entry(2),
                uniform_buffer_layout_entry(3),
            ],
        });

    SsaoLayouts {
        ssao_bind_group_layout,
        ssao_output_bind_group_layout,
        blur_bind_group_layout,
        composite_bind_group_layout,
    }
}

fn create_ssao_pipelines(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layouts: &SsaoLayouts) -> SsaoPipelines {
    let ssao_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ssao-pipeline"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ssao-pipeline-layout"),
                bind_group_layouts: &[&layouts.ssao_bind_group_layout],
                push_constant_ranges: &[],
            }),
        ),
        module: shader,
        entry_point: "cs_ssao",
    });

    let blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ssao-blur-pipeline"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ssao-blur-pipeline-layout"),
                bind_group_layouts: &[&layouts.ssao_bind_group_layout],
                push_constant_ranges: &[],
            }),
        ),
        module: shader,
        entry_point: "cs_ssao",
    });

    let composite_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ssao-composite-pipeline"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ssao-composite-pipeline-layout"),
                bind_group_layouts: &[&layouts.composite_bind_group_layout],
                push_constant_ranges: &[],
            }),
        ),
        module: shader,
        entry_point: "cs_ssao_composite",
    });

    SsaoPipelines {
        ssao_pipeline,
        blur_pipeline,
        composite_pipeline,
    }
}

fn create_ssao_buffers(device: &wgpu::Device) -> SsaoBuffers {
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("ssao-sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let blur_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("ssao-blur-sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let settings_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ssao-settings"),
        size: std::mem::size_of::<SsaoSettingsUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let blur_settings_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ssao-blur-settings"),
        size: std::mem::size_of::<SsaoSettingsUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let view_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ssao-view-params"),
        size: 256,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    SsaoBuffers {
        sampler,
        blur_sampler,
        settings_buffer,
        blur_settings_buffer,
        view_buffer,
    }
}

fn create_ssao_noise_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> SsaoNoiseResources {
    let noise_size = 4u32;
    let mut noise_data = vec![0u8; (noise_size * noise_size * 4) as usize];
    for (i, chunk) in noise_data.chunks_mut(4).enumerate() {
        let angle = (i as f32 * 2.0 * std::f32::consts::PI) / 16.0;
        chunk[0] = ((angle.cos() * 0.5 + 0.5) * 255.0) as u8;
        chunk[1] = ((angle.sin() * 0.5 + 0.5) * 255.0) as u8;
        chunk[2] = 0;
        chunk[3] = 255;
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ssao-noise"),
        size: wgpu::Extent3d {
            width: noise_size,
            height: noise_size,
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
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &noise_data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(noise_size * 4),
            rows_per_image: Some(noise_size),
        },
        wgpu::Extent3d {
            width: noise_size,
            height: noise_size,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("ssao-noise-sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    SsaoNoiseResources {
        texture,
        view,
        sampler,
    }
}

fn create_ssao_depth_resources(device: &wgpu::Device, width: u32, height: u32) -> SsaoDepthResources {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ssao-depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    SsaoDepthResources { texture, view }
}

fn sampled_texture_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_texture_layout_entry(
    binding: u32,
    format: wgpu::TextureFormat,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn uniform_buffer_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn sampler_layout_entry(
    binding: u32,
    sampler_type: wgpu::SamplerBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Sampler(sampler_type),
        count: None,
    }
}


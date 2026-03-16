use super::*;

pub(super) struct SsaoLayouts {
    pub ssao_bind_group_layout: BindGroupLayout,
    pub blur_bind_group_layout: BindGroupLayout,
    pub temporal_bind_group_layout: BindGroupLayout,
    pub composite_bind_group_layout: BindGroupLayout,
}

pub(super) struct SsaoPipelines {
    pub ssao_pipeline: ComputePipeline,
    pub gtao_pipeline: ComputePipeline,
    pub blur_h_pipeline: ComputePipeline,
    pub blur_v_pipeline: ComputePipeline,
    pub temporal_pipeline: ComputePipeline,
    pub composite_pipeline: ComputePipeline,
}

pub(super) fn create_layouts(device: &Device) -> SsaoLayouts {
    let ssao_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("ssao_bind_group_layout"),
        entries: &[
            sampled_texture_entry(0),
            sampled_texture_entry(1),
            sampled_texture_entry(2),
            sampled_texture_entry(3),
            sampler_entry(4),
            storage_texture_entry(5, TextureFormat::R32Float),
            uniform_entry(6),
            uniform_entry(7),
        ],
    });

    let blur_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("ssao_blur_bind_group_layout"),
        entries: &[
            sampled_texture_entry(0),
            sampled_texture_entry(1),
            sampled_texture_entry(2),
            storage_texture_entry(3, TextureFormat::R32Float),
            uniform_entry(4),
        ],
    });

    let temporal_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("ssao_temporal_bgl"),
        entries: &[
            sampled_texture_entry(0),
            sampled_texture_entry(1),
            storage_texture_entry(2, TextureFormat::R32Float),
            uniform_entry(3),
        ],
    });

    let composite_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("ssao_composite_bind_group_layout"),
        entries: &[
            sampled_texture_entry(0),
            storage_texture_entry(1, TextureFormat::Rgba8Unorm),
            sampled_texture_entry(2),
            uniform_entry(3),
        ],
    });

    SsaoLayouts {
        ssao_bind_group_layout,
        blur_bind_group_layout,
        temporal_bind_group_layout,
        composite_bind_group_layout,
    }
}

pub(super) fn create_pipelines(
    device: &Device,
    ssao_bind_group_layout: &BindGroupLayout,
    blur_bind_group_layout: &BindGroupLayout,
    temporal_bind_group_layout: &BindGroupLayout,
    composite_bind_group_layout: &BindGroupLayout,
) -> SsaoPipelines {
    let ssao_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ssao_kernel_shader"),
        source: ShaderSource::Wgsl(SSAO_SHADER_SRC.into()),
    });
    let gtao_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("gtao_kernel_shader"),
        source: ShaderSource::Wgsl(GTAO_SHADER_SRC.into()),
    });
    let filter_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ssao_filter_shader"),
        source: ShaderSource::Wgsl(
            include_str!("../../../shaders/filters/bilateral_separable.wgsl").into(),
        ),
    });
    let temporal_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ssao_temporal_shader"),
        source: ShaderSource::Wgsl(
            include_str!("../../../shaders/temporal/resolve_ao.wgsl").into(),
        ),
    });
    let composite_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ssao_composite_shader"),
        source: ShaderSource::Wgsl(SSAO_COMPOSITE_SHADER_SRC.into()),
    });

    let ssao_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("ssao_pipeline_layout"),
        bind_group_layouts: &[ssao_bind_group_layout],
        push_constant_ranges: &[],
    });
    let blur_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("ssao_blur_pipeline_layout"),
        bind_group_layouts: &[blur_bind_group_layout],
        push_constant_ranges: &[],
    });
    let temporal_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("ssao_temporal_pl"),
        bind_group_layouts: &[temporal_bind_group_layout],
        push_constant_ranges: &[],
    });
    let composite_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("ssao_composite_pipeline_layout"),
        bind_group_layouts: &[composite_bind_group_layout],
        push_constant_ranges: &[],
    });

    SsaoPipelines {
        ssao_pipeline: create_compute_pipeline(
            device,
            "ssao_pipeline",
            &ssao_pipeline_layout,
            &ssao_module,
            "cs_ssao",
        ),
        gtao_pipeline: create_compute_pipeline(
            device,
            "gtao_pipeline",
            &ssao_pipeline_layout,
            &gtao_module,
            "cs_gtao",
        ),
        blur_h_pipeline: create_compute_pipeline(
            device,
            "ssao_blur_h_pipeline",
            &blur_pipeline_layout,
            &filter_shader,
            "cs_blur_h",
        ),
        blur_v_pipeline: create_compute_pipeline(
            device,
            "ssao_blur_v_pipeline",
            &blur_pipeline_layout,
            &filter_shader,
            "cs_blur_v",
        ),
        temporal_pipeline: create_compute_pipeline(
            device,
            "ssao_temporal_pipeline",
            &temporal_pipeline_layout,
            &temporal_shader,
            "cs_resolve_temporal",
        ),
        composite_pipeline: create_compute_pipeline(
            device,
            "ssao_composite_pipeline",
            &composite_pipeline_layout,
            &composite_shader,
            "cs_ssao_composite",
        ),
    }
}

fn sampled_texture_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: false },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_texture_entry(binding: u32, format: TextureFormat) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format,
            view_dimension: TextureViewDimension::D2,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
        count: None,
    }
}

fn create_compute_pipeline(
    device: &Device,
    label: &str,
    layout: &PipelineLayout,
    module: &ShaderModule,
    entry_point: &str,
) -> ComputePipeline {
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module,
        entry_point,
    })
}

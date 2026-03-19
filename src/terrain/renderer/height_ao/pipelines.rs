use super::*;

pub(super) fn create_height_ao_pipeline_resources(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::Buffer) {
    let height_ao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("heightfield_ao.wgsl"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../../shaders/heightfield_ao.wgsl").into(),
        ),
    });
    let height_ao_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("height_ao.bind_group_layout"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
    let height_ao_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("height_ao.pipeline_layout"),
            bind_group_layouts: &[&height_ao_bind_group_layout],
            push_constant_ranges: &[],
        });
    let height_ao_compute_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("height_ao.compute_pipeline"),
            layout: Some(&height_ao_pipeline_layout),
            module: &height_ao_shader,
            entry_point: "main",
        });
    let height_ao_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("height_ao.uniform_buffer"),
        contents: bytemuck::bytes_of(&HeightAoUniforms {
            params0: [6.0, 16.0, 200.0, 1.0],
            params1: [1.0, 1.0, 1.0, 0.0],
            params2: [1.0, 1.0, 1.0, 1.0],
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    (
        height_ao_compute_pipeline,
        height_ao_bind_group_layout,
        height_ao_uniform_buffer,
    )
}

pub(super) fn create_sun_vis_pipeline_resources(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::Buffer) {
    let sun_vis_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("heightfield_sun_vis.wgsl"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../../shaders/heightfield_sun_vis.wgsl").into(),
        ),
    });
    let sun_vis_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sun_vis.bind_group_layout"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
    let sun_vis_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sun_vis.pipeline_layout"),
        bind_group_layouts: &[&sun_vis_bind_group_layout],
        push_constant_ranges: &[],
    });
    let sun_vis_compute_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sun_vis.compute_pipeline"),
            layout: Some(&sun_vis_pipeline_layout),
            module: &sun_vis_shader,
            entry_point: "main",
        });
    let sun_vis_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sun_vis.uniform_buffer"),
        contents: bytemuck::bytes_of(&SunVisUniforms {
            params0: [4.0, 24.0, 400.0, 1.0],
            params1: [1.0, 1.0, 1.0, 0.0],
            params2: [1.0, 1.0, 1.0, 1.0],
            params3: [0.0, 1.0, 0.0, 0.01],
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    (
        sun_vis_compute_pipeline,
        sun_vis_bind_group_layout,
        sun_vis_uniform_buffer,
    )
}

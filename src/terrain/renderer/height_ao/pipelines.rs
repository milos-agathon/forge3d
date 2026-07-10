use super::*;
use crate::core::resource_tracker::{tracked_create_buffer_init, TrackedBuffer};

pub(super) fn create_height_ao_pipeline_resources(
    device: &wgpu::Device,
) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout, TrackedBuffer)> {
    let height_ao_shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "heightfield_ao.wgsl",
        include_str!("../../../shaders/heightfield_ao.wgsl"),
    );
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
    let height_ao_compute_pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
        device,
        &wgpu::ComputePipelineDescriptor {
            label: Some("height_ao.compute_pipeline"),
            layout: Some(&height_ao_pipeline_layout),
            module: &height_ao_shader,
            entry_point: "main",
        },
    );
    let height_ao_uniform_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("height_ao.uniform_buffer"),
            contents: bytemuck::bytes_of(&HeightAoUniforms {
                params0: [6.0, 16.0, 200.0, 1.0],
                params1: [1.0, 1.0, 1.0, 0.0],
                params2: [1.0, 1.0, 1.0, 1.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    )?;

    Ok((
        height_ao_compute_pipeline,
        height_ao_bind_group_layout,
        height_ao_uniform_buffer,
    ))
}

pub(super) fn create_sun_vis_pipeline_resources(
    device: &wgpu::Device,
) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout, TrackedBuffer)> {
    let sun_vis_shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "heightfield_sun_vis.wgsl",
        include_str!("../../../shaders/heightfield_sun_vis.wgsl"),
    );
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
    let sun_vis_compute_pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
        device,
        &wgpu::ComputePipelineDescriptor {
            label: Some("sun_vis.compute_pipeline"),
            layout: Some(&sun_vis_pipeline_layout),
            module: &sun_vis_shader,
            entry_point: "main",
        },
    );
    let sun_vis_uniform_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("sun_vis.uniform_buffer"),
            contents: bytemuck::bytes_of(&SunVisUniforms {
                params0: [4.0, 24.0, 400.0, 1.0],
                params1: [1.0, 1.0, 1.0, 0.0],
                params2: [1.0, 1.0, 1.0, 1.0],
                params3: [0.0, 1.0, 0.0, 0.01],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    )?;

    Ok((
        sun_vis_compute_pipeline,
        sun_vis_bind_group_layout,
        sun_vis_uniform_buffer,
    ))
}

//! DOF pipeline creation and bind group layout.

use wgpu::{
    BindGroupLayout, ComputePipeline, ComputePipelineDescriptor, Device, PipelineLayoutDescriptor,
    TextureFormat,
};

/// Create the DOF bind group layout.
pub fn create_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("dof_bind_group_layout"),
        entries: &[
            // Uniforms
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
            // Color texture
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // Depth texture
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // Sampler
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // Output storage texture
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    })
}

/// Create all DOF compute pipelines.
///
/// Returns (gather_pipeline, separable_h_pipeline, separable_v_pipeline).
pub fn create_pipelines(device: &Device) -> (ComputePipeline, ComputePipeline, ComputePipeline) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("dof_compute_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/dof.wgsl").into()),
    });

    let bind_group_layout = create_bind_group_layout(device);

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("dof_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let gather_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("dof_gather_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "cs_dof",
    });

    let separable_h_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("dof_separable_h_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "cs_dof_separable_h",
    });

    let separable_v_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("dof_separable_v_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "cs_dof_separable_v",
    });

    (gather_pipeline, separable_h_pipeline, separable_v_pipeline)
}

use super::gpu_exec::assemble_with_entry;
use super::gpu_report::TwoProdVariant;
use crate::core::error::{RenderError, RenderResult};
use crate::core::shader_registry::{
    create_compute_pipeline_scoped, create_labeled_shader_module, create_render_pipeline_scoped,
};

const SHADER: &str = include_str!("../../shaders/dd_jitter.wgsl");

pub(super) struct JitterPipelines {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub compute: wgpu::ComputePipeline,
    pub render: wgpu::RenderPipeline,
}

pub(super) fn create_pipelines(
    device: &wgpu::Device,
    variant: TwoProdVariant,
) -> RenderResult<JitterPipelines> {
    let source = assemble_with_entry(variant, SHADER);
    let parsed = naga::front::wgsl::parse_str(&source)
        .map_err(|error| RenderError::render(format!("DD jitter WGSL parse failed: {error}")))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&parsed)
    .map_err(|error| RenderError::render(format!("DD jitter WGSL validation failed: {error}")))?;
    let module = create_labeled_shader_module(device, super::jitter::LABEL, &source);
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("dupla.dd_jitter.layout"),
        entries: &[
            storage_entry(
                0,
                true,
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
            ),
            storage_entry(
                1,
                true,
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
            ),
            storage_entry(2, false, wgpu::ShaderStages::COMPUTE),
            uniform_entry(3, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("dupla.dd_jitter.pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let compute = create_compute_pipeline_scoped(
        device,
        &wgpu::ComputePipelineDescriptor {
            label: Some("dupla.dd_jitter.compute"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "measure_jitter",
        },
    );
    let render = render_pipeline(
        device,
        &pipeline_layout,
        &module,
        "dupla.dd_jitter.render",
        "vs_dd",
    );
    let _raw = render_pipeline(
        device,
        &pipeline_layout,
        &module,
        "dupla.dd_jitter.raw_render",
        "vs_raw_f32",
    );
    if let Some(validation) = crate::core::degradation::degradations_snapshot()
        .into_iter()
        .find(|item| {
            item.kind == "validation_error"
                && matches!(
                    item.name.as_str(),
                    "dupla.dd_jitter.compute"
                        | "dupla.dd_jitter.render"
                        | "dupla.dd_jitter.raw_render"
                )
        })
    {
        return Err(RenderError::render(format!(
            "DD jitter pipeline validation failed for {}: {}",
            validation.name, validation.consequence
        )));
    }
    Ok(JitterPipelines {
        bind_group_layout,
        compute,
        render,
    })
}

fn storage_entry(
    binding: u32,
    read_only: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    label: &str,
    entry: &str,
) -> wgpu::RenderPipeline {
    create_render_pipeline_scoped(
        device,
        &wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: entry,
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        },
    )
}

use super::*;

pub(super) fn create_pipelines(
    device: &Device,
    layouts: &ConstructorLayouts,
) -> ConstructorPipelines {
    let trace_shader = shader(
        device,
        "p5.ssgi.trace",
        "../../../../shaders/ssgi/trace.wgsl",
    );
    let shade_shader = shader(
        device,
        "p5.ssgi.shade",
        "../../../../shaders/ssgi/shade.wgsl",
    );
    let temporal_shader = shader(
        device,
        "p5.ssgi.temporal",
        "../../../../shaders/ssgi/resolve_temporal.wgsl",
    );
    let upsample_shader = shader(
        device,
        "p5.ssgi.upsample",
        "../../../../shaders/filters/edge_aware_upsample.wgsl",
    );
    let composite_shader = shader(
        device,
        "p5.ssgi.composite",
        "../../../../shaders/ssgi/composite.wgsl",
    );

    ConstructorPipelines {
        trace_pipeline: compute_pipeline(
            device,
            "ssgi_trace_pipeline",
            "cs_trace",
            &trace_shader,
            &layouts.trace_bind_group_layout,
        ),
        shade_pipeline: compute_pipeline(
            device,
            "ssgi_shade_pipeline",
            "cs_shade",
            &shade_shader,
            &layouts.shade_bind_group_layout,
        ),
        temporal_pipeline: compute_pipeline(
            device,
            "ssgi_temporal_pipeline",
            "cs_resolve_temporal",
            &temporal_shader,
            &layouts.temporal_bind_group_layout,
        ),
        upsample_pipeline: compute_pipeline(
            device,
            "ssgi_upsample_pipeline",
            "cs_edge_aware_upsample",
            &upsample_shader,
            &layouts.upsample_bind_group_layout,
        ),
        composite_pipeline: compute_pipeline(
            device,
            "ssgi_composite_pipeline",
            "cs_ssgi_composite",
            &composite_shader,
            &layouts.composite_bind_group_layout,
        ),
    }
}

fn shader(device: &Device, label: &str, path: &str) -> ShaderModule {
    let source = match path {
        "../../../../shaders/ssgi/trace.wgsl" => {
            include_str!("../../../../shaders/ssgi/trace.wgsl")
        }
        "../../../../shaders/ssgi/shade.wgsl" => {
            include_str!("../../../../shaders/ssgi/shade.wgsl")
        }
        "../../../../shaders/ssgi/resolve_temporal.wgsl" => {
            include_str!("../../../../shaders/ssgi/resolve_temporal.wgsl")
        }
        "../../../../shaders/filters/edge_aware_upsample.wgsl" => {
            include_str!("../../../../shaders/filters/edge_aware_upsample.wgsl")
        }
        "../../../../shaders/ssgi/composite.wgsl" => {
            include_str!("../../../../shaders/ssgi/composite.wgsl")
        }
        _ => unreachable!(),
    };

    device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(source.into()),
    })
}

fn compute_pipeline(
    device: &Device,
    label: &str,
    entry_point: &str,
    module: &ShaderModule,
    bind_group_layout: &BindGroupLayout,
) -> ComputePipeline {
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(&format!("{label}.layout")),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        })),
        module,
        entry_point,
    })
}

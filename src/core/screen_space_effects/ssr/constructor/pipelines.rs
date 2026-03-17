use super::*;

pub(super) fn create_pipelines(
    device: &Device,
    layouts: &ConstructorLayouts,
) -> ConstructorPipelines {
    let trace_shader = shader(device, "p5.ssr.trace", "../../../../shaders/ssr/trace.wgsl");
    let shade_shader = shader(device, "p5.ssr.shade", "../../../../shaders/ssr/shade.wgsl");
    let fallback_shader = shader(
        device,
        "p5.ssr.fallback",
        "../../../../shaders/ssr/fallback_env.wgsl",
    );
    let temporal_shader = shader(
        device,
        "p5.ssr.temporal",
        "../../../../shaders/ssr/temporal.wgsl",
    );
    let composite_shader = shader(
        device,
        "p5.ssr.composite",
        "../../../../shaders/ssr/composite.wgsl",
    );

    ConstructorPipelines {
        trace_pipeline: compute_pipeline(
            device,
            "p5.ssr.trace",
            "cs_trace",
            &trace_shader,
            &layouts.trace_bind_group_layout,
        ),
        shade_pipeline: compute_pipeline(
            device,
            "p5.ssr.shade",
            "cs_shade",
            &shade_shader,
            &layouts.shade_bind_group_layout,
        ),
        fallback_pipeline: compute_pipeline(
            device,
            "p5.ssr.fallback",
            "cs_fallback",
            &fallback_shader,
            &layouts.fallback_bind_group_layout,
        ),
        temporal_pipeline: compute_pipeline(
            device,
            "p5.ssr.temporal",
            "cs_temporal",
            &temporal_shader,
            &layouts.temporal_bind_group_layout,
        ),
        composite_pipeline: compute_pipeline(
            device,
            "p5.ssr.composite",
            "cs_ssr_composite",
            &composite_shader,
            &layouts.composite_bind_group_layout,
        ),
    }
}

fn shader(device: &Device, label: &str, path: &str) -> ShaderModule {
    let source = match path {
        "../../../../shaders/ssr/trace.wgsl" => include_str!("../../../../shaders/ssr/trace.wgsl"),
        "../../../../shaders/ssr/shade.wgsl" => include_str!("../../../../shaders/ssr/shade.wgsl"),
        "../../../../shaders/ssr/fallback_env.wgsl" => {
            include_str!("../../../../shaders/ssr/fallback_env.wgsl")
        }
        "../../../../shaders/ssr/temporal.wgsl" => {
            include_str!("../../../../shaders/ssr/temporal.wgsl")
        }
        "../../../../shaders/ssr/composite.wgsl" => {
            include_str!("../../../../shaders/ssr/composite.wgsl")
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
        label: Some(&format!("{label}.pipeline")),
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(&format!("{label}.layout")),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        })),
        module,
        entry_point,
    })
}

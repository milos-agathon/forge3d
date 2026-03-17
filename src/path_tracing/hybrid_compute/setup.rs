use super::*;

fn load_hybrid_kernel_src() -> String {
    let sdf_primitives = include_str!("../../shaders/sdf_primitives.wgsl");
    let sdf_operations_raw = include_str!("../../shaders/sdf_operations.wgsl");
    let sdf_operations = sdf_operations_raw
        .lines()
        .filter(|l| !l.trim_start().starts_with("#include"))
        .collect::<Vec<_>>()
        .join("\n");

    let hybrid_traversal_raw = include_str!("../../shaders/hybrid_traversal.wgsl");
    let hybrid_traversal = hybrid_traversal_raw
        .lines()
        .filter(|l| !l.trim_start().starts_with("#include"))
        .collect::<Vec<_>>()
        .join("\n");

    let kernel_raw = include_str!("../../shaders/hybrid_kernel.wgsl");
    let kernel = kernel_raw
        .lines()
        .filter(|l| !l.trim_start().starts_with("#include"))
        .collect::<Vec<_>>()
        .join("\n");

    [sdf_primitives, &sdf_operations, &hybrid_traversal, &kernel].join("\n")
}

impl HybridPathTracer {
    pub fn new() -> Result<Self, RenderError> {
        let device = &ctx().device;
        let shader_src = load_hybrid_kernel_src();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hybrid-pt-kernel"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let layouts = HybridBindGroupLayouts {
            uniforms: Self::create_uniforms_layout(device),
            scene: Self::create_scene_layout(device),
            accum: Self::create_accum_layout(device),
            output: Self::create_output_layout(device),
            lighting: Self::create_lighting_layout(device),
        };

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hybrid-pt-pipeline-layout"),
            bind_group_layouts: &[
                &layouts.uniforms,
                &layouts.scene,
                &layouts.accum,
                &layouts.output,
                &layouts.lighting,
            ],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hybrid-pt-compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        Ok(Self {
            shader,
            layouts,
            pipeline_layout,
            pipeline,
        })
    }
}

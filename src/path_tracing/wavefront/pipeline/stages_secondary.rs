use super::*;

impl WavefrontPipelines {
    pub(super) fn create_shadow_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
        scene_layout: &BindGroupLayout,
        accum_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
        let queue_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow-queue-layout"),
            entries: &[aov_entry(0, false), aov_entry(1, false)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow-pipeline-layout"),
            bind_group_layouts: &[uniforms_layout, scene_layout, &queue_layout, accum_layout],
            push_constant_ranges: &[],
        });
        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("shadow-pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: "main",
            }),
        )
    }

    pub(super) fn create_scatter_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
        scene_layout: &BindGroupLayout,
        accum_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
        let queue_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter-queue-layout"),
            entries: &[
                aov_entry(0, false),
                aov_entry(1, false),
                aov_entry(2, false),
                aov_entry(3, false),
                aov_entry(4, false),
                aov_entry(5, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scatter-pipeline-layout"),
            bind_group_layouts: &[uniforms_layout, scene_layout, &queue_layout, accum_layout],
            push_constant_ranges: &[],
        });
        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("scatter-pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: "main",
            }),
        )
    }

    pub(super) fn create_compact_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
        scene_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
        let queue_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compact-queue-layout"),
            entries: &[
                aov_entry(0, false),
                aov_entry(1, false),
                aov_entry(2, false),
                aov_entry(3, false),
                aov_entry(4, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compact-pipeline-layout"),
            bind_group_layouts: &[uniforms_layout, scene_layout, &queue_layout],
            push_constant_ranges: &[],
        });
        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("compact-pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: "main",
            }),
        )
    }
}

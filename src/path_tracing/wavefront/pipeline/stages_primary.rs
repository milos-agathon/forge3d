use super::*;

impl WavefrontPipelines {
    pub(super) fn create_raygen_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
        scene_layout: &BindGroupLayout,
        accum_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
        let queue_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raygen-queue-layout"),
            entries: &[aov_entry(0, false), aov_entry(1, false)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raygen-pipeline-layout"),
            bind_group_layouts: &[uniforms_layout, scene_layout, &queue_layout, accum_layout],
            push_constant_ranges: &[],
        });
        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("raygen-pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: "main",
            }),
        )
    }

    pub(super) fn create_intersect_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
        scene_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
        let queue_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("intersect-queue-layout"),
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
            label: Some("intersect-pipeline-layout"),
            bind_group_layouts: &[uniforms_layout, scene_layout, &queue_layout],
            push_constant_ranges: &[],
        });
        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("intersect-pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: "main",
            }),
        )
    }

    pub(super) fn create_shade_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
        scene_layout: &BindGroupLayout,
        accum_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
        let queue_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shade-queue-layout"),
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
            label: Some("shade-pipeline-layout"),
            bind_group_layouts: &[uniforms_layout, scene_layout, &queue_layout, accum_layout],
            push_constant_ranges: &[],
        });
        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("shade-pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: "main",
            }),
        )
    }
}

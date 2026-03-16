use super::*;

impl WavefrontPipelines {
    pub(super) fn create_restir_temporal_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("restir-temporal-layout"),
            entries: &[aov_entry(0, true), aov_entry(1, true), aov_entry(2, false)],
        })
    }

    pub(super) fn create_restir_spatial_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("restir-spatial-layout"),
            entries: &[aov_entry(0, true), aov_entry(1, false)],
        })
    }

    pub(super) fn create_restir_temporal_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
        scene_layout: &BindGroupLayout,
        restir_temporal_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("restir-temporal-pipeline-layout"),
            bind_group_layouts: &[uniforms_layout, scene_layout, restir_temporal_layout],
            push_constant_ranges: &[],
        });
        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("restir-temporal-pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: "main",
            }),
        )
    }

    pub(super) fn create_restir_scene_spatial_bind_group_layout(
        device: &Device,
    ) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("restir-scene-spatial-layout"),
            entries: &[
                aov_entry(4, true),
                aov_entry(5, true),
                aov_entry(10, true),
                aov_entry(11, true),
            ],
        })
    }

    pub(super) fn create_restir_spatial_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
        scene_spatial_layout: &BindGroupLayout,
        restir_spatial_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("restir-spatial-pipeline-layout"),
            bind_group_layouts: &[uniforms_layout, scene_spatial_layout, restir_spatial_layout],
            push_constant_ranges: &[],
        });
        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("restir-spatial-pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: "main",
            }),
        )
    }

    pub(super) fn create_restir_init_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
        scene_layout: &BindGroupLayout,
        restir_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("restir-init-pipeline-layout"),
            bind_group_layouts: &[uniforms_layout, scene_layout, restir_layout],
            push_constant_ranges: &[],
        });
        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("restir-init-pipeline"),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: "main",
            }),
        )
    }

    pub(super) fn create_restir_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("restir-layout"),
            entries: &[
                aov_entry(0, false),
                aov_entry(1, true),
                aov_entry(2, true),
                aov_entry(3, true),
            ],
        })
    }
}

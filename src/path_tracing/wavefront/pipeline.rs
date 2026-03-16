use wgpu::{BindGroupLayout, ComputePipeline, Device, ShaderModule};

mod layouts;
mod restir;
mod scene_layout;
mod stages_primary;
mod stages_secondary;

/// All compute pipelines for wavefront path tracing stages
pub struct WavefrontPipelines {
    pub raygen: ComputePipeline,
    pub intersect: ComputePipeline,
    pub shade: ComputePipeline,
    pub scatter: ComputePipeline,
    pub compact: ComputePipeline,
    pub shadow: ComputePipeline,
    pub restir_init: ComputePipeline,
    pub restir_temporal: ComputePipeline,
    pub restir_spatial: ComputePipeline,
    pub ao_compute: ComputePipeline,
    pub uniforms_bind_group_layout: BindGroupLayout,
    pub scene_bind_group_layout: BindGroupLayout,
    pub accum_bind_group_layout: BindGroupLayout,
    pub restir_bind_group_layout: BindGroupLayout,
    pub restir_temporal_bind_group_layout: BindGroupLayout,
    pub restir_spatial_bind_group_layout: BindGroupLayout,
    pub restir_scene_spatial_bind_group_layout: BindGroupLayout,
    pub ao_bind_group_layout: BindGroupLayout,
}

impl WavefrontPipelines {
    pub fn new(device: &Device) -> Result<Self, Box<dyn std::error::Error>> {
        let raygen_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pt-raygen-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/pt_raygen.wgsl").into()),
        });
        let intersect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pt-intersect-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/pt_intersect.wgsl").into(),
            ),
        });
        let shade_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pt-shade-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/pt_shade.wgsl").into()),
        });
        let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pt-scatter-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/pt_scatter.wgsl").into()),
        });
        let compact_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pt-compact-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/pt_compact.wgsl").into()),
        });
        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pt-shadow-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/pt_shadow.wgsl").into()),
        });
        let restir_init_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pt-restir-init-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/pt_restir_init.wgsl").into(),
            ),
        });
        let restir_temporal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pt-restir-temporal-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/pt_restir_temporal.wgsl").into(),
            ),
        });
        let restir_spatial_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pt-restir-spatial-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/pt_restir_spatial.wgsl").into(),
            ),
        });
        let ao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ao-from-aovs-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/ao_from_aovs.wgsl").into(),
            ),
        });

        let uniforms_bind_group_layout = Self::create_uniforms_bind_group_layout(device);
        let scene_bind_group_layout = Self::create_scene_bind_group_layout(device);
        let accum_bind_group_layout = Self::create_accum_bind_group_layout(device);
        let restir_bind_group_layout = Self::create_restir_bind_group_layout(device);
        let restir_temporal_bind_group_layout =
            Self::create_restir_temporal_bind_group_layout(device);
        let restir_spatial_bind_group_layout =
            Self::create_restir_spatial_bind_group_layout(device);
        let restir_scene_spatial_bind_group_layout =
            Self::create_restir_scene_spatial_bind_group_layout(device);

        let ao_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ao-bind-group-layout"),
                entries: &[
                    aov_entry(0, true),
                    aov_entry(1, true),
                    aov_entry(2, false),
                    uniform_entry(3),
                ],
            });

        let raygen = Self::create_raygen_pipeline(
            device,
            &raygen_shader,
            &uniforms_bind_group_layout,
            &scene_bind_group_layout,
            &accum_bind_group_layout,
        )?;
        let intersect = Self::create_intersect_pipeline(
            device,
            &intersect_shader,
            &uniforms_bind_group_layout,
            &scene_bind_group_layout,
        )?;
        let shade = Self::create_shade_pipeline(
            device,
            &shade_shader,
            &uniforms_bind_group_layout,
            &scene_bind_group_layout,
            &accum_bind_group_layout,
        )?;
        let scatter = Self::create_scatter_pipeline(
            device,
            &scatter_shader,
            &uniforms_bind_group_layout,
            &scene_bind_group_layout,
            &accum_bind_group_layout,
        )?;
        let compact = Self::create_compact_pipeline(
            device,
            &compact_shader,
            &uniforms_bind_group_layout,
            &scene_bind_group_layout,
        )?;
        let shadow = Self::create_shadow_pipeline(
            device,
            &shadow_shader,
            &uniforms_bind_group_layout,
            &scene_bind_group_layout,
            &accum_bind_group_layout,
        )?;
        let restir_init = Self::create_restir_init_pipeline(
            device,
            &restir_init_shader,
            &uniforms_bind_group_layout,
            &scene_bind_group_layout,
            &restir_bind_group_layout,
        )?;
        let restir_temporal = Self::create_restir_temporal_pipeline(
            device,
            &restir_temporal_shader,
            &uniforms_bind_group_layout,
            &scene_bind_group_layout,
            &restir_temporal_bind_group_layout,
        )?;
        let restir_spatial = Self::create_restir_spatial_pipeline(
            device,
            &restir_spatial_shader,
            &uniforms_bind_group_layout,
            &restir_scene_spatial_bind_group_layout,
            &restir_spatial_bind_group_layout,
        )?;

        let ao_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ao-pipeline-layout"),
            bind_group_layouts: &[&ao_bind_group_layout],
            push_constant_ranges: &[],
        });
        let ao_compute = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ao-from-aovs-pipeline"),
            layout: Some(&ao_pipeline_layout),
            module: &ao_shader,
            entry_point: "main",
        });

        Ok(Self {
            raygen,
            intersect,
            shade,
            scatter,
            compact,
            shadow,
            restir_init,
            restir_temporal,
            restir_spatial,
            ao_compute,
            uniforms_bind_group_layout,
            scene_bind_group_layout,
            accum_bind_group_layout,
            restir_bind_group_layout,
            restir_temporal_bind_group_layout,
            restir_spatial_bind_group_layout,
            restir_scene_spatial_bind_group_layout,
            ao_bind_group_layout,
        })
    }
}

fn aov_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

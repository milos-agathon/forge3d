// src/path_tracing/wavefront/pipeline.rs
// WGSL pipeline orchestration for wavefront path tracing
// Creates and manages compute pipelines for each stage

use wgpu::{BindGroupLayout, ComputePipeline, Device, ShaderModule};

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
    // AO from AOVs compute
    pub ao_compute: ComputePipeline,

    // Shared bind group layouts
    pub uniforms_bind_group_layout: BindGroupLayout,
    pub scene_bind_group_layout: BindGroupLayout,
    pub accum_bind_group_layout: BindGroupLayout,
    pub restir_bind_group_layout: BindGroupLayout,
    pub restir_temporal_bind_group_layout: BindGroupLayout,
    pub restir_spatial_bind_group_layout: BindGroupLayout,
    // Minimal scene layout for ReSTIR spatial (lights + G-buffer only)
    pub restir_scene_spatial_bind_group_layout: BindGroupLayout,
    // AO bind group layout (Group 0)
    pub ao_bind_group_layout: BindGroupLayout,
}

impl WavefrontPipelines {
    /// Create all wavefront pipelines
    pub fn new(device: &Device) -> Result<Self, Box<dyn std::error::Error>> {
        // Load WGSL shaders
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/pt_restir_init.wgsl").into()),
        });

        let restir_temporal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pt-restir-temporal-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/pt_restir_temporal.wgsl").into()),
        });

        let restir_spatial_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pt-restir-spatial-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/pt_restir_spatial.wgsl").into()),
        });

        let ao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ao-from-aovs-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/ao_from_aovs.wgsl").into()),
        });

        // Create shared bind group layouts
        let uniforms_bind_group_layout = Self::create_uniforms_bind_group_layout(device);
        let scene_bind_group_layout = Self::create_scene_bind_group_layout(device);
        let accum_bind_group_layout = Self::create_accum_bind_group_layout(device);
        let restir_bind_group_layout = Self::create_restir_bind_group_layout(device);
        let restir_temporal_bind_group_layout = Self::create_restir_temporal_bind_group_layout(device);
        let restir_spatial_bind_group_layout = Self::create_restir_spatial_bind_group_layout(device);
        let restir_scene_spatial_bind_group_layout =
            Self::create_restir_scene_spatial_bind_group_layout(device);

        // AO bind group layout (Group 0)
        let ao_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ao-bind-group-layout"),
            entries: &[
                // 0: AOV depth buffer (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: AOV normal buffer (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: AO output buffer (read_write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: AO params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipelines for each stage
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

        // AO pipeline
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

    /// Create ReSTIR temporal bind group layout (Group 2 for temporal reuse)
    fn create_restir_temporal_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("restir-temporal-layout"),
            entries: &[
                // 0: prev reservoirs (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: curr reservoirs (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: out reservoirs (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create ReSTIR spatial bind group layout (Group 2 for spatial reuse)
    fn create_restir_spatial_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("restir-spatial-layout"),
            entries: &[
                // 0: in reservoirs (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: out reservoirs (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create ReSTIR temporal reuse pipeline
    fn create_restir_temporal_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
        scene_layout: &BindGroupLayout,
        restir_temporal_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("restir-temporal-pipeline-layout"),
            // 0=uniforms, 1=scene placeholder, 2=temporal buffers
            bind_group_layouts: &[uniforms_layout, scene_layout, restir_temporal_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("restir-temporal-pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "main",
        });
        Ok(pipeline)
    }

    /// Create minimal scene layout for ReSTIR spatial (Group 1 for spatial pass)
    fn create_restir_scene_spatial_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("restir-scene-spatial-layout"),
            entries: &[
                // 4: area lights (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: directional lights (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 10: G-buffer normal/roughness (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 11: G-buffer position (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create ReSTIR spatial reuse pipeline (uses scene bind group at 1 and spatial at 2)
    fn create_restir_spatial_pipeline(
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

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("restir-spatial-pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "main",
        });
        Ok(pipeline)
    }

    /// Create uniforms bind group layout (Group 0)
    fn create_uniforms_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniforms-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    /// Create scene bind group layout (Group 1)
    fn create_scene_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene-layout"),
            entries: &[
                // 0: spheres/materials buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: mesh vertices
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: mesh indices
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: mesh BVH nodes
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: area lights buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: directional lights buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: object importance weights (per object/material)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7: ReSTIR reservoirs (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 8: ReSTIR diagnostics flags (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 9: ReSTIR debug AOV (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 10: ReSTIR G-buffer (normal.xyz, roughness) (read_write for shading to write)
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 11: ReSTIR G-buffer position (position.xyz,1) (read_write for shading to write)
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 12: ReSTIR settings (uniforms/toggles)
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 13: ReSTIR material-id buffer (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 14: Instances buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 15: BLAS descriptor table (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 16: AOV albedo buffer (RW, vec4<f32> per pixel)
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 17: AOV depth buffer (RW, vec4<f32> per pixel; x = linear depth)
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 18: AOV normal buffer (RW, vec4<f32> per pixel)
                wgpu::BindGroupLayoutEntry {
                    binding: 18,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 19: Medium parameters (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 19,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 20: Hair segments buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 20,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create accumulation bind group layout (Group 3)
    fn create_accum_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("accum-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    /// Create ray generation pipeline
    fn create_raygen_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
        scene_layout: &BindGroupLayout,
        accum_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
        // Create queue bind group layout for raygen
        let queue_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raygen-queue-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raygen-pipeline-layout"),
            bind_group_layouts: &[uniforms_layout, scene_layout, &queue_layout, accum_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raygen-pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "main",
        });

        Ok(pipeline)
    }

    /// Create ReSTIR init pipeline
    fn create_restir_init_pipeline(
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

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("restir-init-pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "main",
        });
        Ok(pipeline)
    }

    /// Create ReSTIR bind group layout (Group 2 for restir init stage)
    fn create_restir_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("restir-layout"),
            entries: &[
                // 0: reservoirs (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: light samples (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: alias entries (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: per-light normalized probabilities (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create shadow pipeline
    fn create_shadow_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
        scene_layout: &BindGroupLayout,
        accum_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
        let queue_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow-queue-layout"),
            entries: &[
                // Shadow queue header
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Shadow queue
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow-pipeline-layout"),
            bind_group_layouts: &[uniforms_layout, scene_layout, &queue_layout, accum_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shadow-pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "main",
        });

        Ok(pipeline)
    }

    /// Create intersection pipeline
    fn create_intersect_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
        scene_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
        // Create queue bind group layout for intersect
        let queue_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("intersect-queue-layout"),
            entries: &[
                // Ray queue header
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Ray queue
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Hit queue header
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Hit queue
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Miss queue header
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Miss queue
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("intersect-pipeline-layout"),
            bind_group_layouts: &[uniforms_layout, scene_layout, &queue_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("intersect-pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "main",
        });

        Ok(pipeline)
    }

    /// Create shading pipeline
    fn create_shade_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
        scene_layout: &BindGroupLayout,
        accum_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
        // Create queue bind group layout for shade
        let queue_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shade-queue-layout"),
            entries: &[
                // Hit queue header
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Hit queue
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Scatter queue header
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Scatter queue
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Shadow queue header
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Shadow queue
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shade-pipeline-layout"),
            bind_group_layouts: &[uniforms_layout, scene_layout, &queue_layout, accum_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shade-pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "main",
        });

        Ok(pipeline)
    }

    /// Create scatter pipeline
    fn create_scatter_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
        scene_layout: &BindGroupLayout,
        accum_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
        // Create queue bind group layout for scatter
        let queue_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter-queue-layout"),
            entries: &[
                // Scatter queue header
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Scatter queue
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Ray queue header
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Ray queue
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Miss queue header
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Miss queue
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scatter-pipeline-layout"),
            // Keep group indices consistent with other stages: 0=uniforms, 1=scene, 2=queues, 3=accum
            bind_group_layouts: &[uniforms_layout, scene_layout, &queue_layout, accum_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scatter-pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "main",
        });

        Ok(pipeline)
    }

    /// Create compaction pipeline
    fn create_compact_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
        scene_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
        // Create queue bind group layout for compact
        let queue_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compact-queue-layout"),
            entries: &[
                // Ray queue header
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Ray queue
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Ray queue compacted
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Ray flags
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Prefix sums
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compact-pipeline-layout"),
            // Align to WGSL groups: 0=uniforms, 1=scene (placeholder), 2=queues
            bind_group_layouts: &[uniforms_layout, scene_layout, &queue_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compact-pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "compact_rays_simple", // Use simple compaction for MVP
        });

        Ok(pipeline)
    }
}

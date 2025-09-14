// src/path_tracing/wavefront/pipeline.rs
// WGSL pipeline orchestration for wavefront path tracing
// Creates and manages compute pipelines for each stage

use wgpu::{Device, ComputePipeline, BindGroupLayout, ShaderModule};

/// All compute pipelines for wavefront path tracing stages
pub struct WavefrontPipelines {
    pub raygen: ComputePipeline,
    pub intersect: ComputePipeline,
    pub shade: ComputePipeline,
    pub scatter: ComputePipeline,
    pub compact: ComputePipeline,
    
    // Shared bind group layouts
    pub uniforms_bind_group_layout: BindGroupLayout,
    pub scene_bind_group_layout: BindGroupLayout,
    pub accum_bind_group_layout: BindGroupLayout,
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/pt_intersect.wgsl").into()),
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
        
        // Create shared bind group layouts
        let uniforms_bind_group_layout = Self::create_uniforms_bind_group_layout(device);
        let scene_bind_group_layout = Self::create_scene_bind_group_layout(device);
        let accum_bind_group_layout = Self::create_accum_bind_group_layout(device);
        
        // Create pipelines for each stage
        let raygen = Self::create_raygen_pipeline(
            device, 
            &raygen_shader, 
            &uniforms_bind_group_layout,
            &scene_bind_group_layout,
            &accum_bind_group_layout
        )?;
        
        let intersect = Self::create_intersect_pipeline(
            device, 
            &intersect_shader, 
            &uniforms_bind_group_layout,
            &scene_bind_group_layout
        )?;
        
        let shade = Self::create_shade_pipeline(
            device, 
            &shade_shader, 
            &uniforms_bind_group_layout,
            &scene_bind_group_layout,
            &accum_bind_group_layout
        )?;
        
        let scatter = Self::create_scatter_pipeline(
            device, 
            &scatter_shader, 
            &uniforms_bind_group_layout,
            &accum_bind_group_layout
        )?;
        
        let compact = Self::create_compact_pipeline(device, &compact_shader)?;
        
        Ok(Self {
            raygen,
            intersect,
            shade,
            scatter,
            compact,
            uniforms_bind_group_layout,
            scene_bind_group_layout,
            accum_bind_group_layout,
        })
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
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
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
            compilation_options: Default::default(),
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
            compilation_options: Default::default(),
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
            compilation_options: Default::default(),
        });
        
        Ok(pipeline)
    }
    
    /// Create scatter pipeline
    fn create_scatter_pipeline(
        device: &Device,
        shader: &ShaderModule,
        uniforms_layout: &BindGroupLayout,
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
            bind_group_layouts: &[uniforms_layout, &queue_layout, accum_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scatter-pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        
        Ok(pipeline)
    }
    
    /// Create compaction pipeline
    fn create_compact_pipeline(
        device: &Device,
        shader: &ShaderModule,
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
            bind_group_layouts: &[&queue_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compact-pipeline"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: "compact_rays_simple",  // Use simple compaction for MVP
            compilation_options: Default::default(),
        });
        
        Ok(pipeline)
    }
}
use super::*;

impl SsrRenderer {
    pub fn new(device: &Device, width: u32, height: u32) -> RenderResult<Self> {
        let mut settings = SsrSettings::default();
        settings.inv_resolution = [1.0 / width as f32, 1.0 / height as f32];

        let settings_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ssr_settings"),
            size: std::mem::size_of::<SsrSettings>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ssr_camera"),
            size: std::mem::size_of::<CameraParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let counters_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("p5.ssr.counters"),
            size: std::mem::size_of::<[u32; 5]>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let counters_readback = device.create_buffer(&BufferDescriptor {
            label: Some("p5.ssr.counters.readback"),
            size: std::mem::size_of::<[u32; 5]>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let trace_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssr.trace"),
            source: ShaderSource::Wgsl(include_str!("../../../shaders/ssr/trace.wgsl").into()),
        });
        let shade_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssr.shade"),
            source: ShaderSource::Wgsl(include_str!("../../../shaders/ssr/shade.wgsl").into()),
        });
        let fallback_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssr.fallback"),
            source: ShaderSource::Wgsl(
                include_str!("../../../shaders/ssr/fallback_env.wgsl").into(),
            ),
        });
        let temporal_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssr.temporal"),
            source: ShaderSource::Wgsl(include_str!("../../../shaders/ssr/temporal.wgsl").into()),
        });
        let composite_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssr.composite"),
            source: ShaderSource::Wgsl(include_str!("../../../shaders/ssr/composite.wgsl").into()),
        });

        let trace_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("p5.ssr.trace.bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shade_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("p5.ssr.shade.bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 8,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 9,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let fallback_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("p5.ssr.fallback.bgl"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 5,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 6,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba16Float,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 7,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 8,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 9,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let temporal_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("p5.ssr.temporal.bgl"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba16Float,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let composite_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("p5.ssr.composite.bgl"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba8Unorm,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let trace_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("p5.ssr.trace.pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("p5.ssr.trace.layout"),
                bind_group_layouts: &[&trace_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &trace_shader,
            entry_point: "cs_trace",
        });
        let shade_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("p5.ssr.shade.pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("p5.ssr.shade.layout"),
                bind_group_layouts: &[&shade_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shade_shader,
            entry_point: "cs_shade",
        });
        let fallback_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("p5.ssr.fallback.pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("p5.ssr.fallback.layout"),
                bind_group_layouts: &[&fallback_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &fallback_shader,
            entry_point: "cs_fallback",
        });
        let temporal_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("p5.ssr.temporal.pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("p5.ssr.temporal.layout"),
                bind_group_layouts: &[&temporal_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &temporal_shader,
            entry_point: "cs_temporal",
        });

        let composite_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("p5.ssr.composite.pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("p5.ssr.composite.layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &composite_shader,
            entry_point: "cs_ssr_composite",
        });

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct SsrCompositeParamsStd140 {
            boost: f32,
            exposure: f32,
            gamma: f32,
            weight_floor: f32,
            tone_white: f32,
            tone_bias: f32,
            reinhard_k: f32,
            _pad0: f32,
        }
        let composite_params_data = SsrCompositeParamsStd140 {
            boost: 1.6,
            exposure: 1.1,
            gamma: 1.0,
            weight_floor: 0.2,
            tone_white: 1.0,
            tone_bias: 0.0,
            reinhard_k: 1.0,
            _pad0: 0.0,
        };
        let composite_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("p5.ssr.composite.params"),
            contents: bytemuck::bytes_of(&composite_params_data),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct SsrTemporalParamsStd140 {
            temporal_alpha: f32,
            pad: [f32; 3],
        }
        let temporal_params_data = SsrTemporalParamsStd140 {
            temporal_alpha: 0.85,
            pad: [0.0; 3],
        };
        let temporal_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("p5.ssr.temporal.params"),
            contents: bytemuck::bytes_of(&temporal_params_data),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let ssr_spec_texture = device.create_texture(&TextureDescriptor {
            label: Some("p5.ssr.spec"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssr_spec_view = ssr_spec_texture.create_view(&TextureViewDescriptor::default());

        let ssr_final_texture = device.create_texture(&TextureDescriptor {
            label: Some("p5.ssr.final"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssr_final_view = ssr_final_texture.create_view(&TextureViewDescriptor::default());

        let ssr_history_texture = device.create_texture(&TextureDescriptor {
            label: Some("p5.ssr.history"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ssr_history_view = ssr_history_texture.create_view(&TextureViewDescriptor::default());

        let ssr_filtered_texture = device.create_texture(&TextureDescriptor {
            label: Some("p5.ssr.filtered"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssr_filtered_view = ssr_filtered_texture.create_view(&TextureViewDescriptor::default());

        let ssr_hit_texture = device.create_texture(&TextureDescriptor {
            label: Some("p5.ssr.hit"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssr_hit_view = ssr_hit_texture.create_view(&TextureViewDescriptor::default());

        let ssr_composited_texture = device.create_texture(&TextureDescriptor {
            label: Some("p5.ssr.composited"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssr_composited_view =
            ssr_composited_texture.create_view(&TextureViewDescriptor::default());

        let env_texture = device.create_texture(&TextureDescriptor {
            label: Some("p5.ssr.env.placeholder"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let env_view = env_texture.create_view(&TextureViewDescriptor {
            label: Some("p5.ssr.env.view"),
            format: None,
            dimension: Some(TextureViewDimension::Cube),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        let env_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("p5.ssr.env.sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });
        let linear_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("p5.ssr.linear"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });

        Ok(Self {
            settings,
            settings_buffer,
            camera_buffer,
            trace_pipeline,
            trace_bind_group_layout,
            shade_pipeline,
            shade_bind_group_layout,
            fallback_pipeline,
            fallback_bind_group_layout,
            temporal_pipeline,
            temporal_bind_group_layout,
            composite_pipeline,
            composite_bind_group_layout,
            composite_params,
            ssr_spec_texture,
            ssr_spec_view,
            ssr_final_texture,
            ssr_final_view,
            ssr_history_texture,
            ssr_history_view,
            ssr_filtered_texture,
            ssr_filtered_view,
            ssr_hit_texture,
            ssr_hit_view,
            ssr_composited_texture,
            ssr_composited_view,
            env_texture,
            env_view,
            env_sampler,
            linear_sampler,
            width,
            height,
            counters_buffer,
            counters_readback,
            temporal_params,
            last_trace_ms: 0.0,
            last_shade_ms: 0.0,
            last_fallback_ms: 0.0,
            stats_readback_pending: false,
            scene_color_override: None,
        })
    }
}

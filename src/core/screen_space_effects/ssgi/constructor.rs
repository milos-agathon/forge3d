use super::*;

impl SsgiRenderer {
    pub fn new(
        device: &Device,
        width: u32,
        height: u32,
        material_format: TextureFormat,
    ) -> RenderResult<Self> {
        let mut settings = SsgiSettings::default();
        settings.inv_resolution = [1.0 / width as f32, 1.0 / height as f32];

        // Uniform buffers
        let settings_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ssgi_settings"),
            size: std::mem::size_of::<SsgiSettings>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ssgi_camera"),
            size: std::mem::size_of::<CameraParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shaders (split per stage per P5.2)
        let trace_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssgi.trace"),
            source: ShaderSource::Wgsl(include_str!("../../../shaders/ssgi/trace.wgsl").into()),
        });
        let shade_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssgi.shade"),
            source: ShaderSource::Wgsl(include_str!("../../../shaders/ssgi/shade.wgsl").into()),
        });
        let temporal_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssgi.temporal"),
            source: ShaderSource::Wgsl(
                include_str!("../../../shaders/ssgi/resolve_temporal.wgsl").into(),
            ),
        });
        let upsample_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssgi.upsample"),
            source: ShaderSource::Wgsl(
                include_str!("../../../shaders/filters/edge_aware_upsample.wgsl").into(),
            ),
        });

        // Trace pass: depth, normal, HZB, outHit, settings, camera
        let trace_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssgi_trace_bgl"),
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
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
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
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let trace_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssgi_trace_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssgi_trace_pl"),
                bind_group_layouts: &[&trace_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &trace_shader,
            entry_point: "cs_trace",
        });

        // Shade pass: prevColor, sampler, env cube/sampler, hit, outRadiance, settings, camera, normalFull, material
        let shade_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssgi_shade_bgl"),
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
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
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
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 9,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let shade_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssgi_shade_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssgi_shade_pl"),
                bind_group_layouts: &[&shade_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shade_shader,
            entry_point: "cs_shade",
        });

        // Temporal resolve layout: current, history, filtered, settings
        let temporal_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("ssgi_temporal_bind_group_layout"),
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
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
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
                ],
            });
        let temporal_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssgi_temporal_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssgi_temporal_pl"),
                bind_group_layouts: &[&temporal_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &temporal_shader,
            entry_point: "cs_resolve_temporal",
        });

        // Edge-aware upsample layout/pipeline
        let upsample_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("ssgi_upsample_bind_group_layout"),
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
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba16Float,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
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
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 5,
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
        let upsample_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssgi_upsample_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssgi_upsample_pl"),
                bind_group_layouts: &[&upsample_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &upsample_shader,
            entry_point: "cs_edge_aware_upsample",
        });

        // Composite pass: material + SSGI
        let composite_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.ssgi.composite"),
            source: ShaderSource::Wgsl(include_str!("../../../shaders/ssgi/composite.wgsl").into()),
        });
        let composite_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("ssgi_composite_bind_group_layout"),
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
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba8Unorm,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
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
        let composite_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("ssgi_composite_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssgi_composite_pl"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &composite_shader,
            entry_point: "cs_ssgi_composite",
        });

        // Output and temporal textures (half-res by default disabled)
        let ssgi_hit = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_hit"),
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
        let ssgi_hit_view = ssgi_hit.create_view(&TextureViewDescriptor::default());
        let ssgi_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_texture"),
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
        let ssgi_view = ssgi_texture.create_view(&TextureViewDescriptor::default());

        let ssgi_history = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_history"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ssgi_history_view = ssgi_history.create_view(&TextureViewDescriptor::default());

        let ssgi_filtered = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_filtered"),
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
        let ssgi_filtered_view = ssgi_filtered.create_view(&TextureViewDescriptor::default());

        // Full-resolution upscaled target
        let ssgi_upscaled = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_upscaled"),
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
                | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ssgi_upscaled_view = ssgi_upscaled.create_view(&TextureViewDescriptor::default());

        // Composited material (material + SSGI)
        let ssgi_composited = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_composited"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssgi_composited_view = ssgi_composited.create_view(&TextureViewDescriptor::default());

        // Composite uniform (x = intensity multiplier)
        let comp_params: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
        let composite_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssgi.composite.uniform"),
            contents: bytemuck::cast_slice(&comp_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Previous-frame color ping-pong (full resolution)
        let history_usage =
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::COPY_SRC;
        let scene_history = [
            device.create_texture(&TextureDescriptor {
                label: Some("ssgi_scene_history_a"),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: material_format,
                usage: history_usage,
                view_formats: &[],
            }),
            device.create_texture(&TextureDescriptor {
                label: Some("ssgi_scene_history_b"),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: material_format,
                usage: history_usage,
                view_formats: &[],
            }),
        ];
        let scene_history_views = [
            scene_history[0].create_view(&TextureViewDescriptor::default()),
            scene_history[1].create_view(&TextureViewDescriptor::default()),
        ];

        // Fallback env cube texture (1x1x6 RGBA8).
        let env_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_env_cube"),
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
            label: Some("ssgi_env_cube_view"),
            format: Some(TextureFormat::Rgba8Unorm),
            dimension: Some(TextureViewDimension::Cube),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        let env_sampler = device.create_sampler(&SamplerDescriptor::default());
        let linear_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("ssgi.linear.sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            ..Default::default()
        });

        Ok(Self {
            settings,
            settings_buffer,
            camera_buffer,
            frame_index: 0,
            trace_pipeline,
            trace_bind_group_layout,
            shade_pipeline,
            shade_bind_group_layout,
            temporal_pipeline,
            temporal_bind_group_layout,
            upsample_pipeline,
            upsample_bind_group_layout,
            composite_pipeline,
            composite_bind_group_layout,
            ssgi_hit,
            ssgi_hit_view,
            ssgi_texture,
            ssgi_view,
            ssgi_history,
            ssgi_history_view,
            ssgi_filtered,
            ssgi_filtered_view,
            ssgi_upscaled,
            ssgi_upscaled_view,
            ssgi_composited,
            ssgi_composited_view,
            composite_uniform,
            scene_history,
            scene_history_views,
            scene_history_index: 0,
            scene_history_ready: false,
            linear_sampler,
            env_texture,
            env_view,
            env_sampler,
            width,
            height,
            half_res: false,
            last_trace_ms: 0.0,
            last_shade_ms: 0.0,
            last_temporal_ms: 0.0,
            last_upsample_ms: 0.0,
        })
    }
}

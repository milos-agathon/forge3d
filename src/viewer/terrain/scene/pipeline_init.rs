use super::*;

impl ViewerTerrainScene {
    pub fn init_wboit_pipeline(&mut self) {
        if self.wboit_compose_pipeline.is_some() {
            return; // Already initialized
        }

        // Create sampler if not already created
        if self.wboit_sampler.is_none() {
            self.wboit_sampler = Some(self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("terrain_viewer.wboit.sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            }));
        }

        // Create compose bind group layout if not already created
        if self.wboit_compose_bind_group_layout.is_none() {
            self.wboit_compose_bind_group_layout = Some(self.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("terrain_viewer.wboit.compose_bind_group_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                },
            ));
        }

        // Create compose pipeline
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("terrain_viewer.wboit.compose_shader"),
                source: wgpu::ShaderSource::Wgsl(WBOIT_COMPOSE_SHADER.into()),
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain_viewer.wboit.compose_pipeline_layout"),
                bind_group_layouts: &[self.wboit_compose_bind_group_layout.as_ref().unwrap()],
                push_constant_ranges: &[],
            });

        self.wboit_compose_pipeline = Some(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("terrain_viewer.wboit.compose_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.surface_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        ));
    }

    /// P6.2: Initialize Shadow Mapping (CSM) resources
    pub fn init_shadows(&mut self) {
        let desired_shadow_map_size = self.pbr_config.shadow_map_res.clamp(512, 8192);
        let desired_cascade_count = 4;

        if let Some(existing) = self.csm_renderer.as_ref() {
            if existing.config.shadow_map_size == desired_shadow_map_size
                && existing.config.cascade_count == desired_cascade_count
            {
                return;
            }
        }

        self.csm_renderer = None;
        self.csm_uniform_buffer = None;
        self.moment_pass = None;

        // Create CSM renderer using the active PBR shadow resolution instead of
        // a hard-coded fallback so terrain shadows remain spatially stable at
        // large extents like the full Switzerland scene.
        let shadow_debug_mode = crate::core::shadows::parse_shadow_debug_env();
        let csm_config = CsmConfig {
            cascade_count: desired_cascade_count,
            shadow_map_size: desired_shadow_map_size,
            max_shadow_distance: 50000.0, // Large enough to cover terrain at any camera distance
            pcf_kernel_size: 3,
            depth_bias: 0.0005,
            slope_bias: 0.001,
            peter_panning_offset: 0.0002,
            enable_evsm: true, // P6.2: Enable moment maps for VSM/EVSM/MSM
            stabilize_cascades: true,
            cascade_blend_range: 0.1,
            debug_mode: shadow_debug_mode,
            ..Default::default()
        };

        let csm = CsmRenderer::new(&self.device, csm_config);
        println!(
            "[terrain_scene] CSM renderer created, shadow_map_size={}, has_moment_maps={}",
            csm.config.shadow_map_size,
            csm.evsm_maps.is_some()
        );
        self.csm_renderer = Some(csm);

        // Create CSM uniform buffer - must match WGSL CsmUniforms struct size
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain_viewer.csm_uniforms"),
            size: std::mem::size_of::<crate::shadows::CsmUniforms>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, // ReadOnlyStorage in shader
            mapped_at_creation: false,
        });
        self.csm_uniform_buffer = Some(buffer);

        // Initialize MomentGenerationPass for VSM/EVSM/MSM techniques
        self.moment_pass = Some(crate::shadows::MomentGenerationPass::new(&self.device));

        println!("[terrain_scene] Shadows initialized (VSM/EVSM/MSM enabled)");
    }

    /// Initialize shadow depth render pipeline for CSM shadow passes
    pub fn init_shadow_depth_pipeline(&mut self) {
        if self.shadow_pipeline.is_some() {
            return;
        }

        // Must have CSM renderer initialized first
        if self.csm_renderer.is_none() {
            self.init_shadows();
        }

        let csm = self.csm_renderer.as_ref().unwrap();
        let cascade_count = csm.config.cascade_count as usize;

        // Create bind group layout for shadow depth pass
        let shadow_bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("terrain_viewer.shadow_depth.bind_group_layout"),
                    entries: &[
                        // Uniform buffer (ShadowPassUniforms)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Heightmap texture (R32Float is non-filterable, use textureLoad in shader)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // Heightmap sampler (non-filtering for R32Float compatibility)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                    ],
                });

        // Create shader module
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("terrain_viewer.shadow_depth.shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../../shaders/terrain_shadow_depth.wgsl").into(),
                ),
            });

        // Create pipeline layout
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain_viewer.shadow_depth.pipeline_layout"),
                bind_group_layouts: &[&shadow_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create depth-only render pipeline
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("terrain_viewer.shadow_depth.pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_shadow",
                    buffers: &[], // Vertices generated procedurally from vertex_index
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_shadow",
                    targets: &[], // Depth-only, no color attachments
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None, // Disable culling - light view may flip winding
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 2, // Small bias to prevent shadow acne
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        self.shadow_pipeline = Some(pipeline);

        // Create per-cascade uniform buffers
        self.shadow_uniform_buffers.clear();
        for i in 0..cascade_count {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("terrain_viewer.shadow_uniforms_{}", i)),
                size: std::mem::size_of::<crate::viewer::terrain::render::ShadowPassUniforms>()
                    as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.shadow_uniform_buffers.push(buffer);
        }

        // Create sampler for heightmap (non-filtering for R32Float compatibility)
        let height_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain_viewer.shadow_depth.height_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create per-cascade bind groups (will be recreated when terrain loads)
        // For now, store the layout for later use
        self.shadow_bind_groups.clear();

        // Store layout for bind group creation when terrain is available
        if let Some(ref terrain) = self.terrain {
            for i in 0..cascade_count {
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("terrain_viewer.shadow_bind_group_{}", i)),
                    layout: &shadow_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.shadow_uniform_buffers[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&terrain.heightmap_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&height_sampler),
                        },
                    ],
                });
                self.shadow_bind_groups.push(bind_group);
            }
        }

        println!(
            "[terrain_scene] Shadow depth pipeline initialized ({} cascades)",
            cascade_count
        );
    }

    /// Recreate shadow bind groups when terrain is loaded/changed
    pub fn update_shadow_bind_groups(&mut self) {
        let terrain = match self.terrain.as_ref() {
            Some(t) => t,
            None => return,
        };

        let csm = match self.csm_renderer.as_ref() {
            Some(c) => c,
            None => return,
        };

        if self.shadow_pipeline.is_none() || self.shadow_uniform_buffers.is_empty() {
            return;
        }

        let cascade_count = csm.config.cascade_count as usize;

        // Recreate bind group layout (needed for bind group creation)
        let shadow_bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("terrain_viewer.shadow_depth.bind_group_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                    ],
                });

        let height_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain_viewer.shadow_depth.height_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        self.shadow_bind_groups.clear();
        for i in 0..cascade_count {
            if i >= self.shadow_uniform_buffers.len() {
                break;
            }
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("terrain_viewer.shadow_bind_group_{}", i)),
                layout: &shadow_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.shadow_uniform_buffers[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&terrain.heightmap_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&height_sampler),
                    },
                ],
            });
            self.shadow_bind_groups.push(bind_group);
        }
    }

    /// P0.1/M1: Initialize WBOIT resources for given dimensions
    /// Creates size-dependent textures and bind group for interactive rendering
    pub fn init_wboit(&mut self, width: u32, height: u32) {
        if self.wboit_size == (width, height) && self.wboit_color_texture.is_some() {
            return; // Already initialized at correct size
        }

        // First ensure pipeline/layout/sampler exist
        self.init_wboit_pipeline();

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // Create color accumulation texture (Rgba16Float for weighted color)
        let color_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain_viewer.wboit.color_accum"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create reveal accumulation texture (R16Float for alpha product)
        let reveal_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain_viewer.wboit.reveal_accum"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let reveal_view = reveal_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Now create size-dependent resources (textures)

        // Create compose bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain_viewer.wboit.compose_bind_group"),
            layout: self.wboit_compose_bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&reveal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(self.wboit_sampler.as_ref().unwrap()),
                },
            ],
        });

        // Create bind group with new textures

        self.wboit_color_texture = Some(color_texture);
        self.wboit_color_view = Some(color_view);
        self.wboit_reveal_texture = Some(reveal_texture);
        self.wboit_reveal_view = Some(reveal_view);
        self.wboit_compose_bind_group = Some(bind_group);
        self.wboit_size = (width, height);

        println!("[terrain_scene] WBOIT initialized: {}x{}", width, height);
    }

    /// Initialize post-process pass (called lazily when lens effects enabled)
    pub fn init_post_process(&mut self) {
        if self.post_process.is_none() {
            self.post_process = Some(crate::viewer::terrain::post_process::PostProcessPass::new(
                self.device.clone(),
                self.surface_format,
            ));
        }
    }

    /// Initialize DoF pass (called lazily when DoF enabled)
    pub fn init_dof_pass(&mut self) {
        if self.dof_pass.is_none() {
            self.dof_pass = Some(crate::viewer::terrain::dof::DofPass::new(
                self.device.clone(),
                self.surface_format,
            ));
            println!("[terrain] DoF pass initialized");
        }
    }

    /// Initialize Denoise pass (called lazily when enabled)
    pub fn init_denoise_pass(&mut self) {
        if self.denoise_pass.is_none() {
            self.denoise_pass = Some(DenoisePass::new(self.device.clone()));
            println!("[terrain] Denoise pass initialized");
        }
    }

    /// Initialize motion blur pass (called lazily when motion blur enabled)
    pub fn init_motion_blur_pass(&mut self) {
        if self.motion_blur_pass.is_none() {
            self.motion_blur_pass = Some(
                crate::viewer::terrain::motion_blur::MotionBlurAccumulator::new(
                    self.device.clone(),
                    self.surface_format,
                ),
            );
        }
    }

    /// Initialize volumetrics pass (called lazily when volumetrics enabled)
    pub fn init_volumetrics_pass(&mut self) {
        if self.volumetrics_pass.is_none() {
            self.volumetrics_pass =
                Some(crate::viewer::terrain::volumetrics::VolumetricsPass::new(
                    self.device.clone(),
                    self.surface_format,
                ));
        }
    }
}

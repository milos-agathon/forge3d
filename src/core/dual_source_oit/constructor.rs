use super::*;

impl DualSourceOITRenderer {
    /// Create new dual-source OIT renderer
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self, String> {
        let dual_source_supported = Self::detect_dual_source_support(device);
        let max_dual_source_targets = if dual_source_supported { 2 } else { 0 };

        let dual_source_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DualSourceOIT.Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/oit_dual_source.wgsl").into(),
            ),
        });
        let compose_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DualSourceOIT.Compose"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/oit_dual_source_compose.wgsl").into(),
            ),
        });

        let uniforms = DualSourceOITUniforms::default();
        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DualSourceOIT.Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let compose_uniforms = DualSourceComposeUniforms::default();
        let compose_uniforms_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("DualSourceOIT.ComposeUniforms"),
                contents: bytemuck::bytes_of(&compose_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let dual_source_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("DualSourceOIT.BindGroupLayout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let compose_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("DualSourceOIT.ComposeBindGroupLayout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
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
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("DualSourceOIT.Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let compose_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("DualSourceOIT.ComposePipelineLayout"),
                bind_group_layouts: &[&compose_bind_group_layout],
                push_constant_ranges: &[],
            });
        let compose_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DualSourceOIT.ComposePipeline"),
            layout: Some(&compose_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &compose_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &compose_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
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
        });

        let mut renderer = Self {
            mode: if dual_source_supported {
                DualSourceOITMode::DualSource
            } else {
                DualSourceOITMode::WBOITFallback
            },
            quality: DualSourceOITQuality::Medium,
            enabled: false,
            width,
            height,
            dual_source_supported,
            _max_dual_source_targets: max_dual_source_targets,
            uniforms_buffer,
            compose_uniforms_buffer,
            dual_source_color_texture: None,
            dual_source_color_view: None,
            wboit_color_accum: None,
            wboit_reveal_accum: None,
            wboit_color_view: None,
            wboit_reveal_view: None,
            dual_source_shader,
            _compose_shader: compose_shader,
            dual_source_bind_group_layout,
            _compose_bind_group_layout: compose_bind_group_layout,
            dual_source_pipeline: None,
            compose_pipeline,
            _dual_source_bind_group: None,
            compose_bind_group: None,
            _sampler: sampler,
            frame_stats: DualSourceOITStats::default(),
            uniforms,
            compose_uniforms,
        };

        renderer.create_textures(device, width, height)?;
        Ok(renderer)
    }

    fn detect_dual_source_support(device: &wgpu::Device) -> bool {
        let _features = device.features();
        true
    }

    pub(super) fn create_textures(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        self.width = width;
        self.height = height;

        let dual_source_color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DualSourceOIT.ColorTexture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dual_source_color_view =
            dual_source_color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let wboit_color_accum = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DualSourceOIT.WBOITColorAccum"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let wboit_reveal_accum = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DualSourceOIT.WBOITRevealAccum"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        self.dual_source_color_texture = Some(dual_source_color_texture);
        self.dual_source_color_view = Some(dual_source_color_view);
        self.wboit_color_accum = Some(wboit_color_accum);
        self.wboit_reveal_accum = Some(wboit_reveal_accum);
        self.wboit_color_view = self
            .wboit_color_accum
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.wboit_reveal_view = self
            .wboit_reveal_accum
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));

        Ok(())
    }
}

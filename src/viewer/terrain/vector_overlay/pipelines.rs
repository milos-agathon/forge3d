use super::*;

impl VectorOverlayStack {
    /// Initialize the vector overlay render pipelines
    pub fn init_pipelines(&mut self, surface_format: wgpu::TextureFormat) {
        // Create bind group layout with texture/sampler for shadow integration
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("vector_overlay_bind_group_layout"),
                    entries: &[
                        // Uniforms
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Sun visibility texture (non-filterable for R32Float compatibility)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // Sampler (non-filtering for R32Float compatibility)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                    ],
                });

        // Create uniform buffer
        let uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vector_overlay_uniforms"),
            size: std::mem::size_of::<VectorOverlayUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Compile shader
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vector_overlay_shader"),
                source: wgpu::ShaderSource::Wgsl(VECTOR_OVERLAY_SHADER.into()),
            });

        // Create pipeline layout
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vector_overlay_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Common depth stencil state (depth test, no write for overlays)
        // Use LessEqual to be more forgiving with depth precision, and aggressive bias
        // to ensure overlay is clearly in front of terrain
        let depth_stencil = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false, // Read only - don't write to depth
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: -100, // Strong bias towards camera
                slope_scale: -10.0,
                clamp: 0.0,
            },
        });

        // Triangle pipeline
        self.pipeline_triangles = Some(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("vector_overlay_triangles_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[VectorVertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None, // Draw both sides
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: depth_stencil.clone(),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        ));

        // Lines pipeline
        self.pipeline_lines = Some(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("vector_overlay_lines_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[VectorVertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: depth_stencil.clone(),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        ));

        // Points pipeline
        self.pipeline_points = Some(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("vector_overlay_points_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[VectorVertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::PointList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        ));

        self.bind_group_layout = Some(bind_group_layout);
        self.uniform_buffer = Some(uniform_buffer);

        // P0.1/M1: Initialize OIT pipelines with WBOIT blend states
        self.init_oit_pipelines(&shader, &pipeline_layout);
    }

    /// P0.1/M1: Initialize OIT pipelines with WBOIT blend states for transparent rendering
    fn init_oit_pipelines(
        &mut self,
        shader: &wgpu::ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
    ) {
        // WBOIT color accumulation blend state (additive)
        let accum_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        // WBOIT reveal blend state (multiplicative for alpha product)
        let reveal_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };

        // Depth stencil for OIT (read only, no write)
        let depth_stencil = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: -100,
                slope_scale: -10.0,
                clamp: 0.0,
            },
        });

        // OIT Triangle pipeline (MRT: color accum + reveal)
        self.oit_pipeline_triangles = Some(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("vector_overlay_oit_triangles_pipeline"),
                layout: Some(pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: "vs_main",
                    buffers: &[VectorVertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: "fs_main_oit",
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: Some(accum_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::R16Float,
                            blend: Some(reveal_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: depth_stencil.clone(),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        ));

        // OIT Lines pipeline
        self.oit_pipeline_lines = Some(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("vector_overlay_oit_lines_pipeline"),
                layout: Some(pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: "vs_main",
                    buffers: &[VectorVertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: "fs_main_oit",
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: Some(accum_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::R16Float,
                            blend: Some(reveal_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: depth_stencil.clone(),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        ));

        // OIT Points pipeline
        self.oit_pipeline_points = Some(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("vector_overlay_oit_points_pipeline"),
                layout: Some(pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: "vs_main",
                    buffers: &[VectorVertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: "fs_main_oit",
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: Some(accum_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::R16Float,
                            blend: Some(reveal_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::PointList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        ));

        println!("[vector_overlay] OIT pipelines initialized");
    }

    /// Check if pipelines are initialized
    pub fn pipelines_ready(&self) -> bool {
        self.pipeline_triangles.is_some() && self.bind_group_layout.is_some()
    }

    /// P0.1/M1: Check if OIT pipelines are ready
    pub fn oit_pipelines_ready(&self) -> bool {
        self.oit_pipeline_triangles.is_some()
    }

    /// P0.1/M1: Render a layer using OIT pipelines to WBOIT accumulation buffers
    pub fn render_layer_oit<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        params: RenderLayerParams,
    ) -> bool {
        let visible_layers: Vec<_> = self.visible_layers().collect();
        if params.layer_index >= visible_layers.len() {
            return false;
        }

        let layer = visible_layers[params.layer_index];
        if layer.index_count == 0 {
            return false;
        }

        // Update uniforms
        let uniforms = VectorOverlayUniforms {
            view_proj: params.view_proj,
            sun_dir: [params.sun_dir[0], params.sun_dir[1], params.sun_dir[2], 0.0],
            lighting: params.lighting,
            layer_params: [layer.config.opacity, 0.0, 0.0, 0.0],
            selected_feature_id: params.selected_feature_id,
            highlight_color: params.highlight_color,
            _pad: [0; 7],
        };

        if let Some(ref buf) = self.uniform_buffer {
            self.queue
                .write_buffer(buf, 0, bytemuck::bytes_of(&uniforms));
        }

        // Set bind group
        if let Some(ref bg) = self.bind_group {
            pass.set_bind_group(0, bg, &[]);
        } else {
            return false;
        }

        // Select OIT pipeline based on primitive type
        let pipeline = match layer.config.primitive {
            OverlayPrimitive::Triangles | OverlayPrimitive::TriangleStrip => {
                self.oit_pipeline_triangles.as_ref()
            }
            OverlayPrimitive::Lines | OverlayPrimitive::LineStrip => {
                self.oit_pipeline_lines.as_ref()
            }
            OverlayPrimitive::Points => self.oit_pipeline_points.as_ref(),
        };

        if let Some(pipe) = pipeline {
            pass.set_pipeline(pipe);
            pass.set_vertex_buffer(0, layer.vertex_buffer.slice(..));
            pass.set_index_buffer(layer.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..layer.index_count, 0, 0..1);
            true
        } else {
            false
        }
    }

    /// Prepare bind group for rendering (call before render pass)
    /// Creates bind group with sun visibility texture for shadow integration
    pub fn prepare_bind_group(&mut self, sun_vis_view: &wgpu::TextureView) {
        let bind_group_layout = match &self.bind_group_layout {
            Some(layout) => layout,
            None => return,
        };

        let uniform_buffer = match &self.uniform_buffer {
            Some(buf) => buf,
            None => return,
        };

        // Create sampler if not already created (non-filtering for R32Float)
        if self.sampler.is_none() {
            self.sampler = Some(self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("vector_overlay_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            }));
        }

        let sampler = self.sampler.as_ref().unwrap();

        // Create bind group with uniforms, sun visibility texture, and sampler
        self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vector_overlay_bind_group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(sun_vis_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        }));
    }

    /// Get count of visible layers
    pub fn visible_layer_count(&self) -> usize {
        self.layers.iter().filter(|l| l.config.visible).count()
    }

    /// Render a single layer with highlight support for picking
    pub fn render_layer_with_highlight<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        params: RenderLayerParams,
    ) -> bool {
        let RenderLayerParams {
            layer_index,
            view_proj,
            sun_dir,
            lighting,
            selected_feature_id,
            highlight_color,
        } = params;

        if !self.enabled {
            return false;
        }

        let bind_group = match &self.bind_group {
            Some(bg) => bg,
            None => return false,
        };

        let uniform_buffer = match &self.uniform_buffer {
            Some(buf) => buf,
            None => return false,
        };

        let visible_layers: Vec<_> = self.layers.iter().filter(|l| l.config.visible).collect();

        if layer_index >= visible_layers.len() {
            return false;
        }

        let layer = visible_layers[layer_index];

        let uniforms = VectorOverlayUniforms {
            view_proj,
            sun_dir: [sun_dir[0], sun_dir[1], sun_dir[2], 0.0],
            lighting,
            layer_params: [
                layer.config.opacity * self.global_opacity,
                layer.config.depth_bias,
                layer.config.line_width,
                layer.config.point_size,
            ],
            highlight_color,
            selected_feature_id,
            _pad: [0; 7],
        };

        self.queue
            .write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let pipeline = match layer.config.primitive {
            OverlayPrimitive::Triangles | OverlayPrimitive::TriangleStrip => {
                self.pipeline_triangles.as_ref()
            }
            OverlayPrimitive::Lines | OverlayPrimitive::LineStrip => self.pipeline_lines.as_ref(),
            OverlayPrimitive::Points => self.pipeline_points.as_ref(),
        };

        if let Some(pipeline) = pipeline {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_vertex_buffer(0, layer.vertex_buffer.slice(..));

            if layer.index_count > 0 {
                pass.set_index_buffer(layer.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..layer.index_count, 0, 0..1);
            } else {
                pass.draw(0..layer.vertex_count, 0..1);
            }
            true
        } else {
            false
        }
    }
}

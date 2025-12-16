use super::types::*;
use crate::core::error::RenderError;
use crate::vector::api::PointDef;
use crate::vector::data::{validate_point_instances, PointInstance};
use crate::vector::layer::Layer;
use bytemuck;
use glam::Vec2;

/// Instanced point renderer with H20,H21,H22 enhancements
pub struct PointRenderer {
    render_pipeline: wgpu::RenderPipeline,
    instance_buffer: Option<wgpu::Buffer>,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    // H5: Picking pipeline and resources
    pick_pipeline: wgpu::RenderPipeline,
    pick_uniform_buffer: wgpu::Buffer,
    pick_bind_group: wgpu::BindGroup,
    // H4: Weighted OIT pipeline for points
    oit_pipeline: wgpu::RenderPipeline,
    instance_capacity: usize,
    // H20: Debug mode support
    debug_flags: DebugFlags,
    // H21: Texture atlas support
    texture_atlas: Option<TextureAtlas>,
    // H22: Clip.w scaling parameters
    enable_clip_w_scaling: bool,
    depth_range: (f32, f32), // (near, far)
    // H2: Shape/Lod params
    shape_mode: u32,
    lod_threshold: f32,
}

impl PointRenderer {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point_instanced.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../../shaders/point_instanced.wgsl"
            ))),
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Point.Uniform"),
            size: std::mem::size_of::<PointUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout with uniform + optional atlas texture/sampler
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vf.Vector.Point.BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Default fallback texture/sampler (1x1 white)
        let default_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.Point.DefaultTexture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let default_view = default_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        // Create bind group with defaults
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vf.Vector.Point.BindGroup"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&default_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&default_sampler),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vf.Vector.Point.PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vf.Vector.Point.Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    // Per-instance point data
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<PointInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            // position
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            // size
                            wgpu::VertexAttribute {
                                offset: 8,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32,
                            },
                            // color
                            wgpu::VertexAttribute {
                                offset: 12,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x4,
                            },
                            // rotation
                            wgpu::VertexAttribute {
                                offset: 28,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32,
                            },
                            // uv_offset (H21)
                            wgpu::VertexAttribute {
                                offset: 32,
                                shader_location: 4,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
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

        // H4: OIT pipeline for points (fs_oit, MRT)
        let oit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vf.Vector.Point.OITPipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let oit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vf.Vector.Point.OITPipeline"),
            layout: Some(&oit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PointInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 28,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32,
                        },
                        wgpu::VertexAttribute {
                            offset: 32,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_oit",
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
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
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R16Float,
                        blend: Some(wgpu::BlendState {
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
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // H5: Picking bind group layout (binding 0 = uniform, binding 3 = pick uniform)
        let pick_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vf.Vector.Point.PickBindGroupLayout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // H5: Picking uniform buffer (u32 pick_id + padding)
        let pick_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Point.PickUniform"),
            size: 16, // 4 u32s for alignment
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // H5: Picking bind group
        let pick_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vf.Vector.Point.PickBindGroup"),
            layout: &pick_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pick_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // H5: Picking pipeline (R32Uint target, fs_pick)
        let pick_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vf.Vector.Point.PickPipelineLayout"),
            bind_group_layouts: &[&pick_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pick_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vf.Vector.Point.PickPipeline"),
            layout: Some(&pick_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PointInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 28,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32,
                        },
                        wgpu::VertexAttribute {
                            offset: 32,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_pick",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
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

        Ok(Self {
            render_pipeline,
            instance_buffer: None,
            uniform_buffer,
            bind_group,
            bind_group_layout,
            pick_pipeline,
            pick_uniform_buffer,
            pick_bind_group,
            oit_pipeline,
            instance_capacity: 0,
            // H20: Initialize with default debug flags
            debug_flags: DebugFlags::default(),
            // H21: No texture atlas initially
            texture_atlas: None,
            // H22: Initialize with default clip.w scaling settings
            enable_clip_w_scaling: false,
            depth_range: (0.1, 1000.0), // Default near/far
            shape_mode: get_global_config().0,
            lod_threshold: get_global_config().1,
        })
    }

    /// Convert point definitions to point instances
    pub fn pack_points(&self, points: &[PointDef]) -> Result<Vec<PointInstance>, RenderError> {
        let mut instances = Vec::with_capacity(points.len());

        for point in points {
            // Validate point data
            if !point.position.x.is_finite() || !point.position.y.is_finite() {
                return Err(RenderError::Upload(format!(
                    "Point has non-finite coordinates: ({}, {})",
                    point.position.x, point.position.y
                )));
            }

            if point.style.point_size <= 0.0 || !point.style.point_size.is_finite() {
                return Err(RenderError::Upload(format!(
                    "Point size must be positive and finite, got {}",
                    point.style.point_size
                )));
            }

            instances.push(PointInstance {
                position: [point.position.x, point.position.y],
                size: point.style.point_size,
                color: point.style.fill_color,
                rotation: 0.0,         // Default rotation (H21)
                uv_offset: [0.0, 0.0], // Default UV offset (H21)
                _pad: 0.0,
            });
        }

        // Validate packed instances
        let validation_result = validate_point_instances(&instances);
        if !validation_result.is_valid {
            return Err(RenderError::Upload(
                validation_result
                    .error_message
                    .unwrap_or_else(|| "Point instance validation failed".to_string()),
            ));
        }

        Ok(instances)
    }

    /// Upload point instances to GPU buffer
    pub fn upload_points(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[PointInstance],
    ) -> Result<(), RenderError> {
        if instances.is_empty() {
            return Ok(());
        }

        // Reallocate buffer if needed
        if instances.len() > self.instance_capacity {
            let new_capacity = (instances.len() * 2).max(1024);
            self.instance_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vf.Vector.Point.InstanceBuffer"),
                size: (new_capacity * std::mem::size_of::<PointInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.instance_capacity = new_capacity;
        }

        // Upload instance data directly to GPU buffer
        if let Some(instance_buffer) = &self.instance_buffer {
            let instance_data = bytemuck::cast_slice(instances);
            queue.write_buffer(instance_buffer, 0, instance_data);
        }

        Ok(())
    }

    /// H4: Render points into OIT MRT attachments (Rgba16Float + R16Float)
    pub fn render_oit<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        queue: &wgpu::Queue,
        transform: &[[f32; 4]; 4],
        viewport_size: [f32; 2],
        pixel_scale: f32,
        instance_count: u32,
    ) -> Result<(), RenderError> {
        if let Some(instance_buffer) = &self.instance_buffer {
            let atlas_size = if let Some(atlas) = &self.texture_atlas {
                [atlas.width as f32, atlas.height as f32]
            } else {
                [1.0, 1.0]
            };

            let uniform = PointUniform {
                transform: *transform,
                viewport_size,
                pixel_scale,
                debug_mode: self.debug_flags.to_bitfield(),
                atlas_size,
                enable_clip_w_scaling: self.enable_clip_w_scaling as u32,
                _pad0: 0.0,
                depth_range: [self.depth_range.0, self.depth_range.1],
                shape_mode: self.shape_mode,
                lod_threshold: self.lod_threshold,
            };

            queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
            render_pass.set_pipeline(&self.oit_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
            render_pass.draw(0..4, 0..instance_count);
        }
        Ok(())
    }

    /// H5: Render picking IDs to an R32Uint attachment
    pub fn render_pick<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        queue: &wgpu::Queue,
        transform: &[[f32; 4]; 4],
        viewport_size: [f32; 2],
        pixel_scale: f32,
        instance_count: u32,
        base_pick_id: u32,
    ) -> Result<(), RenderError> {
        if let Some(instance_buffer) = &self.instance_buffer {
            // Update uniforms
            let atlas_size = if let Some(atlas) = &self.texture_atlas {
                [atlas.width as f32, atlas.height as f32]
            } else {
                [1.0, 1.0]
            };

            let uniform = PointUniform {
                transform: *transform,
                viewport_size,
                pixel_scale,
                debug_mode: self.debug_flags.to_bitfield(),
                atlas_size,
                enable_clip_w_scaling: self.enable_clip_w_scaling as u32,
                _pad0: 0.0,
                depth_range: [self.depth_range.0, self.depth_range.1],
                shape_mode: self.shape_mode,
                lod_threshold: self.lod_threshold,
            };
            queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

            // Write pick uniform (u32 id + padding)
            let pick_data: [u32; 4] = [base_pick_id, 0, 0, 0];
            queue.write_buffer(&self.pick_uniform_buffer, 0, bytemuck::bytes_of(&pick_data));

            // Draw
            render_pass.set_pipeline(&self.pick_pipeline);
            render_pass.set_bind_group(0, &self.pick_bind_group, &[]);
            render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
            render_pass.draw(0..4, 0..instance_count);
        }

        Ok(())
    }

    /// Render instanced points
    pub fn render<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        queue: &wgpu::Queue,
        transform: &[[f32; 4]; 4],
        viewport_size: [f32; 2],
        pixel_scale: f32,
        instance_count: u32,
    ) -> Result<(), RenderError> {
        if let Some(instance_buffer) = &self.instance_buffer {
            // Update uniforms with H20,H21,H22 enhancements
            let atlas_size = if let Some(atlas) = &self.texture_atlas {
                [atlas.width as f32, atlas.height as f32]
            } else {
                [1.0, 1.0] // Default atlas size
            };

            let uniform = PointUniform {
                transform: *transform,
                viewport_size,
                pixel_scale,
                debug_mode: self.debug_flags.to_bitfield(), // H20
                atlas_size,                                 // H21
                enable_clip_w_scaling: self.enable_clip_w_scaling as u32, // H22
                _pad0: 0.0,
                depth_range: [self.depth_range.0, self.depth_range.1], // H22
                shape_mode: self.shape_mode,
                lod_threshold: self.lod_threshold,
            };

            queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

            // Set pipeline and resources
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, instance_buffer.slice(..));

            // Draw instanced - each instance generates a quad (4 vertices as triangle strip)
            render_pass.draw(0..4, 0..instance_count);
        }

        Ok(())
    }

    /// H20: Set debug rendering flags
    pub fn set_debug_flags(&mut self, debug_flags: DebugFlags) {
        self.debug_flags = debug_flags;
    }

    /// H20: Get current debug flags
    pub fn get_debug_flags(&self) -> DebugFlags {
        self.debug_flags
    }

    /// H21: Set texture atlas for sprite rendering  
    pub fn set_texture_atlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas: Option<TextureAtlas>,
    ) {
        self.texture_atlas = atlas;
        // Recreate bind group with atlas texture and sampler
        self.recreate_bind_group(device, queue);
    }

    /// Recreate bind group with current texture atlas state
    fn recreate_bind_group(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Use atlas texture if available, otherwise create a default 1x1 white texture
        let (texture_view, sampler) = if let Some(atlas) = &self.texture_atlas {
            (&atlas.view, &atlas.sampler)
        } else {
            // Create default 1x1 white texture for non-atlas rendering
            static DEFAULT_TEXTURE: std::sync::OnceLock<(wgpu::TextureView, wgpu::Sampler)> =
                std::sync::OnceLock::new();
            DEFAULT_TEXTURE.get_or_init(|| {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("vf.Vector.Point.DefaultTexture"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });

                // Upload white pixel
                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &[255, 255, 255, 255], // RGBA white
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4),
                        rows_per_image: Some(1),
                    },
                    wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );

                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
                (view, sampler)
            });
            {
                let (view, sampler) = DEFAULT_TEXTURE.get().unwrap();
                (view, sampler)
            }
        };

        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vf.Vector.Point.BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
    }

    /// H21: Get current texture atlas
    pub fn get_texture_atlas(&self) -> Option<&TextureAtlas> {
        self.texture_atlas.as_ref()
    }

    /// H22: Enable or disable clip.w aware sizing
    pub fn set_clip_w_scaling(&mut self, enabled: bool) {
        self.enable_clip_w_scaling = enabled;
    }

    /// H22: Set depth range for clip.w scaling
    pub fn set_depth_range(&mut self, near: f32, far: f32) {
        self.depth_range = (near, far);
    }

    /// H22: Get current clip.w scaling setting
    pub fn is_clip_w_scaling_enabled(&self) -> bool {
        self.enable_clip_w_scaling
    }

    /// H2: Set shape mode (0=circle, 4=texture, 5=sphere impostor)
    pub fn set_shape_mode(&mut self, mode: u32) {
        self.shape_mode = mode;
    }

    /// H2: Set LOD threshold in pixels
    pub fn set_lod_threshold(&mut self, threshold: f32) {
        self.lod_threshold = threshold;
    }

    /// Get layer for point rendering
    pub fn layer() -> Layer {
        Layer::Points
    }
}

/// Calculate point clustering for high-density datasets (H20)
pub fn cluster_points(points: &[Vec2], cluster_radius: f32) -> Vec<(Vec2, u32)> {
    if points.is_empty() {
        return Vec::new();
    }

    let mut clusters = Vec::new();
    let mut used = vec![false; points.len()];

    for (i, &point) in points.iter().enumerate() {
        if used[i] {
            continue;
        }

        let mut cluster_center = point;
        let mut cluster_count = 1;
        used[i] = true;

        // Find nearby points to cluster
        for (j, &other_point) in points.iter().enumerate().skip(i + 1) {
            if used[j] {
                continue;
            }

            let distance = (other_point - point).length();
            if distance <= cluster_radius {
                cluster_center = (cluster_center * cluster_count as f32 + other_point)
                    / (cluster_count + 1) as f32;
                cluster_count += 1;
                used[j] = true;
            }
        }

        clusters.push((cluster_center, cluster_count));
    }

    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::api::VectorStyle;
    use glam::Vec2;

    #[test]
    fn test_pack_simple_points() {
        let device = crate::core::gpu::create_device_for_test();
        let renderer = PointRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

        let points = vec![
            PointDef {
                position: Vec2::new(0.0, 0.0),
                style: VectorStyle {
                    point_size: 4.0,
                    fill_color: [1.0, 0.0, 0.0, 1.0],
                    ..Default::default()
                },
            },
            PointDef {
                position: Vec2::new(1.0, 1.0),
                style: VectorStyle {
                    point_size: 6.0,
                    fill_color: [0.0, 1.0, 0.0, 1.0],
                    ..Default::default()
                },
            },
        ];

        let instances = renderer.pack_points(&points).unwrap();

        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0].size, 4.0);
        assert_eq!(instances[0].color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(instances[1].size, 6.0);
        assert_eq!(instances[1].color, [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_reject_invalid_point_size() {
        let device = crate::core::gpu::create_device_for_test();
        let renderer = PointRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

        let invalid_point = PointDef {
            position: Vec2::new(0.0, 0.0),
            style: VectorStyle {
                point_size: -1.0, // Invalid negative size
                ..Default::default()
            },
        };

        let result = renderer.pack_points(&[invalid_point]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("positive and finite"));
    }

    #[test]
    fn test_reject_non_finite_coordinates() {
        let device = crate::core::gpu::create_device_for_test();
        let renderer = PointRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();

        let invalid_point = PointDef {
            position: Vec2::new(f32::NAN, 0.0),
            style: VectorStyle::default(),
        };

        let result = renderer.pack_points(&[invalid_point]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("non-finite coordinates"));
    }

    #[test]
    fn test_point_clustering() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(0.1, 0.1), // Close to first point
            Vec2::new(5.0, 5.0), // Far away
            Vec2::new(5.1, 5.1), // Close to third point
        ];

        let clusters = cluster_points(&points, 1.0);

        // Should create 2 clusters
        assert_eq!(clusters.len(), 2);

        // First cluster should have 2 points
        assert_eq!(clusters[0].1, 2);
        // Second cluster should have 2 points
        assert_eq!(clusters[1].1, 2);
    }

    #[test]
    fn test_no_clustering_when_all_far_apart() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(20.0, 20.0),
        ];

        let clusters = cluster_points(&points, 1.0);

        // Should create 3 clusters (no clustering)
        assert_eq!(clusters.len(), 3);
        assert!(clusters.iter().all(|(_, count)| *count == 1));
    }
}

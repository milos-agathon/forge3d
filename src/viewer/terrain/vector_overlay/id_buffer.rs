use super::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IdBufferUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub depth_bias: f32,
    pub _pad: [f32; 3],
}

/// WGSL shader for ID buffer rendering
pub const ID_BUFFER_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    depth_bias: f32,
    _pad: vec3<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) normal: vec3<f32>,
    @location(4) feature_id: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) feature_id: u32,
};

@vertex
fn vs_id(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    var pos = in.position;
    pos.y += u.depth_bias;
    out.clip_position = u.view_proj * vec4<f32>(pos, 1.0);
    out.feature_id = in.feature_id;
    return out;
}

@fragment
fn fs_id(in: VertexOutput) -> @location(0) u32 {
    return in.feature_id;
}
"#;

impl VectorOverlayStack {
    /// Initialize ID buffer pipelines for picking
    pub fn init_id_pipelines(&mut self) {
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("vector_overlay_id_bind_group_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vector_overlay_id_uniforms"),
            size: std::mem::size_of::<IdBufferUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vector_overlay_id_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vector_overlay_id_shader"),
                source: wgpu::ShaderSource::Wgsl(ID_BUFFER_SHADER.into()),
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vector_overlay_id_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let depth_stencil = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        // Triangle pipeline for ID buffer
        self.id_pipeline_triangles = Some(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("vector_overlay_id_triangles_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_id",
                    buffers: &[VectorVertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_id",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Uint,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
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

        // Lines pipeline for ID buffer
        self.id_pipeline_lines = Some(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("vector_overlay_id_lines_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_id",
                    buffers: &[VectorVertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_id",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Uint,
                        blend: None,
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

        // Points pipeline for ID buffer
        self.id_pipeline_points = Some(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("vector_overlay_id_points_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_id",
                    buffers: &[VectorVertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_id",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Uint,
                        blend: None,
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

        self.id_bind_group_layout = Some(bind_group_layout);
        self.id_uniform_buffer = Some(uniform_buffer);
        self.id_bind_group = Some(bind_group);
    }

    /// Check if ID pipelines are initialized
    pub fn id_pipelines_ready(&self) -> bool {
        self.id_pipeline_triangles.is_some() && self.id_bind_group.is_some()
    }

    /// Render all visible layers to ID buffer
    pub fn render_to_id_buffer<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        view_proj: [[f32; 4]; 4],
        depth_bias: f32,
    ) {
        if !self.enabled || !self.id_pipelines_ready() {
            return;
        }

        let bind_group = match &self.id_bind_group {
            Some(bg) => bg,
            None => return,
        };

        let uniform_buffer = match &self.id_uniform_buffer {
            Some(buf) => buf,
            None => return,
        };

        // Update uniforms
        let uniforms = IdBufferUniforms {
            view_proj,
            depth_bias,
            _pad: [0.0; 3],
        };
        self.queue
            .write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        // Render each visible layer
        for layer in self.layers.iter().filter(|l| l.config.visible) {
            let pipeline = match layer.config.primitive {
                OverlayPrimitive::Triangles | OverlayPrimitive::TriangleStrip => {
                    self.id_pipeline_triangles.as_ref()
                }
                OverlayPrimitive::Lines | OverlayPrimitive::LineStrip => {
                    self.id_pipeline_lines.as_ref()
                }
                OverlayPrimitive::Points => self.id_pipeline_points.as_ref(),
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
            }
        }
    }
}

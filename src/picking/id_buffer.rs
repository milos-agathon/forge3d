// src/picking/id_buffer.rs
// ID buffer rendering pass for feature picking
// Renders feature IDs to an R32Uint texture for GPU-based picking

use wgpu::{
    BindGroup, BindGroupLayout, Buffer, Device, RenderPipeline, Texture, TextureView,
};

/// Vertex format for ID buffer rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IdVertex {
    pub position: [f32; 3],
    pub feature_id: u32,
}

impl IdVertex {
    pub fn new(x: f32, y: f32, z: f32, feature_id: u32) -> Self {
        Self {
            position: [x, y, z],
            feature_id,
        }
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<IdVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

/// ID buffer uniforms
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IdBufferUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub depth_bias: f32,
    pub _pad: [f32; 3],
}

/// ID buffer render pass resources
pub struct IdBufferPass {
    id_texture: Texture,
    id_view: TextureView,
    #[allow(dead_code)]
    depth_texture: Texture,
    depth_view: TextureView,
    pipeline: RenderPipeline,
    #[allow(dead_code)]
    bind_group_layout: BindGroupLayout,
    uniform_buffer: Buffer,
    bind_group: BindGroup,
    width: u32,
    height: u32,
}

impl IdBufferPass {
    /// Create a new ID buffer pass
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        let id_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("id_buffer_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let id_view = id_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("id_buffer_depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("id_buffer_bind_group_layout"),
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

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("id_buffer_uniforms"),
            size: std::mem::size_of::<IdBufferUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("id_buffer_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("id_buffer_shader"),
            source: wgpu::ShaderSource::Wgsl(ID_BUFFER_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("id_buffer_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("id_buffer_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_id",
                buffers: &[IdVertex::desc()],
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            id_texture,
            id_view,
            depth_texture,
            depth_view,
            pipeline,
            bind_group_layout,
            uniform_buffer,
            bind_group,
            width,
            height,
        }
    }

    /// Get the ID texture
    pub fn id_texture(&self) -> &Texture {
        &self.id_texture
    }

    /// Get the ID texture view
    pub fn id_view(&self) -> &TextureView {
        &self.id_view
    }

    /// Get the depth texture view
    pub fn depth_view(&self) -> &TextureView {
        &self.depth_view
    }

    /// Get the render pipeline
    pub fn pipeline(&self) -> &RenderPipeline {
        &self.pipeline
    }

    /// Get the bind group
    pub fn bind_group(&self) -> &BindGroup {
        &self.bind_group
    }

    /// Get the uniform buffer
    pub fn uniform_buffer(&self) -> &Buffer {
        &self.uniform_buffer
    }

    /// Get dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Update uniforms
    pub fn update_uniforms(&self, queue: &wgpu::Queue, view_proj: [[f32; 4]; 4], depth_bias: f32) {
        let uniforms = IdBufferUniforms {
            view_proj,
            depth_bias,
            _pad: [0.0; 3],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    /// Begin render pass and return the render pass
    pub fn begin_render_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("id_buffer_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.id_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        })
    }
}

/// WGSL shader for ID buffer rendering
const ID_BUFFER_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    depth_bias: f32,
    _pad: vec3<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) feature_id: u32,
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

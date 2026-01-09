// src/picking/tile_id.rs
// Small-tile ID buffer rendering for efficient hover/pick operations
// Part of Plan 2: Standard — GPU Ray Picking + Hover Support

use wgpu::{
    BindGroup, BindGroupLayout, Buffer, Device, RenderPipeline, 
    Texture, TextureView,
};

/// Default tile size for ID buffer rendering
pub const DEFAULT_TILE_SIZE: u32 = 64;

/// Uniforms for tile ID rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileIdUniforms {
    /// View-projection matrix
    pub view_proj: [[f32; 4]; 4],
    /// Tile viewport offset (in pixels from full-res origin)
    pub tile_offset: [f32; 2],
    /// Tile size in pixels
    pub tile_size: [f32; 2],
    /// Full resolution dimensions
    pub full_resolution: [f32; 2],
    /// Depth bias for z-fighting prevention
    pub depth_bias: f32,
    /// Padding
    pub _pad: f32,
}

/// WGSL shader for tile ID buffer rendering
pub const TILE_ID_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    tile_offset: vec2<f32>,
    tile_size: vec2<f32>,
    full_resolution: vec2<f32>,
    depth_bias: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) feature_id: u32,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) feature_id: u32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    var pos = in.position;
    pos.y += u.depth_bias;
    
    // Standard clip-space transform
    var clip = u.view_proj * vec4<f32>(pos, 1.0);
    
    // Adjust clip coordinates to render only the tile region
    // NDC range is [-1, 1], we need to map tile_offset/full_resolution to this
    let ndc_offset = (u.tile_offset / u.full_resolution) * 2.0 - 1.0;
    let ndc_scale = u.tile_size / u.full_resolution;
    
    // Remap to tile viewport
    clip.x = (clip.x / clip.w - ndc_offset.x) / ndc_scale.x * clip.w;
    clip.y = (clip.y / clip.w + ndc_offset.y) / ndc_scale.y * clip.w;
    
    out.clip_pos = clip;
    out.feature_id = in.feature_id;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
    return in.feature_id;
}
"#;

/// Tile ID buffer pass for efficient hover picking
pub struct TileIdPass {
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
    tile_size: u32,
}

impl TileIdPass {
    /// Create a new tile ID pass
    pub fn new(device: &Device, tile_size: u32) -> Self {
        let tile_size = tile_size.max(16).min(256);

        // Create tile-sized ID texture (R32Uint)
        let id_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tile_id_texture"),
            size: wgpu::Extent3d {
                width: tile_size,
                height: tile_size,
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

        // Create depth texture for tile
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tile_id_depth"),
            size: wgpu::Extent3d {
                width: tile_size,
                height: tile_size,
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

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile_id_shader"),
            source: wgpu::ShaderSource::Wgsl(TILE_ID_SHADER.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile_id_bind_group_layout"),
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

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_id_uniform_buffer"),
            size: std::mem::size_of::<TileIdUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile_id_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile_id_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Vertex buffer layout for TileIdVertex
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 16, // 3 floats position + 1 u32 feature_id
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 12,
                    shader_location: 1,
                },
            ],
        };

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tile_id_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
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
            tile_size,
        }
    }

    /// Get the tile size
    pub fn tile_size(&self) -> u32 {
        self.tile_size
    }

    /// Get the ID texture for readback
    pub fn id_texture(&self) -> &Texture {
        &self.id_texture
    }

    /// Get the ID texture view
    pub fn id_view(&self) -> &TextureView {
        &self.id_view
    }

    /// Get the depth view
    pub fn depth_view(&self) -> &TextureView {
        &self.depth_view
    }

    /// Get the pipeline
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

    /// Begin a tile render pass
    pub fn begin_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tile_id_render_pass"),
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
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        })
    }
}

/// Vertex format for tile ID rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileIdVertex {
    pub position: [f32; 3],
    pub feature_id: u32,
}

impl TileIdVertex {
    /// Create a new tile ID vertex
    pub fn new(position: [f32; 3], feature_id: u32) -> Self {
        Self { position, feature_id }
    }
}

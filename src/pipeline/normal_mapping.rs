//! Normal mapping pipeline implementation
//! 
//! Provides GPU pipeline for rendering meshes with tangent-space normal mapping
//! support using the TBN vertex attributes from the mesh module.

use crate::mesh::{TbnVertex, TbnData};
use glam::{Mat4, Vec3, Vec4};
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferAddress, BufferBinding,
    BufferDescriptor, BufferUsages, Device, Queue, RenderPass, RenderPipeline, RenderPipelineDescriptor,
    SamplerBindingType, ShaderModuleDescriptor, ShaderSource, Texture, TextureFormat, TextureSampleType,
    TextureViewDimension, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode
};
use bytemuck::{Pod, Zeroable};

/// Uniforms for normal mapping pipeline
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct NormalMappingUniforms {
    pub model_matrix: [[f32; 4]; 4],
    pub view_matrix: [[f32; 4]; 4], 
    pub projection_matrix: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 4], // Expanded to 4x4 for alignment
    pub light_direction: [f32; 4], // w component for strength
    pub normal_strength: f32,
    pub _padding: [f32; 3],
}

impl Default for NormalMappingUniforms {
    fn default() -> Self {
        Self {
            model_matrix: Mat4::IDENTITY.to_cols_array_2d(),
            view_matrix: Mat4::IDENTITY.to_cols_array_2d(),
            projection_matrix: Mat4::IDENTITY.to_cols_array_2d(),
            normal_matrix: Mat4::IDENTITY.to_cols_array_2d(),
            light_direction: [0.0, -1.0, 0.0, 1.0], // Default downward light
            normal_strength: 1.0,
            _padding: [0.0; 3],
        }
    }
}

/// Normal mapping render pipeline
pub struct NormalMappingPipeline {
    pipeline: RenderPipeline,
    uniforms_bind_group_layout: BindGroupLayout,
    texture_bind_group_layout: BindGroupLayout,
    uniforms_buffer: Buffer,
    vertex_buffer: Option<Buffer>,
    index_buffer: Option<Buffer>,
    index_count: u32,
}

impl NormalMappingPipeline {
    /// Create a new normal mapping pipeline
    pub fn new(device: &Device, surface_format: TextureFormat) -> Self {
        // Create shader module
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("normal_mapping_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/normal_mapping_vertex.wgsl").into()),
        });

        // Create bind group layouts
        let uniforms_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("normal_mapping_uniforms_layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let texture_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("normal_mapping_texture_layout"),
            entries: &[
                // Normal map texture
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Normal map sampler  
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("normal_mapping_pipeline_layout"),
            bind_group_layouts: &[&uniforms_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Define vertex buffer layout for TBN vertices
        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<TbnVertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                // Position
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x3,
                },
                // UV coordinates
                VertexAttribute {
                    offset: 12,
                    shader_location: 1, 
                    format: VertexFormat::Float32x2,
                },
                // Normal
                VertexAttribute {
                    offset: 20,
                    shader_location: 2,
                    format: VertexFormat::Float32x3,
                },
                // Tangent
                VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: VertexFormat::Float32x3,
                },
                // Bitangent
                VertexAttribute {
                    offset: 44,
                    shader_location: 4,
                    format: VertexFormat::Float32x3,
                },
            ],
        };

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("normal_mapping_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_buffer_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main", 
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
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

        // Create uniforms buffer
        let uniforms_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("normal_mapping_uniforms"),
            size: std::mem::size_of::<NormalMappingUniforms>() as BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            uniforms_bind_group_layout,
            texture_bind_group_layout,
            uniforms_buffer,
            vertex_buffer: None,
            index_buffer: None,
            index_count: 0,
        }
    }

    /// Upload mesh data to GPU buffers
    pub fn upload_mesh(&mut self, device: &Device, vertices: &[TbnVertex], indices: &[u32]) {
        // Create vertex buffer
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("normal_mapping_vertices"),
            size: (vertices.len() * std::mem::size_of::<TbnVertex>()) as BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create index buffer
        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("normal_mapping_indices"),
            size: (indices.len() * std::mem::size_of::<u32>()) as BufferAddress,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
        self.index_count = indices.len() as u32;
    }

    /// Update uniform data
    pub fn update_uniforms(&self, queue: &Queue, uniforms: &NormalMappingUniforms) {
        queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::cast_slice(&[*uniforms]));
    }

    /// Create uniforms bind group
    pub fn create_uniforms_bind_group(&self, device: &Device) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("normal_mapping_uniforms_bind_group"),
            layout: &self.uniforms_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &self.uniforms_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        })
    }

    /// Create texture bind group for normal map
    pub fn create_texture_bind_group(
        &self,
        device: &Device,
        normal_texture: &Texture,
        sampler: &wgpu::Sampler,
    ) -> BindGroup {
        let texture_view = normal_texture.create_view(&wgpu::TextureViewDescriptor::default());

        device.create_bind_group(&BindGroupDescriptor {
            label: Some("normal_mapping_texture_bind_group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    /// Render with the normal mapping pipeline
    pub fn render<'a>(
        &'a self,
        render_pass: &mut RenderPass<'a>,
        uniforms_bind_group: &'a BindGroup,
        texture_bind_group: &'a BindGroup,
    ) {
        if let (Some(vertex_buffer), Some(index_buffer)) = (&self.vertex_buffer, &self.index_buffer) {
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, uniforms_bind_group, &[]);
            render_pass.set_bind_group(1, texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.index_count, 0, 0..1);
        }
    }
}

/// Helper to compute normal matrix from model matrix
pub fn compute_normal_matrix(model_matrix: Mat4) -> Mat4 {
    // Extract upper-left 3x3 and compute inverse-transpose
    // For simplicity, we'll use the full 4x4 inverse-transpose
    // In production, this should be optimized to 3x3 operations
    
    // Check if matrix is invertible by testing determinant
    if model_matrix.determinant().abs() < 1e-8 {
        Mat4::IDENTITY // Fallback for non-invertible matrices
    } else {
        model_matrix.inverse().transpose()
    }
}

/// Create a simple checkerboard normal map texture for testing
pub fn create_checkerboard_normal_texture(device: &Device, queue: &Queue, size: u32) -> Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("checkerboard_normal_texture"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Generate checkerboard normal map data
    let mut data = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let checker = ((x / 8) + (y / 8)) % 2;
            if checker == 0 {
                // Flat normal (pointing up in tangent space)
                data.extend_from_slice(&[128u8, 128u8, 255u8, 255u8]); // (0,0,1) in [0,255] 
            } else {
                // Perturbed normal (slightly tilted)
                data.extend_from_slice(&[148u8, 148u8, 235u8, 255u8]); // Slightly off-normal
            }
        }
    }

    // Upload texture data
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(size * 4),
            rows_per_image: Some(size),
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );

    texture
}
use super::types::{GroundPlaneMode, GroundPlaneParams, GroundPlaneUniforms};
use glam::{Mat4, Vec3};
use std::borrow::Cow;
use wgpu::{
    vertex_attr_array, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer, BufferAddress,
    BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites,
    CompareFunction, DepthBiasState, DepthStencilState, Device, Face, FragmentState, FrontFace,
    MultisampleState, PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology,
    Queue, RenderPipeline, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, StencilState, TextureFormat, VertexBufferLayout, VertexState, VertexStepMode,
};

/// Main ground plane rendering system
pub struct GroundPlaneRenderer {
    pub uniforms: GroundPlaneUniforms,
    pub params: GroundPlaneParams,

    // GPU resources
    pub uniform_buffer: Buffer,
    pub ground_pipeline: RenderPipeline,

    // Bind groups and layouts
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,

    // Geometry
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,

    // State
    pub enabled: bool,
}

impl GroundPlaneRenderer {
    /// Create a new ground plane renderer
    pub fn new(
        device: &Device,
        color_format: TextureFormat,
        depth_format: Option<TextureFormat>,
        sample_count: u32,
    ) -> Self {
        let params = GroundPlaneParams::default();
        let uniforms = GroundPlaneUniforms::default();

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ground_plane_uniform_buffer"),
            size: std::mem::size_of::<GroundPlaneUniforms>() as wgpu::BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ground_plane_bind_group_layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ground_plane_bind_group"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Create shader
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("ground_plane_shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("../../shaders/ground_plane.wgsl"))),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("ground_plane_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Vertex buffer layout
        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 8]>() as BufferAddress, // position(3) + uv(2) + normal(3)
            step_mode: VertexStepMode::Vertex,
            attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3],
        };

        // Create render pipeline
        let ground_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("ground_plane_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_buffer_layout],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: depth_format.map(|format| DepthStencilState {
                format,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual, // Allow ground plane to write depth
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        // Create ground plane geometry
        let (vertex_buffer, index_buffer, index_count) =
            Self::create_ground_plane_geometry(device, params.size);

        Self {
            uniforms,
            params,
            uniform_buffer,
            ground_pipeline,
            bind_group_layout,
            bind_group,
            vertex_buffer,
            index_buffer,
            index_count,
            enabled: true,
        }
    }

    /// Create ground plane geometry (large quad)
    fn create_ground_plane_geometry(device: &Device, size: f32) -> (Buffer, Buffer, u32) {
        let half_size = size * 0.5;

        // Ground plane vertices: position(3) + uv(2) + normal(3)
        let vertices: &[f32] = &[
            // Position                 UV        Normal
            -half_size, 0.0, -half_size, 0.0, 0.0, 0.0, 1.0, 0.0, // Bottom-left
            half_size, 0.0, -half_size, 1.0, 0.0, 0.0, 1.0, 0.0, // Bottom-right
            half_size, 0.0, half_size, 1.0, 1.0, 0.0, 1.0, 0.0, // Top-right
            -half_size, 0.0, half_size, 0.0, 1.0, 0.0, 1.0, 0.0, // Top-left
        ];

        let indices: &[u16] = &[
            0, 1, 2, // First triangle
            2, 3, 0, // Second triangle
        ];

        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ground_plane_vertex_buffer"),
            size: (vertices.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ground_plane_index_buffer"),
            size: (indices.len() * std::mem::size_of::<u16>()) as wgpu::BufferAddress,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(indices));
        index_buffer.unmap();

        (vertex_buffer, index_buffer, indices.len() as u32)
    }

    /// Update ground plane parameters
    pub fn update_params(&mut self, params: GroundPlaneParams) {
        self.params = params;
        self.update_uniforms();
    }

    /// Set ground plane mode
    pub fn set_mode(&mut self, mode: GroundPlaneMode) {
        self.params.mode = mode;
        self.enabled = mode != GroundPlaneMode::Disabled;
        self.update_uniforms();
    }

    /// Set ground plane height
    pub fn set_height(&mut self, height: f32) {
        self.params.height = height;
        self.update_uniforms();
    }

    /// Set ground plane size
    pub fn set_size(&mut self, size: f32) {
        self.params.size = size;
        // Note: Would need to recreate geometry for size changes in a full implementation
        self.update_uniforms();
    }

    /// Set grid spacing
    pub fn set_grid_spacing(&mut self, major: f32, minor: f32) {
        self.params.major_spacing = major;
        self.params.minor_spacing = minor;
        self.update_uniforms();
    }

    /// Set grid line widths
    pub fn set_grid_width(&mut self, major: f32, minor: f32) {
        self.params.major_width = major;
        self.params.minor_width = minor;
        self.update_uniforms();
    }

    /// Set ground plane albedo color
    pub fn set_albedo(&mut self, color: Vec3, alpha: f32) {
        self.params.albedo = color;
        self.params.alpha = alpha;
        self.update_uniforms();
    }

    /// Set grid colors
    pub fn set_grid_colors(
        &mut self,
        major_color: Vec3,
        major_alpha: f32,
        minor_color: Vec3,
        minor_alpha: f32,
    ) {
        self.params.major_grid_color = major_color;
        self.params.major_grid_alpha = major_alpha;
        self.params.minor_grid_color = minor_color;
        self.params.minor_grid_alpha = minor_alpha;
        self.update_uniforms();
    }

    /// Set fading parameters
    pub fn set_fade_params(
        &mut self,
        fade_distance: f32,
        fade_power: f32,
        grid_fade_distance: f32,
        grid_fade_power: f32,
    ) {
        self.params.fade_distance = fade_distance;
        self.params.fade_power = fade_power;
        self.params.grid_fade_distance = grid_fade_distance;
        self.params.grid_fade_power = grid_fade_power;
        self.update_uniforms();
    }

    /// Set z-bias for z-fighting protection
    pub fn set_z_bias(&mut self, z_bias: f32) {
        self.params.z_bias = z_bias;
        self.update_uniforms();
    }

    /// Enable/disable ground plane
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.params.mode = GroundPlaneMode::Disabled;
        } else if self.params.mode == GroundPlaneMode::Disabled {
            self.params.mode = GroundPlaneMode::Grid;
        }
        self.update_uniforms();
    }

    /// Check if ground plane is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled && self.params.mode != GroundPlaneMode::Disabled
    }

    /// Set camera matrices for rendering
    pub fn set_camera(&mut self, view_proj: Mat4) {
        self.uniforms.view_proj = view_proj.to_cols_array_2d();
    }

    /// Update uniforms from parameters
    fn update_uniforms(&mut self) {
        self.uniforms.update_from_params(&self.params);
    }

    /// Upload uniforms to GPU
    pub fn upload_uniforms(&self, queue: &Queue) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    /// Get current ground plane parameters for external access
    pub fn get_params(&self) -> (f32, f32, f32, f32) {
        (
            self.params.height,
            self.params.major_spacing,
            self.params.minor_spacing,
            self.params.z_bias,
        )
    }

    /// Render the ground plane to the current render pass
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if !self.is_enabled() {
            return;
        }

        // Set pipeline and bind group
        render_pass.set_pipeline(&self.ground_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);

        // Set vertex and index buffers
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        // Draw the ground plane
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

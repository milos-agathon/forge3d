// B10: Ground Plane (Raster) - Simple raster ground plane rendering system
// Provides configurable ground plane with grid patterns and z-fighting protection
// Ensures ground plane draws beneath all geometry

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

/// Ground plane rendering modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GroundPlaneMode {
    Disabled,     // No ground plane
    Solid,        // Solid color ground plane
    Grid,         // Grid pattern with major/minor lines
    CheckerBoard, // Checkerboard pattern (future extension)
}

/// Ground plane configuration parameters
#[derive(Debug, Clone)]
pub struct GroundPlaneParams {
    pub mode: GroundPlaneMode,
    pub size: f32,   // Size of the ground plane
    pub height: f32, // Y position of the ground plane
    pub z_bias: f32, // Z-bias to prevent z-fighting

    // Grid parameters
    pub major_spacing: f32, // Spacing between major grid lines
    pub minor_spacing: f32, // Spacing between minor grid lines
    pub major_width: f32,   // Width of major grid lines
    pub minor_width: f32,   // Width of minor grid lines

    // Colors
    pub albedo: Vec3,           // Base ground color (RGB)
    pub alpha: f32,             // Base ground alpha
    pub major_grid_color: Vec3, // Major grid line color
    pub major_grid_alpha: f32,  // Major grid line alpha
    pub minor_grid_color: Vec3, // Minor grid line color
    pub minor_grid_alpha: f32,  // Minor grid line alpha

    // Fading
    pub fade_distance: f32, // Distance at which ground plane starts fading
    pub fade_power: f32,    // Power curve for ground plane fading
    pub grid_fade_distance: f32, // Distance at which grid lines start fading
    pub grid_fade_power: f32, // Power curve for grid line fading
}

impl Default for GroundPlaneParams {
    fn default() -> Self {
        Self {
            mode: GroundPlaneMode::Grid,
            size: 1000.0,   // Large ground plane
            height: 0.0,    // At ground level
            z_bias: 0.0001, // Small bias to prevent z-fighting

            // Grid settings - metric-style grid
            major_spacing: 10.0, // 10 unit major grid
            minor_spacing: 1.0,  // 1 unit minor grid
            major_width: 2.0,    // Moderate major line width
            minor_width: 1.0,    // Thin minor lines

            // Colors - neutral grid
            albedo: Vec3::new(0.2, 0.2, 0.2), // Dark gray ground
            alpha: 0.8,
            major_grid_color: Vec3::new(0.6, 0.6, 0.6), // Light gray major lines
            major_grid_alpha: 0.8,
            minor_grid_color: Vec3::new(0.4, 0.4, 0.4), // Medium gray minor lines
            minor_grid_alpha: 0.4,

            // Fading to prevent grid noise at distance
            fade_distance: 500.0,      // Start fading ground at 500 units
            fade_power: 2.0,           // Quadratic falloff
            grid_fade_distance: 200.0, // Fade grid lines earlier
            grid_fade_power: 1.5,      // Slightly less aggressive fade
        }
    }
}

/// Ground plane uniforms structure (must match WGSL exactly)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GroundPlaneUniforms {
    pub view_proj: [[f32; 4]; 4],       // 64 bytes - View-projection matrix
    pub world_transform: [[f32; 4]; 4], // 64 bytes - World transformation matrix
    pub plane_params: [f32; 4], // 16 bytes - size (x), height (y), grid_enabled (z), z_bias (w)
    pub grid_params: [f32; 4], // 16 bytes - major_spacing (x), minor_spacing (y), major_width (z), minor_width (w)
    pub color_params: [f32; 4], // 16 bytes - albedo (rgb) + alpha (w)
    pub grid_color_params: [f32; 4], // 16 bytes - major_grid_color (rgb) + major_alpha (w)
    pub minor_grid_color_params: [f32; 4], // 16 bytes - minor_grid_color (rgb) + minor_alpha (w)
    pub fade_params: [f32; 4], // 16 bytes - fade_distance (x), fade_power (y), grid_fade_distance (z), grid_fade_power (w)
}

impl Default for GroundPlaneUniforms {
    fn default() -> Self {
        let params = GroundPlaneParams::default();
        let mut uniforms = Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            world_transform: Mat4::IDENTITY.to_cols_array_2d(),
            plane_params: [params.size, params.height, 1.0, params.z_bias], // grid enabled
            grid_params: [
                params.major_spacing,
                params.minor_spacing,
                params.major_width,
                params.minor_width,
            ],
            color_params: [
                params.albedo.x,
                params.albedo.y,
                params.albedo.z,
                params.alpha,
            ],
            grid_color_params: [
                params.major_grid_color.x,
                params.major_grid_color.y,
                params.major_grid_color.z,
                params.major_grid_alpha,
            ],
            minor_grid_color_params: [
                params.minor_grid_color.x,
                params.minor_grid_color.y,
                params.minor_grid_color.z,
                params.minor_grid_alpha,
            ],
            fade_params: [
                params.fade_distance,
                params.fade_power,
                params.grid_fade_distance,
                params.grid_fade_power,
            ],
        };

        // Set world transform to position the ground plane
        uniforms.world_transform =
            Mat4::from_translation(Vec3::new(0.0, params.height, 0.0)).to_cols_array_2d();

        uniforms
    }
}

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
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/ground_plane.wgsl"))),
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
        // Update plane parameters
        self.uniforms.plane_params = [
            self.params.size,
            self.params.height,
            if self.params.mode == GroundPlaneMode::Grid {
                1.0
            } else {
                0.0
            }, // grid_enabled
            self.params.z_bias,
        ];

        // Update grid parameters
        self.uniforms.grid_params = [
            self.params.major_spacing,
            self.params.minor_spacing,
            self.params.major_width,
            self.params.minor_width,
        ];

        // Update color parameters
        self.uniforms.color_params = [
            self.params.albedo.x,
            self.params.albedo.y,
            self.params.albedo.z,
            self.params.alpha,
        ];

        self.uniforms.grid_color_params = [
            self.params.major_grid_color.x,
            self.params.major_grid_color.y,
            self.params.major_grid_color.z,
            self.params.major_grid_alpha,
        ];

        self.uniforms.minor_grid_color_params = [
            self.params.minor_grid_color.x,
            self.params.minor_grid_color.y,
            self.params.minor_grid_color.z,
            self.params.minor_grid_alpha,
        ];

        // Update fade parameters
        self.uniforms.fade_params = [
            self.params.fade_distance,
            self.params.fade_power,
            self.params.grid_fade_distance,
            self.params.grid_fade_power,
        ];

        // Update world transform
        self.uniforms.world_transform =
            Mat4::from_translation(Vec3::new(0.0, self.params.height, 0.0)).to_cols_array_2d();
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

    /// Create preset configurations
    pub fn create_engineering_grid() -> GroundPlaneParams {
        GroundPlaneParams {
            mode: GroundPlaneMode::Grid,
            major_spacing: 10.0,
            minor_spacing: 1.0,
            major_grid_color: Vec3::new(0.0, 0.8, 0.0), // Green major lines
            minor_grid_color: Vec3::new(0.0, 0.4, 0.0), // Dark green minor lines
            albedo: Vec3::new(0.05, 0.05, 0.05),        // Nearly black background
            ..Default::default()
        }
    }

    pub fn create_architectural_grid() -> GroundPlaneParams {
        GroundPlaneParams {
            mode: GroundPlaneMode::Grid,
            major_spacing: 5.0, // 5 meter grid for architecture
            minor_spacing: 1.0, // 1 meter subdivisions
            major_grid_color: Vec3::new(0.6, 0.6, 0.8), // Light blue major lines
            minor_grid_color: Vec3::new(0.4, 0.4, 0.6), // Darker blue minor lines
            albedo: Vec3::new(0.9, 0.9, 0.95), // Light background
            major_width: 1.5,
            minor_width: 0.8,
            ..Default::default()
        }
    }

    pub fn create_simple_ground() -> GroundPlaneParams {
        GroundPlaneParams {
            mode: GroundPlaneMode::Solid,
            albedo: Vec3::new(0.3, 0.25, 0.2), // Brown earth color
            alpha: 1.0,
            ..Default::default()
        }
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

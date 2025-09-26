// B12: Soft Light Radius (Raster) - Core Rust implementation
// Provides soft light radius control with configurable falloff for raster lighting

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Light falloff modes for soft radius control
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftLightFalloffMode {
    Linear = 0,
    Quadratic = 1,
    Cubic = 2,
    Exponential = 3,
}

impl Default for SoftLightFalloffMode {
    fn default() -> Self {
        Self::Quadratic
    }
}

/// Uniform data structure for soft light radius (matches WGSL layout exactly)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SoftLightRadiusUniforms {
    // Light position and intensity (16 bytes)
    pub light_position: [f32; 3],
    pub light_intensity: f32,

    // Radius parameters (16 bytes)
    pub inner_radius: f32,     // Distance where light starts to falloff
    pub outer_radius: f32,     // Distance where light reaches zero
    pub falloff_exponent: f32, // Controls falloff curve steepness
    pub edge_softness: f32,    // Additional softening factor for edges

    // Color and control (16 bytes)
    pub light_color: [f32; 3],
    pub enabled: f32, // 0.0=disabled, 1.0=enabled

    // Quality and modes (16 bytes)
    pub falloff_mode: u32,    // SoftLightFalloffMode as u32
    pub shadow_softness: f32, // Softness for shadow edges
    pub _pad0: f32,
    pub _pad1: f32,
}

impl Default for SoftLightRadiusUniforms {
    fn default() -> Self {
        Self {
            light_position: [0.0, 10.0, 0.0],
            light_intensity: 1.0,
            inner_radius: 5.0,
            outer_radius: 20.0,
            falloff_exponent: 2.0,
            edge_softness: 1.0,
            light_color: [1.0, 1.0, 1.0],
            enabled: 1.0,
            falloff_mode: SoftLightFalloffMode::Quadratic as u32,
            shadow_softness: 0.5,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

/// Predefined soft light configurations
#[derive(Debug, Clone, Copy)]
pub enum SoftLightPreset {
    /// Harsh spotlight with sharp edges
    Spotlight,
    /// Soft area light with gentle falloff
    AreaLight,
    /// Ambient light with very soft edges
    AmbientLight,
    /// Candle-like point light
    Candle,
    /// Street lamp with medium softness
    StreetLamp,
}

impl SoftLightPreset {
    pub fn to_uniforms(self) -> SoftLightRadiusUniforms {
        match self {
            Self::Spotlight => SoftLightRadiusUniforms {
                light_position: [0.0, 15.0, 0.0],
                light_intensity: 2.0,
                inner_radius: 2.0,
                outer_radius: 15.0,
                falloff_exponent: 4.0,
                edge_softness: 0.2,
                light_color: [1.0, 1.0, 0.9],
                enabled: 1.0,
                falloff_mode: SoftLightFalloffMode::Exponential as u32,
                shadow_softness: 0.1,
                _pad0: 0.0,
                _pad1: 0.0,
            },
            Self::AreaLight => SoftLightRadiusUniforms {
                light_position: [0.0, 8.0, 0.0],
                light_intensity: 1.5,
                inner_radius: 8.0,
                outer_radius: 25.0,
                falloff_exponent: 1.5,
                edge_softness: 3.0,
                light_color: [1.0, 0.95, 0.9],
                enabled: 1.0,
                falloff_mode: SoftLightFalloffMode::Quadratic as u32,
                shadow_softness: 0.8,
                _pad0: 0.0,
                _pad1: 0.0,
            },
            Self::AmbientLight => SoftLightRadiusUniforms {
                light_position: [0.0, 20.0, 0.0],
                light_intensity: 0.8,
                inner_radius: 15.0,
                outer_radius: 50.0,
                falloff_exponent: 1.0,
                edge_softness: 5.0,
                light_color: [0.9, 0.95, 1.0],
                enabled: 1.0,
                falloff_mode: SoftLightFalloffMode::Linear as u32,
                shadow_softness: 1.0,
                _pad0: 0.0,
                _pad1: 0.0,
            },
            Self::Candle => SoftLightRadiusUniforms {
                light_position: [0.0, 2.0, 0.0],
                light_intensity: 1.2,
                inner_radius: 1.0,
                outer_radius: 8.0,
                falloff_exponent: 3.0,
                edge_softness: 0.5,
                light_color: [1.0, 0.7, 0.4],
                enabled: 1.0,
                falloff_mode: SoftLightFalloffMode::Cubic as u32,
                shadow_softness: 0.3,
                _pad0: 0.0,
                _pad1: 0.0,
            },
            Self::StreetLamp => SoftLightRadiusUniforms {
                light_position: [0.0, 12.0, 0.0],
                light_intensity: 1.8,
                inner_radius: 5.0,
                outer_radius: 30.0,
                falloff_exponent: 2.5,
                edge_softness: 2.0,
                light_color: [1.0, 0.9, 0.7],
                enabled: 1.0,
                falloff_mode: SoftLightFalloffMode::Quadratic as u32,
                shadow_softness: 0.6,
                _pad0: 0.0,
                _pad1: 0.0,
            },
        }
    }
}

/// Core renderer for soft light radius effects
pub struct SoftLightRadiusRenderer {
    pipeline: wgpu::RenderPipeline,
    _multiple_lights_pipeline: wgpu::RenderPipeline,
    soft_shadows_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniforms_buffer: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    uniforms: SoftLightRadiusUniforms,
}

impl SoftLightRadiusRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soft_light_radius_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soft_light_radius.wgsl").into(),
            ),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soft_light_radius_bind_group_layout"),
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("soft_light_radius_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline for single light
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("soft_light_radius_pipeline"),
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
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
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

        // Create pipeline for multiple lights
        let multiple_lights_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("soft_light_radius_multiple_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_multiple_lights",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
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

        // Create pipeline for soft shadows
        let soft_shadows_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("soft_light_radius_shadows_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_soft_shadows",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
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

        // Create uniforms buffer
        let uniforms = SoftLightRadiusUniforms::default();
        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("soft_light_radius_uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            _multiple_lights_pipeline: multiple_lights_pipeline,
            soft_shadows_pipeline,
            bind_group_layout,
            uniforms_buffer,
            bind_group: None,
            uniforms,
        }
    }

    /// Set light position
    pub fn set_light_position(&mut self, position: [f32; 3]) {
        self.uniforms.light_position = position;
    }

    /// Set light intensity
    pub fn set_light_intensity(&mut self, intensity: f32) {
        self.uniforms.light_intensity = intensity;
    }

    /// Set light color
    pub fn set_light_color(&mut self, color: [f32; 3]) {
        self.uniforms.light_color = color;
    }

    /// Set inner radius where light starts to falloff
    pub fn set_inner_radius(&mut self, radius: f32) {
        self.uniforms.inner_radius = radius.max(0.0);
    }

    /// Set outer radius where light reaches zero
    pub fn set_outer_radius(&mut self, radius: f32) {
        self.uniforms.outer_radius = radius.max(self.uniforms.inner_radius + 0.1);
    }

    /// Set falloff exponent for curve control
    pub fn set_falloff_exponent(&mut self, exponent: f32) {
        self.uniforms.falloff_exponent = exponent.max(0.1);
    }

    /// Set edge softness factor
    pub fn set_edge_softness(&mut self, softness: f32) {
        self.uniforms.edge_softness = softness.max(0.0);
    }

    /// Set falloff mode
    pub fn set_falloff_mode(&mut self, mode: SoftLightFalloffMode) {
        self.uniforms.falloff_mode = mode as u32;
    }

    /// Set shadow softness
    pub fn set_shadow_softness(&mut self, softness: f32) {
        self.uniforms.shadow_softness = softness.clamp(0.0, 2.0);
    }

    /// Enable or disable soft light radius
    pub fn set_enabled(&mut self, enabled: bool) {
        self.uniforms.enabled = if enabled { 1.0 } else { 0.0 };
    }

    /// Apply a preset configuration
    pub fn apply_preset(&mut self, preset: SoftLightPreset) {
        self.uniforms = preset.to_uniforms();
    }

    /// Update uniforms buffer
    pub fn update_uniforms(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.uniforms_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    /// Create bind group with depth texture
    pub fn create_bind_group(
        &mut self,
        device: &wgpu::Device,
        depth_texture: &wgpu::TextureView,
        depth_sampler: &wgpu::Sampler,
    ) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soft_light_radius_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(depth_sampler),
                },
            ],
        }));
    }

    /// Render soft light radius effect
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, use_soft_shadows: bool) {
        if let Some(bind_group) = &self.bind_group {
            let pipeline = if use_soft_shadows {
                &self.soft_shadows_pipeline
            } else {
                &self.pipeline
            };

            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass.draw(0..3, 0..1); // Full-screen triangle
        }
    }

    /// Get current uniforms for inspection
    pub fn uniforms(&self) -> &SoftLightRadiusUniforms {
        &self.uniforms
    }

    /// Calculate effective light range
    pub fn effective_range(&self) -> f32 {
        self.uniforms.outer_radius + self.uniforms.edge_softness
    }

    /// Check if point is within light influence
    pub fn affects_point(&self, point: [f32; 3]) -> bool {
        let dx = point[0] - self.uniforms.light_position[0];
        let dy = point[1] - self.uniforms.light_position[1];
        let dz = point[2] - self.uniforms.light_position[2];
        let distance_sq = dx * dx + dy * dy + dz * dz;
        let range = self.effective_range();
        distance_sq <= range * range
    }
}

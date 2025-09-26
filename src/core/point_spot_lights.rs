// B13: Point & Spot Lights (Realtime) - Core Rust implementation
// Provides point and spot light management with shadows and penumbra shaping

use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Light types for point and spot lights
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    Point = 0,
    Spot = 1,
}

impl Default for LightType {
    fn default() -> Self {
        Self::Point
    }
}

/// Shadow quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowQuality {
    Off = 0,
    Low = 1,
    Medium = 2,
    High = 3,
}

impl Default for ShadowQuality {
    fn default() -> Self {
        Self::Medium
    }
}

/// Debug visualization modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugMode {
    Normal = 0,
    ShowLightBounds = 1,
    ShowShadows = 2,
}

impl Default for DebugMode {
    fn default() -> Self {
        Self::Normal
    }
}

/// Individual light data (matches WGSL layout exactly - 64 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Light {
    // Position and type (16 bytes)
    pub position: [f32; 3],
    pub light_type: u32, // LightType as u32

    // Direction and range (16 bytes)
    pub direction: [f32; 3], // For spot lights (normalized)
    pub range: f32,          // Maximum light distance

    // Color and intensity (16 bytes)
    pub color: [f32; 3],
    pub intensity: f32,

    // Spot light parameters (16 bytes)
    pub inner_cone_angle: f32,  // Inner cone angle (radians)
    pub outer_cone_angle: f32,  // Outer cone angle (radians)
    pub penumbra_softness: f32, // Penumbra transition softness (0.1-2.0)
    pub shadow_enabled: f32,    // 0.0 = disabled, 1.0 = enabled
}

impl Default for Light {
    fn default() -> Self {
        Self {
            position: [0.0, 5.0, 0.0],
            light_type: LightType::Point as u32,
            direction: [0.0, -1.0, 0.0], // Pointing down
            range: 20.0,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            inner_cone_angle: 30.0f32.to_radians(),
            outer_cone_angle: 45.0f32.to_radians(),
            penumbra_softness: 1.0,
            shadow_enabled: 1.0,
        }
    }
}

impl Light {
    /// Create a new point light
    pub fn point(position: [f32; 3], color: [f32; 3], intensity: f32, range: f32) -> Self {
        Self {
            position,
            light_type: LightType::Point as u32,
            direction: [0.0, -1.0, 0.0], // Direction not used for point lights
            range,
            color,
            intensity,
            inner_cone_angle: 0.0,
            outer_cone_angle: 0.0,
            penumbra_softness: 1.0,
            shadow_enabled: 1.0,
        }
    }

    /// Create a new spot light
    pub fn spot(
        position: [f32; 3],
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        range: f32,
        inner_cone_deg: f32,
        outer_cone_deg: f32,
        penumbra_softness: f32,
    ) -> Self {
        // Normalize direction
        let dir_len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        let normalized_dir = if dir_len > 0.0 {
            [
                direction[0] / dir_len,
                direction[1] / dir_len,
                direction[2] / dir_len,
            ]
        } else {
            [0.0, -1.0, 0.0]
        };

        Self {
            position,
            light_type: LightType::Spot as u32,
            direction: normalized_dir,
            range,
            color,
            intensity,
            inner_cone_angle: inner_cone_deg.to_radians(),
            outer_cone_angle: outer_cone_deg.to_radians(),
            penumbra_softness: penumbra_softness.clamp(0.1, 5.0),
            shadow_enabled: 1.0,
        }
    }

    /// Set light position
    pub fn set_position(&mut self, position: [f32; 3]) {
        self.position = position;
    }

    /// Set light direction (for spot lights)
    pub fn set_direction(&mut self, direction: [f32; 3]) {
        let dir_len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        if dir_len > 0.0 {
            self.direction = [
                direction[0] / dir_len,
                direction[1] / dir_len,
                direction[2] / dir_len,
            ];
        }
    }

    /// Set light color
    pub fn set_color(&mut self, color: [f32; 3]) {
        self.color = color;
    }

    /// Set light intensity
    pub fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity.max(0.0);
    }

    /// Set light range
    pub fn set_range(&mut self, range: f32) {
        self.range = range.max(0.1);
    }

    /// Set spot light cone angles
    pub fn set_cone_angles(&mut self, inner_deg: f32, outer_deg: f32) {
        let inner_rad = inner_deg.to_radians().clamp(0.0, std::f32::consts::PI);
        let outer_rad = outer_deg
            .to_radians()
            .clamp(inner_rad, std::f32::consts::PI);
        self.inner_cone_angle = inner_rad;
        self.outer_cone_angle = outer_rad;
    }

    /// Set penumbra softness
    pub fn set_penumbra_softness(&mut self, softness: f32) {
        self.penumbra_softness = softness.clamp(0.1, 5.0);
    }

    /// Enable or disable shadows for this light
    pub fn set_shadow_enabled(&mut self, enabled: bool) {
        self.shadow_enabled = if enabled { 1.0 } else { 0.0 };
    }

    /// Check if point is within light range
    pub fn affects_point(&self, point: [f32; 3]) -> bool {
        let dx = point[0] - self.position[0];
        let dy = point[1] - self.position[1];
        let dz = point[2] - self.position[2];
        let distance_sq = dx * dx + dy * dy + dz * dz;
        distance_sq <= self.range * self.range
    }
}

/// Per-frame uniforms (matches WGSL layout - 128 bytes total)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PointSpotLightUniforms {
    // Camera and view parameters (64 bytes)
    pub view_matrix: [[f32; 4]; 4],
    pub proj_matrix: [[f32; 4]; 4],

    // Global lighting (16 bytes)
    pub ambient_color: [f32; 3],
    pub ambient_intensity: f32,

    // Light count and control (16 bytes)
    pub active_light_count: u32,
    pub max_lights: u32,
    pub shadow_quality: u32, // ShadowQuality as u32
    pub debug_mode: u32,     // DebugMode as u32

    // Global shadow parameters (16 bytes)
    pub shadow_bias: f32,
    pub shadow_normal_bias: f32,
    pub shadow_softness: f32,
    pub _pad0: f32,

    // Additional padding to reach 128 bytes (16 bytes)
    pub _pad1: [f32; 4],
}

impl Default for PointSpotLightUniforms {
    fn default() -> Self {
        Self {
            view_matrix: glam::Mat4::IDENTITY.to_cols_array_2d(),
            proj_matrix: glam::Mat4::IDENTITY.to_cols_array_2d(),
            ambient_color: [0.2, 0.2, 0.3],
            ambient_intensity: 0.3,
            active_light_count: 0,
            max_lights: 32,
            shadow_quality: ShadowQuality::Medium as u32,
            debug_mode: DebugMode::Normal as u32,
            shadow_bias: 0.001,
            shadow_normal_bias: 0.01,
            shadow_softness: 0.5,
            _pad0: 0.0,
            _pad1: [0.0; 4],
        }
    }
}

/// Predefined light configurations
#[derive(Debug, Clone, Copy)]
pub enum LightPreset {
    /// Standard room light
    RoomLight,
    /// Bright desk lamp
    DeskLamp,
    /// Outdoor street light
    StreetLight,
    /// Spotlight for stage/theater
    Spotlight,
    /// Car headlight
    Headlight,
    /// Flashlight
    Flashlight,
    /// Candle flame
    Candle,
    /// Warm living room lamp
    WarmLamp,
}

impl LightPreset {
    pub fn to_light(self, position: [f32; 3]) -> Light {
        match self {
            Self::RoomLight => Light::point(
                position,
                [1.0, 0.95, 0.9], // Warm white
                2.0,              // Medium intensity
                15.0,             // Good room coverage
            ),
            Self::DeskLamp => Light::spot(
                position,
                [0.0, -1.0, 0.2], // Slightly forward
                [1.0, 1.0, 0.95], // Cool white
                3.0,              // Bright
                8.0,              // Focused range
                25.0,             // Inner cone
                40.0,             // Outer cone
                0.8,              // Medium softness
            ),
            Self::StreetLight => Light::point(
                position,
                [1.0, 0.85, 0.6], // Orange-ish
                4.0,              // Very bright
                30.0,             // Large coverage
            ),
            Self::Spotlight => Light::spot(
                position,
                [0.0, -1.0, 0.0], // Straight down
                [1.0, 1.0, 1.0],  // Pure white
                5.0,              // Very bright
                20.0,             // Long range
                15.0,             // Tight inner cone
                25.0,             // Tight outer cone
                0.3,              // Sharp edge
            ),
            Self::Headlight => Light::spot(
                position,
                [0.0, 0.0, -1.0], // Forward
                [1.0, 1.0, 0.95], // Cool white
                4.0,              // Bright
                50.0,             // Long range
                20.0,             // Inner cone
                35.0,             // Outer cone
                0.6,              // Medium softness
            ),
            Self::Flashlight => Light::spot(
                position,
                [0.0, 0.0, -1.0], // Forward
                [1.0, 1.0, 0.9],  // Slightly warm
                2.5,              // Medium-bright
                25.0,             // Good range
                12.0,             // Tight inner
                30.0,             // Wider outer
                1.2,              // Soft edge
            ),
            Self::Candle => Light::point(
                position,
                [1.0, 0.6, 0.2], // Warm orange
                0.8,             // Dim
                3.0,             // Small range
            ),
            Self::WarmLamp => Light::point(
                position,
                [1.0, 0.8, 0.6], // Very warm
                1.5,             // Cozy intensity
                12.0,            // Medium range
            ),
        }
    }
}

/// Core renderer for point and spot lights
pub struct PointSpotLightRenderer {
    // Rendering pipelines
    deferred_pipeline: wgpu::RenderPipeline,
    _forward_pipeline: wgpu::RenderPipeline,
    _shadow_pipeline: wgpu::RenderPipeline,

    // Bind group layouts
    main_bind_group_layout: wgpu::BindGroupLayout,
    shadow_bind_group_layout: wgpu::BindGroupLayout,

    // Buffers
    uniforms_buffer: wgpu::Buffer,
    lights_buffer: wgpu::Buffer,

    // Shadow mapping resources
    _shadow_map_array: Option<wgpu::Texture>,
    shadow_map_view: Option<wgpu::TextureView>,
    shadow_sampler: wgpu::Sampler,

    // Bind groups
    main_bind_group: Option<wgpu::BindGroup>,
    shadow_bind_group: Option<wgpu::BindGroup>,

    // Light management
    lights: Vec<Light>,
    light_id_counter: u32,
    light_id_map: HashMap<u32, usize>,
    max_lights: usize,

    // Configuration
    uniforms: PointSpotLightUniforms,
    _shadow_map_size: u32,
}

impl PointSpotLightRenderer {
    pub fn new(device: &wgpu::Device, max_lights: usize) -> Self {
        let max_lights = max_lights.min(64); // Cap at reasonable limit

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point_spot_lights_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/point_spot_lights.wgsl").into(),
            ),
        });

        // Create bind group layouts
        let main_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("point_spot_lights_main_layout"),
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
                    // Lights storage buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // G-buffer albedo
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // G-buffer normal
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // G-buffer depth
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // G-buffer sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let shadow_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("point_spot_lights_shadow_layout"),
                entries: &[
                    // Shadow map array
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Shadow sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
            });

        // Create pipeline layouts
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("point_spot_lights_pipeline_layout"),
            bind_group_layouts: &[&main_bind_group_layout, &shadow_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create deferred lighting pipeline
        let deferred_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point_spot_lights_deferred_pipeline"),
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
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create forward rendering pipeline (simplified vertex layout)
        let forward_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point_spot_lights_forward_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_forward",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 32, // position(12) + normal(12) + uv(8)
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
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 24,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_forward",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
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

        // Create shadow pipeline layout and pipeline
        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_pipeline_layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point_spot_lights_shadow_pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_shadow",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12, // position only
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    }],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_shadow",
                targets: &[],
            }),
            primitive: wgpu::PrimitiveState::default(),
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

        // Create buffers
        let mut uniforms = PointSpotLightUniforms::default();
        uniforms.max_lights = max_lights as u32;

        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("point_spot_lights_uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create lights storage buffer (initially empty)
        let lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("point_spot_lights_buffer"),
            size: (max_lights * std::mem::size_of::<Light>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create shadow sampler
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        Self {
            deferred_pipeline,
            _forward_pipeline: forward_pipeline,
            _shadow_pipeline: shadow_pipeline,
            main_bind_group_layout,
            shadow_bind_group_layout,
            uniforms_buffer,
            lights_buffer,
            _shadow_map_array: None,
            shadow_map_view: None,
            shadow_sampler,
            main_bind_group: None,
            shadow_bind_group: None,
            lights: Vec::new(),
            light_id_counter: 0,
            light_id_map: HashMap::new(),
            max_lights,
            uniforms,
            _shadow_map_size: 1024,
        }
    }

    /// Add a light and return its ID
    pub fn add_light(&mut self, light: Light) -> u32 {
        if self.lights.len() >= self.max_lights {
            return u32::MAX; // No more space
        }

        let light_id = self.light_id_counter;
        self.light_id_counter += 1;

        let index = self.lights.len();
        self.lights.push(light);
        self.light_id_map.insert(light_id, index);

        self.uniforms.active_light_count = self.lights.len() as u32;

        light_id
    }

    /// Remove a light by ID
    pub fn remove_light(&mut self, light_id: u32) -> bool {
        if let Some(index) = self.light_id_map.remove(&light_id) {
            self.lights.remove(index);

            // Update indices in the map
            for (_, idx) in self.light_id_map.iter_mut() {
                if *idx > index {
                    *idx -= 1;
                }
            }

            self.uniforms.active_light_count = self.lights.len() as u32;
            true
        } else {
            false
        }
    }

    /// Get a mutable reference to a light by ID
    pub fn get_light_mut(&mut self, light_id: u32) -> Option<&mut Light> {
        if let Some(index) = self.light_id_map.get(&light_id) {
            self.lights.get_mut(*index)
        } else {
            None
        }
    }

    /// Get a reference to a light by ID
    pub fn get_light(&self, light_id: u32) -> Option<&Light> {
        if let Some(index) = self.light_id_map.get(&light_id) {
            self.lights.get(*index)
        } else {
            None
        }
    }

    /// Clear all lights
    pub fn clear_lights(&mut self) {
        self.lights.clear();
        self.light_id_map.clear();
        self.uniforms.active_light_count = 0;
    }

    /// Get number of active lights
    pub fn light_count(&self) -> usize {
        self.lights.len()
    }

    /// Set camera matrices
    pub fn set_camera(&mut self, view: glam::Mat4, proj: glam::Mat4) {
        self.uniforms.view_matrix = view.to_cols_array_2d();
        self.uniforms.proj_matrix = proj.to_cols_array_2d();
    }

    /// Set ambient lighting
    pub fn set_ambient(&mut self, color: [f32; 3], intensity: f32) {
        self.uniforms.ambient_color = color;
        self.uniforms.ambient_intensity = intensity;
    }

    /// Set shadow quality
    pub fn set_shadow_quality(&mut self, quality: ShadowQuality) {
        self.uniforms.shadow_quality = quality as u32;
    }

    /// Set debug mode
    pub fn set_debug_mode(&mut self, mode: DebugMode) {
        self.uniforms.debug_mode = mode as u32;
    }

    /// Set shadow parameters
    pub fn set_shadow_parameters(&mut self, bias: f32, normal_bias: f32, softness: f32) {
        self.uniforms.shadow_bias = bias;
        self.uniforms.shadow_normal_bias = normal_bias;
        self.uniforms.shadow_softness = softness;
    }

    /// Update uniforms and lights buffers
    pub fn update_buffers(&self, queue: &wgpu::Queue) {
        // Update uniforms
        queue.write_buffer(
            &self.uniforms_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );

        // Update lights buffer
        if !self.lights.is_empty() {
            queue.write_buffer(&self.lights_buffer, 0, bytemuck::cast_slice(&self.lights));
        }
    }

    /// Create bind groups with G-buffer textures
    pub fn create_bind_groups(
        &mut self,
        device: &wgpu::Device,
        albedo_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        g_buffer_sampler: &wgpu::Sampler,
    ) {
        self.main_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point_spot_lights_main_bind_group"),
            layout: &self.main_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(g_buffer_sampler),
                },
            ],
        }));

        // Create shadow bind group if shadow map exists
        if let Some(shadow_view) = &self.shadow_map_view {
            self.shadow_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("point_spot_lights_shadow_bind_group"),
                layout: &self.shadow_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(shadow_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                    },
                ],
            }));
        }
    }

    /// Render lights using deferred shading
    pub fn render_deferred<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.lights.is_empty() {
            return;
        }

        if let (Some(main_bind_group), Some(shadow_bind_group)) =
            (&self.main_bind_group, &self.shadow_bind_group)
        {
            render_pass.set_pipeline(&self.deferred_pipeline);
            render_pass.set_bind_group(0, main_bind_group, &[]);
            render_pass.set_bind_group(1, shadow_bind_group, &[]);
            render_pass.draw(0..3, 0..1); // Full-screen triangle
        }
    }

    /// Get current uniforms for inspection
    pub fn uniforms(&self) -> &PointSpotLightUniforms {
        &self.uniforms
    }

    /// Get all lights
    pub fn lights(&self) -> &[Light] {
        &self.lights
    }
}

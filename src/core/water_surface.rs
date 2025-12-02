// B11: Water Surface Color Toggle - Configurable water surface rendering system
// Provides pipeline uniform controlling water albedo/hue with Python setter
// Supports water tint toggling, transparency, and basic wave animation

use glam::{Mat4, Vec2, Vec3};
use std::borrow::Cow;
use wgpu::{
    vertex_attr_array, AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry,
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BlendComponent,
    BlendFactor, BlendOperation, BlendState, Buffer, BufferAddress, BufferBindingType,
    BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Device, Extent3d, FilterMode,
    FragmentState, ImageCopyTexture, ImageDataLayout, Origin3d, PipelineLayoutDescriptor,
    PrimitiveState, PrimitiveTopology, Queue, RenderPipeline, RenderPipelineDescriptor, Sampler,
    SamplerBindingType, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureView, TextureViewDescriptor, TextureViewDimension, VertexBufferLayout, VertexState,
    VertexStepMode,
};

/// Water surface rendering modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaterSurfaceMode {
    Disabled,    // No water surface
    Transparent, // Standard transparent water
    Reflective,  // Enhanced reflective water
    Animated,    // Animated water with waves
}

/// Water surface configuration parameters
#[derive(Debug, Clone)]
pub struct WaterSurfaceParams {
    pub mode: WaterSurfaceMode,
    pub size: f32,   // Size of the water surface
    pub height: f32, // Y position of the water surface
    pub alpha: f32,  // Water transparency

    // Color parameters
    pub base_color: Vec3,   // Base water color (RGB)
    pub hue_shift: f32,     // Hue shift amount in radians
    pub tint_color: Vec3,   // Tint color overlay
    pub tint_strength: f32, // Strength of tint blending

    // Wave animation parameters
    pub wave_amplitude: f32,  // Height of waves
    pub wave_frequency: f32,  // Frequency of wave pattern
    pub wave_speed: f32,      // Speed of wave animation
    pub ripple_scale: f32,    // Scale of surface ripples
    pub ripple_speed: f32,    // Speed of ripple animation
    pub flow_direction: Vec2, // Direction of water flow

    // Lighting parameters
    pub reflection_strength: f32, // Strength of reflection effect
    pub refraction_strength: f32, // Strength of refraction effect
    pub fresnel_power: f32,       // Fresnel effect power
    pub roughness: f32,           // Surface roughness

    // Foam overlay (screen-space, uses mask texture if provided)
    pub foam_enabled: bool,
    pub foam_width_px: f32,    // hint only (used to scale effect)
    pub foam_intensity: f32,   // blend weight
    pub foam_noise_scale: f32, // procedural breakup

    // Debug mode (0 = normal, 100 = water mask, 101 = shore-distance, 102 = IBL spec)
    pub debug_mode: u32,
}

impl Default for WaterSurfaceParams {
    fn default() -> Self {
        Self {
            mode: WaterSurfaceMode::Transparent,
            size: 1000.0, // Large water surface
            height: 0.0,  // At water level
            alpha: 0.7,   // Semi-transparent

            // Default blue water color
            base_color: Vec3::new(0.1, 0.3, 0.6), // Deep blue
            hue_shift: 0.0,                       // No hue shift initially
            tint_color: Vec3::new(0.0, 0.8, 1.0), // Light blue tint
            tint_strength: 0.2,                   // Subtle tint

            // Gentle wave animation
            wave_amplitude: 0.1,                 // Small waves
            wave_frequency: 2.0,                 // Moderate frequency
            wave_speed: 1.0,                     // Slow animation
            ripple_scale: 1.0,                   // Normal ripple scale
            ripple_speed: 0.5,                   // Slow ripples
            flow_direction: Vec2::new(1.0, 0.0), // Flow along X axis

            // Realistic water lighting
            reflection_strength: 0.8, // Strong reflections
            refraction_strength: 0.3, // Moderate refraction
            fresnel_power: 5.0,       // Standard fresnel
            roughness: 0.1,           // Smooth water surface

            // Foam defaults
            foam_enabled: false,
            foam_width_px: 2.0,
            foam_intensity: 0.85,
            foam_noise_scale: 20.0,

            // Debug mode
            debug_mode: 0,
        }
    }
}

/// Water surface uniforms structure (must match WGSL exactly)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterSurfaceUniforms {
    pub view_proj: [[f32; 4]; 4],       // 64 bytes - View-projection matrix
    pub world_transform: [[f32; 4]; 4], // 64 bytes - World transformation matrix
    pub surface_params: [f32; 4],       // 16 bytes - size (x), height (y), enabled (z), alpha (w)
    pub color_params: [f32; 4],         // 16 bytes - base_color (rgb) + hue_shift (w)
    pub wave_params: [f32; 4], // 16 bytes - wave_amplitude (x), wave_frequency (y), wave_speed (z), time (w)
    pub tint_params: [f32; 4], // 16 bytes - tint_color (rgb) + tint_strength (w)
    pub lighting_params: [f32; 4], // 16 bytes - reflection_strength (x), refraction_strength (y), fresnel_power (z), roughness (w)
    pub animation_params: [f32; 4], // 16 bytes - ripple_scale (x), ripple_speed (y), flow_direction (zw)
    pub foam_params: [f32; 4], // 16 bytes - foam_width_px (x), foam_intensity (y), foam_noise_scale (z), mask_enabled (w)
    pub debug_params: [f32; 4], // 16 bytes - debug_mode (x), reserved (yzw)
}

impl Default for WaterSurfaceUniforms {
    fn default() -> Self {
        let params = WaterSurfaceParams::default();
        let mut uniforms = Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            world_transform: Mat4::IDENTITY.to_cols_array_2d(),
            surface_params: [params.size, params.height, 1.0, params.alpha], // enabled
            color_params: [
                params.base_color.x,
                params.base_color.y,
                params.base_color.z,
                params.hue_shift,
            ],
            wave_params: [
                params.wave_amplitude,
                params.wave_frequency,
                params.wave_speed,
                0.0, // time (updated each frame)
            ],
            tint_params: [
                params.tint_color.x,
                params.tint_color.y,
                params.tint_color.z,
                params.tint_strength,
            ],
            lighting_params: [
                params.reflection_strength,
                params.refraction_strength,
                params.fresnel_power,
                params.roughness,
            ],
            animation_params: [
                params.ripple_scale,
                params.ripple_speed,
                params.flow_direction.x,
                params.flow_direction.y,
            ],
            foam_params: [
                params.foam_width_px,
                params.foam_intensity,
                params.foam_noise_scale,
                0.0, // mask_enabled off by default
            ],
            debug_params: [0.0, 0.0, 0.0, 0.0], // debug_mode = 0 (normal)
        };

        // Set world transform to position the water surface
        uniforms.world_transform =
            Mat4::from_translation(Vec3::new(0.0, params.height, 0.0)).to_cols_array_2d();

        uniforms
    }
}

/// Main water surface rendering system
pub struct WaterSurfaceRenderer {
    pub uniforms: WaterSurfaceUniforms,
    pub params: WaterSurfaceParams,

    // GPU resources
    pub uniform_buffer: Buffer,
    pub water_pipeline: RenderPipeline,

    // Bind groups and layouts
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,

    // Optional water mask (R8Unorm), always bound (defaults to 1x1)
    pub mask_bind_group_layout: BindGroupLayout,
    pub mask_bind_group: BindGroup,
    pub mask_texture: Texture,
    pub mask_view: TextureView,
    pub mask_sampler: Sampler,
    pub mask_size: (u32, u32),

    // Geometry
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,

    // Animation
    pub animation_time: f32,

    // State
    pub enabled: bool,
}

impl WaterSurfaceRenderer {
    /// Create a new water surface renderer
    pub fn new(
        device: &Device,
        color_format: TextureFormat,
        depth_format: Option<TextureFormat>,
        sample_count: u32,
    ) -> Self {
        let params = WaterSurfaceParams::default();
        let uniforms = WaterSurfaceUniforms::default();

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("water_surface_uniform_buffer"),
            size: std::mem::size_of::<WaterSurfaceUniforms>() as wgpu::BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout (uniforms)
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("water_surface_bind_group_layout"),
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

        // Create bind group (uniforms)
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("water_surface_bind_group"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Create mask sampler
        let mask_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("water_mask_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        // Create a default 1x1 white mask texture
        let mask_size = (1u32, 1u32);
        let mask_texture = device.create_texture(&TextureDescriptor {
            label: Some("water_mask_texture"),
            size: Extent3d {
                width: mask_size.0,
                height: mask_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mask_view = mask_texture.create_view(&TextureViewDescriptor::default());

        // Create mask bind group layout
        let mask_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("water_surface_mask_bind_group_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2,
                        sample_type: TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create mask bind group
        let mask_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("water_surface_mask_bind_group"),
            layout: &mask_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&mask_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&mask_sampler),
                },
            ],
        });

        // Create shader module
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("water_surface_shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/water_surface.wgsl"
            ))),
        });

        // Create pipeline layout (uniforms + mask)
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("water_surface_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout, &mask_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Vertex buffer layout
        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 8]>() as BufferAddress, // position(3) + uv(2) + normal(3)
            step_mode: VertexStepMode::Vertex,
            attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3],
        };

        // Create render pipeline with alpha blending
        let water_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("water_surface_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_buffer_layout],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Don't cull for water surface
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: false, // Don't write depth for transparent water
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: color_format,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::SrcAlpha,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        // Create water surface geometry
        let (vertex_buffer, index_buffer, index_count) =
            Self::create_water_surface_geometry(device, params.size);

        Self {
            uniforms,
            params,
            uniform_buffer,
            water_pipeline,
            bind_group_layout,
            bind_group,
            mask_bind_group_layout,
            mask_bind_group,
            mask_texture,
            mask_view,
            mask_sampler,
            mask_size,
            vertex_buffer,
            index_buffer,
            index_count,
            animation_time: 0.0,
            enabled: true,
        }
    }

    /// Create water surface geometry (large quad with subdivision for waves)
    fn create_water_surface_geometry(device: &Device, size: f32) -> (Buffer, Buffer, u32) {
        let half_size = size * 0.5;
        let subdivisions = 32; // More subdivisions for wave animation
        let step = size / subdivisions as f32;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Generate subdivided grid
        for y in 0..=subdivisions {
            for x in 0..=subdivisions {
                let world_x = -half_size + x as f32 * step;
                let world_z = -half_size + y as f32 * step;
                let u = x as f32 / subdivisions as f32;
                let v = y as f32 / subdivisions as f32;

                // Water surface vertices: position(3) + uv(2) + normal(3)
                vertices.extend_from_slice(&[
                    world_x, 0.0, world_z, // Position
                    u, v, // UV
                    0.0, 1.0, 0.0, // Normal (upward)
                ]);
            }
        }

        // Generate indices for triangles
        for y in 0..subdivisions {
            for x in 0..subdivisions {
                let i0 = y * (subdivisions + 1) + x;
                let i1 = i0 + 1;
                let i2 = i0 + (subdivisions + 1);
                let i3 = i2 + 1;

                // Two triangles per quad
                indices.extend_from_slice(&[i0, i1, i2, i1, i3, i2]);
            }
        }

        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("water_surface_vertex_buffer"),
            size: (vertices.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("water_surface_index_buffer"),
            size: (indices.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&indices));
        index_buffer.unmap();

        (vertex_buffer, index_buffer, indices.len() as u32)
    }

    /// Update water surface parameters
    pub fn update_params(&mut self, params: WaterSurfaceParams) {
        self.params = params;
        self.update_uniforms();
    }

    /// Set water surface mode
    pub fn set_mode(&mut self, mode: WaterSurfaceMode) {
        self.params.mode = mode;
        self.enabled = mode != WaterSurfaceMode::Disabled;
        self.update_uniforms();
    }

    /// Set water surface height
    pub fn set_height(&mut self, height: f32) {
        self.params.height = height;
        self.update_uniforms();
    }

    /// Set water surface size
    pub fn set_size(&mut self, size: f32) {
        self.params.size = size;
        // Note: Would need to recreate geometry for size changes in a full implementation
        self.update_uniforms();
    }

    /// Set water base color
    pub fn set_base_color(&mut self, color: Vec3) {
        self.params.base_color = color;
        self.update_uniforms();
    }

    /// Set water hue shift (in radians)
    pub fn set_hue_shift(&mut self, hue_shift: f32) {
        self.params.hue_shift = hue_shift;
        self.update_uniforms();
    }

    /// Set water tint color and strength
    pub fn set_tint(&mut self, tint_color: Vec3, tint_strength: f32) {
        self.params.tint_color = tint_color;
        self.params.tint_strength = tint_strength;
        self.update_uniforms();
    }

    /// Set water transparency
    pub fn set_alpha(&mut self, alpha: f32) {
        self.params.alpha = alpha.clamp(0.0, 1.0);
        self.update_uniforms();
    }

    /// Set wave animation parameters
    pub fn set_wave_params(&mut self, amplitude: f32, frequency: f32, speed: f32) {
        self.params.wave_amplitude = amplitude;
        self.params.wave_frequency = frequency;
        self.params.wave_speed = speed;
        self.update_uniforms();
    }

    /// Set water flow direction
    pub fn set_flow_direction(&mut self, direction: Vec2) {
        self.params.flow_direction = direction.normalize();
        self.update_uniforms();
    }

    /// Set lighting parameters
    pub fn set_lighting_params(
        &mut self,
        reflection_strength: f32,
        refraction_strength: f32,
        fresnel_power: f32,
        roughness: f32,
    ) {
        self.params.reflection_strength = reflection_strength;
        self.params.refraction_strength = refraction_strength;
        self.params.fresnel_power = fresnel_power;
        self.params.roughness = roughness;
        self.update_uniforms();
    }

    /// Set foam (shoreline) parameters
    pub fn set_foam_params(&mut self, width_px: f32, intensity: f32, noise_scale: f32) {
        self.params.foam_width_px = width_px.max(0.0);
        self.params.foam_intensity = intensity.clamp(0.0, 1.0);
        self.params.foam_noise_scale = noise_scale.max(1.0);
        self.update_uniforms();
    }

    pub fn set_foam_enabled(&mut self, enabled: bool) {
        self.params.foam_enabled = enabled;
        self.update_uniforms();
    }

    /// Set debug mode (0 = normal, 100 = water mask, 101 = shore-distance, 102 = IBL spec)
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.params.debug_mode = mode;
        self.update_uniforms();
    }

    /// Upload an external water mask (R8Unorm, 0=land, 255=water)
    pub fn upload_water_mask(
        &mut self,
        device: &Device,
        queue: &Queue,
        data: &[u8],
        width: u32,
        height: u32,
    ) {
        assert_eq!(
            data.len() as u32,
            width * height,
            "mask data must be width*height bytes"
        );
        self.mask_size = (width, height);
        self.mask_texture = device.create_texture(&TextureDescriptor {
            label: Some("water_mask_texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.mask_view = self
            .mask_texture
            .create_view(&TextureViewDescriptor::default());
        // Update bind group with new view
        self.mask_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("water_surface_mask_bind_group"),
            layout: &self.mask_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.mask_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.mask_sampler),
                },
            ],
        });
        // Upload pixel data
        queue.write_texture(
            ImageCopyTexture {
                texture: &self.mask_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        // Enable mask usage in uniforms
        self.uniforms.foam_params[3] = 1.0;
    }

    /// Enable/disable water surface
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.params.mode = WaterSurfaceMode::Disabled;
        } else if self.params.mode == WaterSurfaceMode::Disabled {
            self.params.mode = WaterSurfaceMode::Transparent;
        }
        self.update_uniforms();
    }

    /// Check if water surface is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled && self.params.mode != WaterSurfaceMode::Disabled
    }

    /// Set camera matrices for rendering
    pub fn set_camera(&mut self, view_proj: Mat4) {
        self.uniforms.view_proj = view_proj.to_cols_array_2d();
    }

    /// Update animation time
    pub fn update(&mut self, delta_time: f32) {
        self.animation_time += delta_time;
        self.uniforms.wave_params[3] = self.animation_time; // Update time uniform
    }

    /// Update uniforms from parameters
    fn update_uniforms(&mut self) {
        // Update surface parameters
        self.uniforms.surface_params = [
            self.params.size,
            self.params.height,
            if self.enabled && self.params.mode != WaterSurfaceMode::Disabled {
                1.0
            } else {
                0.0
            }, // enabled
            self.params.alpha,
        ];

        // Update color parameters
        self.uniforms.color_params = [
            self.params.base_color.x,
            self.params.base_color.y,
            self.params.base_color.z,
            self.params.hue_shift,
        ];

        // Update wave parameters (time is updated in update() method)
        self.uniforms.wave_params[0] = self.params.wave_amplitude;
        self.uniforms.wave_params[1] = self.params.wave_frequency;
        self.uniforms.wave_params[2] = self.params.wave_speed;

        // Update tint parameters
        self.uniforms.tint_params = [
            self.params.tint_color.x,
            self.params.tint_color.y,
            self.params.tint_color.z,
            self.params.tint_strength,
        ];

        // Update lighting parameters
        self.uniforms.lighting_params = [
            self.params.reflection_strength,
            self.params.refraction_strength,
            self.params.fresnel_power,
            self.params.roughness,
        ];

        // Update animation parameters
        self.uniforms.animation_params = [
            self.params.ripple_scale,
            self.params.ripple_speed,
            self.params.flow_direction.x,
            self.params.flow_direction.y,
        ];

        // Update foam params: width_px, intensity, noise_scale, mask_enabled
        self.uniforms.foam_params = [
            self.params.foam_width_px,
            if self.params.foam_enabled {
                self.params.foam_intensity
            } else {
                0.0
            },
            self.params.foam_noise_scale,
            self.uniforms.foam_params[3],
        ];

        // Update debug params
        self.uniforms.debug_params[0] = self.params.debug_mode as f32;

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

    /// Get current water surface parameters for external access
    pub fn get_params(&self) -> (f32, f32, f32, f32) {
        (
            self.params.height,
            self.params.alpha,
            self.params.hue_shift,
            self.params.tint_strength,
        )
    }

    /// Create preset configurations
    pub fn create_ocean_water() -> WaterSurfaceParams {
        WaterSurfaceParams {
            mode: WaterSurfaceMode::Animated,
            base_color: Vec3::new(0.05, 0.2, 0.4), // Dark blue
            tint_color: Vec3::new(0.0, 0.5, 0.8),  // Ocean blue tint
            tint_strength: 0.3,
            wave_amplitude: 0.3,
            wave_frequency: 1.5,
            wave_speed: 1.2,
            reflection_strength: 1.0,
            alpha: 0.8,
            ..Default::default()
        }
    }

    pub fn create_lake_water() -> WaterSurfaceParams {
        WaterSurfaceParams {
            mode: WaterSurfaceMode::Reflective,
            base_color: Vec3::new(0.1, 0.3, 0.5), // Blue-green
            tint_color: Vec3::new(0.2, 0.6, 0.4), // Green tint
            tint_strength: 0.2,
            wave_amplitude: 0.05,
            wave_frequency: 3.0,
            wave_speed: 0.5,
            reflection_strength: 0.6,
            alpha: 0.7,
            ..Default::default()
        }
    }

    pub fn create_river_water() -> WaterSurfaceParams {
        WaterSurfaceParams {
            mode: WaterSurfaceMode::Animated,
            base_color: Vec3::new(0.2, 0.4, 0.3), // Muddy green
            tint_color: Vec3::new(0.3, 0.5, 0.3), // Earth tint
            tint_strength: 0.4,
            wave_amplitude: 0.02,
            wave_frequency: 8.0,
            wave_speed: 2.0,
            flow_direction: Vec2::new(1.0, 0.0), // Flowing downstream
            reflection_strength: 0.3,
            alpha: 0.6,
            ..Default::default()
        }
    }

    /// Render the water surface to the current render pass
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if !self.is_enabled() {
            return;
        }

        // Set pipeline and bind group
        render_pass.set_pipeline(&self.water_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_bind_group(1, &self.mask_bind_group, &[]);

        // Set vertex and index buffers
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

        // Draw the water surface
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

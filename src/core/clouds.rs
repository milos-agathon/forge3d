// src/core/clouds.rs
// Implements realtime cloud renderer with procedural billboards and volumetric shading.
// Exists so the terrain Scene can composite realtime clouds that react to lighting inputs.
// RELEVANT FILES: src/shaders/clouds.wgsl, src/scene/mod.rs, python/forge3d/__init__.py, tests/test_b8_clouds.py

use glam::{Mat4, Vec2, Vec3};
use std::borrow::Cow;
use wgpu::{
    vertex_attr_array, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferAddress, BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState,
    ColorWrites, ComputePipeline, Device, FragmentState, FrontFace,
    IndexFormat, MultisampleState, PipelineLayoutDescriptor,
    PolygonMode, PrimitiveState, PrimitiveTopology, Queue, RenderPipeline,
    RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, Texture,
    TextureFormat, TextureSampleType, TextureView, TextureViewDimension,
    VertexBufferLayout, VertexState, VertexStepMode,
};

/// Cloud rendering quality levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CloudQuality {
    Low,    // 32^3 noise, 16 steps, billboard-heavy
    Medium, // 64^3 noise, 32 steps, balanced
    High,   // 128^3 noise, 64 steps, volumetric-heavy
    Ultra,  // 256^3 noise, 128 steps, maximum quality
}

impl CloudQuality {
    pub fn noise_resolution(&self) -> u32 {
        match self {
            CloudQuality::Low => 32,
            CloudQuality::Medium => 64,
            CloudQuality::High => 128,
            CloudQuality::Ultra => 256,
        }
    }

    pub fn max_ray_steps(&self) -> u32 {
        match self {
            CloudQuality::Low => 16,
            CloudQuality::Medium => 32,
            CloudQuality::High => 64,
            CloudQuality::Ultra => 128,
        }
    }

    pub fn billboard_threshold(&self) -> f32 {
        match self {
            CloudQuality::Low => 50.0,     // Use billboard beyond 50 units
            CloudQuality::Medium => 100.0, // Use billboard beyond 100 units
            CloudQuality::High => 200.0,   // Use billboard beyond 200 units
            CloudQuality::Ultra => 500.0,  // Use billboard beyond 500 units
        }
    }
}

/// Cloud rendering mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CloudRenderMode {
    Billboard,  // Fast billboard-only rendering
    Volumetric, // High-quality volumetric rendering
    Hybrid,     // Distance-based LOD (billboard far, volumetric near)
}

/// Cloud animation presets
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CloudAnimationPreset {
    Static,   // No animation
    Gentle,   // Slow, calm movement
    Moderate, // Normal wind conditions
    Stormy,   // Fast, chaotic movement
}

impl CloudAnimationPreset {
    pub fn wind_strength(&self) -> f32 {
        match self {
            CloudAnimationPreset::Static => 0.0,
            CloudAnimationPreset::Gentle => 0.2,
            CloudAnimationPreset::Moderate => 0.5,
            CloudAnimationPreset::Stormy => 1.2,
        }
    }

    pub fn animation_speed(&self) -> f32 {
        match self {
            CloudAnimationPreset::Static => 0.0,
            CloudAnimationPreset::Gentle => 0.3,
            CloudAnimationPreset::Moderate => 0.8,
            CloudAnimationPreset::Stormy => 2.0,
        }
    }
}

/// Cloud uniforms structure (must match WGSL exactly)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudUniforms {
    pub view_proj: [[f32; 4]; 4],    // 64 bytes - View-projection matrix
    pub camera_pos: [f32; 4],        // 16 bytes - Camera position (xyz) + cloud_time (w)
    pub sky_params: [f32; 4],        // 16 bytes - Sky color (rgb) + sun_intensity (w)
    pub sun_direction: [f32; 4],     // 16 bytes - Sun direction (xyz) + cloud_density (w)
    pub cloud_params: [f32; 4], // 16 bytes - coverage (x), scale (y), height (z), fade_distance (w)
    pub wind_params: [f32; 4],  // 16 bytes - wind_dir (xy), wind_strength (z), animation_speed (w)
    pub scattering_params: [f32; 4], // 16 bytes - scatter_strength (x), absorption (y), phase_g (z), ambient_strength (w)
    pub render_params: [f32; 4], // 16 bytes - max_steps (x), step_size (y), billboard_size (z), lod_bias (w)
}

impl Default for CloudUniforms {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 10.0, 0.0, 0.0],
            sky_params: [0.6, 0.8, 1.0, 1.0],    // Light blue sky
            sun_direction: [0.3, 0.7, 0.2, 0.8], // Angled sun + medium density
            cloud_params: [0.6, 200.0, 150.0, 1000.0], // coverage, scale, height, fade_distance
            wind_params: [1.0, 0.0, 0.5, 1.0],   // wind direction + strength + speed
            scattering_params: [1.2, 0.8, 0.3, 0.4], // scatter, absorption, phase_g, ambient
            render_params: [32.0, 5.0, 50.0, 0.0], // max_steps, step_size, billboard_size, lod_bias
        }
    }
}

/// Cloud instance data
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudInstance {
    pub world_matrix: [[f32; 4]; 4], // World transformation matrix
    pub cloud_data: [f32; 4],        // size (x), density (y), type (z), blend_factor (w)
    pub animation_data: [f32; 4],    // offset (xy), phase (z), lifetime (w)
}

impl Default for CloudInstance {
    fn default() -> Self {
        Self {
            world_matrix: Mat4::IDENTITY.to_cols_array_2d(),
            cloud_data: [100.0, 0.8, 0.0, 1.0], // size, density, type, blend
            animation_data: [0.0, 0.0, 0.0, 1.0], // offset, phase, lifetime
        }
    }
}

/// Cloud rendering parameters
#[derive(Debug, Clone)]
pub struct CloudParams {
    pub quality: CloudQuality,
    pub render_mode: CloudRenderMode,
    pub animation_preset: CloudAnimationPreset,
    pub density: f32,
    pub coverage: f32,
    pub scale: f32,
    pub height: f32,
    pub fade_distance: f32,
    pub wind_direction: Vec2,
    pub wind_strength: f32,
    pub sun_intensity: f32,
    pub scatter_strength: f32,
    pub absorption: f32,
    pub phase_g: f32,
    pub ambient_strength: f32,
}

impl Default for CloudParams {
    fn default() -> Self {
        Self {
            quality: CloudQuality::Medium,
            render_mode: CloudRenderMode::Hybrid,
            animation_preset: CloudAnimationPreset::Moderate,
            density: 0.8,
            coverage: 0.6,
            scale: 200.0,
            height: 150.0,
            fade_distance: 1000.0,
            wind_direction: Vec2::new(1.0, 0.0),
            wind_strength: 0.5,
            sun_intensity: 1.0,
            scatter_strength: 1.2,
            absorption: 0.8,
            phase_g: 0.3,
            ambient_strength: 0.4,
        }
    }
}

/// Main cloud rendering system
pub struct CloudRenderer {
    pub uniforms: CloudUniforms,
    pub params: CloudParams,

    // GPU resources
    pub uniform_buffer: Buffer,
    pub cloud_pipeline: RenderPipeline,
    pub compute_pipeline: Option<ComputePipeline>,
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,

    // Bind groups and layouts
    pub bind_group_layout_uniforms: BindGroupLayout,
    pub bind_group_layout_textures: BindGroupLayout,
    pub bind_group_layout_ibl: BindGroupLayout,
    pub bind_group_uniforms: BindGroup,
    pub bind_group_textures: Option<BindGroup>,
    pub bind_group_ibl: Option<BindGroup>,

    // Textures
    pub noise_texture: Option<Texture>,
    pub noise_view: Option<TextureView>,
    pub shape_texture: Option<Texture>,
    pub shape_view: Option<TextureView>,
    pub ibl_irradiance_texture: Option<Texture>,
    pub ibl_irradiance_view: Option<TextureView>,
    pub ibl_prefilter_texture: Option<Texture>,
    pub ibl_prefilter_view: Option<TextureView>,
    pub cloud_sampler: Sampler,
    pub shape_sampler: Sampler,
    pub ibl_sampler: Sampler,
    pub noise_resolution: u32,

    // Animation
    pub time: f32,
    pub enabled: bool,
}

impl CloudRenderer {
    /// Create a new cloud renderer
    pub fn new(device: &Device, color_format: TextureFormat, sample_count: u32) -> Self {
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("cloud_uniform_buffer"),
            size: std::mem::size_of::<CloudUniforms>() as wgpu::BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layouts
        let bind_group_layout_uniforms =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("cloud_bind_group_layout_uniforms"),
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

        let bind_group_layout_textures =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("cloud_bind_group_layout_textures"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let bind_group_layout_ibl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("cloud_bind_group_layout_ibl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Cloud geometry buffers
        let (vertex_buffer, index_buffer, index_count) = Self::create_cloud_quad_geometry(device);

        // Create samplers
        let cloud_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("cloud_density_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let shape_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("cloud_shape_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let ibl_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("cloud_ibl_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create uniforms bind group
        let bind_group_uniforms = device.create_bind_group(&BindGroupDescriptor {
            label: Some("cloud_bind_group_uniforms"),
            layout: &bind_group_layout_uniforms,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Create shader
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("cloud_shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/clouds.wgsl"))),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("cloud_pipeline_layout"),
            bind_group_layouts: &[
                &bind_group_layout_uniforms,
                &bind_group_layout_textures,
                &bind_group_layout_ibl,
            ],
            push_constant_ranges: &[],
        });

        // Vertex buffer layout
        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 8]>() as BufferAddress, // position(3) + uv(2) + normal(3)
            step_mode: VertexStepMode::Vertex,
            attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3],
        };

        // Create render pipeline
        let cloud_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("cloud_render_pipeline"),
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
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: sample_count.max(1),
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

        let uniforms = CloudUniforms::default();
        let mut params = CloudParams::default();
        params.quality = match sample_count {
            0 | 1 => CloudQuality::Medium,
            2 => CloudQuality::Medium,
            4 => CloudQuality::High,
            _ => CloudQuality::High,
        };

        let mut renderer = Self {
            uniforms,
            params,
            uniform_buffer,
            cloud_pipeline,
            compute_pipeline: None,
            vertex_buffer,
            index_buffer,
            index_count,
            bind_group_layout_uniforms,
            bind_group_layout_textures,
            bind_group_layout_ibl,
            bind_group_uniforms,
            bind_group_textures: None,
            bind_group_ibl: None,
            noise_texture: None,
            noise_view: None,
            shape_texture: None,
            shape_view: None,
            ibl_irradiance_texture: None,
            ibl_irradiance_view: None,
            ibl_prefilter_texture: None,
            ibl_prefilter_view: None,
            cloud_sampler,
            shape_sampler,
            ibl_sampler,
            noise_resolution: 0,
            time: 0.0,
            enabled: true,
        };
        renderer.update_uniforms();
        renderer
    }

    /// Update cloud parameters
    pub fn update_params(&mut self, params: CloudParams) {
        self.params = params;
        self.update_uniforms();
    }

    /// Set cloud quality
    pub fn set_quality(&mut self, quality: CloudQuality) {
        self.params.quality = quality;
        self.noise_texture = None;
        self.noise_view = None;
        self.noise_resolution = 0;
        self.bind_group_textures = None;
        self.update_uniforms();
    }

    /// Set cloud density
    pub fn set_density(&mut self, density: f32) {
        self.params.density = density.clamp(0.0, 2.0);
        self.update_uniforms();
    }

    /// Set cloud coverage
    pub fn set_coverage(&mut self, coverage: f32) {
        self.params.coverage = coverage.clamp(0.0, 1.0);
        self.update_uniforms();
    }

    /// Set cloud scale
    pub fn set_scale(&mut self, scale: f32) {
        self.params.scale = scale.max(10.0);
        self.update_uniforms();
    }

    /// Set wind parameters
    pub fn set_wind(&mut self, direction: Vec2, strength: f32) {
        self.params.wind_direction = direction.normalize_or_zero();
        self.params.wind_strength = strength.clamp(0.0, 2.0);
        self.update_uniforms();
    }

    /// Set animation preset
    pub fn set_animation_preset(&mut self, preset: CloudAnimationPreset) {
        self.params.animation_preset = preset;
        self.params.wind_strength = preset.wind_strength();
        self.update_uniforms();
    }

    /// Set render mode
    pub fn set_render_mode(&mut self, mode: CloudRenderMode) {
        self.params.render_mode = mode;
        self.update_uniforms();
    }

    /// Update time for animation
    pub fn update(&mut self, delta_time: f32) {
        self.time += delta_time * self.params.animation_preset.animation_speed();
        self.uniforms.camera_pos[3] = self.time;
    }

    /// Update uniforms from parameters
    fn update_uniforms(&mut self) {
        // Update cloud parameters
        self.uniforms.cloud_params = [
            self.params.coverage,
            self.params.scale,
            self.params.height,
            self.params.fade_distance,
        ];

        // Update wind parameters
        self.uniforms.wind_params = [
            self.params.wind_direction.x,
            self.params.wind_direction.y,
            self.params.wind_strength,
            self.params.animation_preset.animation_speed(),
        ];

        // Update scattering parameters
        self.uniforms.scattering_params = [
            self.params.scatter_strength,
            self.params.absorption,
            self.params.phase_g,
            self.params.ambient_strength,
        ];

        // Update render parameters based on quality
        let step_size = match self.params.quality {
            CloudQuality::Low => 12.0,
            CloudQuality::Medium => 8.0,
            CloudQuality::High => 5.0,
            CloudQuality::Ultra => 3.5,
        };
        let render_mode_flag = match self.params.render_mode {
            CloudRenderMode::Billboard => 0.0,
            CloudRenderMode::Volumetric => 1.0,
            CloudRenderMode::Hybrid => 2.0,
        };
        self.uniforms.render_params = [
            self.params.quality.max_ray_steps() as f32,
            step_size,
            self.params.quality.billboard_threshold(),
            render_mode_flag,
        ];

        // Update sun parameters
        self.uniforms.sun_direction[3] = self.params.density;
        self.uniforms.sky_params[3] = self.params.sun_intensity;
    }

    /// Upload uniforms to GPU
    pub fn upload_uniforms(&self, queue: &Queue) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    /// Set camera parameters
    pub fn set_camera(&mut self, view_proj: Mat4, camera_pos: Vec3) {
        self.uniforms.view_proj = view_proj.to_cols_array_2d();
        self.uniforms.camera_pos[0] = camera_pos.x;
        self.uniforms.camera_pos[1] = camera_pos.y;
        self.uniforms.camera_pos[2] = camera_pos.z;
        // Don't overwrite time in camera_pos[3]
    }

    /// Set sky parameters
    pub fn set_sky_params(&mut self, sky_color: Vec3, sun_direction: Vec3, sun_intensity: f32) {
        self.uniforms.sky_params = [sky_color.x, sky_color.y, sky_color.z, sun_intensity];
        self.uniforms.sun_direction = [
            sun_direction.x,
            sun_direction.y,
            sun_direction.z,
            self.params.density,
        ];
    }

    /// Enable/disable cloud rendering
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if clouds are enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get current cloud parameters for external access
    pub fn get_params(&self) -> (f32, f32, f32, f32) {
        (
            self.params.density,
            self.params.coverage,
            self.params.scale,
            self.params.wind_strength,
        )
    }
}

/// Utility functions for cloud rendering
impl CloudRenderer {
    /// Ensure GPU resources required for rendering exist.
    pub fn ensure_resources(&mut self, device: &Device, queue: &Queue) -> Result<(), String> {
        let desired_resolution = self.desired_noise_resolution();
        if self.noise_view.is_none() || self.noise_resolution != desired_resolution {
            self.recreate_noise_texture(device, queue)?;
        }
        if self.shape_view.is_none() {
            self.recreate_shape_texture(device, queue)?;
        }
        if self.ibl_irradiance_view.is_none() || self.ibl_prefilter_view.is_none() {
            self.recreate_default_ibl(device, queue)?;
        }

        if self.bind_group_textures.is_none() {
            let noise_view = self
                .noise_view
                .as_ref()
                .ok_or_else(|| "Noise texture missing".to_string())?;
            let shape_view = self
                .shape_view
                .as_ref()
                .ok_or_else(|| "Shape texture missing".to_string())?;

            self.bind_group_textures = Some(device.create_bind_group(&BindGroupDescriptor {
                label: Some("cloud_bind_group_textures"),
                layout: &self.bind_group_layout_textures,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(noise_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&self.cloud_sampler),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(shape_view),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::Sampler(&self.shape_sampler),
                    },
                ],
            }));
        }

        if self.bind_group_ibl.is_none() {
            let irradiance_view = self
                .ibl_irradiance_view
                .as_ref()
                .ok_or_else(|| "IBL irradiance view missing".to_string())?;
            let prefilter_view = self
                .ibl_prefilter_view
                .as_ref()
                .ok_or_else(|| "IBL prefilter view missing".to_string())?;

            self.bind_group_ibl = Some(device.create_bind_group(&BindGroupDescriptor {
                label: Some("cloud_bind_group_ibl"),
                layout: &self.bind_group_layout_ibl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(irradiance_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&self.ibl_sampler),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(prefilter_view),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::Sampler(&self.ibl_sampler),
                    },
                ],
            }));
        }

        Ok(())
    }

    /// Convenience wrapper for call sites.
    pub fn prepare_frame(&mut self, device: &Device, queue: &Queue) -> Result<(), String> {
        self.ensure_resources(device, queue)
    }

    /// Issue draw call for the cloud pass.
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if !self.enabled {
            return;
        }
        let Some(ref textures_bg) = self.bind_group_textures else {
            return;
        };
        let Some(ref ibl_bg) = self.bind_group_ibl else {
            return;
        };

        pass.set_pipeline(&self.cloud_pipeline);
        pass.set_bind_group(0, &self.bind_group_uniforms, &[]);
        pass.set_bind_group(1, textures_bg, &[]);
        pass.set_bind_group(2, ibl_bg, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    fn desired_noise_resolution(&self) -> u32 {
        self.params.quality.noise_resolution().min(128)
    }

    fn recreate_noise_texture(&mut self, device: &Device, queue: &Queue) -> Result<(), String> {
        let resolution = self.desired_noise_resolution();
        let (data, padded_row) = Self::build_noise_data(resolution);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud_noise_texture"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: resolution,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
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
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(resolution),
            },
            wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: resolution,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.noise_texture = Some(texture);
        self.noise_view = Some(view);
        self.noise_resolution = resolution;
        self.bind_group_textures = None;
        Ok(())
    }

    fn recreate_shape_texture(&mut self, device: &Device, queue: &Queue) -> Result<(), String> {
        let size = 256;
        let (data, padded_row) = Self::build_shape_data(size);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud_shape_texture"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
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
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.shape_texture = Some(texture);
        self.shape_view = Some(view);
        self.bind_group_textures = None;
        Ok(())
    }

    fn recreate_default_ibl(&mut self, device: &Device, queue: &Queue) -> Result<(), String> {
        let padded_row = Self::align_to(4, 256);
        let mut irradiance_data = vec![0u8; padded_row as usize * 6];
        let mut prefilter_data = vec![0u8; padded_row as usize * 6];
        let irradiance_colors = [
            [0.62, 0.72, 0.92],
            [0.58, 0.68, 0.88],
            [0.70, 0.80, 0.95],
            [0.64, 0.74, 0.90],
            [0.55, 0.65, 0.85],
            [0.60, 0.70, 0.89],
        ];
        let prefilter_colors = [
            [0.80, 0.82, 0.86],
            [0.78, 0.80, 0.84],
            [0.82, 0.84, 0.88],
            [0.76, 0.78, 0.82],
            [0.74, 0.76, 0.80],
            [0.83, 0.85, 0.89],
        ];

        for (layer, color) in irradiance_colors.iter().enumerate() {
            let offset = layer * padded_row as usize;
            irradiance_data[offset] = Self::float_to_u8(color[0]);
            irradiance_data[offset + 1] = Self::float_to_u8(color[1]);
            irradiance_data[offset + 2] = Self::float_to_u8(color[2]);
            irradiance_data[offset + 3] = 255;
        }
        for (layer, color) in prefilter_colors.iter().enumerate() {
            let offset = layer * padded_row as usize;
            prefilter_data[offset] = Self::float_to_u8(color[0]);
            prefilter_data[offset + 1] = Self::float_to_u8(color[1]);
            prefilter_data[offset + 2] = Self::float_to_u8(color[2]);
            prefilter_data[offset + 3] = 255;
        }

        let irradiance = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud_ibl_irradiance"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &irradiance,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &irradiance_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
        );
        let irradiance_view = irradiance.create_view(&wgpu::TextureViewDescriptor {
            label: Some("cloud_ibl_irradiance_view"),
            format: Some(TextureFormat::Rgba8Unorm),
            dimension: Some(TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(6),
        });

        let prefilter = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud_ibl_prefilter"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &prefilter,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &prefilter_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
        );
        let prefilter_view = prefilter.create_view(&wgpu::TextureViewDescriptor {
            label: Some("cloud_ibl_prefilter_view"),
            format: Some(TextureFormat::Rgba8Unorm),
            dimension: Some(TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(6),
        });

        self.ibl_irradiance_texture = Some(irradiance);
        self.ibl_irradiance_view = Some(irradiance_view);
        self.ibl_prefilter_texture = Some(prefilter);
        self.ibl_prefilter_view = Some(prefilter_view);
        self.bind_group_ibl = None;
        Ok(())
    }

    fn build_noise_data(resolution: u32) -> (Vec<u8>, u32) {
        let padded_row = Self::align_to(resolution, 256);
        let mut data = vec![0u8; padded_row as usize * resolution as usize * resolution as usize];
        for z in 0..resolution {
            for y in 0..resolution {
                let row_offset = ((z * resolution + y) * padded_row) as usize;
                for x in 0..resolution {
                    let f = (x as f32 * 0.125 + y as f32 * 0.175 + z as f32 * 0.215).sin();
                    let g = (x as f32 * 0.05 + y as f32 * 0.09 + z as f32 * 0.07).cos();
                    let value = ((f + g) * 0.25 + 0.5).clamp(0.0, 1.0);
                    data[row_offset + x as usize] = (value * 255.0) as u8;
                }
            }
        }
        (data, padded_row)
    }

    fn build_shape_data(size: u32) -> (Vec<u8>, u32) {
        let padded_row = Self::align_to(size, 256);
        let mut data = vec![0u8; padded_row as usize * size as usize];
        let half = size as f32 / 2.0;
        for y in 0..size {
            let dy = (y as f32 - half) / half;
            for x in 0..size {
                let dx = (x as f32 - half) / half;
                let dist = (dx * dx + dy * dy).sqrt();
                let softness = 1.0 - dist.powf(1.5);
                let noise = (x as f32 * 0.1).sin() * 0.1 + (y as f32 * 0.12).cos() * 0.1;
                let value = (softness + noise).clamp(0.0, 1.0);
                let offset = (y * padded_row + x) as usize;
                data[offset] = (value * 255.0) as u8;
            }
        }
        (data, padded_row)
    }

    fn align_to(value: u32, alignment: u32) -> u32 {
        ((value + alignment - 1) / alignment) * alignment
    }

    fn float_to_u8(value: f32) -> u8 {
        ((value.max(0.0).min(1.0)) * 255.0 + 0.5) as u8
    }

    /// Create a simple cloud geometry (billboard quad)
    pub fn create_cloud_quad_geometry(device: &Device) -> (Buffer, Buffer, u32) {
        let vertices: &[f32] = &[
            -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        ];
        let indices: &[u16] = &[0, 1, 2, 2, 3, 0];

        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("cloud_vertex_buffer"),
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
            label: Some("cloud_index_buffer"),
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
}

//! M2/P1.2: Bloom post-processing effect implementation
//!
//! Implements bloom effect with bright-pass filtering, dual blur passes,
//! and composite blending. Integrated with the PostFx chain system using
//! resource pool ping-pong textures for intermediate storage.
//!
//! Pipeline: Input HDR -> Brightpass -> H Blur -> V Blur -> Composite -> Output

use super::error::{RenderError, RenderResult};
use crate::core::gpu_timing::GpuTimingManager;
use crate::core::postfx::{PostFxConfig, PostFxEffect, PostFxResourceDesc, PostFxResourcePool};
use std::borrow::Cow;
use wgpu::*;

/// Bloom effect configuration
#[derive(Debug, Clone, Copy)]
pub struct BloomConfig {
    /// Enabled flag (false = passthrough, no bloom)
    pub enabled: bool,
    /// Brightness threshold for bloom extraction (default 1.5 = HDR only)
    pub threshold: f32,
    /// Softness of threshold transition (0.0 = hard, 1.0 = very soft)
    pub softness: f32,
    /// Bloom intensity/strength when compositing (0.0-1.0+)
    pub strength: f32,
    /// Blur radius multiplier (affects spread)
    pub radius: f32,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            enabled: false, // M2: Default off for backward compatibility
            threshold: 1.5, // Conservative: only HDR values bloom
            softness: 0.5,
            strength: 0.3, // Conservative: subtle bloom
            radius: 1.0,
        }
    }
}

/// Uniform data for bloom bright-pass
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomBrightPassUniforms {
    threshold: f32,
    softness: f32,
    _pad: [f32; 2],
}

/// Uniform data for bloom blur passes
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomBlurUniforms {
    radius: f32,
    strength: f32,
    _pad: [f32; 2],
}

/// Uniform data for bloom composite pass
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomCompositeUniforms {
    intensity: f32,
    _pad: [f32; 3],
}

/// Bloom post-processing effect implementation
///
/// Uses the PostFx resource pool for intermediate textures (ping-pong pair 0
/// for brightpass output, pair 1 for blur temp). The composite pass reads
/// the original input and the blurred bloom texture, then writes to the
/// output provided by the chain.
pub struct BloomEffect {
    config: PostFxConfig,
    bloom_config: BloomConfig,

    // Compute pipelines (initialized during initialize())
    brightpass_pipeline: Option<ComputePipeline>,
    blur_h_pipeline: Option<ComputePipeline>,
    blur_v_pipeline: Option<ComputePipeline>,
    composite_pipeline: Option<ComputePipeline>,

    // Bind group layouts
    brightpass_layout: Option<BindGroupLayout>,
    blur_layout: Option<BindGroupLayout>,
    composite_layout: Option<BindGroupLayout>,

    // Uniform buffers
    brightpass_uniform_buffer: Option<Buffer>,
    blur_uniform_buffer: Option<Buffer>,
    composite_uniform_buffer: Option<Buffer>,

    // Resource pool ping-pong pair indices for intermediate textures
    brightpass_texture_index: Option<usize>,
    blur_temp_texture_index: Option<usize>,
}

impl BloomEffect {
    /// Create a new bloom effect
    pub fn new() -> Self {
        let mut config = PostFxConfig::default();
        config.name = "bloom".to_string();
        config.priority = 800;
        config.temporal = false;
        config.ping_pong_count = 2;

        // Initialize default parameters from BloomConfig defaults
        let bloom_config = BloomConfig::default();
        config
            .parameters
            .insert("threshold".to_string(), bloom_config.threshold);
        config
            .parameters
            .insert("softness".to_string(), bloom_config.softness);
        config
            .parameters
            .insert("strength".to_string(), bloom_config.strength);
        config
            .parameters
            .insert("radius".to_string(), bloom_config.radius);
        config.parameters.insert(
            "enabled".to_string(),
            if bloom_config.enabled { 1.0 } else { 0.0 },
        );

        Self {
            config,
            bloom_config,
            brightpass_pipeline: None,
            blur_h_pipeline: None,
            blur_v_pipeline: None,
            composite_pipeline: None,
            brightpass_layout: None,
            blur_layout: None,
            composite_layout: None,
            brightpass_uniform_buffer: None,
            blur_uniform_buffer: None,
            composite_uniform_buffer: None,
            brightpass_texture_index: None,
            blur_temp_texture_index: None,
        }
    }

    /// Update the bloom configuration.
    ///
    /// Call this before executing the chain to ensure the effect uses
    /// the latest settings from Python BloomSettings.
    pub fn update_bloom_config(&mut self, bloom_config: BloomConfig) {
        self.bloom_config = bloom_config;
        // Sync to PostFxConfig parameters for observability
        self.config
            .parameters
            .insert("threshold".to_string(), bloom_config.threshold);
        self.config
            .parameters
            .insert("softness".to_string(), bloom_config.softness);
        self.config
            .parameters
            .insert("strength".to_string(), bloom_config.strength);
        self.config
            .parameters
            .insert("radius".to_string(), bloom_config.radius);
        self.config.parameters.insert(
            "enabled".to_string(),
            if bloom_config.enabled { 1.0 } else { 0.0 },
        );
    }

    /// Return the current bloom configuration.
    pub fn bloom_config(&self) -> &BloomConfig {
        &self.bloom_config
    }
}

impl PostFxEffect for BloomEffect {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn config(&self) -> &PostFxConfig {
        &self.config
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> RenderResult<()> {
        self.config.parameters.insert(name.to_string(), value);
        // Sync parameter changes back to bloom_config
        match name {
            "threshold" => self.bloom_config.threshold = value,
            "softness" => self.bloom_config.softness = value,
            "strength" => self.bloom_config.strength = value,
            "radius" => self.bloom_config.radius = value,
            "enabled" => self.bloom_config.enabled = value > 0.5,
            _ => {}
        }
        Ok(())
    }

    fn get_parameter(&self, name: &str) -> Option<f32> {
        self.config.parameters.get(name).copied()
    }

    fn initialize(
        &mut self,
        device: &Device,
        resource_pool: &mut PostFxResourcePool,
    ) -> RenderResult<()> {
        // --- Bind group layouts ---

        // Brightpass: input texture (read) + output storage (write) + uniforms
        let brightpass_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bloom_brightpass_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Blur: input texture (read) + output storage (write) + uniforms
        let blur_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bloom_blur_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Composite: original (read) + bloom (read) + output storage (write) + uniforms
        let composite_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bloom_composite_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // --- Shader modules ---

        let brightpass_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("bloom_brightpass_shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/bloom_brightpass.wgsl"
            ))),
        });

        let blur_h_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("bloom_blur_h_shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/bloom_blur_h.wgsl"
            ))),
        });

        let blur_v_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("bloom_blur_v_shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/bloom_blur_v.wgsl"
            ))),
        });

        let composite_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("bloom_composite_shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/bloom_composite.wgsl"
            ))),
        });

        // --- Pipeline layouts ---

        let brightpass_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("bloom_brightpass_pipeline_layout"),
            bind_group_layouts: &[&brightpass_layout],
            push_constant_ranges: &[],
        });

        let blur_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("bloom_blur_pipeline_layout"),
            bind_group_layouts: &[&blur_layout],
            push_constant_ranges: &[],
        });

        let composite_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("bloom_composite_pipeline_layout"),
            bind_group_layouts: &[&composite_layout],
            push_constant_ranges: &[],
        });

        // --- Compute pipelines ---

        let brightpass_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("bloom_brightpass_pipeline"),
            layout: Some(&brightpass_pipeline_layout),
            module: &brightpass_shader,
            entry_point: "main",
        });

        let blur_h_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("bloom_blur_h_pipeline"),
            layout: Some(&blur_pipeline_layout),
            module: &blur_h_shader,
            entry_point: "main",
        });

        let blur_v_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("bloom_blur_v_pipeline"),
            layout: Some(&blur_pipeline_layout),
            module: &blur_v_shader,
            entry_point: "main",
        });

        let composite_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("bloom_composite_pipeline"),
            layout: Some(&composite_pipeline_layout),
            module: &composite_shader,
            entry_point: "main",
        });

        // --- Uniform buffers ---

        let brightpass_uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("bloom_brightpass_uniforms"),
            size: std::mem::size_of::<BloomBrightPassUniforms>() as BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let blur_uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("bloom_blur_uniforms"),
            size: std::mem::size_of::<BloomBlurUniforms>() as BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let composite_uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("bloom_composite_uniforms"),
            size: std::mem::size_of::<BloomCompositeUniforms>() as BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Allocate ping-pong texture pairs from resource pool ---
        // Pair 0: brightpass output / blur input (also used for V blur output)
        // Pair 1: blur temp (H blur output / V blur input)

        let pp_desc = PostFxResourceDesc {
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            ..PostFxResourceDesc::default()
        };

        let brightpass_idx = resource_pool.allocate_ping_pong_pair(device, &pp_desc)?;
        let blur_temp_idx = resource_pool.allocate_ping_pong_pair(device, &pp_desc)?;

        // --- Store resources ---

        self.brightpass_pipeline = Some(brightpass_pipeline);
        self.blur_h_pipeline = Some(blur_h_pipeline);
        self.blur_v_pipeline = Some(blur_v_pipeline);
        self.composite_pipeline = Some(composite_pipeline);
        self.brightpass_layout = Some(brightpass_layout);
        self.blur_layout = Some(blur_layout);
        self.composite_layout = Some(composite_layout);
        self.brightpass_uniform_buffer = Some(brightpass_uniform_buffer);
        self.blur_uniform_buffer = Some(blur_uniform_buffer);
        self.composite_uniform_buffer = Some(composite_uniform_buffer);
        self.brightpass_texture_index = Some(brightpass_idx);
        self.blur_temp_texture_index = Some(blur_temp_idx);

        Ok(())
    }

    fn execute(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        input: &TextureView,
        output: &TextureView,
        resource_pool: &PostFxResourcePool,
        mut timing_manager: Option<&mut GpuTimingManager>,
    ) -> RenderResult<()> {
        // Early return if bloom is disabled (passthrough behavior)
        if !self.bloom_config.enabled {
            return Ok(());
        }

        let timing_scope = if let Some(timer) = timing_manager.as_mut() {
            Some(timer.begin_scope(encoder, "bloom"))
        } else {
            None
        };

        // --- Validate all resources are initialized ---

        let brightpass_pipeline = self
            .brightpass_pipeline
            .as_ref()
            .ok_or_else(|| RenderError::Render("Bloom brightpass pipeline not initialized".into()))?;
        let blur_h_pipeline = self
            .blur_h_pipeline
            .as_ref()
            .ok_or_else(|| RenderError::Render("Bloom blur_h pipeline not initialized".into()))?;
        let blur_v_pipeline = self
            .blur_v_pipeline
            .as_ref()
            .ok_or_else(|| RenderError::Render("Bloom blur_v pipeline not initialized".into()))?;
        let composite_pipeline = self
            .composite_pipeline
            .as_ref()
            .ok_or_else(|| RenderError::Render("Bloom composite pipeline not initialized".into()))?;

        let brightpass_layout = self
            .brightpass_layout
            .as_ref()
            .ok_or_else(|| RenderError::Render("Bloom brightpass layout not initialized".into()))?;
        let blur_layout = self
            .blur_layout
            .as_ref()
            .ok_or_else(|| RenderError::Render("Bloom blur layout not initialized".into()))?;
        let composite_layout = self
            .composite_layout
            .as_ref()
            .ok_or_else(|| RenderError::Render("Bloom composite layout not initialized".into()))?;

        let brightpass_uniform_buf = self
            .brightpass_uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::Render("Bloom brightpass UBO not initialized".into()))?;
        let blur_uniform_buf = self
            .blur_uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::Render("Bloom blur UBO not initialized".into()))?;
        let composite_uniform_buf = self
            .composite_uniform_buffer
            .as_ref()
            .ok_or_else(|| RenderError::Render("Bloom composite UBO not initialized".into()))?;

        let bp_idx = self.brightpass_texture_index.ok_or_else(|| {
            RenderError::Render("Bloom brightpass texture index not set".into())
        })?;
        let bt_idx = self.blur_temp_texture_index.ok_or_else(|| {
            RenderError::Render("Bloom blur temp texture index not set".into())
        })?;

        // Get ping-pong texture views from resource pool
        let bright_view = resource_pool.get_current_ping_pong(bp_idx).ok_or_else(|| {
            RenderError::Render("Bloom brightpass ping-pong texture unavailable".into())
        })?;
        let blur_temp_view =
            resource_pool.get_current_ping_pong(bt_idx).ok_or_else(|| {
                RenderError::Render("Bloom blur temp ping-pong texture unavailable".into())
            })?;
        // Use the alternate slot of the brightpass pair for V blur output
        let blur_result_view =
            resource_pool
                .get_previous_ping_pong(bp_idx)
                .ok_or_else(|| {
                    RenderError::Render("Bloom blur result ping-pong texture unavailable".into())
                })?;

        // --- Upload uniforms ---

        let brightpass_uniforms = BloomBrightPassUniforms {
            threshold: self.bloom_config.threshold,
            softness: self.bloom_config.softness,
            _pad: [0.0; 2],
        };
        queue.write_buffer(
            brightpass_uniform_buf,
            0,
            bytemuck::bytes_of(&brightpass_uniforms),
        );

        let blur_uniforms = BloomBlurUniforms {
            radius: self.bloom_config.radius,
            strength: 1.0, // Full strength for blur passes
            _pad: [0.0; 2],
        };
        queue.write_buffer(blur_uniform_buf, 0, bytemuck::bytes_of(&blur_uniforms));

        let composite_uniforms = BloomCompositeUniforms {
            intensity: self.bloom_config.strength,
            _pad: [0.0; 3],
        };
        queue.write_buffer(
            composite_uniform_buf,
            0,
            bytemuck::bytes_of(&composite_uniforms),
        );

        // Calculate workgroup counts (16x16 workgroups matching shader)
        // Use input texture dimensions; fall back to a safe default.
        // The resource pool tracks width/height but does not expose them,
        // so we use a fixed dispatch that the shaders clamp via
        // textureDimensions. We conservatively dispatch for 4096x4096 and
        // let the shader early-return for out-of-bounds invocations.
        // TODO: expose pool width/height or derive from input texture.
        let workgroups_x = (4096 + 15) / 16;
        let workgroups_y = (4096 + 15) / 16;

        // --- Pass 1: Bright-pass extraction ---
        {
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("bloom_brightpass_bg"),
                layout: brightpass_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(input),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(bright_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: brightpass_uniform_buf.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("bloom_brightpass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(brightpass_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // --- Pass 2: Horizontal blur ---
        {
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("bloom_blur_h_bg"),
                layout: blur_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(bright_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(blur_temp_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: blur_uniform_buf.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("bloom_blur_h"),
                timestamp_writes: None,
            });
            pass.set_pipeline(blur_h_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // --- Pass 3: Vertical blur ---
        {
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("bloom_blur_v_bg"),
                layout: blur_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(blur_temp_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(blur_result_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: blur_uniform_buf.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("bloom_blur_v"),
                timestamp_writes: None,
            });
            pass.set_pipeline(blur_v_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // --- Pass 4: Composite (original + bloom -> output) ---
        {
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("bloom_composite_bg"),
                layout: composite_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(input),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(blur_result_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(output),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: composite_uniform_buf.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("bloom_composite"),
                timestamp_writes: None,
            });
            pass.set_pipeline(composite_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        log::debug!(
            target: "bloom",
            "P1.2: Bloom executed: threshold={:.2}, strength={:.2}, radius={:.1}",
            self.bloom_config.threshold,
            self.bloom_config.strength,
            self.bloom_config.radius
        );

        // End timing scope
        let _ = timing_scope;

        Ok(())
    }
}

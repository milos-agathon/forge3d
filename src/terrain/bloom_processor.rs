//! M2: Standalone bloom processor for terrain offline rendering
//!
//! Provides a self-contained bloom implementation that doesn't depend on
//! the PostFx resource pool system. Designed for offline rendering where
//! we have full control over texture allocation.
//!
//! Pipeline: Input HDR -> Brightpass -> H Blur -> V Blur -> Composite -> Output

use anyhow::{anyhow, Result};
use std::borrow::Cow;

/// Bloom configuration for terrain rendering
#[derive(Debug, Clone, Copy)]
pub struct TerrainBloomConfig {
    /// Enabled flag (false = passthrough, identical output)
    pub enabled: bool,
    /// Brightness threshold for bloom extraction (default 1.5 = HDR only)
    pub threshold: f32,
    /// Softness of threshold transition (0.0 = hard, 1.0 = very soft)
    pub softness: f32,
    /// Bloom intensity when compositing (0.0-1.0+)
    pub intensity: f32,
    /// Blur radius multiplier (affects spread)
    pub radius: f32,
}

impl Default for TerrainBloomConfig {
    fn default() -> Self {
        Self {
            enabled: false, // M2: Default off for backward compatibility
            threshold: 1.5, // Conservative: only HDR values bloom
            softness: 0.5,
            intensity: 0.3, // Conservative: subtle bloom
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

/// Standalone bloom processor for terrain rendering
pub struct TerrainBloomProcessor {
    // Compute pipelines
    brightpass_pipeline: wgpu::ComputePipeline,
    blur_h_pipeline: wgpu::ComputePipeline,
    blur_v_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    brightpass_layout: wgpu::BindGroupLayout,
    blur_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,

    // Uniform buffers
    brightpass_uniform_buffer: wgpu::Buffer,
    blur_uniform_buffer: wgpu::Buffer,
    composite_uniform_buffer: wgpu::Buffer,

    // Intermediate textures (resized as needed)
    bright_texture: Option<wgpu::Texture>,
    bright_view: Option<wgpu::TextureView>,
    blur_temp_texture: Option<wgpu::Texture>,
    blur_temp_view: Option<wgpu::TextureView>,
    blur_result_texture: Option<wgpu::Texture>,
    blur_result_view: Option<wgpu::TextureView>,
    current_size: (u32, u32),
}

impl TerrainBloomProcessor {
    /// Create a new bloom processor
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        // Create bind group layouts
        let brightpass_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.bloom.brightpass_layout"),
            entries: &[
                // @binding(0): input texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(1): output storage texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // @binding(2): uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.bloom.blur_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.bloom.composite_layout"),
            entries: &[
                // @binding(0): original texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(1): bloom texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(2): output storage texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // @binding(3): uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create shader modules
        let brightpass_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain.bloom.brightpass_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/bloom_brightpass.wgsl"
            ))),
        });

        let blur_h_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain.bloom.blur_h_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/bloom_blur_h.wgsl"
            ))),
        });

        let blur_v_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain.bloom.blur_v_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/bloom_blur_v.wgsl"
            ))),
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain.bloom.composite_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/bloom_composite.wgsl"
            ))),
        });

        // Create pipeline layouts
        let brightpass_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain.bloom.brightpass_pipeline_layout"),
                bind_group_layouts: &[&brightpass_layout],
                push_constant_ranges: &[],
            });

        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.bloom.blur_pipeline_layout"),
            bind_group_layouts: &[&blur_layout],
            push_constant_ranges: &[],
        });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain.bloom.composite_pipeline_layout"),
                bind_group_layouts: &[&composite_layout],
                push_constant_ranges: &[],
            });

        // Create pipelines
        let brightpass_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("terrain.bloom.brightpass_pipeline"),
                layout: Some(&brightpass_pipeline_layout),
                module: &brightpass_shader,
                entry_point: "main",
            });

        let blur_h_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("terrain.bloom.blur_h_pipeline"),
            layout: Some(&blur_pipeline_layout),
            module: &blur_h_shader,
            entry_point: "main",
        });

        let blur_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("terrain.bloom.blur_v_pipeline"),
            layout: Some(&blur_pipeline_layout),
            module: &blur_v_shader,
            entry_point: "main",
        });

        let composite_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("terrain.bloom.composite_pipeline"),
            layout: Some(&composite_pipeline_layout),
            module: &composite_shader,
            entry_point: "main",
        });

        // Create uniform buffers
        let brightpass_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain.bloom.brightpass_uniforms"),
            size: std::mem::size_of::<BloomBrightPassUniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let blur_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain.bloom.blur_uniforms"),
            size: std::mem::size_of::<BloomBlurUniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let composite_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain.bloom.composite_uniforms"),
            size: std::mem::size_of::<BloomCompositeUniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            brightpass_pipeline,
            blur_h_pipeline,
            blur_v_pipeline,
            composite_pipeline,
            brightpass_layout,
            blur_layout,
            composite_layout,
            brightpass_uniform_buffer,
            blur_uniform_buffer,
            composite_uniform_buffer,
            bright_texture: None,
            bright_view: None,
            blur_temp_texture: None,
            blur_temp_view: None,
            blur_result_texture: None,
            blur_result_view: None,
            current_size: (0, 0),
        })
    }

    /// Ensure intermediate textures match the required size
    fn ensure_textures(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.current_size == (width, height) && self.bright_texture.is_some() {
            return;
        }

        log::info!(
            target: "terrain.bloom",
            "M2: Creating bloom intermediate textures: {}x{}",
            width, height
        );

        // Create bright-pass output texture
        let bright_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.bloom.bright_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bright_view = bright_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create blur temp texture (for H blur output)
        let blur_temp_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.bloom.blur_temp_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let blur_temp_view = blur_temp_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create blur result texture (for V blur output / composite input)
        let blur_result_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.bloom.blur_result_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let blur_result_view =
            blur_result_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.bright_texture = Some(bright_texture);
        self.bright_view = Some(bright_view);
        self.blur_temp_texture = Some(blur_temp_texture);
        self.blur_temp_view = Some(blur_temp_view);
        self.blur_result_texture = Some(blur_result_texture);
        self.blur_result_view = Some(blur_result_view);
        self.current_size = (width, height);
    }

    /// Execute bloom pipeline
    ///
    /// # Arguments
    /// * `device` - WGPU device
    /// * `queue` - WGPU queue for uniform buffer writes
    /// * `encoder` - Command encoder to record commands
    /// * `input_view` - Input HDR texture view
    /// * `output_view` - Output texture view (storage texture, Rgba8Unorm)
    /// * `config` - Bloom configuration
    /// * `width` - Texture width
    /// * `height` - Texture height
    pub fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        config: &TerrainBloomConfig,
        width: u32,
        height: u32,
    ) -> Result<()> {
        if !config.enabled {
            // Bloom disabled - passthrough would need a copy pass.
            // Caller skips this path when disabled.
            return Ok(());
        }

        // Ensure intermediate textures exist
        self.ensure_textures(device, width, height);

        let bright_view = self
            .bright_view
            .as_ref()
            .ok_or_else(|| anyhow!("Bloom bright view not initialized"))?;
        let blur_temp_view = self
            .blur_temp_view
            .as_ref()
            .ok_or_else(|| anyhow!("Bloom blur temp view not initialized"))?;
        let blur_result_view = self
            .blur_result_view
            .as_ref()
            .ok_or_else(|| anyhow!("Bloom blur result view not initialized"))?;

        // Update uniform buffers
        let brightpass_uniforms = BloomBrightPassUniforms {
            threshold: config.threshold,
            softness: config.softness,
            _pad: [0.0; 2],
        };
        queue.write_buffer(
            &self.brightpass_uniform_buffer,
            0,
            bytemuck::bytes_of(&brightpass_uniforms),
        );

        let blur_uniforms = BloomBlurUniforms {
            radius: config.radius,
            strength: 1.0, // Full strength for blur passes
            _pad: [0.0; 2],
        };
        queue.write_buffer(
            &self.blur_uniform_buffer,
            0,
            bytemuck::bytes_of(&blur_uniforms),
        );

        let composite_uniforms = BloomCompositeUniforms {
            intensity: config.intensity,
            _pad: [0.0; 3],
        };
        queue.write_buffer(
            &self.composite_uniform_buffer,
            0,
            bytemuck::bytes_of(&composite_uniforms),
        );

        // Calculate workgroup counts (16x16 workgroups)
        let workgroups_x = (width + 15) / 16;
        let workgroups_y = (height + 15) / 16;

        // Pass 1: Bright-pass extraction
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain.bloom.brightpass_bind_group"),
                layout: &self.brightpass_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(bright_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.brightpass_uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain.bloom.brightpass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.brightpass_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Pass 2: Horizontal blur
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain.bloom.blur_h_bind_group"),
                layout: &self.blur_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(bright_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(blur_temp_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.blur_uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain.bloom.blur_h"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blur_h_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Pass 3: Vertical blur
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain.bloom.blur_v_bind_group"),
                layout: &self.blur_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(blur_temp_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(blur_result_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.blur_uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain.bloom.blur_v"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blur_v_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Pass 4: Composite (original + bloom)
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain.bloom.composite_bind_group"),
                layout: &self.composite_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(blur_result_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.composite_uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain.bloom.composite"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        log::debug!(
            target: "terrain.bloom",
            "M2: Bloom executed: threshold={:.2}, intensity={:.2}, radius={:.1}",
            config.threshold, config.intensity, config.radius
        );

        Ok(())
    }
}

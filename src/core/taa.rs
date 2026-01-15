//! P1.3: Temporal Anti-Aliasing (TAA) system
//!
//! Provides temporal anti-aliasing using reprojection and neighborhood clamping.
//! Integrates with P1.1 motion vectors and P1.2 jitter sequence.

use super::error::RenderResult;
use std::mem::size_of;
use wgpu::util::DeviceExt;
use wgpu::*;

const TAA_SHADER_SRC: &str = include_str!("../shaders/taa.wgsl");

/// TAA settings
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TaaSettings {
    /// Resolution (width, height)
    pub resolution: [f32; 2],
    /// Jitter offset in pixels (from P1.2)
    pub jitter_offset: [f32; 2],
    /// History blend factor (0.0 = current only, 1.0 = history only)
    /// Typical value: 0.9 (90% history, 10% current)
    pub history_weight: f32,
    /// Neighborhood clamp aggressiveness (higher = more aggressive, less ghosting but more flickering)
    /// Typical value: 1.0-1.5
    pub clamp_gamma: f32,
    /// Motion scale for velocity rejection (higher = faster rejection near motion)
    pub motion_scale: f32,
    /// Frame index for temporal dithering
    pub frame_index: u32,
}

impl Default for TaaSettings {
    fn default() -> Self {
        Self {
            resolution: [1920.0, 1080.0],
            jitter_offset: [0.0, 0.0],
            history_weight: 0.9,
            clamp_gamma: 1.25,
            motion_scale: 100.0,
            frame_index: 0,
        }
    }
}

/// TAA renderer with history buffer management
pub struct TaaRenderer {
    settings: TaaSettings,
    settings_buffer: Buffer,
    /// Ping-pong history buffers (index 0 = read, index 1 = write; swap each frame)
    history_textures: [Texture; 2],
    history_views: [TextureView; 2],
    /// Current read index (0 or 1)
    read_index: usize,
    /// Compute pipeline
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    /// Sampler for history/color sampling
    sampler: Sampler,
    /// Whether TAA is enabled
    enabled: bool,
    /// Width/height
    width: u32,
    height: u32,
    /// First frame flag (skip history on first frame)
    first_frame: bool,
}

impl TaaRenderer {
    /// Create new TAA renderer
    pub fn new(device: &Device, width: u32, height: u32) -> RenderResult<Self> {
        let settings = TaaSettings {
            resolution: [width as f32, height as f32],
            ..Default::default()
        };

        let settings_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("taa.settings"),
            contents: bytemuck::cast_slice(&[settings]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Create ping-pong history textures (Rgba16Float for HDR)
        let history_textures = [
            Self::create_history_texture(device, width, height, 0),
            Self::create_history_texture(device, width, height, 1),
        ];

        let history_views = [
            history_textures[0].create_view(&TextureViewDescriptor::default()),
            history_textures[1].create_view(&TextureViewDescriptor::default()),
        ];

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("taa.sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        // Create shader module
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("taa.shader"),
            source: ShaderSource::Wgsl(TAA_SHADER_SRC.into()),
        });

        // Bind group layout:
        // 0: current color (texture)
        // 1: history color (texture)
        // 2: velocity (texture)
        // 3: depth (texture)
        // 4: sampler
        // 5: settings (uniform)
        // 6: output (storage texture)
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("taa.bgl"),
            entries: &[
                // Current color input
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // History color input
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Velocity texture
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Depth texture
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // Settings uniform
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(size_of::<TaaSettings>() as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
                // Output texture (write to history for next frame)
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("taa.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("taa.pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "taa_resolve",
        });

        Ok(Self {
            settings,
            settings_buffer,
            history_textures,
            history_views,
            read_index: 0,
            pipeline,
            bind_group_layout,
            sampler,
            enabled: false,
            width,
            height,
            first_frame: true,
        })
    }

    fn create_history_texture(device: &Device, width: u32, height: u32, index: usize) -> Texture {
        device.create_texture(&TextureDescriptor {
            label: Some(&format!("taa.history.{}", index)),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
    }

    /// Enable/disable TAA
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if enabled {
            self.first_frame = true;
        }
    }

    /// Check if TAA is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Update settings
    pub fn update_settings(&mut self, queue: &Queue, jitter_offset: [f32; 2], frame_index: u32) {
        self.settings.jitter_offset = jitter_offset;
        self.settings.frame_index = frame_index;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::cast_slice(&[self.settings]));
    }

    /// Set history weight (blend factor)
    pub fn set_history_weight(&mut self, weight: f32) {
        self.settings.history_weight = weight.clamp(0.0, 0.99);
    }

    /// Get current history view (read)
    pub fn history_view(&self) -> &TextureView {
        &self.history_views[self.read_index]
    }

    /// Get output history view (write)
    pub fn output_view(&self) -> &TextureView {
        let write_index = 1 - self.read_index;
        &self.history_views[write_index]
    }

    /// P1.4: Get output texture (for copying to another texture)
    pub fn output_texture(&self) -> &Texture {
        let write_index = 1 - self.read_index;
        &self.history_textures[write_index]
    }

    /// Resize TAA buffers
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }

        self.width = width;
        self.height = height;
        self.settings.resolution = [width as f32, height as f32];

        // Recreate history textures
        self.history_textures = [
            Self::create_history_texture(device, width, height, 0),
            Self::create_history_texture(device, width, height, 1),
        ];
        self.history_views = [
            self.history_textures[0].create_view(&TextureViewDescriptor::default()),
            self.history_textures[1].create_view(&TextureViewDescriptor::default()),
        ];

        self.first_frame = true;
    }

    /// Execute TAA resolve pass
    ///
    /// # Arguments
    /// * `device` - GPU device
    /// * `encoder` - Command encoder
    /// * `current_color` - Current frame color (after lighting, before tonemap)
    /// * `velocity_view` - Motion vectors from GBuffer (P1.1)
    /// * `depth_view` - Depth buffer for disocclusion detection
    ///
    /// # Returns
    /// True if TAA was applied (output in history buffer), false if disabled
    pub fn execute(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        queue: &Queue,
        current_color: &TextureView,
        velocity_view: &TextureView,
        depth_view: &TextureView,
    ) -> bool {
        if !self.enabled {
            return false;
        }

        // Update settings buffer
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::cast_slice(&[self.settings]));

        let write_index = 1 - self.read_index;

        // Create bind group
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("taa.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(current_color),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.history_views[self.read_index]),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(velocity_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(depth_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(&self.sampler),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: self.settings_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(&self.history_views[write_index]),
                },
            ],
        });

        // Dispatch compute
        let workgroups_x = (self.width + 7) / 8;
        let workgroups_y = (self.height + 7) / 8;

        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("taa.resolve"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Swap ping-pong buffers
        self.read_index = write_index;
        self.first_frame = false;

        true
    }
}

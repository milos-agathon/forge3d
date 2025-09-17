//! HDR off-screen rendering pipeline
//!
//! Provides high dynamic range off-screen rendering to RGBA16Float textures with
//! tone mapping post-process to sRGB8 output suitable for PNG export and readback.

use crate::core::gpu_timing::GpuTimingManager;
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BlendState, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, CommandEncoder, Device,
    Extent3d, FilterMode, FragmentState, ImageCopyTexture, ImageDataLayout, LoadOp,
    MultisampleState, Operations, Origin3d, PipelineLayoutDescriptor, PrimitiveState, Queue,
    RenderPass, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, SamplerBindingType, SamplerDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, StoreOp, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
    TextureViewDescriptor, TextureViewDimension, VertexState,
};

/// HDR tone mapping operators
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToneMappingOperator {
    /// Simple Reinhard tone mapping: color / (color + 1)
    Reinhard = 0,
    /// Extended Reinhard with white point: color * (1 + color/white²) / (1 + color)
    ReinhardExtended = 1,
    /// Filmic ACES tone mapping
    Aces = 2,
    /// Uncharted 2 filmic tone mapping
    Uncharted2 = 3,
    /// Linear exposure-based mapping
    Exposure = 4,
}

impl Default for ToneMappingOperator {
    fn default() -> Self {
        ToneMappingOperator::Reinhard
    }
}

/// HDR off-screen pipeline configuration
#[derive(Debug, Clone)]
pub struct HdrOffscreenConfig {
    pub width: u32,
    pub height: u32,
    pub hdr_format: TextureFormat,
    pub ldr_format: TextureFormat,
    pub tone_mapping: ToneMappingOperator,
    pub exposure: f32,
    pub white_point: f32,
    pub gamma: f32,
}

impl Default for HdrOffscreenConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            hdr_format: TextureFormat::Rgba16Float,
            ldr_format: TextureFormat::Rgba8UnormSrgb,
            tone_mapping: ToneMappingOperator::Reinhard,
            exposure: 1.0,
            white_point: 4.0,
            gamma: 2.2,
        }
    }
}

/// Tone mapping uniforms for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ToneMappingUniforms {
    pub exposure: f32,
    pub white_point: f32,
    pub gamma: f32,
    pub operator_index: u32, // 0=Reinhard, 1=ReinhardExtended, 2=ACES, etc.
}

/// HDR off-screen rendering pipeline
pub struct HdrOffscreenPipeline {
    pub hdr_texture: Texture,
    pub hdr_view: TextureView,
    pub ldr_texture: Texture,
    pub ldr_view: TextureView,
    pub depth_texture: Texture,
    pub depth_view: TextureView,
    pub config: HdrOffscreenConfig,
    pub tonemap_uniforms: Buffer,
    pub tonemap_bind_group: BindGroup,
    pub tonemap_pipeline: RenderPipeline,
}

impl HdrOffscreenPipeline {
    /// Create new HDR off-screen pipeline
    pub fn new(device: &Device, config: HdrOffscreenConfig) -> Result<Self, String> {
        // Create HDR texture (RGBA16Float color target)
        let hdr_texture = device.create_texture(&TextureDescriptor {
            label: Some("hdr_offscreen_texture"),
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: config.hdr_format,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let hdr_view = hdr_texture.create_view(&TextureViewDescriptor {
            label: Some("hdr_offscreen_view"),
            ..Default::default()
        });

        // Create LDR output texture (sRGB8 output buffer suitable for readback)
        let ldr_texture = device.create_texture(&TextureDescriptor {
            label: Some("ldr_offscreen_texture"),
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: config.ldr_format,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let ldr_view = ldr_texture.create_view(&TextureViewDescriptor {
            label: Some("ldr_offscreen_view"),
            ..Default::default()
        });

        // Create depth texture
        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some("hdr_offscreen_depth"),
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&TextureViewDescriptor {
            label: Some("hdr_offscreen_depth_view"),
            ..Default::default()
        });

        // Create tone mapping uniforms
        let tonemap_uniforms = device.create_buffer(&BufferDescriptor {
            label: Some("tonemap_uniforms"),
            size: std::mem::size_of::<ToneMappingUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create tonemap pipeline
        let tonemap_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("tonemap_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/postprocess_tonemap.wgsl").into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("tonemap_bind_group_layout"),
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
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("tonemap_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let tonemap_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("tonemap_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &tonemap_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &tonemap_shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: config.ldr_format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });

        // Create sampler for HDR texture
        let sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        // Create bind group for tone mapping
        let tonemap_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("tonemap_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&hdr_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: tonemap_uniforms.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            hdr_texture,
            hdr_view,
            ldr_texture,
            ldr_view,
            depth_texture,
            depth_view,
            config,
            tonemap_uniforms,
            tonemap_bind_group,
            tonemap_pipeline,
        })
    }

    /// Begin HDR render pass - renders to off-screen HDR texture
    pub fn begin_hdr_pass<'a>(&'a self, encoder: &'a mut CommandEncoder) -> RenderPass<'a> {
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("hdr_offscreen_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &self.hdr_view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        })
    }

    /// Update tone mapping parameters
    pub fn update_tone_mapping(&self, queue: &Queue) {
        let uniforms = ToneMappingUniforms {
            exposure: self.config.exposure,
            white_point: self.config.white_point,
            gamma: self.config.gamma,
            operator_index: self.config.tone_mapping as u32,
        };

        queue.write_buffer(&self.tonemap_uniforms, 0, bytemuck::cast_slice(&[uniforms]));
    }

    /// Apply tone mapping from HDR to LDR texture - post-process fullscreen pass
    pub fn apply_tone_mapping(&self, encoder: &mut CommandEncoder) {
        self.apply_tone_mapping_with_timing(encoder, None);
    }

    /// Apply tone mapping with optional GPU timing
    pub fn apply_tone_mapping_with_timing(
        &self,
        encoder: &mut CommandEncoder,
        mut timing_manager: Option<&mut GpuTimingManager>,
    ) {
        let timing_scope = if let Some(timer) = timing_manager.as_deref_mut() {
            Some(timer.begin_scope(encoder, "hdr_offscreen_tonemap"))
        } else {
            None
        };

        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("tone_mapping_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &self.ldr_view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(wgpu::Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Fullscreen pass binding hdr texture + sampler to tonemap shader
        render_pass.set_pipeline(&self.tonemap_pipeline);
        render_pass.set_bind_group(0, &self.tonemap_bind_group, &[]);
        render_pass.draw(0..3, 0..1); // Full-screen triangle

        // End render pass before ending timing scope
        drop(render_pass);

        // End GPU timing scope
        if let Some(scope_id) = timing_scope {
            if let Some(timer) = timing_manager.as_deref_mut() {
                timer.end_scope(encoder, scope_id);
            }
        }
    }

    /// Get estimated VRAM usage in bytes
    pub fn get_vram_usage(&self) -> u64 {
        let hdr_size = (self.config.width * self.config.height) as u64
            * match self.config.hdr_format {
                TextureFormat::Rgba16Float => 8,  // 4 channels * 2 bytes
                TextureFormat::Rgba32Float => 16, // 4 channels * 4 bytes
                _ => 8,                           // Default to 16-bit
            };

        let ldr_size = (self.config.width * self.config.height * 4) as u64; // RGBA8
        let depth_size = (self.config.width * self.config.height * 4) as u64; // Depth32Float

        hdr_size + ldr_size + depth_size
    }

    /// Resolve to sRGB8 output buffer suitable for readback
    pub fn read_ldr_data(&self, device: &Device, queue: &Queue) -> Result<Vec<u8>, String> {
        let bpp = 4; // RGBA8
        let unpadded_bytes_per_row = self.config.width * bpp;
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row =
            ((unpadded_bytes_per_row + alignment - 1) / alignment) * alignment;

        let buffer_size = padded_bytes_per_row * self.config.height;

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ldr_staging_buffer"),
            size: buffer_size as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ldr_copy_encoder"),
        });

        encoder.copy_texture_to_buffer(
            ImageCopyTexture {
                texture: &self.ldr_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging_buffer,
                layout: ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(self.config.height),
                },
            },
            Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Map and read the buffer
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .unwrap()
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();

        // Copy LDR data (remove padding)
        let mut ldr_data =
            Vec::with_capacity((self.config.width * self.config.height * 4) as usize);

        for y in 0..self.config.height {
            let row_offset = (y * padded_bytes_per_row) as usize;
            let row_data = &data[row_offset..row_offset + unpadded_bytes_per_row as usize];
            ldr_data.extend_from_slice(row_data);
        }

        drop(data);
        staging_buffer.unmap();

        Ok(ldr_data)
    }

    /// Compute clamp-rate (#pixels channel==0 or 255)/total
    pub fn compute_clamp_rate(&self, device: &Device, queue: &Queue) -> Result<f32, String> {
        let ldr_data = self.read_ldr_data(device, queue)?;

        let total_samples = ldr_data.len(); // Total channel values (width * height * 4)
        let mut clamped_samples = 0;

        for &value in &ldr_data {
            if value == 0 || value == 255 {
                clamped_samples += 1;
            }
        }

        let clamp_rate = clamped_samples as f32 / total_samples as f32;
        Ok(clamp_rate)
    }
}

//! HDR off-screen rendering and tone mapping
//!
//! Provides high dynamic range rendering to floating-point textures with
//! tone mapping operators for converting HDR to LDR display output.

use crate::core::gpu_timing::GpuTimingManager;
use glam::Vec3;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferDescriptor,
    BufferUsages, CommandEncoder, Device, Extent3d, ImageCopyTexture, ImageDataLayout, LoadOp,
    Operations, Origin3d, Queue, RenderPass, RenderPassColorAttachment, RenderPassDescriptor,
    StoreOp, Texture, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
};

/// HDR tone mapping operators
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToneMappingOperator {
    /// Simple Reinhard tone mapping: color / (color + 1)
    Reinhard,
    /// Extended Reinhard with white point: color * (1 + color/white²) / (1 + color)
    ReinhardExtended,
    /// Filmic ACES tone mapping
    Aces,
    /// Uncharted 2 filmic tone mapping
    Uncharted2,
    /// Linear exposure-based mapping
    Exposure,
}

impl Default for ToneMappingOperator {
    fn default() -> Self {
        ToneMappingOperator::Reinhard
    }
}

/// HDR rendering configuration
#[derive(Debug, Clone)]
pub struct HdrConfig {
    pub width: u32,
    pub height: u32,
    pub hdr_format: TextureFormat,
    pub tone_mapping: ToneMappingOperator,
    pub exposure: f32,
    pub white_point: f32,
    pub gamma: f32,
}

impl Default for HdrConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            hdr_format: TextureFormat::Rgba16Float,
            tone_mapping: ToneMappingOperator::Reinhard,
            exposure: 1.0,
            white_point: 4.0,
            gamma: 2.2,
        }
    }
}

/// HDR off-screen render target
pub struct HdrRenderTarget {
    pub hdr_texture: Texture,
    pub hdr_view: TextureView,
    pub ldr_texture: Texture,
    pub ldr_view: TextureView,
    pub depth_texture: Texture,
    pub depth_view: TextureView,
    pub config: HdrConfig,
    pub tonemap_uniforms: Buffer,
    pub tonemap_bind_group: BindGroup,
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

impl HdrRenderTarget {
    /// Create new HDR render target
    pub fn new(device: &Device, config: HdrConfig) -> Result<Self, String> {
        // Create HDR texture (floating-point)
        let hdr_texture = device.create_texture(&TextureDescriptor {
            label: Some("hdr_color_texture"),
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
            label: Some("hdr_color_view"),
            ..Default::default()
        });

        // Create LDR output texture
        let ldr_texture = device.create_texture(&TextureDescriptor {
            label: Some("ldr_color_texture"),
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let ldr_view = ldr_texture.create_view(&TextureViewDescriptor {
            label: Some("ldr_color_view"),
            ..Default::default()
        });

        // Create depth texture
        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some("hdr_depth_texture"),
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
            label: Some("hdr_depth_view"),
            ..Default::default()
        });

        // Create tone mapping uniforms
        let tonemap_uniforms = device.create_buffer(&BufferDescriptor {
            label: Some("tonemap_uniforms"),
            size: std::mem::size_of::<ToneMappingUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout and bind group for tone mapping
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap_bind_group_layout"),
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
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let tonemap_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("tonemap_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: tonemap_uniforms.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&hdr_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&sampler),
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
        })
    }

    /// Begin HDR render pass
    pub fn begin_hdr_pass<'a>(&'a self, encoder: &'a mut CommandEncoder) -> RenderPass<'a> {
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("hdr_render_pass"),
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
    pub fn update_tone_mapping(&self, queue: &Queue, exposure: f32, white_point: f32) {
        let uniforms = ToneMappingUniforms {
            exposure,
            white_point,
            gamma: self.config.gamma,
            operator_index: match self.config.tone_mapping {
                ToneMappingOperator::Reinhard => 0,
                ToneMappingOperator::ReinhardExtended => 1,
                ToneMappingOperator::Aces => 2,
                ToneMappingOperator::Uncharted2 => 3,
                ToneMappingOperator::Exposure => 4,
            },
        };

        queue.write_buffer(&self.tonemap_uniforms, 0, bytemuck::cast_slice(&[uniforms]));
    }

    /// Apply tone mapping from HDR to LDR texture
    pub fn apply_tone_mapping(&self, encoder: &mut CommandEncoder) {
        self.apply_tone_mapping_with_timing(encoder, None);
    }

    /// Apply tone mapping with optional GPU timing
    pub fn apply_tone_mapping_with_timing(
        &self,
        encoder: &mut CommandEncoder,
        mut timing_manager: Option<&mut GpuTimingManager>,
    ) {
        let timing_scope = if let Some(timer) = timing_manager.as_mut() {
            Some(timer.begin_scope(encoder, "hdr_tonemap"))
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

        // This would use a fullscreen quad pipeline to apply tone mapping
        // For now, we'll just record the pass setup
        render_pass.set_bind_group(0, &self.tonemap_bind_group, &[]);

        // Note: In a complete implementation, this would:
        // 1. Set the tone mapping render pipeline
        // 2. Draw a fullscreen triangle
        // 3. The fragment shader would sample HDR and apply tone mapping

        // End render pass before ending timing scope
        drop(render_pass);

        // End GPU timing scope
        if let (Some(timer), Some(scope_id)) = (timing_manager, timing_scope) {
            timer.end_scope(encoder, scope_id);
        }
    }

    /// Read HDR data from texture
    pub fn read_hdr_data(&self, device: &Device, queue: &Queue) -> Result<Vec<f32>, String> {
        let bpp = match self.config.hdr_format {
            TextureFormat::Rgba16Float => 8,  // 4 channels * 2 bytes
            TextureFormat::Rgba32Float => 16, // 4 channels * 4 bytes
            _ => return Err("Unsupported HDR format for readback".to_string()),
        };

        let unpadded_bytes_per_row = self.config.width * bpp;
        let padded_bytes_per_row = {
            let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            ((unpadded_bytes_per_row + alignment - 1) / alignment) * alignment
        };

        let buffer_size = padded_bytes_per_row * self.config.height;

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("hdr_staging_buffer"),
            size: buffer_size as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hdr_copy_encoder"),
        });

        encoder.copy_texture_to_buffer(
            ImageCopyTexture {
                texture: &self.hdr_texture,
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

        // Convert to float data
        let mut hdr_data = Vec::new();

        match self.config.hdr_format {
            TextureFormat::Rgba16Float => {
                for y in 0..self.config.height {
                    let row_offset = (y * padded_bytes_per_row) as usize;
                    for x in 0..self.config.width {
                        let pixel_offset = row_offset + (x * 8) as usize; // 8 bytes per pixel

                        // Read half-float values and convert to f32
                        for c in 0..4 {
                            let half_bytes =
                                [data[pixel_offset + c * 2], data[pixel_offset + c * 2 + 1]];
                            let half_val = half::f16::from_le_bytes(half_bytes);
                            hdr_data.push(half_val.to_f32());
                        }
                    }
                }
            }
            TextureFormat::Rgba32Float => {
                for y in 0..self.config.height {
                    let row_offset = (y * padded_bytes_per_row) as usize;
                    for x in 0..self.config.width {
                        let pixel_offset = row_offset + (x * 16) as usize; // 16 bytes per pixel

                        // Read float values directly
                        for c in 0..4 {
                            let float_bytes = [
                                data[pixel_offset + c * 4],
                                data[pixel_offset + c * 4 + 1],
                                data[pixel_offset + c * 4 + 2],
                                data[pixel_offset + c * 4 + 3],
                            ];
                            let float_val = f32::from_le_bytes(float_bytes);
                            hdr_data.push(float_val);
                        }
                    }
                }
            }
            _ => return Err("Unsupported HDR format".to_string()),
        }

        drop(data);
        staging_buffer.unmap();

        Ok(hdr_data)
    }

    /// Read LDR data from tone-mapped texture
    pub fn read_ldr_data(&self, device: &Device, queue: &Queue) -> Result<Vec<u8>, String> {
        let bpp = 4; // RGBA8
        let unpadded_bytes_per_row = self.config.width * bpp;
        let padded_bytes_per_row = {
            let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            ((unpadded_bytes_per_row + alignment - 1) / alignment) * alignment
        };

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

    /// Resize the HDR render target
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) -> Result<(), String> {
        self.config.width = width;
        self.config.height = height;

        // Recreate textures with new size
        *self = Self::new(device, self.config.clone())?;

        Ok(())
    }
}

/// Apply CPU-side tone mapping to HDR data
pub fn apply_cpu_tone_mapping(
    hdr_data: &[f32],
    width: u32,
    height: u32,
    operator: ToneMappingOperator,
    exposure: f32,
    white_point: f32,
    gamma: f32,
) -> Vec<u8> {
    let mut ldr_data = Vec::with_capacity((width * height * 4) as usize);

    for chunk in hdr_data.chunks(4) {
        let hdr = Vec3::new(chunk[0], chunk[1], chunk[2]) * exposure;

        let tone_mapped = match operator {
            ToneMappingOperator::Reinhard => {
                // Reinhard: color / (color + 1)
                hdr / (hdr + Vec3::ONE)
            }
            ToneMappingOperator::ReinhardExtended => {
                // Extended Reinhard: color * (1 + color/white²) / (1 + color)
                let white_sq = white_point * white_point;
                hdr * (Vec3::ONE + hdr / white_sq) / (Vec3::ONE + hdr)
            }
            ToneMappingOperator::Aces => {
                // ACES filmic tone mapping approximation
                let a = 2.51;
                let b = 0.03;
                let c = 2.43;
                let d = 0.59;
                let e = 0.14;
                (hdr * (hdr * a + b)) / (hdr * (hdr * c + d) + e)
            }
            ToneMappingOperator::Uncharted2 => {
                // Uncharted 2 filmic tone mapping
                fn uncharted2_tonemap_partial(x: Vec3) -> Vec3 {
                    let a = 0.15;
                    let b = 0.50;
                    let c = 0.10;
                    let d = 0.20;
                    let e = 0.02;
                    let f = 0.30;
                    ((x * (x * a + Vec3::splat(c * b)) + Vec3::splat(d * e))
                        / (x * (x * a + b) + Vec3::splat(d * f)))
                        - Vec3::splat(e / f)
                }

                let curr = uncharted2_tonemap_partial(hdr);
                let white_scale = Vec3::ONE / uncharted2_tonemap_partial(Vec3::splat(white_point));
                curr * white_scale
            }
            ToneMappingOperator::Exposure => {
                // Simple exposure mapping
                Vec3::ONE - (-hdr).exp()
            }
        };

        // Apply gamma correction
        let gamma_corrected = tone_mapped.powf(1.0 / gamma);

        // Convert to 8-bit
        let r = (gamma_corrected.x.clamp(0.0, 1.0) * 255.0) as u8;
        let g = (gamma_corrected.y.clamp(0.0, 1.0) * 255.0) as u8;
        let b = (gamma_corrected.z.clamp(0.0, 1.0) * 255.0) as u8;
        let a = (chunk[3].clamp(0.0, 1.0) * 255.0) as u8;

        ldr_data.extend_from_slice(&[r, g, b, a]);
    }

    ldr_data
}

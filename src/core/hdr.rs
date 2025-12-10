//! HDR off-screen rendering and tone mapping
//!
//! Provides high dynamic range rendering to floating-point textures with
//! tone mapping operators for converting HDR to LDR display output.

use crate::core::gpu_timing::GpuTimingManager;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferDescriptor,
    BufferUsages, CommandEncoder, Device, Extent3d, LoadOp, Operations, Queue, RenderPass,
    RenderPassColorAttachment, RenderPassDescriptor, StoreOp, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
    TextureViewDimension,
};

// Re-export types from split modules
pub use super::hdr_readback::{read_hdr_texture, read_ldr_texture};
pub use super::hdr_tonemapping::apply_cpu_tone_mapping;
pub use super::hdr_types::{HdrConfig, ToneMappingOperator, ToneMappingUniforms};

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
            operator_index: self.config.tone_mapping.as_index(),
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
        read_hdr_texture(
            device,
            queue,
            &self.hdr_texture,
            self.config.width,
            self.config.height,
            self.config.hdr_format,
        )
    }

    /// Read LDR data from tone-mapped texture
    pub fn read_ldr_data(&self, device: &Device, queue: &Queue) -> Result<Vec<u8>, String> {
        read_ldr_texture(
            device,
            queue,
            &self.ldr_texture,
            self.config.width,
            self.config.height,
        )
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


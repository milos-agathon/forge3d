//! H16: Weighted Order Independent Transparency (OIT)  
//! Feature-flagged transparent rendering with depth-weighted blending

#[cfg(feature = "weighted-oit")]
use crate::error::RenderError;
#[cfg(feature = "weighted-oit")]
use wgpu::util::DeviceExt;
#[cfg(feature = "weighted-oit")]
use bytemuck::{Pod, Zeroable};

#[cfg(feature = "weighted-oit")]
/// OIT rendering state and resources
pub struct WeightedOIT {
    // Accumulation buffers
    color_buffer: wgpu::Texture,
    reveal_buffer: wgpu::Texture,
    depth_buffer: wgpu::Texture,
    
    // Render targets
    color_view: wgpu::TextureView,
    reveal_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    
    // Compose pipeline for final blend
    compose_pipeline: wgpu::RenderPipeline,
    compose_bind_group: wgpu::BindGroup,
    
    // Screen dimensions
    width: u32,
    height: u32,
    
    // AC2: Store target format for resize
    target_format: wgpu::TextureFormat,
}

#[cfg(feature = "weighted-oit")]
/// OIT fragment uniform data
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct OITUniforms {
    alpha_threshold: f32,      // Alpha cutoff for OIT
    weight_bias: f32,          // Weight function bias
    weight_scale: f32,         // Weight function scale  
    depth_range: f32,          // Depth normalization range
}

#[cfg(feature = "weighted-oit")]
impl WeightedOIT {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        // Create accumulation textures
        let color_buffer = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.OIT.ColorAccum"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float, // High precision for accumulation
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let reveal_buffer = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.OIT.RevealAccum"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float, // Single channel for reveal
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let depth_buffer = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.OIT.Depth"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        // Create texture views
        let color_view = color_buffer.create_view(&wgpu::TextureViewDescriptor::default());
        let reveal_view = reveal_buffer.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth_buffer.create_view(&wgpu::TextureViewDescriptor::default());

        // Create compose shader for final blending
        let compose_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vf.Vector.OIT.Compose"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/oit_compose.wgsl"
            ))),
        });

        // Create compose bind group layout
        let compose_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vf.Vector.OIT.ComposeBindGroupLayout"),
            entries: &[
                // Color accumulation texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Reveal accumulation texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vf.Vector.OIT.Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create compose bind group
        let compose_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vf.Vector.OIT.ComposeBindGroup"),
            layout: &compose_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&reveal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Create compose pipeline layout
        let compose_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vf.Vector.OIT.ComposePipelineLayout"),
            bind_group_layouts: &[&compose_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compose render pipeline
        let compose_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vf.Vector.OIT.ComposePipeline"),
            layout: Some(&compose_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &compose_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &compose_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
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

        Ok(Self {
            color_buffer,
            reveal_buffer,
            depth_buffer,
            color_view,
            reveal_view,
            depth_view,
            compose_pipeline,
            compose_bind_group,
            width,
            height,
            target_format,
        })
    }

    /// Begin OIT accumulation pass
    pub fn begin_accumulation<'pass>(
        &'pass self,
        encoder: &'pass mut wgpu::CommandEncoder,
    ) -> wgpu::RenderPass<'pass> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("vf.Vector.OIT.AccumulationPass"),
            color_attachments: &[
                // Color accumulation buffer
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                // Reveal accumulation buffer
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.reveal_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        })
    }

    /// Compose final image from accumulation buffers
    pub fn compose<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
    ) {
        render_pass.set_pipeline(&self.compose_pipeline);
        render_pass.set_bind_group(0, &self.compose_bind_group, &[]);
        
        // Draw fullscreen triangle
        render_pass.draw(0..3, 0..1);
    }

    /// Resize OIT buffers
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Result<(), RenderError> {
        if self.width == width && self.height == height {
            return Ok(());
        }

        // Recreate textures with new size using stored target format
        let target_format = self.target_format;
        *self = Self::new(device, width, height, target_format)?;
        Ok(())
    }

    /// Get MRT color target states for OIT accumulation pass
    pub fn get_mrt_color_targets() -> [Option<wgpu::ColorTargetState>; 2] {
        [
            // Color accumulation target (Rgba16Float)
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: Some(Self::get_accum_blend_state()),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            // Reveal accumulation target (R16Float)
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R16Float,
                blend: Some(Self::get_reveal_blend_state()),
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ]
    }

    /// Get blend state for color accumulation buffer
    pub fn get_accum_blend_state() -> wgpu::BlendState {
        wgpu::BlendState {
            color: wgpu::BlendComponent {
                // Weighted color accumulation: src_color * src_alpha + dst_color
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                // Weight accumulation: src_alpha + dst_alpha
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }
    }

    /// Get blend state for reveal accumulation buffer
    pub fn get_reveal_blend_state() -> wgpu::BlendState {
        wgpu::BlendState {
            color: wgpu::BlendComponent {
                // Reveal accumulation: (1 - src_alpha) * dst_alpha
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        }
    }

    /// Get ColorTargetState for accumulation buffer (Rgba16Float format)
    pub fn accum_target_state() -> wgpu::ColorTargetState {
        wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
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
        }
    }

    /// Get ColorTargetState for reveal buffer (R16Float format)  
    pub fn reveal_target_state() -> wgpu::ColorTargetState {
        wgpu::ColorTargetState {
            format: wgpu::TextureFormat::R16Float,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            write_mask: wgpu::ColorWrites::ALL,
        }
    }

    /// Calculate OIT weight for fragment
    pub fn calculate_weight(depth: f32, alpha: f32) -> f32 {
        // McGuire & Mara weight function: w = alpha * clamp(0.03 / (1e-5 + pow(z/200, 4)), 1e-2, 3e3)
        let z_norm = depth / 200.0;
        let weight = alpha * (0.03 / (1e-5 + z_norm.powi(4))).clamp(1e-2, 3e3);
        weight
    }
}

#[cfg(not(feature = "weighted-oit"))]
use crate::error::RenderError;

#[cfg(not(feature = "weighted-oit"))]
/// Stub implementation when OIT feature is disabled
pub struct WeightedOIT;

#[cfg(not(feature = "weighted-oit"))]
impl WeightedOIT {
    pub fn new(
        _device: &wgpu::Device,
        _width: u32,
        _height: u32,
        _target_format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        Err(RenderError::Render(
            "Weighted OIT feature not enabled. Build with --features weighted-oit".to_string()
        ))
    }
}

/// Check if weighted OIT feature is enabled
pub fn is_weighted_oit_enabled() -> bool {
    cfg!(feature = "weighted-oit")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "weighted-oit")]
    use crate::error::RenderError;

    #[test]
    fn test_oit_feature_detection() {
        #[cfg(feature = "weighted-oit")]
        assert!(is_weighted_oit_enabled());
        
        #[cfg(not(feature = "weighted-oit"))]
        assert!(!is_weighted_oit_enabled());
    }

    #[cfg(feature = "weighted-oit")]
    #[test]
    fn test_weight_calculation() {
        // Test weight function behavior
        let alpha = 0.5;
        
        // Near depth should have high weight
        let near_weight = WeightedOIT::calculate_weight(1.0, alpha);
        assert!(near_weight > 0.0);
        
        // Far depth should have lower weight
        let far_weight = WeightedOIT::calculate_weight(1000.0, alpha);
        assert!(far_weight > 0.0);
        assert!(near_weight > far_weight);
        
        // Zero alpha should give zero weight
        let zero_weight = WeightedOIT::calculate_weight(1.0, 0.0);
        assert_eq!(zero_weight, 0.0);
    }

    #[cfg(feature = "weighted-oit")]
    #[test]
    fn test_oit_creation() {
        let device = crate::gpu::create_device_for_test();
        let oit = WeightedOIT::new(
            &device,
            512,
            512,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        assert!(oit.is_ok());
    }

    #[cfg(not(feature = "weighted-oit"))]
    #[test]
    fn test_oit_disabled() {
        let device = crate::gpu::create_device_for_test();
        let oit = WeightedOIT::new(
            &device,
            512,
            512,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        assert!(oit.is_err());
        assert!(oit.unwrap_err().to_string().contains("not enabled"));
    }
}
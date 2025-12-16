//! Weighted Order Independent Transparency (OIT).
//!
//! Feature-flagged transparent rendering with depth-weighted blending.
//! Enable with `--features weighted-oit`.

pub mod blend;
#[cfg(feature = "weighted-oit")]
mod pipeline;

pub use blend::{
    accum_target_state, calculate_weight, get_accum_blend_state, get_mrt_color_targets,
    get_reveal_blend_state, reveal_target_state,
};

#[cfg(feature = "weighted-oit")]
use crate::core::error::RenderError;
#[cfg(feature = "weighted-oit")]
use pipeline::{create_accumulation_textures, create_compose_pipeline};

#[cfg(feature = "weighted-oit")]
/// OIT rendering state and resources.
pub struct WeightedOIT {
    #[allow(dead_code)]
    color_buffer: wgpu::Texture,
    #[allow(dead_code)]
    reveal_buffer: wgpu::Texture,
    #[allow(dead_code)]
    depth_buffer: wgpu::Texture,
    color_view: wgpu::TextureView,
    reveal_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    compose_pipeline: wgpu::RenderPipeline,
    compose_bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
    target_format: wgpu::TextureFormat,
}

#[cfg(feature = "weighted-oit")]
impl WeightedOIT {
    /// Create new OIT resources for the given dimensions.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        let (color_buffer, reveal_buffer, depth_buffer) =
            create_accumulation_textures(device, width, height);

        let color_view = color_buffer.create_view(&wgpu::TextureViewDescriptor::default());
        let reveal_view = reveal_buffer.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth_buffer.create_view(&wgpu::TextureViewDescriptor::default());

        let (compose_pipeline, compose_bind_group) =
            create_compose_pipeline(device, target_format, &color_view, &reveal_view);

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

    /// Begin OIT accumulation pass.
    pub fn begin_accumulation<'pass>(
        &'pass self,
        encoder: &'pass mut wgpu::CommandEncoder,
    ) -> wgpu::RenderPass<'pass> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("vf.Vector.OIT.AccumulationPass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.reveal_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
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

    /// Compose final image from accumulation buffers.
    pub fn compose<'pass>(&'pass self, render_pass: &mut wgpu::RenderPass<'pass>) {
        render_pass.set_pipeline(&self.compose_pipeline);
        render_pass.set_bind_group(0, &self.compose_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    /// Resize OIT buffers.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Result<(), RenderError> {
        if self.width == width && self.height == height {
            return Ok(());
        }
        *self = Self::new(device, width, height, self.target_format)?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stub implementation when feature is disabled
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(not(feature = "weighted-oit"))]
use crate::core::error::RenderError;

#[cfg(not(feature = "weighted-oit"))]
/// Stub implementation when OIT feature is disabled.
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
            "Weighted OIT feature not enabled. Build with --features weighted-oit".to_string(),
        ))
    }
}

/// Check if weighted OIT feature is enabled at compile time.
pub fn is_weighted_oit_enabled() -> bool {
    cfg!(feature = "weighted-oit")
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let alpha = 0.5;
        let near_weight = calculate_weight(1.0, alpha);
        let far_weight = calculate_weight(1000.0, alpha);
        assert!(near_weight > 0.0);
        assert!(far_weight > 0.0);
        assert!(near_weight > far_weight);
        assert_eq!(calculate_weight(1.0, 0.0), 0.0);
    }

    #[cfg(feature = "weighted-oit")]
    #[test]
    fn test_oit_creation() {
        let device = crate::core::gpu::create_device_for_test();
        let oit = WeightedOIT::new(&device, 512, 512, wgpu::TextureFormat::Rgba8UnormSrgb);
        assert!(oit.is_ok());
    }

    #[cfg(not(feature = "weighted-oit"))]
    #[test]
    fn test_oit_disabled() {
        let device = crate::core::gpu::create_device_for_test();
        let oit = WeightedOIT::new(&device, 512, 512, wgpu::TextureFormat::Rgba8UnormSrgb);
        assert!(oit.is_err());
    }
}

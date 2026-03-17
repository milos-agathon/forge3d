use super::*;

impl DualSourceOITRenderer {
    pub fn begin_transparency_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        depth_view: &'a wgpu::TextureView,
    ) -> Result<wgpu::RenderPass<'a>, String> {
        match self.get_operating_mode() {
            DualSourceOITMode::DualSource => {
                let color_view = self
                    .dual_source_color_view
                    .as_ref()
                    .ok_or("Dual-source color view not initialized")?;

                Ok(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("DualSourceOIT.TransparencyPass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                }))
            }
            DualSourceOITMode::WBOITFallback => {
                let color_view = self
                    .wboit_color_view
                    .as_ref()
                    .ok_or("WBOIT color view not initialized")?;
                let reveal_view = self
                    .wboit_reveal_view
                    .as_ref()
                    .ok_or("WBOIT reveal view not initialized")?;

                Ok(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("DualSourceOIT.WBOITFallbackPass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: color_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: reveal_view,
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
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                }))
            }
            _ => Err("Dual-source OIT not enabled".to_string()),
        }
    }

    pub fn compose<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) -> Result<(), String> {
        if let Some(bind_group) = &self.compose_bind_group {
            render_pass.set_pipeline(&self.compose_pipeline);
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass.draw(0..3, 0..1);
            Ok(())
        } else {
            Err("Compose bind group not initialized".to_string())
        }
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        if self.width != width || self.height != height {
            self.create_textures(device, width, height)?;
        }
        Ok(())
    }
}

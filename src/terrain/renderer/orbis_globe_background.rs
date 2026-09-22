use super::*;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(in crate::terrain::renderer) struct OrbisGlobeUniforms {
    inverse_view_projection: [f32; 16],
    sphere: [f32; 4],
    base_color: [f32; 4],
    rim_color: [f32; 4],
}

impl TerrainScene {
    pub(in crate::terrain::renderer) fn render_orbis_globe_background(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        render_targets: &crate::terrain::renderer::draw::RenderTargets,
        sky_present: bool,
    ) -> anyhow::Result<bool> {
        if !params.camera_mode.starts_with("clipmap:") {
            return Ok(false);
        }
        if std::env::var_os("ORBIS_DISABLE_GLOBE_BG").is_some() {
            return Ok(false);
        }
        let Some((_center, anchor)) = self.height_streaming_globe_identity() else {
            return Ok(false);
        };
        if !anchor.is_finite() || anchor.length() <= 0.0 {
            anyhow::bail!(
                "ORBIS globe background camera anchor must be finite and non-zero, got {anchor:?}"
            );
        }
        let (eye, view, projection) = Self::build_camera_matrices(params);
        let inverse = (projection * view).inverse();
        if !eye.is_finite() || !inverse.to_cols_array().iter().all(|v| v.is_finite()) {
            anyhow::bail!(
                "ORBIS globe background camera matrices must be finite (eye={eye:?})"
            );
        }
        let uniforms = OrbisGlobeUniforms {
            inverse_view_projection: inverse.to_cols_array(),
            sphere: [
                0.0,
                0.0,
                -(anchor.length() as f32),
                crate::terrain::clipmap::globe::GlobeFrame::WGS84_MEAN_RADIUS_M as f32,
            ],
            base_color: [0.035, 0.095, 0.12, 1.0],
            rim_color: [0.12, 0.34, 0.55, 1.0],
        };
        self.queue
            .write_buffer(&self.orbis_globe_background_uniform, 0, bytemuck::bytes_of(&uniforms));
        let mut cache = self
            .orbis_globe_background_pipeline
            .lock()
            .map_err(|_| anyhow::anyhow!("ORBIS globe background pipeline cache poisoned"))?;
        if cache.as_ref().map(|(samples, _)| *samples) != Some(render_targets.sample_count) {
            *cache = Some((
                render_targets.sample_count,
                Self::create_orbis_globe_background_pipeline(
                    self.device.as_ref(),
                    &self.orbis_globe_background_layout,
                    self.color_format,
                    render_targets.sample_count,
                ),
            ));
        }
        let pipeline = &cache.as_ref().expect("pipeline just inserted").1;
        let load = if sky_present {
            wgpu::LoadOp::Load
        } else {
            wgpu::LoadOp::Clear(wgpu::Color {
                r: 0.1,
                g: 0.1,
                b: 0.15,
                a: 1.0,
            })
        };
        let color_view = render_targets
            .msaa_view
            .as_ref()
            .unwrap_or(&render_targets.internal_view);
        let resolve_target = render_targets
            .msaa_view
            .as_ref()
            .map(|_| &render_targets.internal_view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("orbis.globe.background"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.orbis_globe_background_bind_group, &[]);
        pass.draw(0..3, 0..1);
        Ok(true)
    }
}

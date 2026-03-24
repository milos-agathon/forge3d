#![cfg(feature = "enable-gpu-instancing")]

use super::draw::RenderTargets;
use super::*;
use crate::terrain::renderer::core::TERRAIN_DEPTH_FORMAT;

use crate::terrain::scatter::{
    accumulate_frame_stats, compute_wind_uniforms, summarize_memory, ScatterWindSettingsNative,
    TerrainScatterBatch, TerrainScatterFrameStats, TerrainScatterLevelSpec,
    TerrainScatterMemoryReport,
};

pub(super) struct ScatterRenderState {
    pub(super) view: glam::Mat4,
    pub(super) proj: glam::Mat4,
    pub(super) eye_contract: glam::Vec3,
    pub(super) render_from_contract: glam::Mat4,
    pub(super) instance_scale: f32,
    pub(super) light_dir: [f32; 3],
    pub(super) light_intensity: f32,
    pub(super) time_seconds: f32,
}

pub(super) struct TerrainScatterUploadBatch {
    pub(super) name: Option<String>,
    pub(super) color: [f32; 4],
    pub(super) max_draw_distance: Option<f32>,
    pub(super) transforms_rowmajor: Vec<[f32; 16]>,
    pub(super) levels: Vec<TerrainScatterLevelSpec>,
    pub(super) wind: ScatterWindSettingsNative,
}

impl TerrainScene {
    pub(super) fn set_scatter_batches_native(
        &mut self,
        batches: Vec<TerrainScatterUploadBatch>,
    ) -> Result<()> {
        let mut gpu_batches = Vec::with_capacity(batches.len());
        for batch in batches {
            gpu_batches.push(TerrainScatterBatch::new(
                self.device.as_ref(),
                self.queue.as_ref(),
                batch.levels,
                &batch.transforms_rowmajor,
                batch.color,
                batch.max_draw_distance,
                batch.name,
                batch.wind,
            )?);
        }

        self.scatter_batches = gpu_batches;
        self.scatter_last_frame_stats = TerrainScatterFrameStats::default();
        Ok(())
    }

    pub(super) fn clear_scatter_batches_native(&mut self) {
        self.scatter_batches.clear();
        self.scatter_last_frame_stats = TerrainScatterFrameStats::default();
    }

    pub(super) fn scatter_memory_report(&self) -> TerrainScatterMemoryReport {
        summarize_memory(&self.scatter_batches)
    }

    pub(super) fn scatter_last_frame_stats(&self) -> TerrainScatterFrameStats {
        self.scatter_last_frame_stats.clone()
    }

    pub(super) fn build_scatter_render_state(
        &self,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        heightmap_width: u32,
        heightmap_height: u32,
        view: glam::Mat4,
        proj: glam::Mat4,
        eye_render: glam::Vec3,
        time_seconds: f32,
    ) -> ScatterRenderState {
        let terrain_width = heightmap_width.max(heightmap_height).max(1) as f32;
        let terrain_span = params.terrain_span.max(1e-3);
        let scale_xy = terrain_span / terrain_width;
        let centered_z_offset =
            -0.5 * (decoded.clamp.height_range.1 - decoded.clamp.height_range.0) * params.z_scale;

        // Shared scatter contract:
        // x/z span the terrain footprint in [0, terrain_width], y is height above min * z_scale.
        // The mesh-mode terrain path renders with elevation centered around zero in clip space,
        // so scatter must use the same centered height convention.
        let render_from_contract = glam::Mat4::from_cols_array(&[
            scale_xy,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            scale_xy,
            0.0,
            0.0,
            -terrain_span * 0.5,
            -terrain_span * 0.5,
            centered_z_offset,
            1.0,
        ]);
        let eye_contract = render_from_contract.inverse().transform_point3(eye_render);

        ScatterRenderState {
            view,
            proj,
            eye_contract,
            render_from_contract,
            instance_scale: scale_xy,
            light_dir: decoded.light.direction,
            light_intensity: decoded.light.intensity,
            time_seconds,
        }
    }

    pub(super) fn render_scatter_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        render_targets: &RenderTargets,
        state: &ScatterRenderState,
    ) -> Result<()> {
        if self.scatter_batches.is_empty() {
            self.scatter_last_frame_stats = TerrainScatterFrameStats::default();
            return Ok(());
        }

        self.ensure_scatter_renderer_sample_count(render_targets.sample_count);

        let color_view = render_targets
            .msaa_view
            .as_ref()
            .unwrap_or(&render_targets.internal_view);
        let resolve_target = if render_targets.msaa_view.is_some() {
            Some(&render_targets.internal_view)
        } else {
            None
        };

        let device = self.device.as_ref();
        let queue = self.queue.as_ref();
        let renderer = &mut self.scatter_renderer;
        renderer.reset_draw_batch_uniforms();
        let mut frame_stats = TerrainScatterFrameStats::default();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("terrain.scatter.render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &render_targets.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        for batch in &mut self.scatter_batches {
            let (batch_stats, draws) = batch.prepare_draws(
                device,
                queue,
                state.eye_contract,
                state.render_from_contract,
                state.instance_scale,
            )?;
            accumulate_frame_stats(&mut frame_stats, &batch_stats);

            // Compute batch-constant wind fields
            let base_wind = compute_wind_uniforms(
                &batch.wind,
                state.time_seconds,
                0.0, // placeholder, overridden per-draw
                state.instance_scale,
            );

            for draw in draws {
                let Some(instbuf) = batch.level_instbuf(draw.level_index) else {
                    continue;
                };
                // Inject per-draw mesh_height_max
                let mut wind = base_wind;
                wind.wind_vec_bounds[3] = batch.level_mesh_height_max(draw.level_index);

                renderer.draw_batch_params(
                    device,
                    &mut pass,
                    queue,
                    state.view,
                    state.proj,
                    batch.color,
                    state.light_dir,
                    state.light_intensity,
                    wind.wind_phase,
                    wind.wind_vec_bounds,
                    wind.wind_bend_fade,
                    batch.level_vbuf(draw.level_index),
                    batch.level_ibuf(draw.level_index),
                    instbuf,
                    batch.level_index_count(draw.level_index),
                    draw.instance_count,
                );
            }
        }

        drop(pass);
        self.scatter_last_frame_stats = frame_stats;
        Ok(())
    }

    fn ensure_scatter_renderer_sample_count(&mut self, sample_count: u32) {
        let sample_count = sample_count.max(1);
        if self.scatter_renderer_sample_count == sample_count {
            return;
        }

        self.scatter_renderer =
            crate::render::mesh_instanced::MeshInstancedRenderer::new_with_depth_state(
                self.device.as_ref(),
                self.color_format,
                Some(TERRAIN_DEPTH_FORMAT),
                sample_count,
                wgpu::CompareFunction::LessEqual,
                false,
            );
        self.scatter_renderer_sample_count = sample_count;
    }
}

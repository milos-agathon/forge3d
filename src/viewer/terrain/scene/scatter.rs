#[cfg(feature = "enable-gpu-instancing")]
use super::*;

#[cfg(feature = "enable-gpu-instancing")]
use crate::terrain::scatter::{
    accumulate_frame_stats, compute_wind_uniforms, summarize_memory, TerrainScatterBatch,
    TerrainScatterFrameStats, TerrainScatterLevelSpec, TerrainScatterMemoryReport,
};

#[cfg(feature = "enable-gpu-instancing")]
fn viewer_render_from_contract(_use_pbr: bool, _terrain_width: f32, _h_range: f32) -> glam::Mat4 {
    // The terrain viewer renders in terrain-width units:
    // x/z cover [0, terrain_width], and both terrain shaders resolve world_y to
    // (height - min_height) * z_scale despite differing uniform formulas.
    //
    // TerrainScatterSource already emits that same contract, so the viewer path should not
    // introduce an extra contract->world transform. Callers must size orbit radius and scatter
    // draw distances in terrain-width units because the viewer does not preserve DEM span metadata.
    glam::Mat4::IDENTITY
}

#[cfg(feature = "enable-gpu-instancing")]
pub(in crate::viewer::terrain) fn render_scatter_batches(
    encoder: &mut wgpu::CommandEncoder,
    color_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    batches: &mut [TerrainScatterBatch],
    use_pbr: bool,
    view: glam::Mat4,
    proj: glam::Mat4,
    eye_render: glam::Vec3,
    terrain_width: f32,
    h_range: f32,
    light_dir: [f32; 3],
    light_intensity: f32,
    elapsed_time: f32,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut crate::render::mesh_instanced::MeshInstancedRenderer,
) -> Result<TerrainScatterFrameStats> {
    if batches.is_empty() {
        return Ok(TerrainScatterFrameStats::default());
    }

    let render_from_contract = viewer_render_from_contract(use_pbr, terrain_width, h_range);
    let eye_contract = render_from_contract.inverse().transform_point3(eye_render);

    renderer.reset_draw_batch_uniforms();
    let mut frame_stats = TerrainScatterFrameStats::default();
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("terrain_viewer.scatter_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
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
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    // Viewer uses identity render_from_contract, so instance_scale is 1.0.
    let instance_scale = 1.0_f32;

    for batch in batches {
        let (batch_stats, draws) = batch.prepare_draws(
            device,
            queue,
            eye_contract,
            render_from_contract,
            instance_scale,
        )?;
        accumulate_frame_stats(&mut frame_stats, &batch_stats);

        // Compute batch-constant wind fields
        let base_wind = compute_wind_uniforms(
            &batch.wind,
            elapsed_time,
            0.0, // placeholder, overridden per-draw
            instance_scale,
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
                view,
                proj,
                batch.color,
                light_dir,
                light_intensity,
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
    Ok(frame_stats)
}

impl ViewerTerrainScene {
    pub fn set_scatter_batches_from_configs(
        &mut self,
        batches: &[crate::viewer::viewer_enums::ViewerTerrainScatterBatchConfig],
    ) -> Result<()> {
        let mut gpu_batches = Vec::with_capacity(batches.len());
        for batch in batches {
            let levels = batch
                .levels
                .iter()
                .cloned()
                .map(|level| TerrainScatterLevelSpec {
                    mesh: level.mesh,
                    max_distance: level.max_distance,
                })
                .collect::<Vec<_>>();
            gpu_batches.push(TerrainScatterBatch::new(
                self.device.as_ref(),
                self.queue.as_ref(),
                levels,
                &batch.transforms,
                batch.color,
                batch.max_draw_distance,
                batch.name.clone(),
                batch.wind.clone(),
            )?);
        }

        self.scatter_batches = gpu_batches;
        self.scatter_last_frame_stats = TerrainScatterFrameStats::default();
        Ok(())
    }

    /// Accumulate elapsed time for scatter wind animation.
    pub fn tick_scatter_time(&mut self, dt: f32) {
        self.scatter_elapsed_time += dt;
    }

    pub fn clear_scatter_batches(&mut self) {
        self.scatter_batches.clear();
        self.scatter_last_frame_stats = TerrainScatterFrameStats::default();
    }

    pub fn scatter_memory_report(&self) -> TerrainScatterMemoryReport {
        summarize_memory(&self.scatter_batches)
    }

    pub fn scatter_last_frame_stats(&self) -> TerrainScatterFrameStats {
        self.scatter_last_frame_stats.clone()
    }

    pub(in crate::viewer::terrain) fn render_scatter_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        batches: &mut [TerrainScatterBatch],
        use_pbr: bool,
        view: glam::Mat4,
        proj: glam::Mat4,
        eye_render: glam::Vec3,
        terrain_width: f32,
        h_range: f32,
        light_dir: [f32; 3],
        light_intensity: f32,
    ) -> Result<TerrainScatterFrameStats> {
        render_scatter_batches(
            encoder,
            color_view,
            depth_view,
            batches,
            use_pbr,
            view,
            proj,
            eye_render,
            terrain_width,
            h_range,
            light_dir,
            light_intensity,
            self.scatter_elapsed_time,
            self.device.as_ref(),
            self.queue.as_ref(),
            &mut self.scatter_renderer,
        )
    }
}

#[cfg(all(test, feature = "enable-gpu-instancing"))]
mod tests {
    use super::*;

    #[test]
    fn viewer_render_contract_is_identity_for_all_viewer_modes() {
        assert_eq!(
            viewer_render_from_contract(false, 96.0, 1200.0),
            glam::Mat4::IDENTITY
        );
        assert_eq!(
            viewer_render_from_contract(true, 96.0, 1200.0),
            glam::Mat4::IDENTITY
        );
    }
}

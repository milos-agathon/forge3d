#[cfg(feature = "enable-gpu-instancing")]
use super::*;

#[cfg(feature = "enable-gpu-instancing")]
use crate::terrain::scatter::{
    accumulate_frame_stats, compute_wind_uniforms, pack_hlod_identity_instance,
    TerrainScatterBatch, TerrainScatterBlendConfig, TerrainScatterContactConfig,
    TerrainScatterFrameStats, TerrainScatterLevelSpec,
};

#[cfg(feature = "enable-gpu-instancing")]
fn viewer_render_from_contract(
    render_origin_span: [f32; 4],
    terrain_dimensions: (u32, u32),
) -> (glam::Mat4, f32) {
    let width = terrain_dimensions.0.max(1) as f32;
    let depth = terrain_dimensions.1.max(1) as f32;
    let scale_x = render_origin_span[2] / width;
    let scale_z = render_origin_span[3] / depth;
    let transform = glam::Mat4::from_scale_rotation_translation(
        glam::Vec3::new(scale_x, 1.0, scale_z),
        glam::Quat::IDENTITY,
        glam::Vec3::new(render_origin_span[0], 0.0, render_origin_span[1]),
    );
    // Scatter meshes are local, so preserve their proportions under a raster
    // with non-square pixels while mapping instance positions exactly.
    let instance_scale = (scale_x.abs() * scale_z.abs()).sqrt();
    (transform, instance_scale)
}

#[cfg(feature = "enable-gpu-instancing")]
pub(in crate::viewer::terrain) fn render_scatter_batches(
    encoder: &mut wgpu::CommandEncoder,
    color_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    batches: &mut [TerrainScatterBatch],
    view: glam::Mat4,
    proj: glam::Mat4,
    eye_render: glam::Vec3,
    heightmap_view: &wgpu::TextureView,
    render_origin_span: [f32; 4],
    terrain_dimensions: (u32, u32),
    terrain_min_height: f32,
    z_scale: f32,
    light_dir: [f32; 3],
    light_intensity: f32,
    elapsed_time: f32,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut crate::render::mesh_instanced::MeshInstancedRenderer,
    hlod_instance_buffer: &wgpu::Buffer,
) -> Result<TerrainScatterFrameStats> {
    if batches.is_empty() {
        return Ok(TerrainScatterFrameStats::default());
    }

    let (render_from_contract, instance_scale) =
        viewer_render_from_contract(render_origin_span, terrain_dimensions);
    let eye_contract = render_from_contract.inverse().transform_point3(eye_render);

    renderer.reset_draw_batch_uniforms();
    renderer.set_terrain_context(
        device,
        queue,
        Some(crate::render::mesh_instanced::TerrainBlendContext {
            heightmap_view,
            world_to_uv_scale_bias: [
                1.0 / render_origin_span[2],
                1.0 / render_origin_span[3],
                -render_origin_span[0] / render_origin_span[2],
                -render_origin_span[1] / render_origin_span[3],
            ],
            height_to_world: [z_scale, -terrain_min_height * z_scale, 0.0, 0.0],
        }),
    );
    let mut frame_stats = TerrainScatterFrameStats::default();
    let identity_packed = pack_hlod_identity_instance(render_from_contract);
    queue.write_buffer(
        hlod_instance_buffer,
        0,
        bytemuck::cast_slice(&identity_packed),
    );

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
                batch.terrain_blend.uniform(),
                batch.terrain_contact.uniform(),
                None,
                batch.level_vbuf(draw.level_index),
                batch.level_ibuf(draw.level_index),
                instbuf,
                batch.level_index_count(draw.level_index),
                draw.instance_count,
            );
        }

        // Draw active HLOD clusters
        let active_clusters = batch.hlod_active_clusters(eye_contract);
        for cluster_idx in active_clusters {
            if let (Some(vbuf), Some(ibuf)) = (
                batch.hlod_cluster_vbuf(cluster_idx),
                batch.hlod_cluster_ibuf(cluster_idx),
            ) {
                let index_count = batch.hlod_cluster_index_count(cluster_idx);
                renderer.draw_batch_params(
                    device,
                    &mut pass,
                    queue,
                    view,
                    proj,
                    batch.color,
                    light_dir,
                    light_intensity,
                    [0.0; 4],
                    [0.0; 4],
                    [0.0; 4],
                    batch.terrain_blend.uniform(),
                    batch.terrain_contact.uniform(),
                    None,
                    vbuf,
                    ibuf,
                    hlod_instance_buffer,
                    index_count,
                    1,
                );
            }
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
            let mut gpu_batch = TerrainScatterBatch::new(
                self.device.as_ref(),
                self.queue.as_ref(),
                levels,
                &batch.transforms,
                batch.color,
                batch.max_draw_distance,
                batch.name.clone(),
                batch.wind.clone(),
                batch.hlod_config.clone(),
                TerrainScatterBlendConfig {
                    enabled: batch.terrain_blend.enabled,
                    bury_depth: batch.terrain_blend.bury_depth,
                    fade_distance: batch.terrain_blend.fade_distance,
                },
                TerrainScatterContactConfig {
                    enabled: batch.terrain_contact.enabled,
                    distance: batch.terrain_contact.distance,
                    strength: batch.terrain_contact.strength,
                    vertical_weight: batch.terrain_contact.vertical_weight,
                },
            )?;
            gpu_batch.preallocate_instance_buffers(self.device.as_ref())?;
            gpu_batches.push(gpu_batch);
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
}

#[cfg(all(test, feature = "enable-gpu-instancing"))]
mod tests {
    use super::*;

    #[test]
    fn viewer_render_contract_maps_pixel_contract_to_anchored_physical_span() {
        let (transform, instance_scale) =
            viewer_render_from_contract([-125.0, 40.0, 600.0, 200.0], (20, 10));
        assert_eq!(
            transform.transform_point3(glam::Vec3::new(0.0, 7.0, 0.0)),
            glam::Vec3::new(-125.0, 7.0, 40.0)
        );
        assert_eq!(
            transform.transform_point3(glam::Vec3::new(20.0, 7.0, 10.0)),
            glam::Vec3::new(475.0, 7.0, 240.0)
        );
        assert!((instance_scale - (600.0_f32).sqrt()).abs() < 1e-5);
    }
}

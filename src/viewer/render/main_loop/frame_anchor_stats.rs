use crate::viewer::viewer_types::{ActiveCameraKind, FrameCamera};
use crate::viewer::Viewer;

impl Viewer {
    pub(crate) fn update_frame_anchor_stats(&self, frame: FrameCamera) {
        let (
            point_count,
            visible_point_count,
            source_bytes,
            render_cache_bytes,
            gpu_instance_bytes,
            gpu_instance_id,
        ) = self
            .point_cloud
            .as_ref()
            .filter(|_| self.point_cloud_active())
            .map(|point_cloud| {
                let count = point_cloud.point_count as u64;
                (
                    count,
                    point_cloud.visible_point_count as u64,
                    count * std::mem::size_of::<crate::viewer::pointcloud::PointSource3D>() as u64,
                    count
                        * std::mem::size_of::<crate::viewer::pointcloud::PointInstance3D>() as u64,
                    point_cloud.instance_buffer_bytes(),
                    point_cloud.instance_buffer_id(),
                )
            })
            .unwrap_or((0, 0, 0, 0, 0, 0));
        let (terrain_revision, terrain_heightmap_id, terrain_heightmap_bytes, shadow_revision) =
            self.terrain_viewer
                .as_ref()
                .and_then(|scene| scene.terrain.as_ref().map(|terrain| (scene, terrain)))
                .map(|(scene, terrain)| {
                    (
                        terrain.revision,
                        terrain._heightmap_texture.ledger_id(),
                        u64::from(terrain.dimensions.0)
                            * u64::from(terrain.dimensions.1)
                            * std::mem::size_of::<f32>() as u64,
                        scene.shadow_bind_group_revision,
                    )
                })
                .unwrap_or((0, 0, 0, 0));
        let (vector_source_bytes, vector_render_cache_bytes, vector_gpu_bytes, vector_gpu_ids) =
            self.terrain_viewer.as_ref().map_or_else(
                || (0, 0, 0, Vec::new()),
                |terrain| terrain.vector_overlay_memory_evidence(),
            );
        let vector_bvh_cpu_bytes = self.unified_picking.cpu_bvh_bytes();
        let active_camera = match frame.kind {
            ActiveCameraKind::Terrain => "terrain",
            ActiveCameraKind::PointCloud => "point_cloud",
            ActiveCameraKind::General => "general",
        };
        let (taa_enabled, taa_history_valid) = self
            .terrain_viewer
            .as_ref()
            .filter(|terrain| terrain.has_terrain())
            .map(|terrain| (terrain.is_taa_enabled(), terrain.taa_history_valid()))
            .unwrap_or_else(|| {
                self.taa_renderer.as_ref().map_or((false, false), |taa| {
                    (taa.is_enabled(), taa.history_valid())
                })
            });
        let (
            ssgi_enabled,
            ssgi_temporal_enabled,
            ssgi_history_valid,
            ssr_effect_enabled,
            ssr_history_valid,
        ) = self
            .gi
            .as_ref()
            .map(|gi| {
                use crate::core::screen_space_effects::ScreenSpaceEffect;
                (
                    gi.is_enabled(ScreenSpaceEffect::SSGI),
                    gi.ssgi_settings()
                        .is_some_and(|settings| settings.temporal_enabled != 0),
                    gi.ssgi_history_valid(),
                    gi.is_enabled(ScreenSpaceEffect::SSR),
                    gi.ssr_history_valid(),
                )
            })
            .unwrap_or((false, false, false, false, false));
        let taa_history_ids = self
            .terrain_viewer
            .as_ref()
            .filter(|terrain| terrain.has_terrain())
            .map(crate::viewer::terrain::ViewerTerrainScene::taa_history_allocation_ids)
            .unwrap_or_else(|| {
                self.taa_renderer.as_ref().map_or(
                    [0, 0],
                    crate::core::taa::TaaRenderer::history_allocation_ids,
                )
            });
        let effect_history_ids = self
            .gi
            .as_ref()
            .map_or([0, 0], |gi| gi.temporal_history_allocation_ids());
        let temporal_history_ids = [
            taa_history_ids[0],
            taa_history_ids[1],
            effect_history_ids[0],
            effect_history_ids[1],
            self.fog_history.ledger_id(),
            self.fog_history_half.ledger_id(),
        ];
        crate::viewer::event_loop::update_ipc_frame_stats(
            &self.adapter_name,
            self.adapter_vendor,
            self.adapter_device,
            &self.adapter_backend,
            &self.adapter_device_type,
            &self.adapter_driver,
            &self.adapter_driver_info,
            active_camera,
            frame.anchor.origin().to_array(),
            self.camera_rebase_count,
            self.history_invalidation_count,
            self.last_vector_source_delta,
            self.last_vector_packed_delta,
            vector_source_bytes,
            vector_render_cache_bytes,
            vector_gpu_bytes,
            vector_gpu_ids,
            vector_bvh_cpu_bytes,
            self.frame_count,
            taa_enabled,
            taa_history_valid,
            ssgi_enabled,
            ssgi_temporal_enabled,
            ssgi_history_valid,
            ssr_effect_enabled && self.ssr_params.ssr_enable,
            ssr_history_valid,
            self.fog_enabled,
            self.fog_history_state.is_valid(),
            temporal_history_ids,
            terrain_revision,
            terrain_heightmap_id,
            terrain_heightmap_bytes,
            shadow_revision,
            point_count,
            visible_point_count,
            source_bytes,
            render_cache_bytes,
            gpu_instance_bytes,
            gpu_instance_id,
        );
    }
}

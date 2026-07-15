use glam::{DVec3, Mat4};

use crate::viewer::viewer_types::{ActiveCameraKind, FrameCamera};
use crate::viewer::Viewer;

impl Viewer {
    /// Single point-cloud activity predicate shared by camera ownership,
    /// input, rendering, snapshots, statistics, culling, and picking.
    pub(crate) fn point_cloud_active(&self) -> bool {
        self.point_cloud.as_ref().is_some_and(|point_cloud| {
            point_cloud.visible
                && point_cloud.point_count > 0
                && point_cloud.instance_buffer.is_some()
        })
    }

    fn focus_for_frame(frame: FrameCamera) -> DVec3 {
        match frame.kind {
            ActiveCameraKind::Terrain => {
                DVec3::new(frame.target_world.x, 0.0, frame.target_world.z)
            }
            ActiveCameraKind::PointCloud | ActiveCameraKind::General => frame.eye_world,
        }
    }

    /// Prospective copied frame used by command validators before mutation.
    pub(crate) fn prospective_frame_camera(&self) -> FrameCamera {
        let mut frame = self.selected_frame_camera();
        frame.anchor.rebase_if_needed(Self::focus_for_frame(frame));
        frame
    }

    pub(crate) fn validate_content_points<I>(&self, points: I) -> Result<(), String>
    where
        I: IntoIterator<Item = DVec3>,
    {
        let frame = self.prospective_frame_camera();
        for point in points {
            crate::viewer::camera_controller::validate_world_point(
                crate::viewer::camera_controller::CoordRole::Content,
                point,
                &frame.anchor,
            )
            .map_err(|err| err.to_string())?;
        }
        Ok(())
    }

    /// Select the one camera pose that owns every pass in this frame.
    pub(crate) fn selected_frame_camera(&self) -> FrameCamera {
        if let Some(terrain) = self
            .terrain_viewer
            .as_ref()
            .and_then(|scene| scene.terrain.as_ref())
        {
            return FrameCamera {
                kind: ActiveCameraKind::Terrain,
                anchor: self.camera_anchor,
                eye_world: terrain.camera_eye(),
                target_world: terrain.camera_target(),
                up: terrain.camera_up(),
                fov_deg: terrain.cam_fov_deg,
                near: (terrain.cam_radius * 0.001).clamp(0.001, 1.0),
                far: (terrain.cam_radius * 10.0).max(10.0),
            };
        }

        if self.point_cloud_active() {
            let point_cloud = self.point_cloud.as_ref().expect("activity checked");
            let (eye_world, target_world, near, far) = point_cloud.camera_pose_world();
            return FrameCamera {
                kind: ActiveCameraKind::PointCloud,
                anchor: self.camera_anchor,
                eye_world,
                target_world,
                up: glam::Vec3::Y,
                fov_deg: 45.0,
                near,
                far,
            };
        }

        FrameCamera {
            kind: ActiveCameraKind::General,
            anchor: self.camera_anchor,
            eye_world: self.camera.eye(),
            target_world: self.camera.target(),
            up: self.camera.up(),
            fov_deg: self.view_config.fov_deg,
            near: self.view_config.znear,
            far: self.view_config.zfar,
        }
    }

    /// Sole production rebase site for the interactive viewer.
    pub(crate) fn prepare_frame_anchor(&mut self) {
        let before = self.selected_frame_camera();
        let focus = Self::focus_for_frame(before);
        let rebased = self.camera_anchor.rebase_if_needed(focus);
        let frame = self.selected_frame_camera();
        self.frame_camera = Some(frame);
        if rebased {
            self.camera_rebase_count = self.camera_rebase_count.wrapping_add(1);
            self.refresh_after_rebase(frame);
        }
        self.update_frame_anchor_stats(frame);
    }

    pub(crate) fn current_frame_camera(&self) -> FrameCamera {
        self.frame_camera
            .unwrap_or_else(|| self.selected_frame_camera())
    }

    /// Model matrix for local object vertices. Translation is applied exactly
    /// once, after f64 anchor subtraction; rotation and scale stay local f32.
    pub(crate) fn anchored_object_model(&self, frame: FrameCamera) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            self.object_scale,
            self.object_rotation,
            frame.anchor.model_offset(self.object_translation),
        )
    }

    fn refresh_after_rebase(&mut self, frame: FrameCamera) {
        if let Some(point_cloud) = self.point_cloud.as_mut() {
            point_cloud.repack_for_anchor(&self.queue, &frame.anchor);
        }
        let vector_bvh_sources = if let Some(terrain_viewer) = self.terrain_viewer.as_mut() {
            terrain_viewer.refresh_anchor_caches(&frame.anchor);
            terrain_viewer.invalidate_temporal_history();
            terrain_viewer.all_vector_overlay_render_data()
        } else {
            Vec::new()
        };
        for (id, name, vertices, indices, primitive) in vector_bvh_sources {
            if let Some(layer) = crate::viewer::terrain::vector_overlay::build_layer_bvh(
                id, &name, &vertices, &indices, primitive,
            ) {
                self.unified_picking.register_layer_bvh(layer);
            } else {
                self.unified_picking.remove_layer_bvh(id);
            }
        }
        if let Some(gi) = self.gi.as_mut() {
            gi.invalidate_temporal_histories();
        }
        if let Some(taa) = self.taa_renderer.as_mut() {
            taa.reset_history();
        }

        let current_vp = frame.projection(self.config.width, self.config.height)
            * frame.view()
            * self.anchored_object_model(frame);
        self.prev_view_proj = current_vp;
        self.taa_jitter = if self.taa_jitter.enabled {
            crate::core::jitter::JitterState::enabled()
        } else {
            crate::core::jitter::JitterState::new()
        };
        self.fog_frame_index = 0;
        self.fog_history_state.invalidate();
        self.history_invalidation_count = self.history_invalidation_count.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Anchor;

    #[test]
    fn stationary_terrain_target_focus_does_not_rebase_during_orbit() {
        let mut anchor = Anchor::new();
        let target = DVec3::new(6_378_137.0, 40.0, 500_000.0);
        let focus = DVec3::new(target.x, 0.0, target.z);
        assert!(anchor.rebase_if_needed(focus));
        for _ in 0..360 {
            assert!(!anchor.rebase_if_needed(focus));
        }
    }
}

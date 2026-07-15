use glam::{DVec3, Vec3};

use super::command_preflight_helpers::{
    distinct_value, point_repack_failpoint_blocks_publish, validate_points,
};
use crate::viewer::camera_controller::CoordRole;
use crate::viewer::pointcloud::preflight_laz_bounds;
use crate::viewer::terrain::ViewerTerrainScene;
use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

impl Viewer {
    pub(crate) fn preflight_command_batch(&self, commands: &[ViewerCmd]) -> Result<(), String> {
        let current = self.selected_frame_camera();
        let mut terrain_present = self
            .terrain_viewer
            .as_ref()
            .is_some_and(ViewerTerrainScene::has_terrain);
        let mut terrain_focus = self
            .terrain_viewer
            .as_ref()
            .and_then(|scene| scene.terrain.as_ref())
            .map(|terrain| terrain.cam_target);
        let mut terrain_span = self
            .terrain_viewer
            .as_ref()
            .and_then(|scene| scene.terrain.as_ref())
            .map(|terrain| terrain.world_span_xz.max_element());
        let mut point_loaded = self
            .point_cloud
            .as_ref()
            .is_some_and(|cloud| cloud.point_count > 0);
        let mut point_present = self
            .point_cloud
            .as_ref()
            .is_some_and(|cloud| cloud.visible && cloud.point_count > 0);
        let mut point_center = self.point_cloud.as_ref().map(|cloud| cloud.center);
        let mut point_extent = self.point_cloud.as_ref().map(|cloud| cloud.extent_render);
        let mut point_camera_radius = self
            .point_cloud
            .as_ref()
            .map_or(1.0, |cloud| cloud.cam_radius);
        let mut general_eye = current.eye_world;
        let mut general_target = current.target_world;
        let mut object_translation = self.object_translation;
        let object_present = !self.object_source_positions.is_empty();
        let mut object_transform_requested = false;
        let mut requested_terrain_path: Option<&str> = None;
        let mut requested_point_path: Option<&str> = None;
        let mut requested_terrain_target: Option<[f64; 3]> = None;
        let mut requested_general_pose: Option<([f64; 3], [f64; 3])> = None;

        for cmd in commands {
            match cmd {
                ViewerCmd::LoadTerrain(path) => {
                    distinct_value(&mut requested_terrain_path, path.as_str(), "terrain paths")?;
                    let preflight = ViewerTerrainScene::preflight_terrain_path(path)
                        .map_err(|error| error.to_string())?;
                    terrain_present = true;
                    terrain_focus = Some(preflight.default_focus());
                    terrain_span = Some(preflight.max_horizontal_span());
                }
                ViewerCmd::SetTerrainCamera {
                    target: Some(target),
                    ..
                }
                | ViewerCmd::SetTerrain {
                    target: Some(target),
                    ..
                } => {
                    distinct_value(&mut requested_terrain_target, *target, "terrain targets")?;
                }
                ViewerCmd::LoadPointCloud {
                    path,
                    point_size,
                    max_points,
                    ..
                } => {
                    if !point_size.is_finite() || *point_size <= 0.0 || *max_points == 0 {
                        return Err("invalid point-cloud load parameters".to_string());
                    }
                    distinct_value(
                        &mut requested_point_path,
                        path.as_str(),
                        "point-cloud paths",
                    )?;
                    let preflight = preflight_laz_bounds(path)?;
                    point_loaded = true;
                    point_present = true;
                    point_center = Some(preflight.center);
                    point_extent = Some(preflight.extent_render);
                    point_camera_radius = 1.0;
                }
                ViewerCmd::SetPointCloudParams {
                    visible,
                    radius: Some(radius),
                    ..
                } => {
                    point_camera_radius = radius.clamp(0.1, 100.0);
                    if let Some(visible) = visible {
                        point_present = point_loaded && *visible;
                    }
                }
                ViewerCmd::SetPointCloudParams {
                    visible: Some(visible),
                    ..
                } => {
                    point_present = point_loaded && *visible;
                }
                ViewerCmd::ClearPointCloud => {
                    point_loaded = false;
                    point_present = false;
                }
                ViewerCmd::SetCamLookAt { eye, target, .. } => {
                    distinct_value(
                        &mut requested_general_pose,
                        (*eye, *target),
                        "general camera poses",
                    )?;
                    general_eye = DVec3::from(*eye);
                    general_target = DVec3::from(*target);
                }
                ViewerCmd::SetTransform { translation, .. } => {
                    object_transform_requested = true;
                    if let Some(translation) = translation {
                        object_translation = DVec3::from(*translation);
                    }
                }
                _ => {}
            }
        }
        if let Some(target) = requested_terrain_target {
            terrain_focus = Some(DVec3::from(target));
        }

        let focus = if terrain_present {
            let target =
                terrain_focus.ok_or_else(|| "prospective terrain has no target".to_string())?;
            DVec3::new(target.x, 0.0, target.z)
        } else if point_present {
            point_center.ok_or_else(|| "prospective point cloud has no center".to_string())?
        } else if object_transform_requested {
            object_translation
        } else {
            general_eye
        };
        if point_present
            && commands
                .iter()
                .any(|command| matches!(command, ViewerCmd::SetPointCloudParams { .. }))
            && point_repack_failpoint_blocks_publish(self, focus)
        {
            return Err(
                "point_cloud_repack_failpoint: point=before_anchor_publish; unchanged_state=true"
                    .to_string(),
            );
        }
        let mut anchor = self.camera_anchor;
        anchor.rebase_if_needed(focus);

        if terrain_present {
            let target = terrain_focus.unwrap();
            let radius = terrain_span.unwrap_or(100.0).max(100.0) * 1.5;
            let offset = DVec3::new(radius * 0.5, radius * 0.707, radius * 0.5);
            validate_points(&anchor, CoordRole::Eye, [target + offset])?;
            validate_points(&anchor, CoordRole::Target, [target])?;
        } else if point_present {
            let center = point_center.unwrap();
            let extent = point_extent.unwrap_or(100.0).max(100.0);
            let radius = extent * 2.0 * point_camera_radius;
            let offset = DVec3::new(
                f64::from(radius * 0.5),
                f64::from(radius * 0.5),
                f64::from(radius * 0.707),
            );
            validate_points(&anchor, CoordRole::Eye, [center + offset])?;
            validate_points(&anchor, CoordRole::Target, [center])?;
        } else if !(object_transform_requested && requested_general_pose.is_none()) {
            validate_points(&anchor, CoordRole::Eye, [general_eye])?;
            validate_points(&anchor, CoordRole::Target, [general_target])?;
        }
        if let Some(cloud) = self.point_cloud.as_ref().filter(|_| point_present) {
            validate_points(
                &anchor,
                CoordRole::Content,
                cloud.source_points.iter().map(|point| point.position),
            )?;
        }
        if let Some(scene) = self.terrain_viewer.as_ref() {
            validate_points(
                &anchor,
                CoordRole::Content,
                scene.all_vector_overlay_source_points(),
            )?;
        }
        validate_points(
            &anchor,
            CoordRole::Content,
            self.label_manager.world_points(),
        )?;
        if object_present {
            validate_points(&anchor, CoordRole::Object, [object_translation])?;
        }

        for cmd in commands {
            match cmd {
                ViewerCmd::LoadPointCloud { path, .. } => {
                    let bounds = preflight_laz_bounds(path)?;
                    validate_points(&anchor, CoordRole::Content, [bounds.min, bounds.max])?;
                }
                ViewerCmd::SetCamLookAt { eye, target, up } => {
                    if !Vec3::from(*up).is_finite() {
                        return Err("non_finite_world_coordinate: role=Up".to_string());
                    }
                    validate_points(&anchor, CoordRole::Eye, [DVec3::from(*eye)])?;
                    validate_points(&anchor, CoordRole::Target, [DVec3::from(*target)])?;
                }
                ViewerCmd::SetTerrainCamera {
                    phi_deg,
                    theta_deg,
                    radius,
                    fov_deg,
                    target,
                } => {
                    if ![*phi_deg, *theta_deg, *radius, *fov_deg]
                        .into_iter()
                        .all(f32::is_finite)
                    {
                        return Err("non_finite_terrain_camera".to_string());
                    }
                    if let Some(target) = target {
                        validate_points(&anchor, CoordRole::Target, [DVec3::from(*target)])?;
                    }
                }
                ViewerCmd::SetTerrain {
                    target,
                    radius,
                    phi,
                    theta,
                    fov,
                    ..
                } => {
                    if [*radius, *phi, *theta, *fov]
                        .into_iter()
                        .flatten()
                        .any(|value| !value.is_finite())
                    {
                        return Err("non_finite_terrain_camera".to_string());
                    }
                    if let Some(target) = target {
                        validate_points(&anchor, CoordRole::Target, [DVec3::from(*target)])?;
                    }
                }
                ViewerCmd::SetPointCloudParams {
                    point_size,
                    phi,
                    theta,
                    radius,
                    ..
                } => {
                    if [*point_size, *phi, *theta, *radius]
                        .into_iter()
                        .flatten()
                        .any(|value| !value.is_finite())
                    {
                        return Err("non_finite_point_cloud_camera".to_string());
                    }
                }
                ViewerCmd::SetTransform {
                    translation,
                    rotation_quat,
                    scale,
                } => {
                    let translation = translation.map(DVec3::from).unwrap_or(object_translation);
                    let rotation = rotation_quat
                        .map(glam::Quat::from_array)
                        .unwrap_or(self.object_rotation);
                    let scale = scale.map(Vec3::from).unwrap_or(self.object_scale);
                    validate_points(&anchor, CoordRole::Object, [translation])?;
                    if !rotation.is_finite()
                        || rotation.length_squared() == 0.0
                        || !scale.is_finite()
                    {
                        return Err("invalid_object_transform".to_string());
                    }
                }
                ViewerCmd::AddVectorOverlay { vertices, .. } => validate_points(
                    &anchor,
                    CoordRole::Content,
                    vertices.iter().map(|vertex| DVec3::from(vertex.position)),
                )?,
                ViewerCmd::AddLabel { world_pos, .. } => {
                    validate_points(&anchor, CoordRole::Content, [DVec3::from(*world_pos)])?
                }
                ViewerCmd::AddLineLabel { polyline, .. }
                | ViewerCmd::AddCurvedLabel { polyline, .. } => validate_points(
                    &anchor,
                    CoordRole::Content,
                    polyline.iter().copied().map(DVec3::from),
                )?,
                ViewerCmd::AddCallout { anchor: point, .. } => {
                    validate_points(&anchor, CoordRole::Content, [DVec3::from(*point)])?
                }
                _ => {}
            }
        }
        Ok(())
    }
}

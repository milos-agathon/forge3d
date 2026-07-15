use glam::{DVec3, Vec3};

use crate::viewer::camera_controller::{validate_world_point, CoordRole};
use crate::viewer::pointcloud::preflight_laz_bounds;
use crate::viewer::terrain::ViewerTerrainScene;
use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

pub(crate) fn command_priority(cmd: &ViewerCmd) -> u8 {
    match cmd {
        ViewerCmd::LoadTerrain(_) => 0,
        ViewerCmd::LoadPointCloud { .. } => 1,
        ViewerCmd::SetTerrainCamera { .. } | ViewerCmd::SetTerrain { .. } => 2,
        ViewerCmd::SetPointCloudParams { .. } => 3,
        ViewerCmd::SetCamLookAt { .. } => 4,
        _ => 5,
    }
}

/// Stable canonical order: resource/frame establishment precedes dependent
/// content while preserving order within each semantic class.
pub(crate) fn order_command_batch(mut commands: Vec<ViewerCmd>) -> Vec<ViewerCmd> {
    commands.sort_by_key(command_priority);
    commands
}

fn validate_points<I>(
    anchor: &crate::camera::Anchor,
    role: CoordRole,
    points: I,
) -> Result<(), String>
where
    I: IntoIterator<Item = DVec3>,
{
    points
        .into_iter()
        .try_for_each(|point| validate_world_point(role, point, anchor).map_err(|e| e.to_string()))
}

fn distinct_value<T: Copy + PartialEq>(
    slot: &mut Option<T>,
    value: T,
    name: &str,
) -> Result<(), String> {
    if slot.is_some_and(|existing| existing != value) {
        return Err(format!("ambiguous prospective batch: conflicting {name}"));
    }
    *slot = Some(value);
    Ok(())
}

impl Viewer {
    /// Validate an entire drained command batch against one final prospective
    /// anchor. This method is pure with respect to Viewer and GPU state.
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
        } else {
            general_eye
        };
        if point_present
            && commands
                .iter()
                .any(|command| matches!(command, ViewerCmd::SetPointCloudParams { .. }))
            && std::env::var("RUN_M06_VIEWER_CI").as_deref() == Ok("1")
            && std::env::var("FORGE3D_M06_POINT_REPACK_FAILPOINT").as_deref()
                == Ok("before_anchor_publish")
        {
            let candidate =
                crate::viewer::camera_controller::prospective_anchor(&self.camera_anchor, focus);
            if candidate.origin() != self.camera_anchor.origin() {
                return Err(
                    "point_cloud_repack_failpoint: point=before_anchor_publish; unchanged_state=true"
                        .to_string(),
                );
            }
        }
        let mut anchor = self.camera_anchor;
        anchor.rebase_if_needed(focus);

        // Validate the final owning camera.
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
        } else {
            validate_points(&anchor, CoordRole::Eye, [general_eye])?;
            validate_points(&anchor, CoordRole::Target, [general_target])?;
        }

        // Existing absolute sources must survive the same final frame.
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
        validate_points(&anchor, CoordRole::Object, [self.object_translation])?;

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
                    let translation = translation
                        .map(DVec3::from)
                        .unwrap_or(self.object_translation);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_order_is_stable_and_loads_precede_frame_mutations_and_content() {
        let commands = vec![
            ViewerCmd::ClearLabels,
            ViewerCmd::SetCamLookAt {
                eye: [0.0; 3],
                target: [0.0; 3],
                up: [0.0, 1.0, 0.0],
            },
            ViewerCmd::LoadPointCloud {
                path: "p.laz".into(),
                point_size: 1.0,
                max_points: 1,
                color_mode: None,
            },
            ViewerCmd::LoadTerrain("t.tif".into()),
        ];
        let ordered = order_command_batch(commands);
        assert!(matches!(ordered[0], ViewerCmd::LoadTerrain(_)));
        assert!(matches!(ordered[1], ViewerCmd::LoadPointCloud { .. }));
        assert!(matches!(ordered[2], ViewerCmd::SetCamLookAt { .. }));
        assert!(matches!(ordered[3], ViewerCmd::ClearLabels));
    }

    #[test]
    fn every_frame_establisher_permutation_has_one_canonical_order() {
        let commands = [
            ViewerCmd::LoadTerrain("t.tif".into()),
            ViewerCmd::LoadPointCloud {
                path: "p.laz".into(),
                point_size: 1.0,
                max_points: 1,
                color_mode: None,
            },
            ViewerCmd::SetCamLookAt {
                eye: [0.0; 3],
                target: [0.0; 3],
                up: [0.0, 1.0, 0.0],
            },
            ViewerCmd::SetTransform {
                translation: Some([1.0, 2.0, 3.0]),
                rotation_quat: None,
                scale: None,
            },
        ];
        let permutations = [
            [0, 1, 2, 3],
            [0, 1, 3, 2],
            [0, 2, 1, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
            [0, 3, 2, 1],
            [1, 0, 2, 3],
            [1, 0, 3, 2],
            [1, 2, 0, 3],
            [1, 2, 3, 0],
            [1, 3, 0, 2],
            [1, 3, 2, 0],
            [2, 0, 1, 3],
            [2, 0, 3, 1],
            [2, 1, 0, 3],
            [2, 1, 3, 0],
            [2, 3, 0, 1],
            [2, 3, 1, 0],
            [3, 0, 1, 2],
            [3, 0, 2, 1],
            [3, 1, 0, 2],
            [3, 1, 2, 0],
            [3, 2, 0, 1],
            [3, 2, 1, 0],
        ];

        for permutation in permutations {
            let batch = permutation
                .into_iter()
                .map(|index| commands[index].clone())
                .collect();
            let ordered = order_command_batch(batch);
            assert!(matches!(ordered[0], ViewerCmd::LoadTerrain(_)));
            assert!(matches!(ordered[1], ViewerCmd::LoadPointCloud { .. }));
            assert!(matches!(ordered[2], ViewerCmd::SetCamLookAt { .. }));
            assert!(matches!(ordered[3], ViewerCmd::SetTransform { .. }));
        }
    }
}

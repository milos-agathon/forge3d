use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

pub(crate) fn handle_cmd(viewer: &mut Viewer, cmd: &ViewerCmd) -> bool {
    match cmd {
        ViewerCmd::LoadPointCloud {
            path,
            point_size,
            max_points,
            color_mode,
        } => {
            use crate::viewer::pointcloud::{ColorMode, PointCloudState};

            eprintln!(
                "[pointcloud] Loading: {} (size={}, max={}, mode={:?})",
                path, point_size, max_points, color_mode
            );

            let point_cloud_owns_frame = viewer
                .terrain_viewer
                .as_ref()
                .and_then(|scene| scene.terrain.as_ref())
                .is_none();
            let prospective_anchor = viewer.prospective_frame_camera().anchor;
            let depth_format = wgpu::TextureFormat::Depth32Float;
            let mut candidate =
                match PointCloudState::new(&viewer.device, viewer.config.format, depth_format) {
                    Ok(point_cloud) => point_cloud,
                    Err(error) => {
                        viewer.reject_command(format!(
                            "point_cloud_state_construction_failed: {error}"
                        ));
                        return true;
                    }
                };
            let mode = color_mode
                .as_ref()
                .map(|mode| ColorMode::from_str(mode))
                .unwrap_or(ColorMode::Elevation);
            if let Err(error) = candidate.load_from_file(
                &viewer.device,
                &viewer.queue,
                &prospective_anchor,
                point_cloud_owns_frame,
                path,
                *max_points,
                mode,
            ) {
                viewer.reject_command(format!("point_cloud_execution_failed: {error}"));
                return true;
            }
            candidate.set_point_size(*point_size);
            eprintln!("[pointcloud] Loaded {} points", candidate.point_count);
            viewer.point_cloud = Some(candidate);
            true
        }
        ViewerCmd::ClearPointCloud => {
            if let Some(ref mut point_cloud) = viewer.point_cloud {
                point_cloud.clear();
            }
            println!("[pointcloud] Cleared");
            true
        }
        ViewerCmd::SetPointCloudParams {
            point_size,
            visible,
            color_mode,
            phi,
            theta,
            radius,
        } => {
            let anchor = viewer.prospective_frame_camera().anchor;
            let point_cloud_owns_frame = viewer
                .terrain_viewer
                .as_ref()
                .and_then(|scene| scene.terrain.as_ref())
                .is_none();
            if let Some(ref mut point_cloud) = viewer.point_cloud {
                if let Err(error) = point_cloud.try_set_params(
                    &anchor,
                    point_cloud_owns_frame,
                    *point_size,
                    *visible,
                    color_mode.as_deref(),
                    *phi,
                    *theta,
                    *radius,
                ) {
                    viewer.command_error = Some(format!(
                        "point_cloud_parameter_rejected: {error}; unchanged_state=true"
                    ));
                    return true;
                }
                println!(
                    "[pointcloud] Params updated: size={}, visible={}, mode={:?}, phi={:.3}, theta={:.3}, radius={:.3}",
                    point_cloud.point_size,
                    point_cloud.visible,
                    point_cloud.color_mode,
                    point_cloud.cam_phi,
                    point_cloud.cam_theta,
                    point_cloud.cam_radius,
                );
            } else {
                viewer.reject_command("point-cloud parameters require a loaded point cloud");
            }
            true
        }
        _ => false,
    }
}

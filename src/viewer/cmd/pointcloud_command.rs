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

            if viewer.point_cloud.is_none() {
                eprintln!("[pointcloud] Creating PointCloudState...");
                let depth_format = wgpu::TextureFormat::Depth32Float;
                match PointCloudState::new(&viewer.device, viewer.config.format, depth_format) {
                    Ok(pc) => {
                        viewer.point_cloud = Some(pc);
                        eprintln!("[pointcloud] PointCloudState created");
                    }
                    Err(e) => {
                        eprintln!("[pointcloud] Failed to create PointCloudState: {e}");
                        return false;
                    }
                }
            }

            let point_cloud_owns_frame = viewer
                .terrain_viewer
                .as_ref()
                .and_then(|scene| scene.terrain.as_ref())
                .is_none();
            let prospective_anchor = viewer.prospective_frame_camera().anchor;
            if let Some(ref mut point_cloud) = viewer.point_cloud {
                let mode = color_mode
                    .as_ref()
                    .map(|mode| ColorMode::from_str(mode))
                    .unwrap_or(ColorMode::Elevation);

                eprintln!("[pointcloud] Loading file...");
                match point_cloud.load_from_file(
                    &viewer.device,
                    &viewer.queue,
                    &prospective_anchor,
                    point_cloud_owns_frame,
                    path,
                    *max_points,
                    mode,
                ) {
                    Ok(()) => {
                        point_cloud.set_point_size(*point_size);
                        eprintln!("[pointcloud] Loaded {} points", point_cloud.point_count);
                        eprintln!(
                            "[pointcloud] Bounds: ({:.1}, {:.1}, {:.1}) - ({:.1}, {:.1}, {:.1})",
                            point_cloud.bounds_min.x,
                            point_cloud.bounds_min.y,
                            point_cloud.bounds_min.z,
                            point_cloud.bounds_max.x,
                            point_cloud.bounds_max.y,
                            point_cloud.bounds_max.z
                        );
                        eprintln!(
                            "[pointcloud] Center: ({:.1}, {:.1}, {:.1})",
                            point_cloud.center.x, point_cloud.center.y, point_cloud.center.z
                        );
                        eprintln!("[pointcloud] Load complete, returning to render loop");
                    }
                    Err(e) => {
                        eprintln!("[pointcloud] Error: {}", e);
                    }
                }
            }
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
            if let Some(ref mut point_cloud) = viewer.point_cloud {
                if let Some(size) = point_size {
                    point_cloud.set_point_size(*size);
                }
                if let Some(vis) = visible {
                    point_cloud.set_visible(*vis);
                }
                if let Some(mode) = color_mode {
                    use crate::viewer::pointcloud::ColorMode;
                    point_cloud.color_mode = ColorMode::from_str(mode);
                }
                if let Some(v) = phi {
                    point_cloud.cam_phi = *v;
                }
                if let Some(v) = theta {
                    point_cloud.cam_theta = v.clamp(0.1, 1.5);
                }
                if let Some(v) = radius {
                    point_cloud.cam_radius = v.clamp(0.1, 100.0);
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
            }
            true
        }
        _ => false,
    }
}

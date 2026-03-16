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
                viewer.point_cloud = Some(PointCloudState::new(
                    &viewer.device,
                    viewer.config.format,
                    depth_format,
                ));
                eprintln!("[pointcloud] PointCloudState created");
            }

            if let Some(ref mut point_cloud) = viewer.point_cloud {
                let mode = color_mode
                    .as_ref()
                    .map(|mode| ColorMode::from_str(mode))
                    .unwrap_or(ColorMode::Elevation);

                point_cloud.set_point_size(*point_size);
                eprintln!("[pointcloud] Loading file...");

                match point_cloud.load_from_file(
                    &viewer.device,
                    &viewer.queue,
                    path,
                    *max_points,
                    mode,
                ) {
                    Ok(()) => {
                        eprintln!("[pointcloud] Loaded {} points", point_cloud.point_count);
                        eprintln!(
                            "[pointcloud] Bounds: ({:.1}, {:.1}, {:.1}) - ({:.1}, {:.1}, {:.1})",
                            point_cloud.bounds_min[0],
                            point_cloud.bounds_min[1],
                            point_cloud.bounds_min[2],
                            point_cloud.bounds_max[0],
                            point_cloud.bounds_max[1],
                            point_cloud.bounds_max[2]
                        );
                        eprintln!(
                            "[pointcloud] Center: ({:.1}, {:.1}, {:.1})",
                            point_cloud.center[0], point_cloud.center[1], point_cloud.center[2]
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
                println!(
                    "[pointcloud] Params updated: size={}, visible={}, mode={:?}",
                    point_cloud.point_size, point_cloud.visible, point_cloud.color_mode
                );
            }
            true
        }
        _ => false,
    }
}

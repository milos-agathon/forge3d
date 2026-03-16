use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

pub(crate) fn handle_cmd(viewer: &mut Viewer, cmd: &ViewerCmd) -> bool {
    match cmd {
        ViewerCmd::SetOitEnabled { enabled, mode } => {
            viewer.oit_enabled = *enabled;
            viewer.oit_mode = mode.clone();
            if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                terrain_viewer.set_oit_mode(*enabled, mode);
            }
            println!("[oit] enabled={} mode={}", enabled, mode);
            true
        }
        ViewerCmd::GetOitMode => {
            println!(
                "[oit] enabled={} mode={}",
                viewer.oit_enabled, viewer.oit_mode
            );
            true
        }
        ViewerCmd::SetTaaEnabled { enabled } => {
            viewer.set_taa_enabled(*enabled);
            println!("[taa] enabled={}", enabled);
            true
        }
        ViewerCmd::GetTaaStatus => {
            let taa_enabled = viewer
                .taa_renderer
                .as_ref()
                .map(|taa| taa.is_enabled())
                .unwrap_or(false);
            let jitter_enabled = viewer.taa_jitter.enabled;
            println!(
                "[taa] enabled={} jitter_enabled={}",
                taa_enabled, jitter_enabled
            );
            true
        }
        ViewerCmd::SetTaaParams {
            history_weight,
            jitter_scale,
            enable_jitter,
        } => {
            let terrain_active = viewer
                .terrain_viewer
                .as_ref()
                .map(|terrain_viewer| terrain_viewer.has_terrain())
                .unwrap_or(false);

            if terrain_active {
                if let Some(ref mut terrain_viewer) = viewer.terrain_viewer {
                    terrain_viewer.set_taa_params(*history_weight, *jitter_scale);
                }
            } else {
                if let Some(weight) = history_weight {
                    if let Some(ref mut taa) = viewer.taa_renderer {
                        taa.set_history_weight(*weight);
                    }
                }
                if let Some(scale) = jitter_scale {
                    viewer.taa_jitter.set_scale(*scale);
                }
                if let Some(enabled) = enable_jitter {
                    viewer.taa_jitter.set_enabled(*enabled);
                } else if let Some(scale) = jitter_scale {
                    if *scale > 0.0 && !viewer.taa_jitter.enabled {
                        viewer.taa_jitter.set_enabled(true);
                    }
                }

                let taa_weight = viewer
                    .taa_renderer
                    .as_ref()
                    .map(|taa| taa.history_weight())
                    .unwrap_or(0.0);
                println!(
                    "[taa] params updated: weight={:.2} jitter_scale={:.2} jitter_enabled={}",
                    taa_weight, viewer.taa_jitter.scale, viewer.taa_jitter.enabled
                );
            }
            true
        }
        _ => false,
    }
}

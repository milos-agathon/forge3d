use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

pub(crate) fn handle_cmd(viewer: &mut Viewer, cmd: &ViewerCmd) -> bool {
    match cmd {
        ViewerCmd::SetSceneReviewState { state } => {
            if let Err(err) = viewer.set_scene_review_state(state.clone()) {
                eprintln!("[scene_review] failed to install review state: {err}");
                viewer.reject_command(format!("scene_review_transaction_failed: {err}"));
            }
            true
        }
        ViewerCmd::ApplySceneVariant { variant_id } => {
            if let Err(err) = viewer.apply_scene_variant(variant_id) {
                eprintln!(
                    "[scene_review] failed to apply variant '{}': {err}",
                    variant_id
                );
                viewer.reject_command(format!("scene_variant_transaction_failed: {err}"));
            }
            true
        }
        ViewerCmd::SetReviewLayerVisible { layer_id, visible } => {
            if let Err(err) = viewer.set_review_layer_visible(layer_id, *visible) {
                eprintln!(
                    "[scene_review] failed to set layer '{}' visible={}: {err}",
                    layer_id, visible
                );
                viewer.reject_command(format!("review_layer_transaction_failed: {err}"));
            }
            true
        }
        _ => false,
    }
}

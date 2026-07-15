//! Label management methods for the Viewer.

use crate::labels::{LabelId, LabelStyle};
use crate::viewer::Viewer;

impl Viewer {
    /// Add a label at a world position.
    pub fn add_label(
        &mut self,
        text: &str,
        world_pos: (f64, f64, f64),
        style: Option<LabelStyle>,
    ) -> u64 {
        self.add_label_with_id(None, text, world_pos, style)
    }

    /// Add a label at a world position with an externally allocated ID.
    pub fn add_label_with_id(
        &mut self,
        id: Option<u64>,
        text: &str,
        world_pos: (f64, f64, f64),
        style: Option<LabelStyle>,
    ) -> u64 {
        let pos = glam::DVec3::new(world_pos.0, world_pos.1, world_pos.2);
        let style = style.unwrap_or_default();
        let id =
            self.label_manager
                .add_label_with_id(id.map(LabelId), text.to_string(), pos, style);
        id.0
    }

    /// Remove a label by ID.
    pub fn remove_label(&mut self, id: u64) -> bool {
        self.label_manager.remove_label(LabelId(id))
    }

    /// Clear all labels.
    pub fn clear_labels(&mut self) {
        self.label_manager.clear();
    }

    /// Set labels enabled/disabled.
    pub fn set_labels_enabled(&mut self, enabled: bool) {
        self.label_manager.set_enabled(enabled);
    }

    /// Check if labels are enabled.
    pub fn labels_enabled(&self) -> bool {
        self.label_manager.is_enabled()
    }

    /// Get the number of labels.
    pub fn label_count(&self) -> usize {
        self.label_manager.label_count()
    }

    /// Load an MSDF font atlas from files.
    pub fn load_label_atlas(
        &mut self,
        atlas_png_path: &str,
        metrics_json_path: &str,
    ) -> Result<(), String> {
        self.label_manager.load_atlas_from_files(
            &self.device,
            &self.queue,
            atlas_png_path,
            metrics_json_path,
        )
    }

    /// Update label positions based on current camera.
    /// Called automatically during render, but can be called manually.
    pub fn update_labels(&mut self) {
        let frame = self.current_frame_camera();
        self.update_labels_for_frame(frame);
    }

    /// One Viewer-owned label update path for both automatic and manual calls.
    /// The frozen frame, render eye, viewport, selection, collision index, and
    /// pick boxes therefore cannot diverge between those entry points.
    pub(crate) fn update_labels_for_frame(
        &mut self,
        frame: crate::viewer::viewer_types::FrameCamera,
    ) -> usize {
        let selected_u32 = self.unified_picking.selection_manager().get_selection();
        let selected_ids: std::collections::HashSet<u64> =
            selected_u32.iter().map(|&id| u64::from(id)).collect();
        let view_proj = frame.view_projection(self.config.width, self.config.height);
        self.label_manager.update_with_camera_anchored(
            view_proj,
            Some(frame.render_eye()),
            Some(&selected_ids),
            &frame.anchor,
        )
    }

    /// Resize the label collision grid for new screen dimensions.
    pub fn resize_labels(&mut self, width: u32, height: u32) {
        self.label_manager.resize(width, height);
    }
}

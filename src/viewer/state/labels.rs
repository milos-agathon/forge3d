//! Label management methods for the Viewer.

use crate::labels::{LabelId, LabelStyle};
use crate::viewer::Viewer;
use glam::Vec3;

impl Viewer {
    /// Add a label at a world position.
    pub fn add_label(&mut self, text: &str, world_pos: (f32, f32, f32), style: Option<LabelStyle>) -> u64 {
        let pos = Vec3::new(world_pos.0, world_pos.1, world_pos.2);
        let style = style.unwrap_or_default();
        let id = self.label_manager.add_label(text.to_string(), pos, style);
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
    pub fn load_label_atlas(&mut self, atlas_png_path: &str, metrics_json_path: &str) -> Result<(), String> {
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
        let aspect = self.config.width as f32 / self.config.height as f32;
        let fov_rad = self.view_config.fov_deg.to_radians();
        let view = self.camera.view_matrix();
        let proj = glam::Mat4::perspective_rh(fov_rad, aspect, self.view_config.znear, self.view_config.zfar);
        let view_proj = proj * view;
        self.label_manager.update(view_proj);
    }

    /// Resize the label collision grid for new screen dimensions.
    pub fn resize_labels(&mut self, width: u32, height: u32) {
        self.label_manager.resize(width, height);
    }
}

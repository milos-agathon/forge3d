//! One reset owner for resources invalidated by anchor and viewport changes.
use super::Viewer;

impl Viewer {
    pub(crate) fn invalidate_viewport_histories(&mut self, width: u32, height: u32) {
        if let Some(terrain) = self.terrain_viewer.as_mut() {
            terrain.invalidate_temporal_history();
        }
        if let Some(gi) = self.gi.as_mut() {
            gi.invalidate_temporal_histories();
        }
        if let Some(taa) = self.taa_renderer.as_mut() {
            taa.reset_history();
        }
        self.taa_jitter = if self.taa_jitter.enabled {
            crate::core::jitter::JitterState::enabled()
        } else {
            crate::core::jitter::JitterState::new()
        };
        let frame = self.current_frame_camera();
        self.prev_view_proj =
            frame.projection(width, height) * frame.view() * self.anchored_object_model(frame);
        self.fog_frame_index = 0;
        self.fog_history_state.invalidate();
    }
}

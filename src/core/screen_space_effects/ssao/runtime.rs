use super::*;

impl SsaoRenderer {
    pub fn update_settings(&mut self, queue: &Queue, settings: SsaoSettings) {
        let technique_changed = self.settings.technique != settings.technique;
        self.settings = settings;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&settings));
        if technique_changed {
            self.invalidate_history();
        }
    }

    pub fn get_settings(&self) -> SsaoSettings {
        self.settings
    }

    pub fn set_seed(&mut self, queue: &Queue, seed: u32) {
        self.settings.frame_index = seed;
        self.frame_index = seed;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
        self.invalidate_history();
    }

    pub fn update_camera(&mut self, queue: &Queue, camera: &CameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
        let p11 = camera.proj_matrix[1][1];
        self.settings.proj_scale = 0.5 * self.height as f32 * p11;
        self.settings.inv_resolution = [1.0 / self.width as f32, 1.0 / self.height as f32];
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
        self.invalidate_history();
    }

    pub(crate) fn invalidate_history(&mut self) {
        self.history_valid = false;
    }
}

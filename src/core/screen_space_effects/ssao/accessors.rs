use super::*;

impl SsaoRenderer {
    pub fn get_output(&self) -> &TextureView {
        &self.ssao_blurred_view
    }

    pub fn get_raw_ao_view(&self) -> &TextureView {
        &self.ssao_view
    }

    pub fn get_tmp_ao_view(&self) -> &TextureView {
        &self.ssao_tmp_view
    }

    pub fn get_resolved_ao_view(&self) -> &TextureView {
        &self.ssao_resolved_view
    }

    pub fn get_composited(&self) -> &TextureView {
        &self.ssao_composited_view
    }

    pub fn set_composite_multiplier(&mut self, queue: &Queue, mul: f32) {
        let params: [f32; 4] = [1.0, mul.clamp(0.0, 1.0), 0.0, 0.0];
        queue.write_buffer(&self.comp_uniform, 0, bytemuck::cast_slice(&params));
    }

    pub fn set_blur_enabled(&mut self, on: bool) {
        if self.blur_enabled != on {
            self.blur_enabled = on;
            self.invalidate_history();
        }
    }

    pub fn set_temporal_enabled(&mut self, on: bool) {
        if self.temporal_enabled != on {
            self.temporal_enabled = on;
            self.invalidate_history();
        }
    }

    pub fn blur_enabled(&self) -> bool {
        self.blur_enabled
    }

    pub fn raw_ao_texture(&self) -> &Texture {
        &self.ssao_texture
    }

    pub fn blurred_ao_texture(&self) -> &Texture {
        &self.ssao_blurred
    }

    pub fn resolved_ao_texture(&self) -> &Texture {
        &self.ssao_resolved
    }

    pub fn composited_texture(&self) -> &Texture {
        &self.ssao_composited
    }

    pub fn timings_ms(&self) -> (f32, f32, f32) {
        (self.last_ao_ms, self.last_blur_ms, self.last_temporal_ms)
    }
}

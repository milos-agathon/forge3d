use super::*;

impl SsgiRenderer {
    #[allow(dead_code)]
    pub fn get_output(&self) -> &TextureView {
        &self.ssgi_filtered_view
    }

    pub fn get_output_view(&self) -> &TextureView {
        &self.ssgi_filtered_view
    }

    pub fn get_upscaled_view(&self) -> &TextureView {
        &self.ssgi_upscaled_view
    }

    /// Get display-ready output (upscaled if running half-res)
    pub fn get_output_for_display(&self) -> &TextureView {
        if self.half_res {
            &self.ssgi_upscaled_view
        } else {
            &self.ssgi_filtered_view
        }
    }

    pub fn timings_ms(&self) -> (f32, f32, f32, f32) {
        (
            self.last_trace_ms,
            self.last_shade_ms,
            self.last_temporal_ms,
            self.last_upsample_ms,
        )
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn is_half_res(&self) -> bool {
        self.half_res
    }

    pub fn hit_texture(&self) -> &Texture {
        &self.ssgi_hit
    }

    pub fn filtered_texture(&self) -> &Texture {
        &self.ssgi_filtered
    }

    pub fn history_texture(&self) -> &Texture {
        &self.ssgi_history
    }

    pub fn upscaled_texture(&self) -> &Texture {
        &self.ssgi_upscaled
    }

    pub fn get_composited(&self) -> &TextureView {
        &self.ssgi_composited_view
    }

    pub fn set_composite_intensity(&mut self, queue: &Queue, intensity: f32) {
        let params: [f32; 4] = [intensity, 0.0, 0.0, 0.0];
        queue.write_buffer(&self.composite_uniform, 0, bytemuck::cast_slice(&params));
    }

    pub fn reset_history(&mut self, device: &Device, queue: &Queue) -> RenderResult<()> {
        self.scene_history_ready = false;
        self.scene_history_index = 0;
        self.set_half_res(device, queue, self.half_res)
    }
}

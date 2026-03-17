use super::*;

impl DualSourceOITRenderer {
    pub fn set_mode(&mut self, mode: DualSourceOITMode) {
        self.mode = mode;
        self.update_compose_uniforms();
    }

    pub fn mode(&self) -> DualSourceOITMode {
        self.mode
    }

    pub fn set_quality(&mut self, quality: DualSourceOITQuality) {
        self.quality = quality;
        self.update_uniforms_for_quality();
    }

    pub fn quality(&self) -> DualSourceOITQuality {
        self.quality
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.update_compose_uniforms();
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn is_dual_source_supported(&self) -> bool {
        self.dual_source_supported
    }

    pub fn get_operating_mode(&self) -> DualSourceOITMode {
        if !self.enabled {
            return DualSourceOITMode::Disabled;
        }

        match self.mode {
            DualSourceOITMode::Disabled => DualSourceOITMode::Disabled,
            DualSourceOITMode::DualSource => {
                if self.dual_source_supported {
                    DualSourceOITMode::DualSource
                } else {
                    DualSourceOITMode::WBOITFallback
                }
            }
            DualSourceOITMode::WBOITFallback => DualSourceOITMode::WBOITFallback,
            DualSourceOITMode::Automatic => {
                if self.dual_source_supported {
                    DualSourceOITMode::DualSource
                } else {
                    DualSourceOITMode::WBOITFallback
                }
            }
        }
    }

    fn update_uniforms_for_quality(&mut self) {
        match self.quality {
            DualSourceOITQuality::Low => {
                self.uniforms.alpha_correction = 1.0;
                self.uniforms.depth_weight_scale = 0.5;
                self.uniforms.max_fragments = 4.0;
                self.uniforms.premultiply_factor = 1.0;
            }
            DualSourceOITQuality::Medium => {
                self.uniforms.alpha_correction = 1.1;
                self.uniforms.depth_weight_scale = 1.0;
                self.uniforms.max_fragments = 8.0;
                self.uniforms.premultiply_factor = 1.0;
            }
            DualSourceOITQuality::High => {
                self.uniforms.alpha_correction = 1.2;
                self.uniforms.depth_weight_scale = 1.5;
                self.uniforms.max_fragments = 16.0;
                self.uniforms.premultiply_factor = 1.0;
            }
            DualSourceOITQuality::Ultra => {
                self.uniforms.alpha_correction = 1.3;
                self.uniforms.depth_weight_scale = 2.0;
                self.uniforms.max_fragments = 32.0;
                self.uniforms.premultiply_factor = 1.0;
            }
        }
    }

    fn update_compose_uniforms(&mut self) {
        self.compose_uniforms.use_dual_source = match self.get_operating_mode() {
            DualSourceOITMode::DualSource => 1,
            _ => 0,
        };
    }

    pub fn upload_uniforms(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(&self.uniforms));
        queue.write_buffer(
            &self.compose_uniforms_buffer,
            0,
            bytemuck::bytes_of(&self.compose_uniforms),
        );
    }

    pub fn get_stats(&self) -> DualSourceOITStats {
        self.frame_stats
    }
}

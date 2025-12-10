// src/shadows/csm_depth_control.rs
// B17: Depth-clip control methods for CSM
// RELEVANT FILES: shaders/shadows.wgsl

use super::cascade_math::calculate_unclipped_cascade_splits;
use super::csm_renderer::CsmRenderer;
use wgpu::Device;

/// Extension trait for unclipped depth control on CsmRenderer
impl CsmRenderer {
    /// Detect hardware support for unclipped depth
    pub fn detect_unclipped_depth_support(device: &Device) -> bool {
        let _features = device.features();
        // WebGPU currently doesn't directly expose unclipped depth as a feature
        // Conservative approach: assume modern hardware supports it
        true
    }

    /// Enable or disable unclipped depth rendering
    pub fn set_unclipped_depth_enabled(&mut self, enabled: bool, device: &Device) {
        let supported = Self::detect_unclipped_depth_support(device);

        self.config.enable_unclipped_depth = enabled && supported;
        self.uniforms.enable_unclipped_depth = if self.config.enable_unclipped_depth {
            1
        } else {
            0
        };

        // Adjust depth clip factor for better cascade coverage when unclipped depth is enabled
        if self.config.enable_unclipped_depth {
            self.config.depth_clip_factor = 1.5; // Extend 50% beyond normal clip range
        } else {
            self.config.depth_clip_factor = 1.0; // Standard clipping
        }

        self.uniforms.depth_clip_factor = self.config.depth_clip_factor;
    }

    /// Check if unclipped depth is currently enabled
    pub fn is_unclipped_depth_enabled(&self) -> bool {
        self.config.enable_unclipped_depth
    }

    /// Get current depth clip factor
    pub fn get_depth_clip_factor(&self) -> f32 {
        self.config.depth_clip_factor
    }

    /// Set custom depth clip factor (for advanced tuning)
    pub fn set_depth_clip_factor(&mut self, factor: f32) {
        self.config.depth_clip_factor = factor.clamp(0.5, 3.0);
        self.uniforms.depth_clip_factor = self.config.depth_clip_factor;
    }

    /// Retune cascades for optimal unclipped depth performance
    pub fn retune_cascades_for_unclipped_depth(&mut self) {
        if self.config.enable_unclipped_depth {
            // Reduce peter-panning offset since unclipped depth reduces artifacts
            self.config.peter_panning_offset *= 0.5;
            self.uniforms.peter_panning_offset = self.config.peter_panning_offset;

            // Slightly reduce depth bias as unclipped depth improves precision
            self.config.depth_bias *= 0.8;
            self.uniforms.depth_bias = self.config.depth_bias;

            // Adjust slope bias for better contact shadows
            self.config.slope_bias *= 0.9;
            self.uniforms.slope_bias = self.config.slope_bias;

            // Increase cascade count if performance allows (better quality)
            if self.config.cascade_count < 4 {
                self.config.cascade_count = (self.config.cascade_count + 1).min(4);
                self.uniforms.cascade_count = self.config.cascade_count;
            }
        }
    }

    /// Calculate optimal cascade splits for unclipped depth
    pub fn calculate_unclipped_cascade_splits(&self, near_plane: f32, far_plane: f32) -> Vec<f32> {
        if !self.config.enable_unclipped_depth {
            return self.calculate_cascade_splits(near_plane, far_plane);
        }

        calculate_unclipped_cascade_splits(
            self.config.cascade_count,
            near_plane,
            far_plane,
            self.config.depth_clip_factor,
        )
    }
}

//! Depth of Field types and configuration.

use bytemuck::{Pod, Zeroable};

/// DOF parameters matching WGSL DofUniforms structure.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DofUniforms {
    // Camera parameters
    pub aperture: f32,
    pub focus_distance: f32,
    pub focal_length: f32,
    pub sensor_size: f32,

    // Quality and performance settings
    pub blur_radius_scale: f32,
    pub max_blur_radius: f32,
    pub sample_count: u32,
    pub quality_level: u32,

    // Near and far field settings
    pub near_transition_range: f32,
    pub far_transition_range: f32,
    pub coc_bias: f32,
    pub bokeh_rotation: f32,

    // Screen space parameters
    pub screen_size: [f32; 2],
    pub inv_screen_size: [f32; 2],

    // Debug and visualization
    pub debug_mode: u32,
    pub show_coc: u32,
    pub _padding: [f32; 2],
}

impl Default for DofUniforms {
    fn default() -> Self {
        Self {
            aperture: 0.1,
            focus_distance: 10.0,
            focal_length: 50.0,
            sensor_size: 36.0,
            blur_radius_scale: 1.0,
            max_blur_radius: 16.0,
            sample_count: 16,
            quality_level: 1,
            near_transition_range: 2.0,
            far_transition_range: 5.0,
            coc_bias: 0.0,
            bokeh_rotation: 0.0,
            screen_size: [1920.0, 1080.0],
            inv_screen_size: [1.0 / 1920.0, 1.0 / 1080.0],
            debug_mode: 0,
            show_coc: 0,
            _padding: [0.0; 2],
        }
    }
}

/// DOF quality settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DofQuality {
    /// Low quality: 8 samples, fast but lower quality.
    Low,
    /// Medium quality: 16 samples, balanced.
    Medium,
    /// High quality: 24 samples, high quality.
    High,
    /// Ultra quality: 32 samples, best quality.
    Ultra,
}

impl DofQuality {
    /// Get sample count for this quality setting.
    pub fn sample_count(self) -> u32 {
        match self {
            DofQuality::Low => 8,
            DofQuality::Medium => 16,
            DofQuality::High => 24,
            DofQuality::Ultra => 32,
        }
    }

    /// Get quality level index.
    pub fn level(self) -> u32 {
        match self {
            DofQuality::Low => 0,
            DofQuality::Medium => 1,
            DofQuality::High => 2,
            DofQuality::Ultra => 3,
        }
    }

    /// Get max blur radius for this quality.
    pub fn max_blur_radius(self) -> f32 {
        match self {
            DofQuality::Low => 8.0,
            DofQuality::Medium => 12.0,
            DofQuality::High => 16.0,
            DofQuality::Ultra => 20.0,
        }
    }
}

/// DOF rendering method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DofMethod {
    /// Single-pass gather blur (higher quality).
    Gather,
    /// Two-pass separable blur (better performance).
    Separable,
}

/// Camera DOF parameters.
#[derive(Debug, Clone, Copy)]
pub struct CameraDofParams {
    pub aperture: f32,
    pub focus_distance: f32,
    pub focal_length: f32,
    pub auto_focus: bool,
    pub auto_focus_speed: f32,
}

impl Default for CameraDofParams {
    fn default() -> Self {
        Self {
            aperture: 0.1,
            focus_distance: 10.0,
            focal_length: 50.0,
            auto_focus: false,
            auto_focus_speed: 2.0,
        }
    }
}

/// Utility functions for DOF calculations.
pub mod utils {
    /// Convert f-stop to aperture value.
    pub fn f_stop_to_aperture(f_stop: f32) -> f32 {
        1.0 / f_stop.max(1.0)
    }

    /// Convert aperture to f-stop.
    pub fn aperture_to_f_stop(aperture: f32) -> f32 {
        1.0 / aperture.max(0.001)
    }

    /// Calculate hyperfocal distance.
    pub fn hyperfocal_distance(focal_length: f32, f_stop: f32, coc: f32) -> f32 {
        (focal_length * focal_length) / (f_stop * coc) + focal_length
    }

    /// Calculate depth of field range.
    pub fn depth_of_field_range(
        focal_length: f32,
        f_stop: f32,
        focus_distance: f32,
        coc: f32,
    ) -> (f32, f32) {
        let h = hyperfocal_distance(focal_length, f_stop, coc);
        let near = (h * focus_distance) / (h + focus_distance - focal_length);
        let far = if focus_distance < (h - focal_length) {
            (h * focus_distance) / (h - focus_distance + focal_length)
        } else {
            f32::INFINITY
        };
        (near, far)
    }
}

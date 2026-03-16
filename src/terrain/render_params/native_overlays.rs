#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct VectorOverlaySettingsNative {
    /// When true, vectors are occluded by terrain
    pub depth_test: bool,
    /// Depth offset to prevent z-fighting
    pub depth_bias: f32,
    /// Slope-scaled bias for grazing angles
    pub depth_bias_slope: f32,
    /// Enable halo/outline rendering
    pub halo_enabled: bool,
    /// Halo width in pixels
    pub halo_width: f32,
    /// Halo color (RGBA)
    pub halo_color: [f32; 4],
    /// Halo blur/softness
    pub halo_blur: f32,
    /// Enable contour rendering
    pub contour_enabled: bool,
    /// Contour line width in pixels
    pub contour_width: f32,
    /// Contour color (RGBA)
    pub contour_color: [f32; 4],
}

#[cfg(feature = "extension-module")]
impl Default for VectorOverlaySettingsNative {
    fn default() -> Self {
        Self {
            depth_test: false,
            depth_bias: 0.001,
            depth_bias_slope: 1.0,
            halo_enabled: false,
            halo_width: 2.0,
            halo_color: [0.0, 0.0, 0.0, 0.5],
            halo_blur: 1.0,
            contour_enabled: false,
            contour_width: 1.0,
            contour_color: [0.0, 0.0, 0.0, 0.8],
        }
    }
}

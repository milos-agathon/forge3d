#[cfg(feature = "extension-module")]
#[derive(Clone, Debug)]
pub struct ProbeSettingsNative {
    pub enabled: bool,
    pub grid_dims: (u32, u32),
    pub origin: Option<(f32, f32)>,
    pub spacing: Option<(f32, f32)>,
    pub height_offset: f32,
    pub ray_count: u32,
    pub fallback_blend_distance: Option<f32>,
    pub sky_color: [f32; 3],
    pub sky_intensity: f32,
}

#[cfg(feature = "extension-module")]
impl Default for ProbeSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            grid_dims: (8, 8),
            origin: None,
            spacing: None,
            height_offset: 5.0,
            ray_count: 64,
            fallback_blend_distance: None,
            sky_color: [0.6, 0.75, 1.0],
            sky_intensity: 1.0,
        }
    }
}

#[cfg(feature = "extension-module")]
#[derive(Clone, Debug)]
pub struct ReflectionProbeSettingsNative {
    pub enabled: bool,
    pub grid_dims: (u32, u32),
    pub origin: Option<(f32, f32)>,
    pub spacing: Option<(f32, f32)>,
    pub height_offset: f32,
    pub resolution: u32,
    pub ray_count: u32,
    pub trace_steps: u32,
    pub trace_refine_steps: u32,
    pub fallback_blend_distance: Option<(f32, f32)>,
    pub strength: f32,
}

#[cfg(feature = "extension-module")]
impl Default for ReflectionProbeSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            grid_dims: (4, 4),
            origin: None,
            spacing: None,
            height_offset: 5.0,
            resolution: 16,
            ray_count: 64,
            trace_steps: 192,
            trace_refine_steps: 5,
            fallback_blend_distance: None,
            strength: 1.0,
        }
    }
}

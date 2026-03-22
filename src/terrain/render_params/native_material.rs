#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct MaterialNoiseSettingsNative {
    pub macro_scale: f32,
    pub detail_scale: f32,
    pub octaves: u32,
    pub snow_macro_amplitude: f32,
    pub snow_detail_amplitude: f32,
    pub rock_macro_amplitude: f32,
    pub rock_detail_amplitude: f32,
    pub wetness_macro_amplitude: f32,
    pub wetness_detail_amplitude: f32,
}

#[cfg(feature = "extension-module")]
impl Default for MaterialNoiseSettingsNative {
    fn default() -> Self {
        Self {
            macro_scale: 3.5,
            detail_scale: 18.0,
            octaves: 4,
            snow_macro_amplitude: 0.0,
            snow_detail_amplitude: 0.0,
            rock_macro_amplitude: 0.0,
            rock_detail_amplitude: 0.0,
            wetness_macro_amplitude: 0.0,
            wetness_detail_amplitude: 0.0,
        }
    }
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct MaterialLayerSettingsNative {
    // Snow layer
    pub snow_enabled: bool,
    pub snow_altitude_min: f32,
    pub snow_altitude_blend: f32,
    pub snow_slope_max: f32,   // degrees
    pub snow_slope_blend: f32, // degrees
    pub snow_aspect_influence: f32,
    pub snow_color: [f32; 3],
    pub snow_roughness: f32,
    pub snow_subsurface_strength: f32,
    pub snow_subsurface_color: [f32; 3],
    // Rock layer
    pub rock_enabled: bool,
    pub rock_slope_min: f32,   // degrees
    pub rock_slope_blend: f32, // degrees
    pub rock_color: [f32; 3],
    pub rock_roughness: f32,
    pub rock_subsurface_strength: f32,
    pub rock_subsurface_color: [f32; 3],
    // Wetness layer
    pub wetness_enabled: bool,
    pub wetness_strength: f32,
    pub wetness_slope_influence: f32,
    pub wetness_subsurface_strength: f32,
    pub wetness_subsurface_color: [f32; 3],
    // TV4: Procedural variation controls shared across terrain material layers.
    pub variation: MaterialNoiseSettingsNative,
}

#[cfg(feature = "extension-module")]
impl Default for MaterialLayerSettingsNative {
    fn default() -> Self {
        Self {
            snow_enabled: false,
            snow_altitude_min: 2000.0,
            snow_altitude_blend: 500.0,
            snow_slope_max: 45.0,
            snow_slope_blend: 15.0,
            snow_aspect_influence: 0.3,
            snow_color: [0.95, 0.95, 0.98],
            snow_roughness: 0.4,
            snow_subsurface_strength: 0.35,
            snow_subsurface_color: [0.78, 0.88, 0.98],
            rock_enabled: false,
            rock_slope_min: 45.0,
            rock_slope_blend: 10.0,
            rock_color: [0.35, 0.32, 0.28],
            rock_roughness: 0.8,
            rock_subsurface_strength: 0.0,
            rock_subsurface_color: [0.42, 0.36, 0.30],
            wetness_enabled: false,
            wetness_strength: 0.3,
            wetness_slope_influence: 0.5,
            wetness_subsurface_strength: 0.12,
            wetness_subsurface_color: [0.40, 0.28, 0.18],
            variation: MaterialNoiseSettingsNative::default(),
        }
    }
}

/// M5: Vector overlay settings for depth-correct rendering and halos
/// When depth_test = false, output is identical to baseline
#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct DetailSettingsNative {
    /// Enable micro-detail enhancement
    pub enabled: bool,
    /// World-space repeat interval for detail (default 2.0 meters)
    pub detail_scale: f32,
    /// Detail normal blending strength (0.0-1.0)
    pub normal_strength: f32,
    /// Albedo brightness noise amplitude (±percentage)
    pub albedo_noise: f32,
    /// Distance at which detail begins fading (world units)
    pub fade_start: f32,
    /// Distance at which detail is fully faded (world units)
    pub fade_end: f32,
}

#[cfg(feature = "extension-module")]
impl Default for DetailSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default (P5 compatibility)
            detail_scale: 2.0,
            normal_strength: 0.3,
            albedo_noise: 0.1,
            fade_start: 50.0,
            fade_end: 200.0,
        }
    }
}

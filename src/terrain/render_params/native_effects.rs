#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct FogSettingsNative {
    /// Fog density coefficient (0.0 = disabled)
    pub density: f32,
    /// Height falloff rate (higher = fog thins faster at altitude)
    pub height_falloff: f32,
    /// Base height for fog calculation (world-space Y)
    pub base_height: f32,
    /// Inscatter color (linear RGB, typically sky-tinted)
    pub inscatter: [f32; 3],
}

#[cfg(feature = "extension-module")]
impl Default for FogSettingsNative {
    fn default() -> Self {
        Self {
            density: 0.0, // Disabled by default (P1 compatibility)
            height_falloff: 0.0,
            base_height: 0.0,
            inscatter: [1.0, 1.0, 1.0],
        }
    }
}

/// P4: Water reflection settings extracted from Python ReflectionSettings or CLI args
/// When enabled = false, reflections are disabled (no-op for P3 compatibility)
#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct ReflectionSettingsNative {
    /// Enable water planar reflections
    pub enabled: bool,
    /// Reflection intensity (0.0-1.0, default 0.8)
    pub intensity: f32,
    /// Fresnel power for reflection falloff (default 5.0)
    pub fresnel_power: f32,
    /// Wave-based UV distortion strength (default 0.02)
    pub wave_strength: f32,
    /// Shore attenuation width - reduce reflections near land (default 0.3)
    pub shore_atten_width: f32,
    /// Water plane height in world space (default 0.0 = sea level)
    pub water_plane_height: f32,
}

#[cfg(feature = "extension-module")]
impl Default for ReflectionSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default (P3 compatibility)
            intensity: 0.8,
            fresnel_power: 5.0,
            wave_strength: 0.02,
            shore_atten_width: 0.3,
            water_plane_height: 0.0,
        }
    }
}

/// M2: Bloom post-processing settings
/// When enabled = false, bloom is disabled (identical output for backward compatibility)
#[cfg(feature = "extension-module")]
#[derive(Clone, Copy)]
pub struct BloomSettingsNative {
    /// Enable bloom post-processing
    pub enabled: bool,
    /// Brightness threshold for bloom extraction (default 1.5 = HDR only)
    pub threshold: f32,
    /// Softness of threshold transition (0.0 = hard, 1.0 = very soft)
    pub softness: f32,
    /// Bloom intensity when compositing (0.0-1.0+)
    pub intensity: f32,
    /// Blur radius multiplier (affects spread)
    pub radius: f32,
}

#[cfg(feature = "extension-module")]
impl Default for BloomSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default for backward compatibility
            threshold: 1.5, // Conservative: only HDR values bloom
            softness: 0.5,
            intensity: 0.3, // Conservative: subtle bloom
            radius: 1.0,
        }
    }
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct HeightAoSettingsNative {
    pub enabled: bool,
    pub resolution_scale: f32,
    pub directions: u32,
    pub steps: u32,
    pub max_distance: f32,
    pub strength: f32,
    pub blur: bool,
}

#[cfg(feature = "extension-module")]
impl Default for HeightAoSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            resolution_scale: 0.5,
            directions: 6,
            steps: 16,
            max_distance: 200.0,
            strength: 1.0,
            blur: false,
        }
    }
}

/// Heightfield ray-traced sun visibility settings
/// When enabled = false, sun visibility is disabled (fallback 1x1 white texture)
#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct SunVisibilitySettingsNative {
    pub enabled: bool,
    pub mode: String,
    pub resolution_scale: f32,
    pub samples: u32,
    pub steps: u32,
    pub max_distance: f32,
    pub softness: f32,
    pub bias: f32,
}

#[cfg(feature = "extension-module")]
impl Default for SunVisibilitySettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: "hard".to_string(),
            resolution_scale: 0.5,
            samples: 4,
            steps: 24,
            max_distance: 400.0,
            softness: 1.0,
            bias: 0.01,
        }
    }
}

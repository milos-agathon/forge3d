// src/lighting/screen_space.rs
// Screen-space effect settings (SSAO, SSGI, SSR)
// Split from types.rs for single-responsibility

use bytemuck::{Pod, Zeroable};

/// Screen-space effect technique enumeration (P5)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScreenSpaceEffect {
    None = 0,
    SSAO = 1,
    GTAO = 2,
    SSGI = 3,
    SSR = 4,
}

impl ScreenSpaceEffect {
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn name(&self) -> &'static str {
        match self {
            ScreenSpaceEffect::None => "none",
            ScreenSpaceEffect::SSAO => "ssao",
            ScreenSpaceEffect::GTAO => "gtao",
            ScreenSpaceEffect::SSGI => "ssgi",
            ScreenSpaceEffect::SSR => "ssr",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "none" => Some(ScreenSpaceEffect::None),
            "ssao" => Some(ScreenSpaceEffect::SSAO),
            "gtao" => Some(ScreenSpaceEffect::GTAO),
            "ssgi" => Some(ScreenSpaceEffect::SSGI),
            "ssr" => Some(ScreenSpaceEffect::SSR),
            _ => None,
        }
    }
}

/// SSAO/GTAO settings (P5)
/// Size: 32 bytes (2 vec4s)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SSAOSettings {
    pub radius: f32,
    pub intensity: f32,
    pub bias: f32,
    pub sample_count: u32,
    pub spiral_turns: f32,
    pub technique: u32,
    pub blur_radius: u32,
    pub temporal_alpha: f32,
}

impl Default for SSAOSettings {
    fn default() -> Self {
        Self {
            radius: 0.5,
            intensity: 1.0,
            bias: 0.025,
            sample_count: 16,
            spiral_turns: 7.0,
            technique: 0,
            blur_radius: 2,
            temporal_alpha: 0.0,
        }
    }
}

impl SSAOSettings {
    pub fn ssao(radius: f32, intensity: f32) -> Self {
        Self {
            radius,
            intensity,
            technique: 0,
            ..Default::default()
        }
    }

    pub fn gtao(radius: f32, intensity: f32) -> Self {
        Self {
            radius,
            intensity,
            technique: 1,
            ..Default::default()
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.radius <= 0.0 || self.radius > 100.0 {
            return Err("radius must be in (0, 100]");
        }
        if self.intensity < 0.0 || self.intensity > 5.0 {
            return Err("intensity must be in [0, 5]");
        }
        if self.sample_count == 0 || self.sample_count > 64 {
            return Err("sample_count must be in [1, 64]");
        }
        if self.technique > 1 {
            return Err("technique must be 0 (SSAO) or 1 (GTAO)");
        }
        Ok(())
    }
}

/// SSGI settings (P5)
/// Size: 32 bytes (2 vec4s)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SSGISettings {
    pub ray_steps: u32,
    pub ray_radius: f32,
    pub ray_thickness: f32,
    pub intensity: f32,
    pub temporal_alpha: f32,
    pub use_half_res: u32,
    pub ibl_fallback: f32,
    pub _pad: f32,
}

impl Default for SSGISettings {
    fn default() -> Self {
        Self {
            ray_steps: 24,
            ray_radius: 5.0,
            ray_thickness: 0.5,
            intensity: 1.0,
            temporal_alpha: 0.0,
            use_half_res: 1,
            ibl_fallback: 0.3,
            _pad: 0.0,
        }
    }
}

impl SSGISettings {
    pub fn new(ray_radius: f32, intensity: f32) -> Self {
        Self {
            ray_radius,
            intensity,
            ..Default::default()
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.ray_steps == 0 || self.ray_steps > 128 {
            return Err("ray_steps must be in [1, 128]");
        }
        if self.ray_radius <= 0.0 || self.ray_radius > 1000.0 {
            return Err("ray_radius must be in (0, 1000]");
        }
        if self.intensity < 0.0 || self.intensity > 10.0 {
            return Err("intensity must be in [0, 10]");
        }
        Ok(())
    }
}

/// SSR settings (P5)
/// Size: 32 bytes (2 vec4s)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SSRSettings {
    pub max_steps: u32,
    pub max_distance: f32,
    pub thickness: f32,
    pub stride: f32,
    pub intensity: f32,
    pub roughness_fade: f32,
    pub edge_fade: f32,
    pub temporal_alpha: f32,
}

impl Default for SSRSettings {
    fn default() -> Self {
        Self {
            max_steps: 48,
            max_distance: 50.0,
            thickness: 0.5,
            stride: 1.0,
            intensity: 1.0,
            roughness_fade: 0.8,
            edge_fade: 0.1,
            temporal_alpha: 0.0,
        }
    }
}

impl SSRSettings {
    pub fn new(max_distance: f32, intensity: f32) -> Self {
        Self {
            max_distance,
            intensity,
            ..Default::default()
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_steps == 0 || self.max_steps > 256 {
            return Err("max_steps must be in [1, 256]");
        }
        if self.max_distance <= 0.0 || self.max_distance > 1000.0 {
            return Err("max_distance must be in (0, 1000]");
        }
        if self.intensity < 0.0 || self.intensity > 10.0 {
            return Err("intensity must be in [0, 10]");
        }
        Ok(())
    }
}

/// Combined screen-space settings (P5)
#[derive(Debug, Clone, Copy)]
pub struct ScreenSpaceSettings {
    pub ssao: Option<SSAOSettings>,
    pub ssgi: Option<SSGISettings>,
    pub ssr: Option<SSRSettings>,
}

impl Default for ScreenSpaceSettings {
    fn default() -> Self {
        Self {
            ssao: None,
            ssgi: None,
            ssr: None,
        }
    }
}

impl ScreenSpaceSettings {
    pub fn with_ssao(radius: f32, intensity: f32) -> Self {
        Self {
            ssao: Some(SSAOSettings::ssao(radius, intensity)),
            ..Default::default()
        }
    }

    pub fn with_gtao(radius: f32, intensity: f32) -> Self {
        Self {
            ssao: Some(SSAOSettings::gtao(radius, intensity)),
            ..Default::default()
        }
    }

    pub fn with_ssgi(ray_radius: f32, intensity: f32) -> Self {
        Self {
            ssgi: Some(SSGISettings::new(ray_radius, intensity)),
            ..Default::default()
        }
    }

    pub fn with_ssr(max_distance: f32, intensity: f32) -> Self {
        Self {
            ssr: Some(SSRSettings::new(max_distance, intensity)),
            ..Default::default()
        }
    }
}

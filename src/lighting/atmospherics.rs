// src/lighting/atmospherics.rs
// Sky and volumetric fog settings (P6)
// Split from types.rs for single-responsibility

use bytemuck::{Pod, Zeroable};

/// Sky model enumeration (P6)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkyModel {
    Off = 0,
    Preetham = 1,
    HosekWilkie = 2,
}

impl SkyModel {
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn name(&self) -> &'static str {
        match self {
            SkyModel::Off => "off",
            SkyModel::Preetham => "preetham",
            SkyModel::HosekWilkie => "hosek-wilkie",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "off" => Some(SkyModel::Off),
            "preetham" => Some(SkyModel::Preetham),
            "hosek-wilkie" | "hosek_wilkie" | "hosekwilkie" => Some(SkyModel::HosekWilkie),
            _ => None,
        }
    }
}

/// Sky settings (P6)
/// Size: 48 bytes (3 vec4s)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SkySettings {
    pub sun_direction: [f32; 3],
    pub turbidity: f32,
    pub ground_albedo: f32,
    pub model: u32,
    pub sun_intensity: f32,
    pub exposure: f32,
    pub _pad: [f32; 4],
}

impl Default for SkySettings {
    fn default() -> Self {
        Self {
            sun_direction: [0.3, 0.8, 0.5],
            turbidity: 2.5,
            ground_albedo: 0.2,
            model: SkyModel::HosekWilkie.as_u32(),
            sun_intensity: 20.0,
            exposure: 1.0,
            _pad: [0.0; 4],
        }
    }
}

impl SkySettings {
    pub fn preetham(turbidity: f32, ground_albedo: f32) -> Self {
        Self {
            turbidity,
            ground_albedo,
            model: SkyModel::Preetham.as_u32(),
            ..Default::default()
        }
    }

    pub fn hosek_wilkie(turbidity: f32, ground_albedo: f32) -> Self {
        Self {
            turbidity,
            ground_albedo,
            model: SkyModel::HosekWilkie.as_u32(),
            ..Default::default()
        }
    }

    pub fn with_sun_angles(mut self, azimuth_deg: f32, elevation_deg: f32) -> Self {
        let az_rad = azimuth_deg.to_radians();
        let el_rad = elevation_deg.to_radians();
        self.sun_direction = [
            el_rad.cos() * az_rad.sin(),
            el_rad.sin(),
            el_rad.cos() * az_rad.cos(),
        ];
        self
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.turbidity < 1.0 || self.turbidity > 10.0 {
            return Err("turbidity must be in [1.0, 10.0]");
        }
        if self.ground_albedo < 0.0 || self.ground_albedo > 1.0 {
            return Err("ground_albedo must be in [0, 1]");
        }
        if self.sun_intensity < 0.0 || self.sun_intensity > 1000.0 {
            return Err("sun_intensity must be in [0, 1000]");
        }
        Ok(())
    }
}

/// Volumetric phase function enumeration (P6)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumetricPhase {
    Isotropic = 0,
    HenyeyGreenstein = 1,
}

impl VolumetricPhase {
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn name(&self) -> &'static str {
        match self {
            VolumetricPhase::Isotropic => "isotropic",
            VolumetricPhase::HenyeyGreenstein => "hg",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "isotropic" | "iso" => Some(VolumetricPhase::Isotropic),
            "hg" | "henyey-greenstein" | "henyey_greenstein" => {
                Some(VolumetricPhase::HenyeyGreenstein)
            }
            _ => None,
        }
    }
}

/// Volumetric fog settings (P6)
/// Size: 80 bytes (5 vec4s)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct VolumetricSettings {
    pub density: f32,
    pub height_falloff: f32,
    pub phase_g: f32,
    pub max_steps: u32,
    pub start_distance: f32,
    pub max_distance: f32,
    pub absorption: f32,
    pub sun_intensity: f32,
    pub scattering_color: [f32; 3],
    pub temporal_alpha: f32,
    pub ambient_color: [f32; 3],
    pub use_shadows: u32,
    pub jitter_strength: f32,
    pub phase_function: u32,
    pub _pad: [f32; 2],
}

impl Default for VolumetricSettings {
    fn default() -> Self {
        Self {
            density: 0.015,
            height_falloff: 0.1,
            phase_g: 0.7,
            max_steps: 48,
            start_distance: 0.1,
            max_distance: 100.0,
            absorption: 0.5,
            sun_intensity: 1.0,
            scattering_color: [1.0, 1.0, 1.0],
            temporal_alpha: 0.0,
            ambient_color: [0.3, 0.4, 0.5],
            use_shadows: 1,
            jitter_strength: 0.5,
            phase_function: VolumetricPhase::HenyeyGreenstein.as_u32(),
            _pad: [0.0; 2],
        }
    }
}

impl VolumetricSettings {
    pub fn with_god_rays(density: f32, phase_g: f32) -> Self {
        Self {
            density,
            phase_g,
            use_shadows: 1,
            ..Default::default()
        }
    }

    pub fn uniform_fog(density: f32) -> Self {
        Self {
            density,
            phase_g: 0.0,
            use_shadows: 0,
            height_falloff: 0.0,
            ..Default::default()
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.density < 0.0 || self.density > 10.0 {
            return Err("density must be in [0, 10]");
        }
        if self.height_falloff < 0.0 || self.height_falloff > 10.0 {
            return Err("height_falloff must be in [0, 10]");
        }
        if self.phase_g < -1.0 || self.phase_g > 1.0 {
            return Err("phase_g must be in [-1, 1]");
        }
        if self.max_steps == 0 || self.max_steps > 256 {
            return Err("max_steps must be in [1, 256]");
        }
        if self.start_distance < 0.0 || self.start_distance >= self.max_distance {
            return Err("start_distance must be in [0, max_distance)");
        }
        Ok(())
    }

    pub fn froxel_memory_budget(&self) -> usize {
        8192 * 8
    }
}

/// Combined atmospherics settings (P6)
#[derive(Debug, Clone, Copy)]
pub struct AtmosphericsSettings {
    pub sky: Option<SkySettings>,
    pub volumetric: Option<VolumetricSettings>,
}

impl Default for AtmosphericsSettings {
    fn default() -> Self {
        Self {
            sky: None,
            volumetric: None,
        }
    }
}

impl AtmosphericsSettings {
    pub fn with_sky(turbidity: f32, ground_albedo: f32) -> Self {
        Self {
            sky: Some(SkySettings::hosek_wilkie(turbidity, ground_albedo)),
            ..Default::default()
        }
    }

    pub fn with_volumetric(density: f32, phase_g: f32) -> Self {
        Self {
            volumetric: Some(VolumetricSettings::with_god_rays(density, phase_g)),
            ..Default::default()
        }
    }

    pub fn full_atmospherics(
        turbidity: f32,
        ground_albedo: f32,
        fog_density: f32,
        phase_g: f32,
    ) -> Self {
        Self {
            sky: Some(SkySettings::hosek_wilkie(turbidity, ground_albedo)),
            volumetric: Some(VolumetricSettings::with_god_rays(fog_density, phase_g)),
        }
    }
}

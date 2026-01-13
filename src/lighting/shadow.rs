// src/lighting/shadow.rs
// Shadow and GI settings with GPU-aligned layouts
// Split from types.rs for single-responsibility

use bytemuck::{Pod, Zeroable};

/// Shadow technique enumeration (P0/P3)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowTechnique {
    Hard = 0,
    PCF = 1,
    PCSS = 2,
    VSM = 3,
    EVSM = 4,
    MSM = 5,
    CSM = 6,
}

impl ShadowTechnique {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
    
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => ShadowTechnique::Hard,
            1 => ShadowTechnique::PCF,
            2 => ShadowTechnique::PCSS,
            3 => ShadowTechnique::VSM,
            4 => ShadowTechnique::EVSM,
            5 => ShadowTechnique::MSM,
            6 => ShadowTechnique::CSM,
            _ => ShadowTechnique::PCF, // Default fallback
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ShadowTechnique::Hard => "hard",
            ShadowTechnique::PCF => "pcf",
            ShadowTechnique::PCSS => "pcss",
            ShadowTechnique::VSM => "vsm",
            ShadowTechnique::EVSM => "evsm",
            ShadowTechnique::MSM => "msm",
            ShadowTechnique::CSM => "csm",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "hard" => Some(ShadowTechnique::Hard),
            "pcf" => Some(ShadowTechnique::PCF),
            "pcss" => Some(ShadowTechnique::PCSS),
            "vsm" => Some(ShadowTechnique::VSM),
            "evsm" => Some(ShadowTechnique::EVSM),
            "msm" => Some(ShadowTechnique::MSM),
            "csm" => Some(ShadowTechnique::CSM),
            _ => None,
        }
    }

    pub fn requires_moments(&self) -> bool {
        matches!(
            self,
            ShadowTechnique::VSM | ShadowTechnique::EVSM | ShadowTechnique::MSM
        )
    }

    pub fn channels(&self) -> u32 {
        match self {
            ShadowTechnique::VSM => 2,
            ShadowTechnique::EVSM | ShadowTechnique::MSM => 4,
            _ => 1,
        }
    }
}

/// Global illumination technique enumeration (P0)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GiTechnique {
    None = 0,
    Ibl = 1,
}

impl GiTechnique {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Shadow settings (P0)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ShadowSettings {
    pub tech: u32,
    pub map_res: u32,
    pub bias: f32,
    pub normal_bias: f32,
    pub softness: f32,
    pub pcss_blocker_radius: f32,
    pub pcss_filter_radius: f32,
    pub light_size: f32,
    pub moment_bias: f32,
    pub _pad: [f32; 3],
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            tech: ShadowTechnique::PCF.as_u32(),
            map_res: 2048,
            bias: 0.002,
            normal_bias: 0.5,
            softness: 1.25,
            pcss_blocker_radius: 0.03,
            pcss_filter_radius: 0.06,
            light_size: 0.25,
            moment_bias: 0.0005,
            _pad: [0.0; 3],
        }
    }
}

impl ShadowSettings {
    fn technique_enum(&self) -> Option<ShadowTechnique> {
        match self.tech {
            x if x == ShadowTechnique::Hard.as_u32() => Some(ShadowTechnique::Hard),
            x if x == ShadowTechnique::PCF.as_u32() => Some(ShadowTechnique::PCF),
            x if x == ShadowTechnique::PCSS.as_u32() => Some(ShadowTechnique::PCSS),
            x if x == ShadowTechnique::VSM.as_u32() => Some(ShadowTechnique::VSM),
            x if x == ShadowTechnique::EVSM.as_u32() => Some(ShadowTechnique::EVSM),
            x if x == ShadowTechnique::MSM.as_u32() => Some(ShadowTechnique::MSM),
            x if x == ShadowTechnique::CSM.as_u32() => Some(ShadowTechnique::CSM),
            _ => None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.map_res == 0 {
            return Err("map_res must be greater than zero".to_string());
        }
        if self.map_res > 4096 {
            return Err(format!("map_res must be <= 4096, got {}", self.map_res));
        }
        if !self.map_res.is_power_of_two() {
            return Err(format!(
                "map_res must be power of two, got {}",
                self.map_res
            ));
        }
        let technique = self
            .technique_enum()
            .ok_or_else(|| format!("unknown shadow technique id {}", self.tech))?;
        if technique == ShadowTechnique::PCSS {
            if self.pcss_blocker_radius < 0.0 {
                return Err("pcss_blocker_radius must be non-negative".to_string());
            }
            if self.pcss_filter_radius < 0.0 {
                return Err("pcss_filter_radius must be non-negative".to_string());
            }
            if self.light_size <= 0.0 {
                return Err("light_size must be positive for PCSS".to_string());
            }
        }
        if technique.requires_moments() && self.moment_bias <= 0.0 {
            return Err("moment_bias must be positive for moment-based techniques".to_string());
        }
        Ok(())
    }

    pub fn memory_budget(&self) -> u64 {
        let pixels = (self.map_res as u64) * (self.map_res as u64);
        let depth_bytes = pixels * 4;
        let moment_bytes = match self.technique_enum().unwrap_or(ShadowTechnique::Hard) {
            ShadowTechnique::VSM => pixels * 8,
            ShadowTechnique::EVSM | ShadowTechnique::MSM => pixels * 16,
            _ => 0,
        };
        depth_bytes + moment_bytes
    }
}

/// Global illumination settings (P0)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GiSettings {
    pub tech: u32,
    pub ibl_intensity: f32,
    pub ibl_rotation_deg: f32,
    pub _pad: f32,
}

impl Default for GiSettings {
    fn default() -> Self {
        Self {
            tech: GiTechnique::Ibl.as_u32(),
            ibl_intensity: 1.0,
            ibl_rotation_deg: 0.0,
            _pad: 0.0,
        }
    }
}

/// Atmospheric settings (P0)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Atmosphere {
    pub fog_density: f32,
    pub exposure: f32,
    pub sky_model: u32,
    pub _pad: f32,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self {
            fog_density: 0.0,
            exposure: 1.0,
            sky_model: 0,
            _pad: 0.0,
        }
    }
}

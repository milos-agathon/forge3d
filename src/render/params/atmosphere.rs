use super::common::normalize_key;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SkyModel {
    HosekWilkie,
    Preetham,
    Hdri,
}

impl SkyModel {
    #[allow(dead_code)]
    pub fn canonical(self) -> &'static str {
        match self {
            Self::HosekWilkie => "hosek-wilkie",
            Self::Preetham => "preetham",
            Self::Hdri => "hdri",
        }
    }
}

impl FromStr for SkyModel {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let key = normalize_key(value);
        Ok(match key.as_str() {
            "hosekwilkie" | "hosek-wilkie" => Self::HosekWilkie,
            "preetham" => Self::Preetham,
            "hdri" | "environment" | "envmap" => Self::Hdri,
            _ => return Err("unknown sky model"),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VolumetricMode {
    Raymarch,
    Froxels,
}

impl FromStr for VolumetricMode {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let key = normalize_key(value);
        Ok(match key.as_str() {
            "raymarch" | "rm" | "0" => Self::Raymarch,
            "froxels" | "fx" | "1" => Self::Froxels,
            _ => return Err("unknown volumetric mode"),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VolumetricPhase {
    Isotropic,
    #[serde(rename = "henyey-greenstein")]
    HenyeyGreenstein,
}

impl VolumetricPhase {
    #[allow(dead_code)]
    pub fn canonical(self) -> &'static str {
        match self {
            Self::Isotropic => "isotropic",
            Self::HenyeyGreenstein => "henyey-greenstein",
        }
    }
}

impl FromStr for VolumetricPhase {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let key = normalize_key(value);
        Ok(match key.as_str() {
            "isotropic" => Self::Isotropic,
            "henyeygreenstein" | "henyey-greenstein" | "hg" => Self::HenyeyGreenstein,
            _ => return Err("unknown volumetric phase"),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VolumetricParams {
    #[serde(default = "VolumetricParams::default_density")]
    pub density: f32,
    #[serde(default = "VolumetricParams::default_phase")]
    pub phase: VolumetricPhase,
    #[serde(default = "VolumetricParams::default_anisotropy")]
    pub anisotropy: f32,
    #[serde(default = "VolumetricParams::default_mode")]
    pub mode: VolumetricMode,
    #[serde(default = "VolumetricParams::default_height_falloff")]
    pub height_falloff: f32,
    #[serde(default = "VolumetricParams::default_max_steps")]
    pub max_steps: u32,
    #[serde(default = "VolumetricParams::default_start_distance")]
    pub start_distance: f32,
    #[serde(default = "VolumetricParams::default_max_distance")]
    pub max_distance: f32,
    #[serde(default = "VolumetricParams::default_absorption")]
    pub absorption: f32,
    #[serde(default = "VolumetricParams::default_scattering_color")]
    pub scattering_color: [f32; 3],
    #[serde(default = "VolumetricParams::default_ambient_color")]
    pub ambient_color: [f32; 3],
    #[serde(default = "VolumetricParams::default_temporal_alpha")]
    pub temporal_alpha: f32,
    #[serde(default = "VolumetricParams::default_use_shadows")]
    pub use_shadows: bool,
    #[serde(default = "VolumetricParams::default_jitter_strength")]
    pub jitter_strength: f32,
}

impl VolumetricParams {
    const fn default_density() -> f32 {
        0.02
    }

    fn default_phase() -> VolumetricPhase {
        VolumetricPhase::Isotropic
    }

    const fn default_anisotropy() -> f32 {
        0.0
    }

    fn default_mode() -> VolumetricMode {
        VolumetricMode::Raymarch
    }

    const fn default_height_falloff() -> f32 {
        0.0
    }
    const fn default_max_steps() -> u32 {
        64
    }
    const fn default_start_distance() -> f32 {
        0.0
    }
    const fn default_max_distance() -> f32 {
        1000.0
    }
    const fn default_absorption() -> f32 {
        0.0
    }
    const fn default_scattering_color() -> [f32; 3] {
        [1.0, 1.0, 1.0]
    }
    const fn default_ambient_color() -> [f32; 3] {
        [0.0, 0.0, 0.0]
    }
    const fn default_temporal_alpha() -> f32 {
        0.2
    }
    const fn default_use_shadows() -> bool {
        false
    }
    const fn default_jitter_strength() -> f32 {
        0.25
    }
}

impl Default for VolumetricParams {
    fn default() -> Self {
        Self {
            density: Self::default_density(),
            phase: Self::default_phase(),
            anisotropy: Self::default_anisotropy(),
            mode: Self::default_mode(),
            height_falloff: Self::default_height_falloff(),
            max_steps: Self::default_max_steps(),
            start_distance: Self::default_start_distance(),
            max_distance: Self::default_max_distance(),
            absorption: Self::default_absorption(),
            scattering_color: Self::default_scattering_color(),
            ambient_color: Self::default_ambient_color(),
            temporal_alpha: Self::default_temporal_alpha(),
            use_shadows: Self::default_use_shadows(),
            jitter_strength: Self::default_jitter_strength(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AtmosphereParams {
    #[serde(default = "AtmosphereParams::default_enabled")]
    pub enabled: bool,
    #[serde(default = "AtmosphereParams::default_sky")]
    pub sky: SkyModel,
    #[serde(default)]
    pub hdr_path: Option<String>,
    #[serde(default)]
    pub volumetric: Option<VolumetricParams>,
}

impl AtmosphereParams {
    const fn default_enabled() -> bool {
        true
    }

    fn default_sky() -> SkyModel {
        SkyModel::HosekWilkie
    }
}

impl Default for AtmosphereParams {
    fn default() -> Self {
        Self {
            enabled: Self::default_enabled(),
            sky: Self::default_sky(),
            hdr_path: None,
            volumetric: None,
        }
    }
}

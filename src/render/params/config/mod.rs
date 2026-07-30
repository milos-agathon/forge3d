use super::atmosphere::{AtmosphereParams, SkyModel, VolumetricPhase};
use super::gi::{GiMode, GiParams};
use super::lights::{LightType, LightingParams};
use super::shading::{BrdfModel, ShadingParams};
use super::shadows::{ShadowParams, ShadowTechnique};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone)]
pub struct ConfigError {
    message: String,
}

impl ConfigError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RendererConfig validation failed: {}", self.message)
    }
}

impl Error for ConfigError {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RendererConfig {
    #[serde(default)]
    pub lighting: LightingParams,
    #[serde(default)]
    pub shading: ShadingParams,
    #[serde(default)]
    pub shadows: ShadowParams,
    #[serde(default)]
    pub gi: GiParams,
    #[serde(default)]
    pub atmosphere: AtmosphereParams,
    #[serde(default)]
    pub brdf_override: Option<BrdfModel>,
}

impl RendererConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        for (index, light) in self.lighting.lights.iter().enumerate() {
            let label = format!("lights[{index}]");
            match light.light_type {
                LightType::Directional => {
                    if light.direction.is_none() {
                        return Err(ConfigError::new(format!(
                            "{label}.direction required for directional lights"
                        )));
                    }
                }
                LightType::Point | LightType::Spot | LightType::AreaRect | LightType::AreaDisk => {
                    if light.position.is_none() {
                        return Err(ConfigError::new(format!(
                            "{label}.position required for positional lights"
                        )));
                    }
                }
                LightType::AreaSphere => {
                    if light.position.is_none() {
                        return Err(ConfigError::new(format!(
                            "{label}.position required for area-sphere lights"
                        )));
                    }
                }
                LightType::Environment => {
                    if light.hdr_path.is_none() && self.atmosphere.hdr_path.is_none() {
                        return Err(ConfigError::new(format!(
                            "{label}.hdr_path required for environment lights unless atmosphere.hdr_path is set"
                        )));
                    }
                }
            }

            if let Some(cone) = light.cone_angle {
                if !(0.0..=180.0).contains(&cone) {
                    return Err(ConfigError::new(format!(
                        "{label}.cone_angle must be within [0, 180] degrees"
                    )));
                }
            }

            if let Some(extent) = light.area_extent {
                if extent[0] <= 0.0 || extent[1] <= 0.0 {
                    return Err(ConfigError::new(format!(
                        "{label}.area_extent entries must be positive"
                    )));
                }
            }
        }

        if self.shadows.enabled {
            if self.shadows.map_size == 0 {
                return Err(ConfigError::new(
                    "shadows.map_size must be greater than zero when shadows are enabled",
                ));
            }
            if !self.shadows.is_power_of_two_map() {
                return Err(ConfigError::new(
                    "shadows.map_size must be a power of two when shadows are enabled",
                ));
            }
            if matches!(
                self.shadows.technique,
                ShadowTechnique::Pcss
                    | ShadowTechnique::Pcf
                    | ShadowTechnique::Vsm
                    | ShadowTechnique::Evsm
                    | ShadowTechnique::Msm
                    | ShadowTechnique::Csm
            ) && self.shadows.map_size < 256
            {
                return Err(ConfigError::new(
                    "shadows.map_size should be at least 256 for filtered techniques",
                ));
            }
            if self.shadows.cascades == 0 || self.shadows.cascades > 4 {
                return Err(ConfigError::new("shadows.cascades must be within [1, 4]"));
            }
            if matches!(self.shadows.technique, ShadowTechnique::Csm) && self.shadows.cascades < 2 {
                return Err(ConfigError::new(
                    "shadows.cascades must be >= 2 when using cascaded shadow maps",
                ));
            }
            if matches!(self.shadows.technique, ShadowTechnique::Pcss) {
                if self.shadows.pcss_blocker_radius < 0.0 {
                    return Err(ConfigError::new(
                        "shadows.pcss_blocker_radius must be non-negative",
                    ));
                }
                if self.shadows.pcss_filter_radius < 0.0 {
                    return Err(ConfigError::new(
                        "shadows.pcss_filter_radius must be non-negative",
                    ));
                }
                if self.shadows.light_size <= 0.0 {
                    return Err(ConfigError::new(
                        "shadows.light_size must be positive for PCSS",
                    ));
                }
            }
            if self.shadows.requires_moments() && self.shadows.moment_bias <= 0.0 {
                return Err(ConfigError::new(
                    "shadows.moment_bias must be positive for moment-based techniques",
                ));
            }
            let max_bytes = 256 * 1024 * 1024;
            if self.shadows.atlas_memory_bytes() > max_bytes {
                return Err(ConfigError::new(
                    "shadow atlas exceeds 256 MiB budget; reduce cascades or resolution",
                ));
            }
        }

        if self.atmosphere.enabled && matches!(self.atmosphere.sky, SkyModel::Hdri) {
            if self.atmosphere.hdr_path.is_none()
                && !self.lighting.lights.iter().any(|light| {
                    light.light_type == LightType::Environment && light.hdr_path.is_some()
                })
            {
                return Err(ConfigError::new(
                    "atmosphere.sky=hdri requires atmosphere.hdr_path or an environment light with hdr_path",
                ));
            }
        }

        if let Some(vol) = &self.atmosphere.volumetric {
            if vol.density < 0.0 {
                return Err(ConfigError::new(
                    "atmosphere.volumetric.density must be non-negative",
                ));
            }
            if matches!(vol.phase, VolumetricPhase::HenyeyGreenstein)
                && !(-0.999..=0.999).contains(&vol.anisotropy)
            {
                return Err(ConfigError::new(
                    "atmosphere.volumetric.anisotropy must be within [-0.999, 0.999] for Henyey-Greenstein",
                ));
            }
        }

        for mode in &self.gi.modes {
            if matches!(mode, GiMode::Ibl)
                && !self
                    .lighting
                    .lights
                    .iter()
                    .any(|light| light.light_type == LightType::Environment)
                && self.atmosphere.hdr_path.is_none()
            {
                return Err(ConfigError::new(
                    "gi mode 'ibl' requires either an environment light or atmosphere.hdr_path",
                ));
            }
        }

        Ok(())
    }
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            lighting: LightingParams::default(),
            shading: ShadingParams::default(),
            shadows: ShadowParams::default(),
            gi: GiParams::default(),
            atmosphere: AtmosphereParams::default(),
            brdf_override: None,
        }
    }
}

#[cfg(test)]
mod tests;

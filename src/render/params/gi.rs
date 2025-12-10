use super::common::normalize_key;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GiMode {
    None,
    Ibl,
    #[serde(rename = "irradiance-probes")]
    IrradianceProbes,
    Ddgi,
    #[serde(rename = "voxel-cone-tracing")]
    VoxelConeTracing,
    Ssao,
    Gtao,
    Ssgi,
    Ssr,
}

impl GiMode {
    #[allow(dead_code)]
    pub fn canonical(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Ibl => "ibl",
            Self::IrradianceProbes => "irradiance-probes",
            Self::Ddgi => "ddgi",
            Self::VoxelConeTracing => "voxel-cone-tracing",
            Self::Ssao => "ssao",
            Self::Gtao => "gtao",
            Self::Ssgi => "ssgi",
            Self::Ssr => "ssr",
        }
    }
}

impl FromStr for GiMode {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let key = normalize_key(value);
        Ok(match key.as_str() {
            "none" => Self::None,
            "ibl" => Self::Ibl,
            "irradianceprobes" | "irradiance-probes" | "probes" => Self::IrradianceProbes,
            "ddgi" => Self::Ddgi,
            "voxelconetracing" | "voxel-cone-tracing" | "vct" => Self::VoxelConeTracing,
            "ssao" => Self::Ssao,
            "gtao" => Self::Gtao,
            "ssgi" => Self::Ssgi,
            "ssr" => Self::Ssr,
            _ => return Err("unknown gi mode"),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GiParams {
    #[serde(default = "GiParams::default_modes")]
    pub modes: Vec<GiMode>,
    #[serde(default = "GiParams::default_ao_strength")]
    pub ambient_occlusion_strength: f32,

    // P5.1: SSAO/GTAO parameters
    #[serde(default = "GiParams::default_ssao_radius")]
    pub ssao_radius: f32,
    #[serde(default = "GiParams::default_ssao_intensity")]
    pub ssao_intensity: f32,
    #[serde(default = "GiParams::default_ssao_technique")]
    pub ssao_technique: String, // "ssao" or "gtao"
    #[serde(default = "GiParams::default_ssao_temporal_enabled")]
    pub ssao_temporal_enabled: bool,
    #[serde(
        default = "GiParams::default_ssao_composite_enabled",
        alias = "ssao_composite"
    )]
    pub ssao_composite_enabled: bool,
    #[serde(default = "GiParams::default_ssao_mul")]
    pub ssao_mul: f32,
}

impl GiParams {
    fn default_modes() -> Vec<GiMode> {
        vec![GiMode::None]
    }
    const fn default_ao_strength() -> f32 {
        1.0
    }

    // P5.1 defaults
    const fn default_ssao_radius() -> f32 {
        0.5
    }
    const fn default_ssao_intensity() -> f32 {
        1.0
    }
    fn default_ssao_technique() -> String {
        "ssao".to_string()
    }
    const fn default_ssao_temporal_enabled() -> bool {
        false
    }
    const fn default_ssao_composite_enabled() -> bool {
        true
    }
    const fn default_ssao_mul() -> f32 {
        1.0
    }
}

impl Default for GiParams {
    fn default() -> Self {
        Self {
            modes: Self::default_modes(),
            ambient_occlusion_strength: Self::default_ao_strength(),
            ssao_radius: Self::default_ssao_radius(),
            ssao_intensity: Self::default_ssao_intensity(),
            ssao_technique: Self::default_ssao_technique(),
            ssao_temporal_enabled: Self::default_ssao_temporal_enabled(),
            ssao_composite_enabled: Self::default_ssao_composite_enabled(),
            ssao_mul: Self::default_ssao_mul(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SsrParams {
    #[serde(default)]
    pub ssr_enable: bool,
    #[serde(default = "SsrParams::default_max_steps")]
    pub ssr_max_steps: u32,
    #[serde(default = "SsrParams::default_thickness")]
    pub ssr_thickness: f32,
}

impl SsrParams {
    const fn default_max_steps() -> u32 {
        64
    }
    const fn default_thickness() -> f32 {
        0.05
    }

    pub fn set_enabled(&mut self, enable: bool) {
        self.ssr_enable = enable;
    }

    pub fn set_max_steps(&mut self, steps: u32) {
        let clamped = steps.clamp(1, 512);
        self.ssr_max_steps = clamped;
    }

    pub fn set_thickness(&mut self, thickness: f32) {
        self.ssr_thickness = thickness.clamp(0.0, 1.0);
    }
}

impl Default for SsrParams {
    fn default() -> Self {
        Self {
            ssr_enable: false,
            ssr_max_steps: Self::default_max_steps(),
            ssr_thickness: Self::default_thickness(),
        }
    }
}

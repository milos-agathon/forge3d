use super::common::normalize_key;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LightType {
    #[serde(alias = "dir")]
    Directional,
    #[serde(alias = "point-light")]
    Point,
    #[serde(alias = "spot-light")]
    Spot,
    #[serde(alias = "area-rect")]
    AreaRect,
    #[serde(alias = "area-disk")]
    AreaDisk,
    #[serde(alias = "area-sphere")]
    AreaSphere,
    #[serde(alias = "environment-map")]
    Environment,
}

impl LightType {
    #[allow(dead_code)]
    pub fn canonical(self) -> &'static str {
        match self {
            Self::Directional => "directional",
            Self::Point => "point",
            Self::Spot => "spot",
            Self::AreaRect => "area-rect",
            Self::AreaDisk => "area-disk",
            Self::AreaSphere => "area-sphere",
            Self::Environment => "environment",
        }
    }
}

impl FromStr for LightType {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let key = normalize_key(value);
        Ok(match key.as_str() {
            "directional" | "dir" | "sun" => Self::Directional,
            "point" | "pointlight" => Self::Point,
            "spot" | "spotlight" => Self::Spot,
            "arearect" | "rectlight" | "rect" => Self::AreaRect,
            "areadisk" | "disklight" | "disk" => Self::AreaDisk,
            "areasphere" | "spherelight" | "sphere" => Self::AreaSphere,
            "environment" | "env" | "hdri" => Self::Environment,
            _ => return Err("unknown light type"),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LightConfig {
    #[serde(rename = "type")]
    pub light_type: LightType,
    #[serde(default = "LightConfig::default_intensity")]
    pub intensity: f32,
    #[serde(default = "LightConfig::default_color")]
    pub color: [f32; 3],
    #[serde(default)]
    pub direction: Option<[f32; 3]>,
    #[serde(default)]
    pub position: Option<[f32; 3]>,
    #[serde(default)]
    pub cone_angle: Option<f32>,
    #[serde(default)]
    pub area_extent: Option<[f32; 2]>,
    #[serde(default)]
    pub hdr_path: Option<String>,
}

impl LightConfig {
    pub fn directional_default() -> Self {
        Self {
            light_type: LightType::Directional,
            intensity: Self::default_intensity(),
            color: Self::default_color(),
            direction: Some([-0.35, -1.0, -0.25]),
            position: None,
            cone_angle: None,
            area_extent: None,
            hdr_path: None,
        }
    }

    const fn default_intensity() -> f32 {
        5.0
    }

    const fn default_color() -> [f32; 3] {
        [1.0, 0.97, 0.94]
    }
}

impl Default for LightConfig {
    fn default() -> Self {
        Self::directional_default()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LightingParams {
    #[serde(default)]
    pub lights: Vec<LightConfig>,
    #[serde(default = "LightingParams::default_exposure")]
    pub exposure: f32,
}

impl LightingParams {
    const fn default_exposure() -> f32 {
        1.0
    }
}

impl Default for LightingParams {
    fn default() -> Self {
        Self {
            lights: vec![LightConfig::default()],
            exposure: Self::default_exposure(),
        }
    }
}

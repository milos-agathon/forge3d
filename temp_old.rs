// src/render/params.rs
// Type-safe renderer configuration enums and structs
// Exists to unify lighting, shading, shadow, GI, and atmosphere knobs before shader work lands
// RELEVANT FILES: src/lib.rs, src/render/mod.rs, python/forge3d/__init__.py, examples/terrain_demo.py

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
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
    fn canonical(self) -> &'static str {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BrdfModel {
    Lambert,
    Phong,
    #[serde(rename = "blinn-phong")]
    BlinnPhong,
    #[serde(rename = "oren-nayar")]
    OrenNayar,
    #[serde(rename = "cooktorrance-ggx")]
    CookTorranceGGX,
    #[serde(rename = "cooktorrance-beckmann")]
    CookTorranceBeckmann,
    #[serde(rename = "disney-principled")]
    DisneyPrincipled,
    #[serde(rename = "ashikhmin-shirley")]
    AshikhminShirley,
    Ward,
    Toon,
    Minnaert,
    #[serde(rename = "subsurface")]
    Subsurface,
    #[serde(rename = "hair")]
    Hair,
}

impl BrdfModel {
    fn canonical(self) -> &'static str {
        match self {
            Self::Lambert => "lambert",
            Self::Phong => "phong",
            Self::BlinnPhong => "blinn-phong",
            Self::OrenNayar => "oren-nayar",
            Self::CookTorranceGGX => "cooktorrance-ggx",
            Self::CookTorranceBeckmann => "cooktorrance-beckmann",
            Self::DisneyPrincipled => "disney-principled",
            Self::AshikhminShirley => "ashikhmin-shirley",
            Self::Ward => "ward",
            Self::Toon => "toon",
            Self::Minnaert => "minnaert",
            Self::Subsurface => "subsurface",
            Self::Hair => "hair",
        }
    }
}

impl FromStr for BrdfModel {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let key = normalize_key(value);
        Ok(match key.as_str() {
            "lambert" => Self::Lambert,
            "phong" => Self::Phong,
            "blinnphong" | "blinn-phong" => Self::BlinnPhong,
            "orennayar" | "oren-nayar" => Self::OrenNayar,
            "cooktorranceggx" | "cooktorrance-ggx" | "ggx" => Self::CookTorranceGGX,
            "cooktorrancebeckmann" | "cooktorrance-beckmann" | "beckmann" => {
                Self::CookTorranceBeckmann
            }
            "disneyprincipled" | "disney-principled" | "disney" => Self::DisneyPrincipled,
            "ashikhminshirley" | "ashikhmin-shirley" => Self::AshikhminShirley,
            "ward" => Self::Ward,
            "toon" => Self::Toon,
            "minnaert" => Self::Minnaert,
            "subsurface" | "sss" => Self::Subsurface,
            "hair" | "kajiyakay" | "kajiya-kay" => Self::Hair,
            _ => return Err("unknown brdf model"),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ShadowTechnique {
    Hard,
    Pcf,
    Pcss,
    Vsm,
    Evsm,
    Msm,
    Csm,
}

impl ShadowTechnique {
    fn canonical(self) -> &'static str {
        match self {
            Self::Hard => "hard",
            Self::Pcf => "pcf",
            Self::Pcss => "pcss",
            Self::Vsm => "vsm",
            Self::Evsm => "evsm",
            Self::Msm => "msm",
            Self::Csm => "csm",
        }
    }
}

impl FromStr for ShadowTechnique {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let key = normalize_key(value);
        Ok(match key.as_str() {
            "hard" => Self::Hard,
            "pcf" => Self::Pcf,
            "pcss" => Self::Pcss,
            "vsm" => Self::Vsm,
            "evsm" => Self::Evsm,
            "msm" => Self::Msm,
            "csm" => Self::Csm,
            _ => return Err("unknown shadow technique"),
        })
    }
}

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
    fn canonical(self) -> &'static str {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SkyModel {
    HosekWilkie,
    Preetham,
    Hdri,
}

impl SkyModel {
    fn canonical(self) -> &'static str {
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
pub enum VolumetricPhase {
    Isotropic,
    #[serde(rename = "henyey-greenstein")]
    HenyeyGreenstein,
}

impl VolumetricPhase {
    fn canonical(self) -> &'static str {
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShadingParams {
    #[serde(default = "ShadingParams::default_brdf")]
    pub brdf: BrdfModel,
    #[serde(default = "ShadingParams::default_enable_normal_maps")]
    pub normal_maps: bool,
    #[serde(default = "ShadingParams::default_enable_clearcoat")]
    pub clearcoat: bool,
}

impl ShadingParams {
    fn default_brdf() -> BrdfModel {
        BrdfModel::CookTorranceGGX
    }

    const fn default_enable_normal_maps() -> bool {
        true
    }

    const fn default_enable_clearcoat() -> bool {
        false
    }
}

impl Default for ShadingParams {
    fn default() -> Self {
        Self {
            brdf: Self::default_brdf(),
            normal_maps: Self::default_enable_normal_maps(),
            clearcoat: Self::default_enable_clearcoat(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShadowParams {
    #[serde(default = "ShadowParams::default_enabled")]
    pub enabled: bool,
    #[serde(default = "ShadowParams::default_technique")]
    pub technique: ShadowTechnique,
    #[serde(default = "ShadowParams::default_map_size")]
    pub map_size: u32,
    #[serde(default = "ShadowParams::default_cascades")]
    pub cascades: u32,
    #[serde(default = "ShadowParams::default_contact_hardening")]
    pub contact_hardening: bool,
}

impl ShadowParams {
    const fn default_enabled() -> bool {
        true
    }

    fn default_technique() -> ShadowTechnique {
        ShadowTechnique::Pcf
    }

    const fn default_map_size() -> u32 {
        2048
    }

    const fn default_cascades() -> u32 {
        4
    }

    const fn default_contact_hardening() -> bool {
        true
    }
}

impl Default for ShadowParams {
    fn default() -> Self {
        Self {
            enabled: Self::default_enabled(),
            technique: Self::default_technique(),
            map_size: Self::default_map_size(),
            cascades: Self::default_cascades(),
            contact_hardening: Self::default_contact_hardening(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GiParams {
    #[serde(default)]
    pub modes: Vec<GiMode>,
    #[serde(default = "GiParams::default_ao_strength")]
    pub ambient_occlusion_strength: f32,
}

impl GiParams {
    const fn default_ao_strength() -> f32 {
        1.0
    }
}

impl Default for GiParams {
    fn default() -> Self {
        Self {
            modes: vec![GiMode::None],
            ambient_occlusion_strength: Self::default_ao_strength(),
        }
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
}

impl Default for VolumetricParams {
    fn default() -> Self {
        Self {
            density: Self::default_density(),
            phase: Self::default_phase(),
            anisotropy: Self::default_anisotropy(),
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
            if matches!(self.shadows.technique, ShadowTechnique::Csm) && self.shadows.cascades < 2 {
                return Err(ConfigError::new(
                    "shadows.cascades must be >= 2 when using cascaded shadow maps",
                ));
            }
        }

        if self.atmosphere.enabled && matches!(self.atmosphere.sky, SkyModel::Hdri) {
            if self.atmosphere.hdr_path.is_none()
                && !self
                    .lighting
                    .lights
                    .iter()
                    .any(|light| light.light_type == LightType::Environment && light.hdr_path.is_some())
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
                && !( -0.999..=0.999).contains(&vol.anisotropy)
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

fn normalize_key(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| !matches!(c, '-' | '_' | ' ' | '.'))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_serializes_and_validates() {
        let cfg = RendererConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize default config");
        let de: RendererConfig =
            serde_json::from_str(&json).expect("deserialize default config");
        de.validate().expect("default config should validate");
    }

    #[test]
    fn parse_enums_from_strings() {
        assert_eq!("directional".parse::<LightType>().unwrap(), LightType::Directional);
        assert_eq!("cooktorrance-ggx".parse::<BrdfModel>().unwrap(), BrdfModel::CookTorranceGGX);
        assert_eq!("pcf".parse::<ShadowTechnique>().unwrap(), ShadowTechnique::Pcf);
        assert_eq!("ssao".parse::<GiMode>().unwrap(), GiMode::Ssao);
        assert_eq!("hosek-wilkie".parse::<SkyModel>().unwrap(), SkyModel::HosekWilkie);
        assert_eq!("hg".parse::<VolumetricPhase>().unwrap(), VolumetricPhase::HenyeyGreenstein);
    }

    #[test]
    fn validation_catches_missing_direction() {
        let mut cfg = RendererConfig::default();
        if let Some(light) = cfg.lighting.lights.get_mut(0) {
            light.direction = None;
        }
        let err = cfg.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("lights[0].direction required"));
    }
}

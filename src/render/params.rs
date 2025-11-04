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

impl LightType {
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[serde(default = "ShadowParams::default_pcss_blocker_radius")]
    pub pcss_blocker_radius: f32,
    #[serde(default = "ShadowParams::default_pcss_filter_radius")]
    pub pcss_filter_radius: f32,
    #[serde(default = "ShadowParams::default_light_size")]
    pub light_size: f32,
    #[serde(default = "ShadowParams::default_moment_bias")]
    pub moment_bias: f32,
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

    const fn default_pcss_blocker_radius() -> f32 {
        0.03
    }

    const fn default_pcss_filter_radius() -> f32 {
        0.06
    }

    const fn default_light_size() -> f32 {
        0.25
    }

    const fn default_moment_bias() -> f32 {
        0.0005
    }

    pub fn requires_moments(&self) -> bool {
        matches!(
            self.technique,
            ShadowTechnique::Vsm | ShadowTechnique::Evsm | ShadowTechnique::Msm
        )
    }

    pub fn atlas_memory_bytes(&self) -> u64 {
        let cascades = self.cascades.max(1) as u64;
        let resolution = self.map_size.max(1) as u64;
        let depth_bytes = cascades * resolution * resolution * 4;
        let moment_bytes = match self.technique {
            ShadowTechnique::Vsm => cascades * resolution * resolution * 8,
            ShadowTechnique::Evsm | ShadowTechnique::Msm => cascades * resolution * resolution * 16,
            _ => 0,
        };
        depth_bytes + moment_bytes
    }

    pub fn is_power_of_two_map(&self) -> bool {
        self.map_size.is_power_of_two()
    }

    /// Convert ShadowParams from RendererConfig to ShadowManagerConfig (P3-11)
    pub fn to_shadow_manager_config(&self) -> crate::shadows::ShadowManagerConfig {
        use crate::shadows::{ShadowManagerConfig, CsmConfig};
        
        let csm = CsmConfig {
            cascade_count: self.cascades,
            shadow_map_size: self.map_size,
            max_shadow_distance: 1000.0, // Will be overridden by camera far plane
            cascade_splits: vec![], // Empty = auto-calculate splits
            pcf_kernel_size: 3, // Default 3x3 PCF
            depth_bias: 0.0005,
            slope_bias: 0.001,
            peter_panning_offset: 0.002,
            enable_evsm: matches!(self.technique, ShadowTechnique::Evsm),
            evsm_positive_exp: 40.0,
            evsm_negative_exp: 40.0,
            debug_mode: 0,
            enable_unclipped_depth: false,
            depth_clip_factor: 1.0,
            stabilize_cascades: true,
            cascade_blend_range: 0.1,
        };
        
        // Convert ShadowTechnique from params to lighting::types::ShadowTechnique
        let technique = match self.technique {
            ShadowTechnique::Hard => crate::lighting::types::ShadowTechnique::Hard,
            ShadowTechnique::Pcf => crate::lighting::types::ShadowTechnique::PCF,
            ShadowTechnique::Pcss => crate::lighting::types::ShadowTechnique::PCSS,
            ShadowTechnique::Vsm => crate::lighting::types::ShadowTechnique::VSM,
            ShadowTechnique::Evsm => crate::lighting::types::ShadowTechnique::EVSM,
            ShadowTechnique::Msm => crate::lighting::types::ShadowTechnique::MSM,
            ShadowTechnique::Csm => crate::lighting::types::ShadowTechnique::PCF, // CSM is a layout, use PCF
        };
        
        ShadowManagerConfig {
            csm,
            technique,
            pcss_blocker_radius: self.pcss_blocker_radius,
            pcss_filter_radius: self.pcss_filter_radius,
            light_size: self.light_size,
            moment_bias: self.moment_bias,
            max_memory_bytes: 256 * 1024 * 1024, // 256 MiB budget
        }
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
            pcss_blocker_radius: Self::default_pcss_blocker_radius(),
            pcss_filter_radius: Self::default_pcss_filter_radius(),
            light_size: Self::default_light_size(),
            moment_bias: Self::default_moment_bias(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GiParams {
    #[serde(default = "GiParams::default_modes")]
    pub modes: Vec<GiMode>,
    #[serde(default = "GiParams::default_ao_strength")]
    pub ambient_occlusion_strength: f32,
}

impl GiParams {
    fn default_modes() -> Vec<GiMode> { vec![GiMode::None] }
    const fn default_ao_strength() -> f32 { 1.0 }
}

impl Default for GiParams {
    fn default() -> Self {
        Self { modes: Self::default_modes(), ambient_occlusion_strength: Self::default_ao_strength() }
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

    fn default_mode() -> VolumetricMode { VolumetricMode::Raymarch }

    const fn default_height_falloff() -> f32 { 0.0 }
    const fn default_max_steps() -> u32 { 64 }
    const fn default_start_distance() -> f32 { 0.0 }
    const fn default_max_distance() -> f32 { 1000.0 }
    const fn default_absorption() -> f32 { 0.0 }
    const fn default_scattering_color() -> [f32; 3] { [1.0, 1.0, 1.0] }
    const fn default_ambient_color() -> [f32; 3] { [0.0, 0.0, 0.0] }
    const fn default_temporal_alpha() -> f32 { 0.2 }
    const fn default_use_shadows() -> bool { false }
    const fn default_jitter_strength() -> f32 { 0.25 }
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
                return Err(ConfigError::new(
                    "shadows.cascades must be within [1, 4]",
                ));
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

    // P0-01: Comprehensive enum coverage and round-trip tests
    #[test]
    fn light_type_all_variants_parse_and_round_trip() {
        let cases = vec![
            (LightType::Directional, "directional"),
            (LightType::Point, "point"),
            (LightType::Spot, "spot"),
            (LightType::AreaRect, "area-rect"),
            (LightType::AreaDisk, "area-disk"),
            (LightType::AreaSphere, "area-sphere"),
            (LightType::Environment, "environment"),
        ];
        for (variant, canonical) in cases {
            // Test canonical name matches
            assert_eq!(variant.canonical(), canonical);
            // Test parsing canonical returns correct variant
            assert_eq!(canonical.parse::<LightType>().unwrap(), variant);
            // Test round-trip: variant -> canonical -> parse -> variant
            let round_trip = variant.canonical().parse::<LightType>().unwrap();
            assert_eq!(round_trip, variant);
        }
    }

    #[test]
    fn light_type_aliases_parse_correctly() {
        assert_eq!("dir".parse::<LightType>().unwrap(), LightType::Directional);
        assert_eq!("sun".parse::<LightType>().unwrap(), LightType::Directional);
        assert_eq!("pointlight".parse::<LightType>().unwrap(), LightType::Point);
        assert_eq!("spotlight".parse::<LightType>().unwrap(), LightType::Spot);
        assert_eq!("rect".parse::<LightType>().unwrap(), LightType::AreaRect);
        assert_eq!("disk".parse::<LightType>().unwrap(), LightType::AreaDisk);
        assert_eq!("sphere".parse::<LightType>().unwrap(), LightType::AreaSphere);
        assert_eq!("env".parse::<LightType>().unwrap(), LightType::Environment);
        assert_eq!("hdri".parse::<LightType>().unwrap(), LightType::Environment);
    }

    #[test]
    fn brdf_model_all_variants_parse_and_round_trip() {
        let cases = vec![
            (BrdfModel::Lambert, "lambert"),
            (BrdfModel::Phong, "phong"),
            (BrdfModel::BlinnPhong, "blinn-phong"),
            (BrdfModel::OrenNayar, "oren-nayar"),
            (BrdfModel::CookTorranceGGX, "cooktorrance-ggx"),
            (BrdfModel::CookTorranceBeckmann, "cooktorrance-beckmann"),
            (BrdfModel::DisneyPrincipled, "disney-principled"),
            (BrdfModel::AshikhminShirley, "ashikhmin-shirley"),
            (BrdfModel::Ward, "ward"),
            (BrdfModel::Toon, "toon"),
            (BrdfModel::Minnaert, "minnaert"),
            (BrdfModel::Subsurface, "subsurface"),
            (BrdfModel::Hair, "hair"),
        ];
        for (variant, canonical) in cases {
            assert_eq!(variant.canonical(), canonical);
            assert_eq!(canonical.parse::<BrdfModel>().unwrap(), variant);
            let round_trip = variant.canonical().parse::<BrdfModel>().unwrap();
            assert_eq!(round_trip, variant);
        }
    }

    #[test]
    fn brdf_model_aliases_parse_correctly() {
        assert_eq!("blinnphong".parse::<BrdfModel>().unwrap(), BrdfModel::BlinnPhong);
        assert_eq!("orennayar".parse::<BrdfModel>().unwrap(), BrdfModel::OrenNayar);
        assert_eq!("ggx".parse::<BrdfModel>().unwrap(), BrdfModel::CookTorranceGGX);
        assert_eq!("beckmann".parse::<BrdfModel>().unwrap(), BrdfModel::CookTorranceBeckmann);
        assert_eq!("disney".parse::<BrdfModel>().unwrap(), BrdfModel::DisneyPrincipled);
        assert_eq!("ashikhminshirley".parse::<BrdfModel>().unwrap(), BrdfModel::AshikhminShirley);
        assert_eq!("sss".parse::<BrdfModel>().unwrap(), BrdfModel::Subsurface);
        assert_eq!("kajiyakay".parse::<BrdfModel>().unwrap(), BrdfModel::Hair);
        assert_eq!("kajiya-kay".parse::<BrdfModel>().unwrap(), BrdfModel::Hair);
    }

    #[test]
    fn shadow_technique_all_variants_parse_and_round_trip() {
        let cases = vec![
            (ShadowTechnique::Hard, "hard"),
            (ShadowTechnique::Pcf, "pcf"),
            (ShadowTechnique::Pcss, "pcss"),
            (ShadowTechnique::Vsm, "vsm"),
            (ShadowTechnique::Evsm, "evsm"),
            (ShadowTechnique::Msm, "msm"),
            (ShadowTechnique::Csm, "csm"),
        ];
        for (variant, canonical) in cases {
            assert_eq!(variant.canonical(), canonical);
            assert_eq!(canonical.parse::<ShadowTechnique>().unwrap(), variant);
            let round_trip = variant.canonical().parse::<ShadowTechnique>().unwrap();
            assert_eq!(round_trip, variant);
        }
    }

    #[test]
    fn gi_mode_all_variants_parse_and_round_trip() {
        let cases = vec![
            (GiMode::None, "none"),
            (GiMode::Ibl, "ibl"),
            (GiMode::IrradianceProbes, "irradiance-probes"),
            (GiMode::Ddgi, "ddgi"),
            (GiMode::VoxelConeTracing, "voxel-cone-tracing"),
            (GiMode::Ssao, "ssao"),
            (GiMode::Gtao, "gtao"),
            (GiMode::Ssgi, "ssgi"),
            (GiMode::Ssr, "ssr"),
        ];
        for (variant, canonical) in cases {
            assert_eq!(variant.canonical(), canonical);
            assert_eq!(canonical.parse::<GiMode>().unwrap(), variant);
            let round_trip = variant.canonical().parse::<GiMode>().unwrap();
            assert_eq!(round_trip, variant);
        }
    }

    #[test]
    fn gi_mode_aliases_parse_correctly() {
        assert_eq!("irradianceprobes".parse::<GiMode>().unwrap(), GiMode::IrradianceProbes);
        assert_eq!("probes".parse::<GiMode>().unwrap(), GiMode::IrradianceProbes);
        assert_eq!("voxelconetracing".parse::<GiMode>().unwrap(), GiMode::VoxelConeTracing);
        assert_eq!("vct".parse::<GiMode>().unwrap(), GiMode::VoxelConeTracing);
    }

    #[test]
    fn sky_model_all_variants_parse_and_round_trip() {
        let cases = vec![
            (SkyModel::HosekWilkie, "hosek-wilkie"),
            (SkyModel::Preetham, "preetham"),
            (SkyModel::Hdri, "hdri"),
        ];
        for (variant, canonical) in cases {
            assert_eq!(variant.canonical(), canonical);
            assert_eq!(canonical.parse::<SkyModel>().unwrap(), variant);
            let round_trip = variant.canonical().parse::<SkyModel>().unwrap();
            assert_eq!(round_trip, variant);
        }
    }

    #[test]
    fn sky_model_aliases_parse_correctly() {
        assert_eq!("hosekwilkie".parse::<SkyModel>().unwrap(), SkyModel::HosekWilkie);
        assert_eq!("environment".parse::<SkyModel>().unwrap(), SkyModel::Hdri);
        assert_eq!("envmap".parse::<SkyModel>().unwrap(), SkyModel::Hdri);
    }

    #[test]
    fn volumetric_phase_all_variants_parse_and_round_trip() {
        let cases = vec![
            (VolumetricPhase::Isotropic, "isotropic"),
            (VolumetricPhase::HenyeyGreenstein, "henyey-greenstein"),
        ];
        for (variant, canonical) in cases {
            assert_eq!(variant.canonical(), canonical);
            assert_eq!(canonical.parse::<VolumetricPhase>().unwrap(), variant);
            let round_trip = variant.canonical().parse::<VolumetricPhase>().unwrap();
            assert_eq!(round_trip, variant);
        }
    }

    #[test]
    fn volumetric_phase_aliases_parse_correctly() {
        assert_eq!("henyeygreenstein".parse::<VolumetricPhase>().unwrap(), VolumetricPhase::HenyeyGreenstein);
        assert_eq!("hg".parse::<VolumetricPhase>().unwrap(), VolumetricPhase::HenyeyGreenstein);
    }

    #[test]
    fn volumetric_mode_all_variants_parse() {
        let cases = vec![
            (VolumetricMode::Raymarch, "raymarch"),
            (VolumetricMode::Froxels, "froxels"),
        ];
        for (variant, canonical) in cases {
            assert_eq!(canonical.parse::<VolumetricMode>().unwrap(), variant);
        }
    }

    #[test]
    fn volumetric_mode_aliases_parse_correctly() {
        assert_eq!("rm".parse::<VolumetricMode>().unwrap(), VolumetricMode::Raymarch);
        assert_eq!("0".parse::<VolumetricMode>().unwrap(), VolumetricMode::Raymarch);
        assert_eq!("fx".parse::<VolumetricMode>().unwrap(), VolumetricMode::Froxels);
        assert_eq!("1".parse::<VolumetricMode>().unwrap(), VolumetricMode::Froxels);
    }

    #[test]
    fn normalize_key_handles_variations() {
        assert_eq!(normalize_key("Cook-Torrance_GGX"), "cooktorranceggx");
        assert_eq!(normalize_key(" Blinn Phong "), "blinnphong");
        assert_eq!(normalize_key("OREN.NAYAR"), "orennayar");
        assert_eq!(normalize_key("henyey-greenstein"), "henyeygreenstein");
    }

    // P0-02: Comprehensive validation tests for all invariants

    // Light validation tests
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

    #[test]
    fn validation_point_light_requires_position() {
        let mut cfg = RendererConfig::default();
        cfg.lighting.lights = vec![LightConfig {
            light_type: LightType::Point,
            intensity: 5.0,
            color: [1.0, 1.0, 1.0],
            direction: None,
            position: None,
            cone_angle: None,
            area_extent: None,
            hdr_path: None,
        }];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("position required"));
    }

    #[test]
    fn validation_spot_light_requires_position() {
        let mut cfg = RendererConfig::default();
        cfg.lighting.lights = vec![LightConfig {
            light_type: LightType::Spot,
            intensity: 5.0,
            color: [1.0, 1.0, 1.0],
            direction: None,
            position: None,
            cone_angle: Some(45.0),
            area_extent: None,
            hdr_path: None,
        }];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("position required"));
    }

    #[test]
    fn validation_area_lights_require_position() {
        for light_type in [LightType::AreaRect, LightType::AreaDisk, LightType::AreaSphere] {
            let mut cfg = RendererConfig::default();
            cfg.lighting.lights = vec![LightConfig {
                light_type,
                intensity: 5.0,
                color: [1.0, 1.0, 1.0],
                direction: None,
                position: None,
                cone_angle: None,
                area_extent: Some([1.0, 1.0]),
                hdr_path: None,
            }];
            let err = cfg.validate().unwrap_err();
            assert!(err.to_string().contains("position required"));
        }
    }

    #[test]
    fn validation_environment_light_requires_hdr_path() {
        let mut cfg = RendererConfig::default();
        cfg.lighting.lights = vec![LightConfig {
            light_type: LightType::Environment,
            intensity: 1.0,
            color: [1.0, 1.0, 1.0],
            direction: None,
            position: None,
            cone_angle: None,
            area_extent: None,
            hdr_path: None,
        }];
        cfg.atmosphere.hdr_path = None;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("hdr_path required"));
    }

    #[test]
    fn validation_environment_light_accepts_atmosphere_hdr() {
        let mut cfg = RendererConfig::default();
        cfg.lighting.lights = vec![LightConfig {
            light_type: LightType::Environment,
            intensity: 1.0,
            color: [1.0, 1.0, 1.0],
            direction: None,
            position: None,
            cone_angle: None,
            area_extent: None,
            hdr_path: None,
        }];
        cfg.atmosphere.hdr_path = Some("assets/sky.hdr".to_string());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validation_cone_angle_must_be_valid() {
        let mut cfg = RendererConfig::default();
        cfg.lighting.lights = vec![LightConfig {
            light_type: LightType::Spot,
            intensity: 5.0,
            color: [1.0, 1.0, 1.0],
            direction: None,
            position: Some([0.0, 0.0, 0.0]),
            cone_angle: Some(200.0),
            area_extent: None,
            hdr_path: None,
        }];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("cone_angle"));
    }

    #[test]
    fn validation_area_extent_must_be_positive() {
        let mut cfg = RendererConfig::default();
        cfg.lighting.lights = vec![LightConfig {
            light_type: LightType::AreaRect,
            intensity: 5.0,
            color: [1.0, 1.0, 1.0],
            direction: None,
            position: Some([0.0, 0.0, 0.0]),
            cone_angle: None,
            area_extent: Some([1.0, -1.0]),
            hdr_path: None,
        }];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("area_extent"));
    }

    // Shadow validation tests
    #[test]
    fn shadows_require_power_of_two_map() {
        let mut cfg = RendererConfig::default();
        cfg.shadows.map_size = 300;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("power of two"));
    }

    #[test]
    fn shadows_map_size_must_be_nonzero() {
        let mut cfg = RendererConfig::default();
        cfg.shadows.map_size = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("greater than zero"));
    }

    #[test]
    fn shadows_cascades_must_be_in_range() {
        let mut cfg = RendererConfig::default();
        cfg.shadows.cascades = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("[1, 4]"));

        cfg.shadows.cascades = 5;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("[1, 4]"));
    }

    #[test]
    fn shadows_csm_requires_multiple_cascades() {
        let mut cfg = RendererConfig::default();
        cfg.shadows.technique = ShadowTechnique::Csm;
        cfg.shadows.cascades = 1;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains(">= 2"));
    }

    #[test]
    fn shadows_pcss_requires_positive_blocker_radius() {
        let mut cfg = RendererConfig::default();
        cfg.shadows.technique = ShadowTechnique::Pcss;
        cfg.shadows.pcss_blocker_radius = -0.1;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("pcss_blocker_radius"));
    }

    #[test]
    fn shadows_pcss_requires_positive_filter_radius() {
        let mut cfg = RendererConfig::default();
        cfg.shadows.technique = ShadowTechnique::Pcss;
        cfg.shadows.pcss_filter_radius = -0.1;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("pcss_filter_radius"));
    }

    #[test]
    fn shadows_pcss_requires_positive_light_size() {
        let mut cfg = RendererConfig::default();
        cfg.shadows.technique = ShadowTechnique::Pcss;
        cfg.shadows.light_size = 0.0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("light_size"));
    }

    #[test]
    fn shadows_moment_techniques_require_positive_bias() {
        for technique in [ShadowTechnique::Vsm, ShadowTechnique::Evsm, ShadowTechnique::Msm] {
            let mut cfg = RendererConfig::default();
            cfg.shadows.technique = technique;
            cfg.shadows.moment_bias = 0.0;
            let err = cfg.validate().unwrap_err();
            assert!(err.to_string().contains("moment_bias"));
        }
    }

    #[test]
    fn shadow_memory_budget_is_enforced() {
        let mut cfg = RendererConfig::default();
        cfg.shadows.technique = ShadowTechnique::Evsm;
        cfg.shadows.map_size = 8192;
        cfg.shadows.cascades = 4;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("256 MiB"));
    }

    #[test]
    fn shadows_filtered_techniques_recommend_min_resolution() {
        let mut cfg = RendererConfig::default();
        cfg.shadows.technique = ShadowTechnique::Pcf;
        cfg.shadows.map_size = 128;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("at least 256"));
    }

    // Atmosphere validation tests
    #[test]
    fn atmosphere_hdri_sky_requires_hdr_path() {
        let mut cfg = RendererConfig::default();
        cfg.atmosphere.sky = SkyModel::Hdri;
        cfg.atmosphere.hdr_path = None;
        cfg.lighting.lights = vec![]; // No environment light
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("sky=hdri"));
    }

    #[test]
    fn atmosphere_hdri_sky_accepts_environment_light() {
        let mut cfg = RendererConfig::default();
        cfg.atmosphere.sky = SkyModel::Hdri;
        cfg.atmosphere.hdr_path = None;
        cfg.lighting.lights = vec![LightConfig {
            light_type: LightType::Environment,
            intensity: 1.0,
            color: [1.0, 1.0, 1.0],
            direction: None,
            position: None,
            cone_angle: None,
            area_extent: None,
            hdr_path: Some("assets/sky.hdr".to_string()),
        }];
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn volumetric_density_must_be_non_negative() {
        let mut cfg = RendererConfig::default();
        cfg.atmosphere.volumetric = Some(VolumetricParams {
            density: -0.1,
            ..VolumetricParams::default()
        });
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("density"));
    }

    #[test]
    fn volumetric_hg_anisotropy_must_be_in_range() {
        let mut cfg = RendererConfig::default();
        cfg.atmosphere.volumetric = Some(VolumetricParams {
            phase: VolumetricPhase::HenyeyGreenstein,
            anisotropy: 1.5,
            ..VolumetricParams::default()
        });
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("anisotropy"));

        cfg.atmosphere.volumetric = Some(VolumetricParams {
            phase: VolumetricPhase::HenyeyGreenstein,
            anisotropy: -1.5,
            ..VolumetricParams::default()
        });
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("anisotropy"));
    }

    #[test]
    fn volumetric_hg_anisotropy_boundary_values() {
        let mut cfg = RendererConfig::default();
        cfg.atmosphere.volumetric = Some(VolumetricParams {
            phase: VolumetricPhase::HenyeyGreenstein,
            anisotropy: 0.999,
            ..VolumetricParams::default()
        });
        assert!(cfg.validate().is_ok());

        cfg.atmosphere.volumetric = Some(VolumetricParams {
            phase: VolumetricPhase::HenyeyGreenstein,
            anisotropy: -0.999,
            ..VolumetricParams::default()
        });
        assert!(cfg.validate().is_ok());
    }

    // GI validation tests
    #[test]
    fn gi_ibl_requires_environment_light_or_atmosphere_hdr() {
        let mut cfg = RendererConfig::default();
        cfg.gi.modes = vec![GiMode::Ibl];
        cfg.lighting.lights = vec![]; // No environment light
        cfg.atmosphere.hdr_path = None;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("ibl"));
    }

    #[test]
    fn gi_ibl_accepts_environment_light() {
        let mut cfg = RendererConfig::default();
        cfg.gi.modes = vec![GiMode::Ibl];
        cfg.lighting.lights = vec![LightConfig {
            light_type: LightType::Environment,
            intensity: 1.0,
            color: [1.0, 1.0, 1.0],
            direction: None,
            position: None,
            cone_angle: None,
            area_extent: None,
            hdr_path: Some("assets/sky.hdr".to_string()),
        }];
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn gi_ibl_accepts_atmosphere_hdr() {
        let mut cfg = RendererConfig::default();
        cfg.gi.modes = vec![GiMode::Ibl];
        cfg.atmosphere.hdr_path = Some("assets/sky.hdr".to_string());
        cfg.lighting.lights = vec![LightConfig::default()];
        assert!(cfg.validate().is_ok());
    }

    // Struct defaults verification
    #[test]
    fn lighting_params_has_correct_defaults() {
        let params = LightingParams::default();
        assert_eq!(params.exposure, 1.0);
        assert_eq!(params.lights.len(), 1);
        assert_eq!(params.lights[0].light_type, LightType::Directional);
    }

    #[test]
    fn shading_params_has_correct_defaults() {
        let params = ShadingParams::default();
        assert_eq!(params.brdf, BrdfModel::CookTorranceGGX);
        assert!(params.normal_maps);
        assert!(!params.clearcoat);
    }

    #[test]
    fn shadow_params_has_correct_defaults() {
        let params = ShadowParams::default();
        assert!(params.enabled);
        assert_eq!(params.technique, ShadowTechnique::Pcf);
        assert_eq!(params.map_size, 2048);
        assert_eq!(params.cascades, 4);
        assert!(params.is_power_of_two_map());
    }

    #[test]
    fn gi_params_has_correct_defaults() {
        let params = GiParams::default();
        assert_eq!(params.modes, vec![GiMode::None]);
        assert_eq!(params.ambient_occlusion_strength, 1.0);
    }

    #[test]
    fn atmosphere_params_has_correct_defaults() {
        let params = AtmosphereParams::default();
        assert!(params.enabled);
        assert_eq!(params.sky, SkyModel::HosekWilkie);
        assert!(params.hdr_path.is_none());
        assert!(params.volumetric.is_none());
    }

    #[test]
    fn renderer_config_has_correct_defaults() {
        let config = RendererConfig::default();
        assert!(config.lighting.lights.len() > 0);
        assert_eq!(config.shading.brdf, BrdfModel::CookTorranceGGX);
        assert!(config.shadows.enabled);
        assert_eq!(config.gi.modes, vec![GiMode::None]);
        assert!(config.atmosphere.enabled);
    }
}

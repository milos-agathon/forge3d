use super::common::normalize_key;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

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
    pub fn canonical(self) -> &'static str {
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

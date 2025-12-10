// src/cli/gi_params.rs
// GI CLI parameter structs
// Extracted from args.rs for maintainability (<300 lines)

use super::gi_types::{GiEntry, GiVizMode};

/// SSAO-related CLI parameters.
#[derive(Clone, Debug, Default)]
pub struct SsaoCliParams {
    pub radius: Option<f32>,
    pub intensity: Option<f32>,
    pub technique: Option<String>,
    pub composite_enabled: Option<bool>,
    pub composite_mul: Option<f32>,
    pub bias: Option<f32>,
    pub samples: Option<u32>,
    pub directions: Option<u32>,
    pub temporal_alpha: Option<f32>,
    pub temporal_enabled: Option<bool>,
    pub blur_enabled: Option<bool>,
}

/// SSGI-related CLI parameters.
#[derive(Clone, Debug, Default)]
pub struct SsgiCliParams {
    pub steps: Option<u32>,
    pub radius: Option<f32>,
    pub half_res: Option<bool>,
    pub temporal_alpha: Option<f32>,
    pub temporal_enabled: Option<bool>,
    pub edges: Option<bool>,
    pub upsample_sigma_depth: Option<f32>,
    pub upsample_sigma_normal: Option<f32>,
}

/// SSR-related CLI parameters.
#[derive(Clone, Debug, Default)]
pub struct SsrCliParams {
    pub enable: Option<bool>,
    pub max_steps: Option<u32>,
    pub thickness: Option<f32>,
}

/// Aggregated GI CLI configuration.
#[derive(Clone, Debug, Default)]
pub struct GiCliConfig {
    pub entries: Vec<GiEntry>,
    pub ssao: SsaoCliParams,
    pub ssgi: SsgiCliParams,
    pub ssr: SsrCliParams,
    pub ao_weight: Option<f32>,
    pub ssgi_weight: Option<f32>,
    pub ssr_weight: Option<f32>,
    pub gi_viz: Option<GiVizMode>,
    pub gi_seed: Option<u32>,
}

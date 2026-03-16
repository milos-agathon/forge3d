use super::*;
use crate::render::params::SsrParams;

mod accessors;
mod core;
mod execute;

/// Screen-space effects manager
pub struct ScreenSpaceEffectsManager {
    gbuffer: GBuffer,
    ssao_renderer: Option<SsaoRenderer>,
    ssgi_renderer: Option<SsgiRenderer>,
    ssr_renderer: Option<SsrRenderer>,
    enabled_effects: Vec<ScreenSpaceEffect>,
    pub hzb: Option<HzbPyramid>,
    ssr_params: SsrParams,
    last_hzb_ms: f32,
}

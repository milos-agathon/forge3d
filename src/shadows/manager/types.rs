use crate::lighting::types::ShadowTechnique;
use super::super::CsmConfig;

pub const DEFAULT_MEMORY_BUDGET_BYTES: u64 = 256 * 1024 * 1024;
pub const MIN_SHADOW_RESOLUTION: u32 = 256;
pub const MAX_SEARCH_TEXELS: f32 = 6.0;

/// High-level configuration used to instantiate the shadow manager.
#[derive(Debug, Clone)]
pub struct ShadowManagerConfig {
    pub csm: CsmConfig,
    pub technique: ShadowTechnique,
    pub pcss_blocker_radius: f32,
    pub pcss_filter_radius: f32,
    pub light_size: f32,
    pub moment_bias: f32,
    pub max_memory_bytes: u64,
}

impl Default for ShadowManagerConfig {
    fn default() -> Self {
        Self {
            csm: CsmConfig::default(),
            technique: ShadowTechnique::PCF,
            pcss_blocker_radius: 0.03,
            pcss_filter_radius: 0.06,
            light_size: 0.25,
            moment_bias: 0.0005,
            max_memory_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
        }
    }
}

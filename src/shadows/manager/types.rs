use super::super::CsmConfig;
use crate::lighting::types::ShadowTechnique;

pub const DEFAULT_MEMORY_BUDGET_BYTES: u64 = 256 * 1024 * 1024;
pub const MIN_SHADOW_RESOLUTION: u32 = 256;
pub const DEFAULT_PCSS_BLOCKER_RADIUS_TEXELS: f32 = 6.0;
pub const DEFAULT_PCSS_FILTER_RADIUS_TEXELS: f32 = 4.0;
pub const DEFAULT_PCSS_LIGHT_SIZE: f32 = 1.0;
pub const MAX_PCSS_BLOCKER_RADIUS_TEXELS: f32 = 50.0;
pub const MAX_PCSS_FILTER_RADIUS_TEXELS: f32 = 100.0;

/// High-level configuration used to instantiate the shadow manager.
#[derive(Debug, Clone)]
pub struct ShadowManagerConfig {
    pub csm: CsmConfig,
    pub technique: ShadowTechnique,
    /// Blocker-search radius in shadow-map texels.
    pub pcss_blocker_radius: f32,
    /// Base PCF radius in shadow-map texels.
    pub pcss_filter_radius: f32,
    /// Dimensionless area-light size used by the PCSS penumbra estimate.
    pub light_size: f32,
    pub moment_bias: f32,
    /// P0.2/M3: Blur kernel radius for VSM/EVSM/MSM moment maps (2-4 typical)
    pub blur_kernel_radius: u32,
    pub max_memory_bytes: u64,
}

impl Default for ShadowManagerConfig {
    fn default() -> Self {
        Self {
            csm: CsmConfig::default(),
            technique: ShadowTechnique::PCF,
            pcss_blocker_radius: DEFAULT_PCSS_BLOCKER_RADIUS_TEXELS,
            pcss_filter_radius: DEFAULT_PCSS_FILTER_RADIUS_TEXELS,
            light_size: DEFAULT_PCSS_LIGHT_SIZE,
            moment_bias: 0.0005,
            blur_kernel_radius: 3, // P0.2/M3: Default blur radius for VSM/EVSM/MSM
            max_memory_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
        }
    }
}

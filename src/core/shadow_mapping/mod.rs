//! Shadow mapping implementation with Cascaded Shadow Maps (CSM)
//!
//! Provides GPU-accelerated shadow mapping with multiple cascades,
//! depth texture arrays, and PCF filtering support.

mod bind_group;
mod system;
mod types;

pub use bind_group::create_shadow_bind_group_layout;
pub use system::ShadowMapping;
pub use types::{
    CsmCascadeData, CsmUniforms, PcfQuality, ShadowAtlasInfo, ShadowMappingConfig, ShadowStats,
};

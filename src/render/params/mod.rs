pub mod atmosphere;
pub mod common;
pub mod config;
pub mod gi;
pub mod lights;
pub mod shading;
pub mod shadows;

// Re-export specific types to match original API
pub use atmosphere::{
    AtmosphereParams, SkyModel, VolumetricMode, VolumetricParams, VolumetricPhase,
};
pub use config::{ConfigError, RendererConfig};
pub use gi::{GiMode, GiParams, SsrParams};
pub use lights::{LightConfig, LightType, LightingParams};
pub use shading::{BrdfModel, ShadingParams};
pub use shadows::{ShadowParams, ShadowTechnique};

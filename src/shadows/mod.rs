// src/shadows/mod.rs
// Shadow mapping implementations for Workstream B
// Exists to centralize GPU/CPU shadow utilities shared across bindings and pipelines
// RELEVANT FILES: shaders/shadows.wgsl, python/forge3d/lighting.py, tests/test_b4_csm.py

mod cascade_math;
mod csm_depth_control;
mod csm_renderer;
mod csm_types;

pub mod blur_pass;
pub mod manager;
pub mod moment_pass;
pub mod state;

// Re-export CSM types from split modules
pub use cascade_math::detect_peter_panning;
pub use csm_renderer::CsmRenderer;
pub use csm_types::{CascadeStatistics, CsmConfig, CsmUniforms, ShadowCascade};

pub use blur_pass::ShadowBlurPass;
pub use manager::{ShadowManager, ShadowManagerConfig};
pub use moment_pass::{create_moment_storage_view, MomentGenerationPass};

// Re-export common shadow types and utilities
pub use csm_renderer::CsmRenderer as CascadedShadowMaps;

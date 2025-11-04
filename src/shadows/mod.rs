// src/shadows/mod.rs
// Shadow mapping implementations for Workstream B
// Exists to centralize GPU/CPU shadow utilities shared across bindings and pipelines
// RELEVANT FILES: shaders/shadows.wgsl, python/forge3d/lighting.py, tests/test_b4_csm.py

pub mod csm;
pub mod state;
pub mod manager;
pub mod moment_pass;

pub use csm::{
    detect_peter_panning, CascadeStatistics, CsmConfig, CsmRenderer, CsmUniforms, ShadowCascade,
};
pub use manager::{ShadowManager, ShadowManagerConfig};
pub use moment_pass::{MomentGenerationPass, create_moment_storage_view};

// Re-export common shadow types and utilities
pub use csm::CsmRenderer as CascadedShadowMaps;

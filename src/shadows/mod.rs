// src/shadows/mod.rs
// Shadow mapping implementations for Workstream B
// Exists to centralize GPU/CPU shadow utilities shared across bindings and pipelines
// RELEVANT FILES: shaders/csm.wgsl, python/forge3d/lighting.py, tests/test_b4_csm.py

pub mod csm;
pub mod state;

pub use csm::{detect_peter_panning, CascadeStatistics, CsmConfig, CsmRenderer, CsmUniforms, ShadowCascade};

// Re-export common shadow types and utilities
pub use csm::CsmRenderer as CascadedShadowMaps;

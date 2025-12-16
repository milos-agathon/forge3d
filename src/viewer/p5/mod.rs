// src/viewer/p5/mod.rs
// P5 (Phase 5) screen-space effects module
// Split from viewer_p5*.rs files as part of the viewer refactoring

pub mod ao;
pub mod cornell;
mod gbuffer_dump;
pub mod gi_ablation;
pub mod gi_verification;
pub mod ssgi_cornell;
pub mod ssgi_temporal;
pub mod ssr_glossy;
pub mod ssr_helpers;
mod ssr_scene_impl;
pub mod ssr_thickness;

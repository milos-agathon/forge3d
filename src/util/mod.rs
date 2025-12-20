// src/util/mod.rs
// Utility module namespace for helper functions shared across components
// Exists to group reusable helpers like image encoding utilities
// RELEVANT FILES: src/util/image_write.rs, src/lib.rs, src/renderer/readback.rs, src/gpu.rs

pub mod debug_pattern;
#[cfg(feature = "images")]
pub mod exr_write;
pub mod image_write;
pub mod memory_budget;

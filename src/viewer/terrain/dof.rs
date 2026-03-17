// src/viewer/terrain/dof.rs
// Depth of Field post-process pass with separable blur

mod pass;
mod shader;
mod types;

pub use pass::DofPass;
pub use types::{DofConfig, DofUniforms};

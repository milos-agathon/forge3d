// src/core/text_mesh.rs
// 3D Text mesh renderer and mesh builder (extruded outlines)

mod builder;
mod renderer;
mod types;

pub use builder::build_text_mesh;
pub use renderer::TextMeshRenderer;
pub use types::{MeshUniforms, VertexPN};

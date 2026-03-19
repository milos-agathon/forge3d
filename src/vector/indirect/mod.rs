//! Indirect drawing and GPU culling.

mod culling;
mod draw;
mod init;
mod renderer;
#[cfg(test)]
mod tests;
mod types;

pub use renderer::IndirectRenderer;
pub use types::{
    create_cullable_instance, CullableInstance, CullingStats, IndirectDrawCommand,
    IndirectDrawIndexedCommand,
};

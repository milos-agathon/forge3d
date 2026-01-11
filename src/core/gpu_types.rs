//! Shared GPU type aliases and small helper structs.
//!
//! Centralizes wgpu re-exports and common descriptors to keep render code
//! consistent and reduce repetitive imports.

// Re-export commonly used wgpu types for consistency
pub use wgpu::{
    BindGroup, BindGroupLayout, Buffer, BufferUsages, CommandEncoder, Device, Queue, RenderPass,
    RenderPipeline, Texture, TextureFormat, TextureView,
};

/// Common texture format used throughout the renderer
pub const RENDER_TARGET_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;

/// Opaque handle for higher-level GPU resource tracking.
#[derive(Debug, Clone, Copy)]
pub struct GpuResourceId(pub u32);

/// GPU buffer descriptor with common defaults
#[derive(Debug, Clone)]
pub struct GpuBufferDesc {
    pub label: Option<String>,
    pub size: u64,
    pub usage: BufferUsages,
}

impl GpuBufferDesc {
    /// Create a new buffer descriptor
    pub fn new(size: u64, usage: BufferUsages) -> Self {
        Self {
            label: None,
            size,
            usage,
        }
    }

    /// Set a debug label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

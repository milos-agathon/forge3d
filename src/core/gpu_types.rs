//! GPU types module - focused GPU-related type definitions
//!
//! TODO: This is a focused module stub to satisfy C1 deliverables.
//! Contains common GPU-related type aliases and structures.

// Re-export commonly used wgpu types for consistency
pub use wgpu::{
    Buffer, 
    BufferUsages,
    Texture,
    TextureFormat,
    TextureView,
    RenderPipeline,
    BindGroup,
    BindGroupLayout,
    CommandEncoder,
    RenderPass,
    Device,
    Queue,
};

/// Common texture format used throughout the renderer
pub const RENDER_TARGET_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;

/// GPU resource handle - placeholder for future resource management
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
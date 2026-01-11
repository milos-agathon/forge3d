use crate::lighting::types::Light;
use wgpu::{BindGroup, BindGroupLayout, Buffer};

/// Maximum number of lights supported (P1 default)
pub const MAX_LIGHTS: usize = 16;

/// Light buffer manager with triple-buffering for TAA-friendly updates
///
/// # P1-02: Triple-buffered SSBO Manager
///
/// Implements triple-buffered GPU light storage with R2 sequence seeds for
/// TAA-friendly temporal light sampling. Supports up to MAX_LIGHTS=16 lights
/// with minimal memory overhead.
pub struct LightBuffer {
    /// Storage buffers for light array (triple-buffered)
    pub(crate) buffers: [Buffer; 3],
    /// Uniform buffer for light count (triple-buffered)
    pub(crate) count_buffers: [Buffer; 3],
    /// Uniform buffer reserved for environment lighting parameters (P1-05).
    pub(crate) environment_stub: Buffer,
    /// Current frame index (0, 1, 2)
    pub(crate) frame_index: usize,
    /// Monotonic frame counter used for quasi-random sampling offsets
    pub(crate) frame_counter: u64,
    /// Cached 2D R2 sequence seed for the current frame
    pub(crate) sequence_seed: [f32; 2],
    /// Current number of active lights
    pub(crate) light_count: u32,
    /// Bind group for current frame
    pub(crate) bind_group: Option<BindGroup>,
    /// Bind group layout
    pub(crate) bind_group_layout: BindGroupLayout,
    /// P1-07: Last uploaded lights for debug inspection (CPU-side only)
    pub(crate) last_uploaded_lights: Vec<Light>,
}

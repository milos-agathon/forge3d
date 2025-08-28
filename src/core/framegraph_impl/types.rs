//! Framegraph resource and pass type definitions
//!
//! This module defines the core types for the framegraph system including
//! resources, passes, and their relationships.

use std::collections::HashMap;
use wgpu::{TextureFormat, TextureUsages, Extent3d};

/// Unique identifier for a resource in the framegraph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResourceHandle(pub(crate) usize);

/// Unique identifier for a pass in the framegraph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PassHandle(pub(crate) usize);

/// Resource type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceType {
    /// Color attachment (render target)
    ColorAttachment,
    /// Depth/stencil attachment  
    DepthStencilAttachment,
    /// Storage buffer
    StorageBuffer,
    /// Uniform buffer
    UniformBuffer,
    /// Texture for sampling
    SampledTexture,
}

/// Resource description for creating GPU resources
#[derive(Debug, Clone)]
pub struct ResourceDesc {
    /// Human-readable name for debugging
    pub name: String,
    /// Type of resource
    pub resource_type: ResourceType,
    /// Texture format (for texture resources)
    pub format: Option<TextureFormat>,
    /// Texture dimensions (for texture resources)
    pub extent: Option<Extent3d>,
    /// Buffer size in bytes (for buffer resources)
    pub size: Option<u64>,
    /// Usage flags
    pub usage: Option<TextureUsages>,
    /// Whether this resource can be aliased with others
    pub can_alias: bool,
}

/// Resource state information
#[derive(Debug, Clone)]
pub struct ResourceInfo {
    /// Resource description
    pub desc: ResourceDesc,
    /// First pass that uses this resource
    pub first_use: Option<PassHandle>,
    /// Last pass that uses this resource
    pub last_use: Option<PassHandle>,
    /// Whether this resource is transient (created and destroyed within frame)
    pub is_transient: bool,
    /// Handle to the aliased resource (if any)
    pub aliased_with: Option<ResourceHandle>,
}

/// Pass type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PassType {
    /// Graphics rendering pass
    Graphics,
    /// Compute pass
    Compute,
    /// Copy/transfer pass
    Transfer,
}

/// Pass description
#[derive(Debug, Clone)]
pub struct PassDesc {
    /// Human-readable name
    pub name: String,
    /// Type of pass
    pub pass_type: PassType,
    /// Resources read by this pass
    pub reads: Vec<ResourceHandle>,
    /// Resources written by this pass
    pub writes: Vec<ResourceHandle>,
    /// Whether this pass can run in parallel with others
    pub can_parallelize: bool,
}

/// Pass information with computed dependencies
#[derive(Debug, Clone)]
pub struct PassInfo {
    /// Pass description
    pub desc: PassDesc,
    /// Passes that must complete before this one
    pub dependencies: Vec<PassHandle>,
    /// Passes that depend on this one
    pub dependents: Vec<PassHandle>,
}

/// Barrier type for resource transitions
#[derive(Debug, Clone, PartialEq)]
pub enum BarrierType {
    /// Texture layout transition
    TextureBarrier {
        /// Previous usage
        old_usage: TextureUsages,
        /// New usage
        new_usage: TextureUsages,
    },
    /// Buffer usage transition
    BufferBarrier {
        /// Previous usage
        old_usage: wgpu::BufferUsages,
        /// New usage
        new_usage: wgpu::BufferUsages,
    },
    /// Memory barrier (flush/invalidate)
    MemoryBarrier,
}

/// Resource barrier information
#[derive(Debug, Clone)]
pub struct ResourceBarrier {
    /// Resource being transitioned
    pub resource: ResourceHandle,
    /// Type of barrier
    pub barrier_type: BarrierType,
    /// Pass that requires this barrier before execution
    pub before_pass: PassHandle,
}

/// Framegraph execution metrics
#[derive(Debug, Default)]
pub struct FrameGraphMetrics {
    /// Number of passes in the graph
    pub pass_count: usize,
    /// Number of resources
    pub resource_count: usize,
    /// Number of transient resources
    pub transient_count: usize,
    /// Number of successfully aliased resources
    pub aliased_count: usize,
    /// Number of barriers inserted
    pub barrier_count: usize,
    /// Memory saved by aliasing (bytes)
    pub memory_saved_bytes: u64,
}
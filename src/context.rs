//! GPU context management - thin re-export layer
//!
//! This module provides a consistent interface for GPU context operations
//! by re-exporting functionality from the gpu module.

// Re-export GPU context functionality  
pub use crate::gpu::{ctx, align_copy_bpr};

// Re-export for compatibility
pub use wgpu::{Device, Queue, Adapter, Instance};
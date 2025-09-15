//! Framegraph legacy compatibility layer
//!
//! This module provides backward compatibility for the old framegraph API
//! while redirecting to the new full implementation.

use crate::error::RenderResult;

// Re-export main types from the new implementation
pub use super::framegraph_impl::{
    FrameGraph as NewFrameGraph, PassType, ResourceDesc, ResourceType,
};

/// Legacy FrameGraph wrapper for backward compatibility
#[derive(Debug)]
pub struct FrameGraph {
    inner: NewFrameGraph,
}

impl FrameGraph {
    /// Create a new framegraph
    pub fn new() -> Self {
        Self {
            inner: NewFrameGraph::new(),
        }
    }

    /// Add a render pass (legacy compatibility)
    pub fn add_pass(&mut self, name: impl Into<String>) -> RenderResult<()> {
        let _handle = self
            .inner
            .add_pass(&name.into(), PassType::Graphics, |_builder| Ok(()))?;
        Ok(())
    }

    /// Get access to the full framegraph implementation
    pub fn full(&mut self) -> &mut NewFrameGraph {
        &mut self.inner
    }
}

impl Default for FrameGraph {
    fn default() -> Self {
        Self::new()
    }
}

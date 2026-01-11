//! Framegraph legacy compatibility layer
//!
//! This module provides backward compatibility for the old framegraph API
//! while redirecting to the new full implementation.

use super::error::RenderResult;

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

    /// P5.0: Add optional GI passes branch after lighting and before tonemap.
    /// Currently a no-op; effects are orchestrated by the viewer/manager when enabled.
    /// Keeping this hook allows future migration to the full framegraph without breaking legacy code.
    pub fn add_gi_passes(&mut self) -> RenderResult<()> {
        // Intentionally no-op to maintain bit-identical baseline when GI is disabled.
        Ok(())
    }
}

impl Default for FrameGraph {
    fn default() -> Self {
        Self::new()
    }
}

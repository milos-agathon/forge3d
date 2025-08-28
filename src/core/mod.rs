//! Core engine modules
//!
//! Contains foundational types and systems for the renderer.

pub mod framegraph;  // Legacy compatibility layer
pub mod gpu_types;
pub mod memory_tracker;

// New framegraph implementation
pub mod framegraph_impl;
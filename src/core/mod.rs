//! Core engine modules
//!
//! Contains foundational types and systems for the renderer.

pub mod framegraph;  // Legacy compatibility layer
pub mod gpu_types;
pub mod memory_tracker;

// New framegraph implementation
pub mod framegraph_impl;

// C8: Tonemap post-processing
pub mod tonemap;

// C9: Matrix stack utility
pub mod matrix_stack;

// C10: Hierarchical scene graph
pub mod scene_graph;

// C6: Multi-thread command recording
pub mod multi_thread;

// C7: Async compute prepasses
pub mod async_compute;
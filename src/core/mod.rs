//! Core engine modules
//!
//! Contains foundational types and systems for the renderer.

pub mod framegraph;  // Legacy compatibility layer
pub mod gpu_types;
pub mod memory_tracker;
pub mod resource_tracker;

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

// R9: Async and double-buffered readback system (opt-in)
#[cfg(feature = "async_readback")]
pub mod async_readback;

// I7: Big buffer pattern for per-object data
#[cfg(feature = "wsI_bigbuf")]
pub mod big_buffer;

// I8: Double-buffering for per-frame data
#[cfg(any(feature = "wsI_bigbuf", feature = "wsI_double_buf"))]
pub mod double_buffer;

// L4: Mipmap generation utilities
pub mod mipmap;

// L5: Sampler mode matrix and policy utilities
pub mod sampler_modes;

// L6: Texture upload helpers for HDR formats
pub mod texture_upload;

// N5: Environment mapping and IBL
pub mod envmap;

// N8: HDR rendering and tone mapping
pub mod hdr;

// N1: PBR materials
pub mod material;
pub mod pbr;

// N2: Shadow mapping
pub mod cascade_split;
pub mod shadow_mapping;
pub mod shadows;

// N4: Render bundles
pub mod render_bundles;

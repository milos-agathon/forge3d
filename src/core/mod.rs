//! Core engine modules
//!
//! Contains foundational types and systems for the renderer.

pub mod framegraph;  // Legacy compatibility layer
pub mod gpu_types;
pub mod memory_tracker;
pub mod resource_tracker;

// Q3: GPU profiling and timing
pub mod gpu_timing;

// Q1: Post-processing compute pipeline
pub mod postfx;

// Q5: Bloom post-processing effect
pub mod bloom;

// Workstream O: Resource & Memory Management
pub mod staging_rings;        // O1: Staging buffer rings with fence synchronization
pub mod fence_tracker;        // O1: Fence tracking for staging buffers
pub mod compressed_textures;  // O3: Compressed texture pipeline
pub mod texture_format;       // O3: Texture format registry and detection
pub mod virtual_texture;      // O4: Virtual texture streaming system
pub mod feedback_buffer;      // O4: GPU feedback buffer for tile visibility
pub mod tile_cache;          // O4: LRU tile cache for virtual textures

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

// O1: Staging buffer rings with fence synchronization
#[cfg(feature = "enable-staging-rings")]
pub mod staging_rings;
#[cfg(feature = "enable-staging-rings")]
pub mod fence_tracker;

// O2: Memory pools are integrated into memory_tracker
// Available when enable-memory-pools feature is enabled

// O3: Compressed texture pipeline (already declared above)

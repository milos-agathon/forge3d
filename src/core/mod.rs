//! Core engine modules
//!
//! Contains foundational types and systems for the renderer.

// Foundational modules
pub mod context;
pub mod device_caps;
pub mod error;
pub mod gpu;
#[cfg(feature = "extension-module")]
pub mod session;

pub mod framegraph; // Legacy compatibility layer
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
// staging_rings and fence_tracker are declared below with feature flags
pub mod compressed_textures; // O3: Compressed texture pipeline
pub mod feedback_buffer; // O4: GPU feedback buffer for tile visibility
pub mod texture_format; // O3: Texture format registry and detection
pub mod texture_format_defs; // O3: Texture format definitions
pub mod tile_cache;
pub mod virtual_texture; // O4: Virtual texture streaming system // O4: LRU tile cache for virtual textures

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

// B5: Planar reflections
pub mod reflections;

// B6: Depth of Field
pub mod dof;

// B7: Cloud Shadows
pub mod cloud_shadows;

// B8: Realtime Clouds
pub mod clouds;

// B10: Ground Plane (Raster)
pub mod ground_plane;

// B11: Water Surface Color Toggle
pub mod water_surface;

// B12: Soft Light Radius (Raster)
pub mod soft_light_radius;

// B13: Point & Spot Lights (Realtime)
pub mod point_spot_lights;

// B14: Rect Area Lights (LTC)
pub mod ltc_area_lights;

// B15: Image-Based Lighting (IBL) Polish
pub mod ibl;

// B16: Dual-source blending OIT
pub mod dual_source_oit;

// D: Overlays compositor (native)
pub mod overlays;

// D: Text overlay (native)
pub mod text_overlay;

// Overlay layer for terrain rendering
#[cfg(feature = "extension-module")]
pub mod overlay_layer;

// D11: 3D Text Mesh (native)
pub mod text_mesh;

// N4: Render bundles
pub mod render_bundles;
pub mod render_bundles_types;

// O1: Staging buffer rings with fence synchronization
#[cfg(feature = "enable-staging-rings")]
pub mod fence_tracker;
#[cfg(feature = "enable-staging-rings")]
pub mod staging_rings;

// O2: Memory pools are integrated into memory_tracker
// Available when enable-memory-pools feature is enabled

// O3: Compressed texture pipeline (already declared above)

// P5: Screen-space effects (SSAO/GTAO, SSGI, SSR)
pub mod gbuffer;
pub mod screen_space_effects;

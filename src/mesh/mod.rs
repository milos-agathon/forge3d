//! Mesh generation and processing utilities
//!
//! Provides utilities for generating and processing 3D mesh data including
//! TBN (Tangent, Bitangent, Normal) generation for normal mapping.

#[cfg(feature = "enable-tbn")]
pub mod tbn;

#[cfg(feature = "enable-tbn")]
pub mod vertex;

#[cfg(feature = "enable-tbn")]
pub use tbn::{TbnVertex as TbnMeshVertex, TbnData, generate_tbn, generate_plane_tbn, generate_cube_tbn};

#[cfg(feature = "enable-tbn")]
pub use vertex::{TbnVertex, CompactTbnVertex, create_tbn_vertices_from_mesh, create_compact_tbn_vertices_from_mesh};
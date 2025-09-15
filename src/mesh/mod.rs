//! Mesh generation and processing utilities
//!
//! Provides utilities for generating and processing 3D mesh data including
//! TBN (Tangent, Bitangent, Normal) generation for normal mapping.

#[cfg(feature = "enable-tbn")]
pub mod tbn;

#[cfg(feature = "enable-tbn")]
pub mod vertex;

#[cfg(feature = "enable-tbn")]
pub use tbn::{
    generate_cube_tbn, generate_plane_tbn, generate_tbn, TbnData, TbnVertex as TbnMeshVertex,
};

#[cfg(feature = "enable-tbn")]
pub use vertex::{
    create_compact_tbn_vertices_from_mesh, create_tbn_vertices_from_mesh, CompactTbnVertex,
    TbnVertex,
};

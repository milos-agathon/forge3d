//! Core geometry utilities for mesh generation, validation, and processing.
//!
//! Provides shared data structures ([`MeshBuffers`]) and operations for:
//! - Primitive generation (planes, spheres, cylinders, etc.)
//! - Polygon extrusion
//! - Mesh validation and welding
//! - Coordinate transforms

mod curves;
mod displacement;
mod extrude;
pub mod grid;
mod primitives;
mod subdivision;
mod tangents;
mod thick_polyline;
mod transform;
pub mod transforms;
mod validate;
mod weld;

// Python bindings modules (extension-module only)
#[cfg(feature = "extension-module")]
mod array_convert;
#[cfg(feature = "extension-module")]
mod mesh_python;
#[cfg(feature = "extension-module")]
mod py_advanced;
#[cfg(feature = "extension-module")]
mod py_bindings;

pub use extrude::{extrude_polygon, extrude_polygon_with_options, ExtrudeOptions};
pub use primitives::{
    generate_cone, generate_cylinder, generate_plane, generate_primitive, generate_sphere,
    generate_text3d_stub, generate_torus, generate_unit_box, PrimitiveParams, PrimitiveType,
};
pub use subdivision::subdivide_triangles;
#[cfg(feature = "extension-module")]
pub use thick_polyline::geometry_generate_thick_polyline_py;
pub use transform::{center_to_target, compute_bounds, flip_axis, scale_about_pivot, swap_axes};
pub use validate::{validate_mesh, MeshStats, MeshValidationIssue, MeshValidationReport};
pub use weld::{weld_mesh, WeldOptions, WeldResult};

// Re-export Python bindings
#[cfg(feature = "extension-module")]
pub use mesh_python::{map_geometry_err, mesh_from_python, mesh_to_python};
#[cfg(feature = "extension-module")]
pub use py_advanced::{
    geometry_attach_tangents_py, geometry_displace_heightmap_py, geometry_displace_procedural_py,
    geometry_generate_ribbon_py, geometry_generate_tangents_py, geometry_generate_tube_py,
    geometry_subdivide_adaptive_py, geometry_subdivide_py,
};
#[cfg(feature = "extension-module")]
pub use py_bindings::{
    geometry_extrude_polygon_py, geometry_generate_primitive_py, geometry_transform_bounds_py,
    geometry_transform_center_py, geometry_transform_flip_axis_py, geometry_transform_scale_py,
    geometry_transform_swap_axes_py, geometry_validate_mesh_py, geometry_weld_mesh_py,
};

/// Shared mesh container used by the geometry module family.
#[derive(Debug, Clone, Default)]
pub struct MeshBuffers {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

impl MeshBuffers {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(vertex_capacity: usize, index_capacity: usize) -> Self {
        Self {
            positions: Vec::with_capacity(vertex_capacity),
            normals: Vec::with_capacity(vertex_capacity),
            uvs: Vec::with_capacity(vertex_capacity),
            tangents: Vec::with_capacity(vertex_capacity),
            indices: Vec::with_capacity(index_capacity),
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty() || self.indices.is_empty()
    }
}

/// Error type returned by geometry helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeometryError {
    message: String,
}

impl GeometryError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for GeometryError {}

/// Convenience alias for geometry results.
pub type GeometryResult<T> = Result<T, GeometryError>;

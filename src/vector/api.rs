// src/vector/api.rs
// Python-facing vector API, validation, and GPU bridges
// Exists to expose safe typed bindings and orchestrate mesh workflows
// RELEVANT FILES: src/vector/extrusion.rs, src/vector/gpu_extrusion.rs, python/forge3d/__init__.py, docs/api/polygon_extrusion.md
//! H1: Public API definition (vectors)
//! Freeze Python surface for vectors/graphs with CRS validation.

mod core;
mod extrusion;
mod py;

pub use core::{
    CrsType, GraphDef, PointDef, PolygonDef, PolylineDef, VectorApi, VectorId, VectorStyle,
};
pub use extrusion::{extrude_polygon_gpu_py, extrude_polygon_py};
pub use py::{
    add_graph_py, add_lines_py, add_points_py, add_polygons_py, clear_vectors_py,
    get_vector_counts_py, parse_polygon_from_numpy,
};

#[cfg(test)]
mod tests;

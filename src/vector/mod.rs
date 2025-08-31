//! Workstream H: Vector & Graph Layers
//! Complete vector graphics rendering pipeline with GPU acceleration

pub mod api;
pub mod data;
pub mod layer;
pub mod polygon;
pub mod line;
pub mod point;
pub mod graph;
pub mod batch;
pub mod oit;
pub mod indirect;

// Re-export main types for convenience
pub use api::{
    VectorApi, VectorId, CrsType, VectorStyle,
    PolygonDef, PolylineDef, PointDef, GraphDef,
};
pub use data::{
    PolygonVertex, LineVertex, PointInstance, GraphNode, GraphEdge,
    PackedPolygon, PackedPolyline, ValidationResult,
    pack_lines, validate_polygon_vertices, validate_point_instances,
};
pub use layer::{Layer, LayeredDrawCmd, sort_draw_commands};
pub use polygon::PolygonRenderer;
pub use line::{LineRenderer, LineInstance, calculate_line_joins};
pub use point::{PointRenderer, PointShape, DebugFlags, TextureAtlas, cluster_points};
pub use graph::{GraphRenderer, PackedGraph, layout_force_directed, calculate_graph_bounds};
pub use batch::{BatchManager, Batch, AABB, Frustum, BatchingStats, PrimitiveType};
pub use oit::{WeightedOIT, is_weighted_oit_enabled};
pub use indirect::{IndirectRenderer, IndirectDrawCommand, CullableInstance, CullingStats, create_cullable_instance};
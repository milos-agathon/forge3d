//! Workstream H: Vector & Graph Layers
//! Complete vector graphics rendering pipeline with GPU acceleration

pub mod api;
pub mod batch;
pub mod data;
pub mod graph;
pub mod indirect;
pub mod layer;
pub mod line;
pub mod oit;
pub mod point;
pub mod polygon;

// Re-export main types for convenience
pub use api::{
    CrsType, GraphDef, PointDef, PolygonDef, PolylineDef, VectorApi, VectorId, VectorStyle,
};
pub use batch::{Batch, BatchManager, BatchingStats, Frustum, PrimitiveType, AABB};
pub use data::{
    pack_lines, validate_point_instances, validate_polygon_vertices, GraphEdge, GraphNode,
    LineVertex, PackedPolygon, PackedPolyline, PointInstance, PolygonVertex, ValidationResult,
};
pub use graph::{calculate_graph_bounds, layout_force_directed, GraphRenderer, PackedGraph};
pub use indirect::{
    create_cullable_instance, CullableInstance, CullingStats, IndirectDrawCommand, IndirectRenderer,
};
pub use layer::{sort_draw_commands, Layer, LayeredDrawCmd};
pub use line::{calculate_line_joins, LineInstance, LineRenderer};
pub use oit::{is_weighted_oit_enabled, WeightedOIT};
pub use point::{cluster_points, DebugFlags, PointRenderer, PointShape, TextureAtlas};
pub use polygon::PolygonRenderer;

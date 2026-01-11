use crate::core::error::RenderError;
use glam::Vec2;

/// Vector primitive ID returned from API calls
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorId(pub u32);

/// Supported CRS (Coordinate Reference Systems)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrsType {
    /// Planar coordinates (required - no geographic projections yet)
    Planar,
    /// Web Mercator (EPSG:3857) - common for web maps
    WebMercator,
}

/// Vector feature styles
#[derive(Debug, Clone)]
pub struct VectorStyle {
    pub fill_color: [f32; 4],   // RGBA fill color
    pub stroke_color: [f32; 4], // RGBA stroke color
    pub stroke_width: f32,      // Stroke width in world units
    pub point_size: f32,        // Point size in pixels
}

impl Default for VectorStyle {
    fn default() -> Self {
        Self {
            fill_color: [0.2, 0.4, 0.8, 1.0],   // Blue fill
            stroke_color: [0.0, 0.0, 0.0, 1.0], // Black stroke
            stroke_width: 1.0,
            point_size: 4.0,
        }
    }
}

/// Polygon definition with optional holes
#[derive(Debug, Clone)]
pub struct PolygonDef {
    pub exterior: Vec<Vec2>,   // Exterior ring (CCW)
    pub holes: Vec<Vec<Vec2>>, // Interior rings (CW)
    pub style: VectorStyle,
}

/// Polyline definition
#[derive(Debug, Clone)]
pub struct PolylineDef {
    pub path: Vec<Vec2>, // Line path coordinates
    pub style: VectorStyle,
}

/// Point definition
#[derive(Debug, Clone)]
pub struct PointDef {
    pub position: Vec2, // Point position
    pub style: VectorStyle,
}

/// Graph definition with nodes and edges
#[derive(Debug, Clone)]
pub struct GraphDef {
    pub nodes: Vec<Vec2>,       // Node positions
    pub edges: Vec<(u32, u32)>, // Edge pairs (from_node, to_node)
    pub node_style: VectorStyle,
    pub edge_style: VectorStyle,
}

/// Vector API implementation
pub struct VectorApi {
    next_id: u32,
    polygons: Vec<(VectorId, PolygonDef)>,
    polylines: Vec<(VectorId, PolylineDef)>,
    points: Vec<(VectorId, PointDef)>,
    graphs: Vec<(VectorId, GraphDef)>,
}

impl VectorApi {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            polygons: Vec::new(),
            polylines: Vec::new(),
            points: Vec::new(),
            graphs: Vec::new(),
        }
    }

    fn next_id(&mut self) -> VectorId {
        let id = VectorId(self.next_id);
        self.next_id += 1;
        id
    }

    /// H1: Add polygons with CRS validation
    pub fn add_polygons(
        &mut self,
        polygons: Vec<PolygonDef>,
        crs: CrsType,
    ) -> Result<Vec<VectorId>, RenderError> {
        // Validate CRS (planar-only).
        if crs != CrsType::Planar {
            return Err(RenderError::Upload(
                "Only planar CRS supported; geographic projections not implemented".to_string(),
            ));
        }

        let mut ids = Vec::new();

        for polygon in polygons {
            // Validate polygon geometry
            self.validate_polygon(&polygon)?;

            let id = self.next_id();
            self.polygons.push((id, polygon));
            ids.push(id);
        }

        Ok(ids)
    }

    /// H1: Add polylines with validation
    pub fn add_lines(
        &mut self,
        lines: Vec<PolylineDef>,
        crs: CrsType,
    ) -> Result<Vec<VectorId>, RenderError> {
        if crs != CrsType::Planar {
            return Err(RenderError::Upload("Only planar CRS supported".to_string()));
        }

        let mut ids = Vec::new();

        for line in lines {
            // Validate line geometry
            self.validate_polyline(&line)?;

            let id = self.next_id();
            self.polylines.push((id, line));
            ids.push(id);
        }

        Ok(ids)
    }

    /// H1: Add points
    pub fn add_points(
        &mut self,
        points: Vec<PointDef>,
        crs: CrsType,
    ) -> Result<Vec<VectorId>, RenderError> {
        if crs != CrsType::Planar {
            return Err(RenderError::Upload("Only planar CRS supported".to_string()));
        }

        let mut ids = Vec::new();

        for point in points {
            // Validate point geometry
            self.validate_point(&point)?;

            let id = self.next_id();
            self.points.push((id, point));
            ids.push(id);
        }

        Ok(ids)
    }

    /// H1: Add graph (nodes + edges)  
    pub fn add_graph(&mut self, graph: GraphDef, crs: CrsType) -> Result<VectorId, RenderError> {
        if crs != CrsType::Planar {
            return Err(RenderError::Upload("Only planar CRS supported".to_string()));
        }

        // Validate graph structure
        self.validate_graph(&graph)?;

        let id = self.next_id();
        self.graphs.push((id, graph));
        Ok(id)
    }

    /// Get current primitive counts for metrics
    pub fn get_counts(&self) -> (usize, usize, usize, usize) {
        (
            self.polygons.len(),
            self.polylines.len(),
            self.points.len(),
            self.graphs.len(),
        )
    }

    /// Clear all primitives
    pub fn clear(&mut self) {
        self.polygons.clear();
        self.polylines.clear();
        self.points.clear();
        self.graphs.clear();
    }

    // Validation helpers

    fn validate_polygon(&self, polygon: &PolygonDef) -> Result<(), RenderError> {
        if polygon.exterior.len() < 3 {
            return Err(RenderError::Upload(
                "Polygon exterior must have at least 3 vertices".to_string(),
            ));
        }

        // Check for finite coordinates
        for (i, vertex) in polygon.exterior.iter().enumerate() {
            if !vertex.x.is_finite() || !vertex.y.is_finite() {
                return Err(RenderError::Upload(format!(
                    "Polygon exterior vertex {} has non-finite coordinates: ({}, {})",
                    i, vertex.x, vertex.y
                )));
            }
        }

        // Validate holes
        for (hole_idx, hole) in polygon.holes.iter().enumerate() {
            if hole.len() < 3 {
                return Err(RenderError::Upload(format!(
                    "Polygon hole {} must have at least 3 vertices",
                    hole_idx
                )));
            }

            for (i, vertex) in hole.iter().enumerate() {
                if !vertex.x.is_finite() || !vertex.y.is_finite() {
                    return Err(RenderError::Upload(format!(
                        "Polygon hole {} vertex {} has non-finite coordinates: ({}, {})",
                        hole_idx, i, vertex.x, vertex.y
                    )));
                }
            }
        }

        Ok(())
    }

    fn validate_polyline(&self, line: &PolylineDef) -> Result<(), RenderError> {
        if line.path.len() < 2 {
            return Err(RenderError::Upload(
                "Polyline must have at least 2 vertices".to_string(),
            ));
        }

        for (i, vertex) in line.path.iter().enumerate() {
            if !vertex.x.is_finite() || !vertex.y.is_finite() {
                return Err(RenderError::Upload(format!(
                    "Polyline vertex {} has non-finite coordinates: ({}, {})",
                    i, vertex.x, vertex.y
                )));
            }
        }

        Ok(())
    }

    fn validate_point(&self, point: &PointDef) -> Result<(), RenderError> {
        if !point.position.x.is_finite() || !point.position.y.is_finite() {
            return Err(RenderError::Upload(format!(
                "Point has non-finite coordinates: ({}, {})",
                point.position.x, point.position.y
            )));
        }

        if point.style.point_size <= 0.0 || !point.style.point_size.is_finite() {
            return Err(RenderError::Upload(format!(
                "Point size must be positive and finite, got {}",
                point.style.point_size
            )));
        }

        Ok(())
    }

    fn validate_graph(&self, graph: &GraphDef) -> Result<(), RenderError> {
        if graph.nodes.is_empty() {
            return Err(RenderError::Upload(
                "Graph must have at least one node".to_string(),
            ));
        }

        // Validate node positions
        for (i, node) in graph.nodes.iter().enumerate() {
            if !node.x.is_finite() || !node.y.is_finite() {
                return Err(RenderError::Upload(format!(
                    "Graph node {} has non-finite coordinates: ({}, {})",
                    i, node.x, node.y
                )));
            }
        }

        // Validate edge indices
        let node_count = graph.nodes.len() as u32;
        for (i, &(from, to)) in graph.edges.iter().enumerate() {
            if from >= node_count {
                return Err(RenderError::Upload(format!(
                    "Edge {} from_node {} exceeds node count {}",
                    i, from, node_count
                )));
            }
            if to >= node_count {
                return Err(RenderError::Upload(format!(
                    "Edge {} to_node {} exceeds node count {}",
                    i, to, node_count
                )));
            }
        }

        Ok(())
    }
}

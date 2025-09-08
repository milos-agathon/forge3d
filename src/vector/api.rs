//! H1: Public API definition (vectors)  
//! Freeze Python surface for vectors/graphs with CRS validation
//!
//! NOTE: Numpy inputs must be 2D arrays shaped (N, 2). Parameters are accepted as
//! `PyReadonlyArray2<'py, T>` (owned pyo3 handles). Do not use `&PyReadonlyArray2<T>`
//! in #[pyfunction] signatures—pyo3 cannot extract references from Python call sites.

use crate::error::RenderError;
use glam::Vec2;
use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};

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
    pub fill_color: [f32; 4],     // RGBA fill color
    pub stroke_color: [f32; 4],   // RGBA stroke color  
    pub stroke_width: f32,        // Stroke width in world units
    pub point_size: f32,          // Point size in pixels
}

impl Default for VectorStyle {
    fn default() -> Self {
        Self {
            fill_color: [0.2, 0.4, 0.8, 1.0],    // Blue fill
            stroke_color: [0.0, 0.0, 0.0, 1.0],  // Black stroke
            stroke_width: 1.0,
            point_size: 4.0,
        }
    }
}

/// Polygon definition with optional holes
#[derive(Debug, Clone)]
pub struct PolygonDef {
    pub exterior: Vec<Vec2>,      // Exterior ring (CCW)
    pub holes: Vec<Vec<Vec2>>,    // Interior rings (CW)
    pub style: VectorStyle,
}

/// Polyline definition
#[derive(Debug, Clone)]
pub struct PolylineDef {
    pub path: Vec<Vec2>,          // Line path coordinates
    pub style: VectorStyle,
}

/// Point definition
#[derive(Debug, Clone)]
pub struct PointDef {
    pub position: Vec2,           // Point position
    pub style: VectorStyle,
}

/// Graph definition with nodes and edges
#[derive(Debug, Clone)]
pub struct GraphDef {
    pub nodes: Vec<Vec2>,         // Node positions
    pub edges: Vec<(u32, u32)>,   // Edge pairs (from_node, to_node)
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
    pub fn add_polygons(&mut self, polygons: Vec<PolygonDef>, crs: CrsType) -> Result<Vec<VectorId>, RenderError> {
        // Validate CRS (must be planar for now)
        if crs != CrsType::Planar {
            return Err(RenderError::Upload(
                "Only planar CRS supported; geographic projections not implemented".to_string()
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
    pub fn add_lines(&mut self, lines: Vec<PolylineDef>, crs: CrsType) -> Result<Vec<VectorId>, RenderError> {
        if crs != CrsType::Planar {
            return Err(RenderError::Upload(
                "Only planar CRS supported".to_string()
            ));
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
    pub fn add_points(&mut self, points: Vec<PointDef>, crs: CrsType) -> Result<Vec<VectorId>, RenderError> {
        if crs != CrsType::Planar {
            return Err(RenderError::Upload(
                "Only planar CRS supported".to_string()
            ));
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
            return Err(RenderError::Upload(
                "Only planar CRS supported".to_string()
            ));
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
                "Polygon exterior must have at least 3 vertices".to_string()
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
                    "Polygon hole {} must have at least 3 vertices", hole_idx
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
                "Polyline must have at least 2 vertices".to_string()
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
                "Graph must have at least one node".to_string()
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

/// Python wrapper functions for numpy array inputs
pub fn parse_polygon_from_numpy<'py>(exterior: PyReadonlyArray2<'py, f64>) -> Result<Vec<Vec2>, RenderError> {
    if !exterior.is_contiguous() {
        return Err(RenderError::Upload(
            "Polygon exterior array must be C-contiguous (row-major); use np.ascontiguousarray()".to_string()
        ));
    }
    
    let exterior_arr = exterior.as_array();
    if exterior_arr.shape()[1] != 2 {
        return Err(RenderError::Upload(format!(
            "Polygon exterior must have shape (N, 2); got shape ({}, {})",
            exterior_arr.shape()[0], exterior_arr.shape()[1]
        )));
    }
    
    let mut vertices = Vec::with_capacity(exterior_arr.shape()[0]);
    for i in 0..exterior_arr.shape()[0] {
        let x = exterior_arr[[i, 0]] as f32;
        let y = exterior_arr[[i, 1]] as f32;
        
        if !x.is_finite() || !y.is_finite() {
            return Err(RenderError::Upload(format!(
                "Polygon vertex {} has non-finite coordinates: ({}, {})",
                i, x, y
            )));
        }
        
        vertices.push(Vec2::new(x, y));
    }
    
    Ok(vertices)
}

/// Global vector API instance for Python interface
use std::sync::Mutex;
static GLOBAL_VECTOR_API: Mutex<Option<VectorApi>> = Mutex::new(None);

fn with_global_api<F, T>(f: F) -> PyResult<T>
where
    F: FnOnce(&mut VectorApi) -> Result<T, RenderError>,
{
    let mut api_guard = GLOBAL_VECTOR_API.lock().map_err(|_| {
        pyo3::exceptions::PyRuntimeError::new_err("Failed to acquire vector API lock")
    })?;
    
    if api_guard.is_none() {
        *api_guard = Some(VectorApi::new());
    }
    
    let api = api_guard.as_mut().unwrap();
    f(api).map_err(|e| e.to_py_err())
}

/// Create polygons from numpy arrays
#[pyfunction]
#[pyo3(text_signature = "(exterior_coords, holes=None, fill_color=None, stroke_color=None, stroke_width=1.0)")]
pub fn add_polygons_py<'py>(
    _py: Python<'py>,
    exterior_coords: PyReadonlyArray2<'py, f64>,
    holes: Option<Vec<PyReadonlyArray2<'py, f64>>>,
    fill_color: Option<[f32; 4]>,
    stroke_color: Option<[f32; 4]>,
    stroke_width: Option<f32>,
) -> PyResult<Vec<u32>> {
    // Parse exterior
    let exterior = parse_polygon_from_numpy(exterior_coords)
        .map_err(|e| e.to_py_err())?;
    
    // Parse holes if provided
    let mut hole_rings = Vec::new();
    if let Some(hole_arrays) = holes {
        for hole_array in hole_arrays {
            let hole_vertices = parse_polygon_from_numpy(hole_array)
                .map_err(|e| e.to_py_err())?;
            hole_rings.push(hole_vertices);
        }
    }
    
    // Create style
    let style = VectorStyle {
        fill_color: fill_color.unwrap_or([0.2, 0.4, 0.8, 1.0]),
        stroke_color: stroke_color.unwrap_or([0.0, 0.0, 0.0, 1.0]),
        stroke_width: stroke_width.unwrap_or(1.0),
        point_size: 4.0, // Not used for polygons
    };
    
    let polygon = PolygonDef {
        exterior,
        holes: hole_rings,
        style,
    };
    
    with_global_api(|api| {
        api.add_polygons(vec![polygon], CrsType::Planar)
            .map(|ids| ids.into_iter().map(|id| id.0).collect())
    })
}

/// Create lines from numpy arrays
#[pyfunction]
#[pyo3(text_signature = "(path_coords, stroke_color=None, stroke_width=1.0)")]
pub fn add_lines_py<'py>(
    _py: Python<'py>,
    path_coords: PyReadonlyArray2<'py, f64>,
    stroke_color: Option<[f32; 4]>,
    stroke_width: Option<f32>,
) -> PyResult<Vec<u32>> {
    // Parse path coordinates
    if !path_coords.is_contiguous() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Path array must be C-contiguous (row-major); use np.ascontiguousarray()"
        ));
    }
    
    let path_arr = path_coords.as_array();
    if path_arr.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Path must have shape (N, 2); got shape ({}, {})",
            path_arr.shape()[0], path_arr.shape()[1]
        )));
    }
    
    let mut path = Vec::with_capacity(path_arr.shape()[0]);
    for i in 0..path_arr.shape()[0] {
        let x = path_arr[[i, 0]] as f32;
        let y = path_arr[[i, 1]] as f32;
        
        if !x.is_finite() || !y.is_finite() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Path vertex {} has non-finite coordinates: ({}, {})",
                i, x, y
            )));
        }
        
        path.push(Vec2::new(x, y));
    }
    
    // Create style
    let style = VectorStyle {
        fill_color: [0.0, 0.0, 0.0, 0.0], // Not used for lines
        stroke_color: stroke_color.unwrap_or([0.0, 0.0, 0.0, 1.0]),
        stroke_width: stroke_width.unwrap_or(1.0),
        point_size: 4.0, // Not used for lines
    };
    
    let polyline = PolylineDef {
        path,
        style,
    };
    
    with_global_api(|api| {
        api.add_lines(vec![polyline], CrsType::Planar)
            .map(|ids| ids.into_iter().map(|id| id.0).collect())
    })
}

/// Create points from numpy arrays
#[pyfunction]
#[pyo3(text_signature = "(positions, fill_color=None, point_size=4.0)")]
pub fn add_points_py<'py>(
    _py: Python<'py>,
    positions: PyReadonlyArray2<'py, f64>,
    fill_color: Option<[f32; 4]>,
    point_size: Option<f32>,
) -> PyResult<Vec<u32>> {
    // Parse positions
    if !positions.is_contiguous() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Positions array must be C-contiguous (row-major); use np.ascontiguousarray()"
        ));
    }
    
    let pos_arr = positions.as_array();
    if pos_arr.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Positions must have shape (N, 2); got shape ({}, {})",
            pos_arr.shape()[0], pos_arr.shape()[1]
        )));
    }
    
    let mut points = Vec::with_capacity(pos_arr.shape()[0]);
    for i in 0..pos_arr.shape()[0] {
        let x = pos_arr[[i, 0]] as f32;
        let y = pos_arr[[i, 1]] as f32;
        
        if !x.is_finite() || !y.is_finite() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Position {} has non-finite coordinates: ({}, {})",
                i, x, y
            )));
        }
        
        // Create style
        let style = VectorStyle {
            fill_color: fill_color.unwrap_or([1.0, 0.0, 0.0, 1.0]),
            stroke_color: [0.0, 0.0, 0.0, 1.0], // Not used for points
            stroke_width: 1.0, // Not used for points
            point_size: point_size.unwrap_or(4.0),
        };
        
        let point = PointDef {
            position: Vec2::new(x, y),
            style,
        };
        
        points.push(point);
    }
    
    with_global_api(|api| {
        api.add_points(points, CrsType::Planar)
            .map(|ids| ids.into_iter().map(|id| id.0).collect())
    })
}

/// Create graph from numpy arrays
#[pyfunction]
#[pyo3(text_signature = "(nodes, edges, node_fill_color=None, node_size=4.0, edge_stroke_color=None, edge_width=1.0)")]
pub fn add_graph_py<'py>(
    _py: Python<'py>,
    nodes: PyReadonlyArray2<'py, f64>,
    edges: PyReadonlyArray2<'py, u32>,
    node_fill_color: Option<[f32; 4]>,
    node_size: Option<f32>,
    edge_stroke_color: Option<[f32; 4]>,
    edge_width: Option<f32>,
) -> PyResult<u32> {
    // Parse nodes
    if !nodes.is_contiguous() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Nodes array must be C-contiguous (row-major); use np.ascontiguousarray()"
        ));
    }
    
    let nodes_arr = nodes.as_array();
    if nodes_arr.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Nodes must have shape (N, 2); got shape ({}, {})",
            nodes_arr.shape()[0], nodes_arr.shape()[1]
        )));
    }
    
    let mut node_positions = Vec::with_capacity(nodes_arr.shape()[0]);
    for i in 0..nodes_arr.shape()[0] {
        let x = nodes_arr[[i, 0]] as f32;
        let y = nodes_arr[[i, 1]] as f32;
        
        if !x.is_finite() || !y.is_finite() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Node {} has non-finite coordinates: ({}, {})",
                i, x, y
            )));
        }
        
        node_positions.push(Vec2::new(x, y));
    }
    
    // Parse edges
    if !edges.is_contiguous() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Edges array must be C-contiguous (row-major); use np.ascontiguousarray()"
        ));
    }
    
    let edges_arr = edges.as_array();
    if edges_arr.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Edges must have shape (M, 2); got shape ({}, {})",
            edges_arr.shape()[0], edges_arr.shape()[1]
        )));
    }
    
    let mut edge_pairs = Vec::with_capacity(edges_arr.shape()[0]);
    for i in 0..edges_arr.shape()[0] {
        let from = edges_arr[[i, 0]];
        let to = edges_arr[[i, 1]];
        edge_pairs.push((from, to));
    }
    
    // Create styles
    let node_style = VectorStyle {
        fill_color: node_fill_color.unwrap_or([1.0, 0.0, 0.0, 1.0]),
        stroke_color: [0.0, 0.0, 0.0, 1.0],
        stroke_width: 1.0,
        point_size: node_size.unwrap_or(4.0),
    };
    
    let edge_style = VectorStyle {
        fill_color: [0.0, 0.0, 0.0, 0.0],
        stroke_color: edge_stroke_color.unwrap_or([0.0, 0.0, 0.0, 1.0]),
        stroke_width: edge_width.unwrap_or(1.0),
        point_size: 4.0,
    };
    
    let graph = GraphDef {
        nodes: node_positions,
        edges: edge_pairs,
        node_style,
        edge_style,
    };
    
    with_global_api(|api| {
        api.add_graph(graph, CrsType::Planar)
            .map(|id| id.0)
    })
}

/// Clear all vector primitives
#[pyfunction]
#[pyo3(text_signature = "()")]
pub fn clear_vectors_py() -> PyResult<()> {
    with_global_api(|api| {
        api.clear();
        Ok(())
    })
}

/// Get vector primitive counts
#[pyfunction]
#[pyo3(text_signature = "()")]
pub fn get_vector_counts_py() -> PyResult<(usize, usize, usize, usize)> {
    with_global_api(|api| {
        Ok(api.get_counts())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vector_api_basic() {
        let mut api = VectorApi::new();
        
        // Test polygon addition
        let polygon = PolygonDef {
            exterior: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0), 
                Vec2::new(0.5, 1.0),
            ],
            holes: vec![],
            style: VectorStyle::default(),
        };
        
        let ids = api.add_polygons(vec![polygon], CrsType::Planar).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(api.get_counts(), (1, 0, 0, 0));
    }
    
    #[test]  
    fn test_crs_validation() {
        let mut api = VectorApi::new();
        
        let polygon = PolygonDef {
            exterior: vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.5, 1.0)],
            holes: vec![],
            style: VectorStyle::default(),
        };
        
        // Should reject non-planar CRS
        let result = api.add_polygons(vec![polygon], CrsType::WebMercator);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("planar CRS"));
    }
    
    #[test]
    fn test_polygon_validation() {
        let mut api = VectorApi::new();
        
        // Should reject polygon with < 3 vertices
        let invalid_polygon = PolygonDef {
            exterior: vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)], // Only 2 vertices
            holes: vec![],
            style: VectorStyle::default(),
        };
        
        let result = api.add_polygons(vec![invalid_polygon], CrsType::Planar);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("at least 3 vertices"));
    }
    
    #[test]
    fn test_graph_validation() {
        let mut api = VectorApi::new();
        
        // Valid graph
        let valid_graph = GraphDef {
            nodes: vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0)],
            edges: vec![(0, 1)],
            node_style: VectorStyle::default(),
            edge_style: VectorStyle::default(),
        };
        
        let id = api.add_graph(valid_graph, CrsType::Planar).unwrap();
        assert!(id.0 > 0);
        
        // Invalid graph - edge index out of bounds  
        let invalid_graph = GraphDef {
            nodes: vec![Vec2::new(0.0, 0.0)], // Only 1 node
            edges: vec![(0, 1)],               // Edge to non-existent node 1
            node_style: VectorStyle::default(),
            edge_style: VectorStyle::default(),
        };
        
        let result = api.add_graph(invalid_graph, CrsType::Planar);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds node count"));
    }
}
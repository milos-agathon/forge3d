//! H3: Packed data contracts
//! Reuse packed formats to limit Python overhead and ensure validation

use crate::error::RenderError;
use glam::{Vec2, Vec3, Vec4};
use std::mem;

/// Packed vertex data for polygons (position + UV)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PolygonVertex {
    pub position: [f32; 2],  // World coordinates
    pub uv: [f32; 2],        // Texture coordinates [0,1]
}

/// Packed vertex data for lines (position + direction + width)
#[repr(C)] 
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineVertex {
    pub position: [f32; 2],   // World coordinates
    pub direction: [f32; 2],  // Normalized direction vector
    pub width: f32,           // Line width in world units
    pub _pad: f32,            // Alignment padding
}

/// Packed vertex data for instanced points
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointInstance {
    pub position: [f32; 2],   // World coordinates
    pub size: f32,            // Point size in pixels
    pub color: [f32; 4],      // RGBA color
    pub rotation: f32,        // Rotation in radians (H21)
    pub uv_offset: [f32; 2],  // UV offset for texture atlas (H21)
    pub _pad: f32,            // Alignment padding  
}

/// Graph node data
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GraphNode {
    pub position: [f32; 2],   // World coordinates
    pub size: f32,            // Node size
    pub color: [f32; 4],      // RGBA color
}

/// Graph edge data (connects two node indices)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GraphEdge {
    pub from_node: u32,       // Source node index
    pub to_node: u32,         // Target node index
    pub width: f32,           // Edge width
    pub color: [f32; 4],      // RGBA color
}

/// Packed polygon data with hole support
#[derive(Debug, Clone)]
pub struct PackedPolygon {
    pub vertices: Vec<PolygonVertex>,
    pub indices: Vec<u32>,
    pub hole_offsets: Vec<u32>,  // Indices where holes start
}

/// Packed polyline data (H7)
#[derive(Debug, Clone)]
pub struct PackedPolyline {
    pub vertices: Vec<LineVertex>, 
    pub path_offsets: Vec<u32>,    // Start indices for each path
    pub path_lengths: Vec<u32>,    // Length of each path
}

/// Validation result for input data
#[derive(Debug)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub error_message: Option<String>,
    pub vertex_count: usize,
    pub index_count: usize,
}

impl ValidationResult {
    pub fn valid(vertex_count: usize, index_count: usize) -> Self {
        Self {
            is_valid: true,
            error_message: None,
            vertex_count,
            index_count,
        }
    }
    
    pub fn invalid(message: String) -> Self {
        Self {
            is_valid: false,
            error_message: Some(message),
            vertex_count: 0,
            index_count: 0,
        }
    }
}

/// H7: Polyline packing and validation
pub fn pack_lines(paths: &[Vec<Vec2>]) -> Result<PackedPolyline, RenderError> {
    if paths.is_empty() {
        return Err(RenderError::Upload("No paths provided".to_string()));
    }
    
    let mut vertices = Vec::new();
    let mut path_offsets = Vec::new();
    let mut path_lengths = Vec::new();
    
    for (path_idx, path) in paths.iter().enumerate() {
        if path.len() < 2 {
            return Err(RenderError::Upload(format!(
                "Path {} has only {} points; need at least 2", 
                path_idx, path.len()
            )));
        }
        
        // Check for invalid coordinates (NaN, infinite)
        for (pt_idx, point) in path.iter().enumerate() {
            if !point.x.is_finite() || !point.y.is_finite() {
                return Err(RenderError::Upload(format!(
                    "Path {} point {} has non-finite coordinates: ({}, {})",
                    path_idx, pt_idx, point.x, point.y
                )));
            }
        }
        
        path_offsets.push(vertices.len() as u32);
        let start_vertex_count = vertices.len();
        
        // Generate line segments with direction vectors
        for i in 0..path.len() - 1 {
            let p0 = path[i];
            let p1 = path[i + 1];
            
            let segment = p1 - p0;
            let length = segment.length();
            
            if length < 1e-6 {
                // Skip degenerate segments (duplicate consecutive vertices)
                continue;
            }
            
            let direction = segment / length;
            
            // Create line vertex
            vertices.push(LineVertex {
                position: [p0.x, p0.y],
                direction: [direction.x, direction.y],
                width: 1.0, // Default width, can be overridden
                _pad: 0.0,
            });
        }
        
        let actual_vertex_count = vertices.len() - start_vertex_count;
        path_lengths.push(actual_vertex_count as u32);
    }
    
    Ok(PackedPolyline {
        vertices,
        path_offsets,
        path_lengths,
    })
}

/// Validate polygon vertex data
pub fn validate_polygon_vertices(vertices: &[PolygonVertex], indices: &[u32]) -> ValidationResult {
    if vertices.is_empty() {
        return ValidationResult::invalid("Empty vertex array".to_string());
    }
    
    if indices.is_empty() {
        return ValidationResult::invalid("Empty index array".to_string());
    }
    
    if indices.len() % 3 != 0 {
        return ValidationResult::invalid(format!(
            "Index count {} is not divisible by 3 (not triangles)",
            indices.len()
        ));
    }
    
    // Validate index bounds
    let max_vertex_index = vertices.len() as u32 - 1;
    for (i, &index) in indices.iter().enumerate() {
        if index > max_vertex_index {
            return ValidationResult::invalid(format!(
                "Index {} at position {} exceeds vertex count {}",
                index, i, vertices.len()
            ));
        }
    }
    
    // Validate vertex coordinates are finite
    for (i, vertex) in vertices.iter().enumerate() {
        if !vertex.position[0].is_finite() || !vertex.position[1].is_finite() {
            return ValidationResult::invalid(format!(
                "Vertex {} has non-finite position: ({}, {})",
                i, vertex.position[0], vertex.position[1]
            ));
        }
    }
    
    ValidationResult::valid(vertices.len(), indices.len())
}

/// Validate point instance data
pub fn validate_point_instances(instances: &[PointInstance]) -> ValidationResult {
    if instances.is_empty() {
        return ValidationResult::invalid("Empty instance array".to_string());
    }
    
    for (i, instance) in instances.iter().enumerate() {
        // Validate position
        if !instance.position[0].is_finite() || !instance.position[1].is_finite() {
            return ValidationResult::invalid(format!(
                "Instance {} has non-finite position: ({}, {})",
                i, instance.position[0], instance.position[1]
            ));
        }
        
        // Validate size
        if !instance.size.is_finite() || instance.size <= 0.0 {
            return ValidationResult::invalid(format!(
                "Instance {} has invalid size: {}",
                i, instance.size
            ));
        }
        
        // Validate color components [0,1]
        for (c, &color_component) in instance.color.iter().enumerate() {
            if !color_component.is_finite() || color_component < 0.0 || color_component > 1.0 {
                return ValidationResult::invalid(format!(
                    "Instance {} has invalid color component {}: {}",
                    i, c, color_component
                ));
            }
        }
    }
    
    ValidationResult::valid(instances.len(), 0)
}

/// Memory layout constants for GPU buffers
pub mod layout {
    use super::*;
    
    pub const POLYGON_VERTEX_SIZE: usize = mem::size_of::<PolygonVertex>();
    pub const LINE_VERTEX_SIZE: usize = mem::size_of::<LineVertex>();
    pub const POINT_INSTANCE_SIZE: usize = mem::size_of::<PointInstance>();
    pub const GRAPH_NODE_SIZE: usize = mem::size_of::<GraphNode>();
    pub const GRAPH_EDGE_SIZE: usize = mem::size_of::<GraphEdge>();
    
    // GPU alignment requirements (16-byte aligned for uniform buffers)
    pub const GPU_ALIGNMENT: usize = 16;
    
    pub fn align_size(size: usize) -> usize {
        (size + GPU_ALIGNMENT - 1) & !(GPU_ALIGNMENT - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;
    
    #[test]
    fn test_pack_lines_valid() {
        let paths = vec![
            vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0), Vec2::new(2.0, 0.0)],
            vec![Vec2::new(10.0, 10.0), Vec2::new(11.0, 11.0)],
        ];
        
        let result = pack_lines(&paths).unwrap();
        
        assert_eq!(result.path_offsets.len(), 2);
        assert_eq!(result.path_lengths.len(), 2);
        assert_eq!(result.path_offsets[0], 0);
        assert_eq!(result.path_lengths[0], 2); // 3 points -> 2 segments
        assert_eq!(result.path_lengths[1], 1); // 2 points -> 1 segment
        
        // Check direction vectors are normalized
        for vertex in &result.vertices {
            let dir_len = (vertex.direction[0] * vertex.direction[0] + 
                          vertex.direction[1] * vertex.direction[1]).sqrt();
            assert!((dir_len - 1.0).abs() < 1e-5);
        }
    }
    
    #[test]
    fn test_pack_lines_rejects_short_paths() {
        let paths = vec![
            vec![Vec2::new(0.0, 0.0)], // Only 1 point
        ];
        
        let result = pack_lines(&paths);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("need at least 2"));
    }
    
    #[test]
    fn test_pack_lines_rejects_nan() {
        let paths = vec![
            vec![Vec2::new(0.0, 0.0), Vec2::new(f32::NAN, 1.0)],
        ];
        
        let result = pack_lines(&paths);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("non-finite"));
    }
    
    #[test]
    fn test_validate_polygon_vertices() {
        let vertices = vec![
            PolygonVertex { position: [0.0, 0.0], uv: [0.0, 0.0] },
            PolygonVertex { position: [1.0, 0.0], uv: [1.0, 0.0] }, 
            PolygonVertex { position: [0.5, 1.0], uv: [0.5, 1.0] },
        ];
        let indices = vec![0, 1, 2];
        
        let result = validate_polygon_vertices(&vertices, &indices);
        assert!(result.is_valid);
        assert_eq!(result.vertex_count, 3);
        assert_eq!(result.index_count, 3);
    }
    
    #[test]
    fn test_validate_polygon_rejects_out_of_bounds_index() {
        let vertices = vec![
            PolygonVertex { position: [0.0, 0.0], uv: [0.0, 0.0] },
        ];
        let indices = vec![0, 1, 2]; // Index 1,2 are out of bounds
        
        let result = validate_polygon_vertices(&vertices, &indices);
        assert!(!result.is_valid);
        assert!(result.error_message.unwrap().contains("exceeds vertex count"));
    }
    
    #[test]
    fn test_memory_layout_constants() {
        // Verify struct sizes for GPU compatibility
        assert_eq!(layout::POLYGON_VERTEX_SIZE, 16); // 2*f32 + 2*f32 = 16 bytes
        assert_eq!(layout::LINE_VERTEX_SIZE, 24);    // 2*f32 + 2*f32 + f32 + f32 = 24 bytes
        assert_eq!(layout::POINT_INSTANCE_SIZE, 48); // 2*f32 + f32 + 4*f32 + f32 + 2*f32 + f32 = 48 bytes
        
        // Test alignment helper
        assert_eq!(layout::align_size(17), 32);
        assert_eq!(layout::align_size(16), 16);
    }
}
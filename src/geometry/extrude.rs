// src/geometry/extrude.rs
// Polygon extrusion helpers producing Forge3D mesh buffers
// Exists to bridge vector tessellation with reusable geometry generation
// RELEVANT FILES:src/geometry/mod.rs,src/vector/extrusion.rs,tests/test_f1_extrude.py,examples/f1_extrude_demo.py

use glam::Vec2;

use crate::vector::extrusion as vector_extrusion;

use super::{GeometryError, GeometryResult, MeshBuffers};

/// Options controlling polygon extrusion.
#[derive(Debug, Clone, Copy)]
pub struct ExtrudeOptions {
    pub height: f32,
    pub cap_uv_scale: f32,
}

impl Default for ExtrudeOptions {
    fn default() -> Self {
        Self {
            height: 1.0,
            cap_uv_scale: 1.0,
        }
    }
}

/// Extrude a polygon ring with a simple uniform height.
pub fn extrude_polygon(points: &[[f32; 2]], height: f32) -> GeometryResult<MeshBuffers> {
    let mut options = ExtrudeOptions::default();
    options.height = height;
    extrude_polygon_with_options(points, options)
}

/// Extrude a polygon using the provided options.
pub fn extrude_polygon_with_options(
    points: &[[f32; 2]],
    options: ExtrudeOptions,
) -> GeometryResult<MeshBuffers> {
    if points.len() < 3 {
        return Err(GeometryError::new(
            "Polygon extrusion requires at least three vertices",
        ));
    }

    if !options.height.is_finite() {
        return Err(GeometryError::new("Extrusion height must be finite"));
    }

    if options.height.abs() <= f32::EPSILON {
        return Err(GeometryError::new(
            "Extrusion height must be non-zero to generate volume",
        ));
    }

    let polygon: Vec<Vec2> = points.iter().map(|p| Vec2::new(p[0], p[1])).collect();

    let (positions, indices, normals, uvs) =
        vector_extrusion::extrude_polygon(&polygon, options.height);

    if positions.is_empty() || indices.is_empty() {
        return Err(GeometryError::new(
            "Extrusion produced no geometry, check input winding and height",
        ));
    }

    if normals.len() != positions.len() {
        return Err(GeometryError::new(
            "Extrusion produced mismatched normal count",
        ));
    }

    if uvs.len() != positions.len() {
        return Err(GeometryError::new("Extrusion produced mismatched UV count"));
    }

    let mut mesh = MeshBuffers::with_capacity(positions.len(), indices.len());
    mesh.positions
        .extend(positions.into_iter().map(|v| [v.x, v.y, v.z]));
    mesh.normals
        .extend(normals.into_iter().map(|n| [n.x, n.y, n.z]));
    mesh.uvs.extend(
        uvs.into_iter()
            .map(|uv| [uv.x * options.cap_uv_scale, uv.y * options.cap_uv_scale]),
    );
    mesh.indices = indices;

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extrude_simple_square() {
        let square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let mesh = extrude_polygon(&square, 2.0).expect("extrusion should succeed");
        assert_eq!(mesh.vertex_count(), 32);
        assert_eq!(mesh.triangle_count(), 8);
        assert!(!mesh.is_empty());
    }

    #[test]
    fn rejects_short_polygons() {
        let line = [[0.0, 0.0], [1.0, 0.0]];
        assert!(extrude_polygon(&line, 1.0).is_err());
    }
}

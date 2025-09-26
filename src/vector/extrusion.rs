// src/vector/extrusion.rs
// CPU polygon prism mesh builder with triangulation, normals, and UV generation
// Exists to provide deterministic CPU fallback and shared prep for GPU extrusion
// RELEVANT FILES: src/vector/gpu_extrusion.rs, shaders/extrusion.wgsl, src/vector/api.rs, docs/api/polygon_extrusion.md

//! F1: Polygon Extrusion
//! Reference CPU implementation and shared tessellation utilities.

use glam::{vec2, Vec2, Vec3};
use lyon_path::math::Point;
use lyon_path::Path;
use lyon_tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers};

pub(crate) const EPSILON: f32 = 1e-6;

/// Tessellated polygon data reused by CPU and GPU extrusion paths.
#[derive(Debug, Clone)]
pub(crate) struct TessellatedPolygon {
    pub base_vertices: Vec<Vec2>,
    pub base_indices: Vec<u32>,
    pub ring: Vec<Vec2>,
    pub ring_u: Vec<f32>,
    pub bbox_min: Vec2,
    pub bbox_size: Vec2,
}

/// Compute tessellation data for a single polygon.
pub(crate) fn tessellate_polygon(polygon: &[Vec2]) -> Option<TessellatedPolygon> {
    let mut ring = preprocess_ring(polygon);
    if ring.len() < 3 {
        return None;
    }

    // Ensure counter-clockwise orientation for consistent outward normals.
    if signed_area(&ring) < 0.0 {
        ring.reverse();
    }

    let mut geometry: VertexBuffers<Vec2, u32> = VertexBuffers::new();
    let mut tessellator = FillTessellator::new();

    let mut path_builder = Path::builder();
    path_builder.begin(Point::new(ring[0].x, ring[0].y));
    for vertex in ring.iter().skip(1) {
        path_builder.line_to(Point::new(vertex.x, vertex.y));
    }
    path_builder.close();
    let path = path_builder.build();

    if tessellator
        .tessellate_path(
            &path,
            &FillOptions::default(),
            &mut BuffersBuilder::new(&mut geometry, |vertex: FillVertex| {
                Vec2::new(vertex.position().x, vertex.position().y)
            }),
        )
        .is_err()
    {
        return None;
    }

    if geometry.vertices.is_empty() || geometry.indices.is_empty() {
        return None;
    }

    let (bbox_min, bbox_size) = compute_bounds(&ring);
    let ring_u = compute_ring_uvs(&ring);

    Some(TessellatedPolygon {
        base_vertices: geometry.vertices,
        base_indices: geometry.indices,
        ring,
        ring_u,
        bbox_min,
        bbox_size,
    })
}

/// Extrude a polygon into a 3D prism mesh (positions, indices, normals, UVs).
pub fn extrude_polygon(
    polygon: &[Vec2],
    height: f32,
) -> (Vec<Vec3>, Vec<u32>, Vec<Vec3>, Vec<Vec2>) {
    let tess = match tessellate_polygon(polygon) {
        Some(data) => data,
        None => return (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
    };

    build_prism(&tess, height)
}

/// Build the extruded prism geometry from tessellation data.
fn build_prism(
    tess: &TessellatedPolygon,
    height: f32,
) -> (Vec<Vec3>, Vec<u32>, Vec<Vec3>, Vec<Vec2>) {
    let base_vertex_count = tess.base_vertices.len();
    let ring_vertex_count = tess.ring.len();
    let side_vertex_count = ring_vertex_count * 4;

    let total_vertices = base_vertex_count * 2 + side_vertex_count;
    let total_indices = tess.base_indices.len() * 2 + ring_vertex_count * 6;

    let mut positions = Vec::with_capacity(total_vertices);
    let mut normals = Vec::with_capacity(total_vertices);
    let mut uvs = Vec::with_capacity(total_vertices);
    let mut indices = Vec::with_capacity(total_indices);

    let inv_scale = Vec2::new(
        if tess.bbox_size.x.abs() > EPSILON {
            1.0 / tess.bbox_size.x
        } else {
            0.0
        },
        if tess.bbox_size.y.abs() > EPSILON {
            1.0 / tess.bbox_size.y
        } else {
            0.0
        },
    );

    // Bottom face vertices
    for vertex in &tess.base_vertices {
        positions.push(Vec3::new(vertex.x, 0.0, vertex.y));
        normals.push(Vec3::new(0.0, -1.0, 0.0));
        uvs.push(Vec2::new(
            if inv_scale.x != 0.0 {
                (vertex.x - tess.bbox_min.x) * inv_scale.x
            } else {
                0.0
            },
            if inv_scale.y != 0.0 {
                (vertex.y - tess.bbox_min.y) * inv_scale.y
            } else {
                0.0
            },
        ));
    }

    let top_vertex_offset = positions.len();

    // Top face vertices
    for vertex in &tess.base_vertices {
        positions.push(Vec3::new(vertex.x, height, vertex.y));
        normals.push(Vec3::new(0.0, 1.0, 0.0));
        uvs.push(Vec2::new(
            if inv_scale.x != 0.0 {
                (vertex.x - tess.bbox_min.x) * inv_scale.x
            } else {
                0.0
            },
            if inv_scale.y != 0.0 {
                (vertex.y - tess.bbox_min.y) * inv_scale.y
            } else {
                0.0
            },
        ));
    }

    let side_vertex_offset = positions.len();

    // Bottom indices (reverse winding for downward normal)
    for triangle in tess.base_indices.chunks_exact(3) {
        let a = triangle[0];
        let b = triangle[1];
        let c = triangle[2];
        indices.extend_from_slice(&[a, c, b]);
    }

    // Top indices (match tessellation winding)
    for triangle in tess.base_indices.chunks_exact(3) {
        indices.extend(triangle.iter().map(|idx| top_vertex_offset as u32 + idx));
    }

    // Side faces
    let mut side_vertices_written = 0usize;
    let mut _side_indices_written = 0usize;
    for (i, current) in tess.ring.iter().enumerate() {
        let next = tess.ring[(i + 1) % ring_vertex_count];
        let u_curr = tess.ring_u[i];
        let mut u_next = if i + 1 == ring_vertex_count {
            1.0
        } else {
            tess.ring_u[i + 1]
        };
        if u_next < u_curr {
            u_next = 1.0;
        }

        let edge = next - *current;
        let edge_len = edge.length();
        let normal_2d = if edge_len > EPSILON {
            vec2(edge.y / edge_len, -edge.x / edge_len)
        } else {
            vec2(0.0, 0.0)
        };
        let normal = Vec3::new(normal_2d.x, 0.0, normal_2d.y);

        let base_index = side_vertex_offset + side_vertices_written;
        let base_index_u32 = base_index as u32;

        positions.push(Vec3::new(current.x, 0.0, current.y));
        positions.push(Vec3::new(next.x, 0.0, next.y));
        positions.push(Vec3::new(current.x, height, current.y));
        positions.push(Vec3::new(next.x, height, next.y));

        normals.push(normal);
        normals.push(normal);
        normals.push(normal);
        normals.push(normal);

        uvs.push(Vec2::new(u_curr, 0.0));
        uvs.push(Vec2::new(u_next, 0.0));
        uvs.push(Vec2::new(u_curr, 1.0));
        uvs.push(Vec2::new(u_next, 1.0));

        indices.extend_from_slice(&[
            base_index_u32,
            base_index_u32 + 2,
            base_index_u32 + 1,
            base_index_u32 + 2,
            base_index_u32 + 3,
            base_index_u32 + 1,
        ]);

        side_vertices_written += 4;
        _side_indices_written += 6;
    }

    (positions, indices, normals, uvs)
}

fn preprocess_ring(polygon: &[Vec2]) -> Vec<Vec2> {
    let mut ring: Vec<Vec2> = Vec::new();
    for &point in polygon {
        if let Some(last) = ring.last() {
            if (*last - point).length_squared() < EPSILON * EPSILON {
                continue;
            }
        }
        ring.push(point);
    }

    if ring.len() >= 2 {
        let close_distance = (ring[0] - *ring.last().unwrap()).length_squared();
        if close_distance < EPSILON * EPSILON {
            ring.pop();
        }
    }

    ring
}

fn signed_area(ring: &[Vec2]) -> f32 {
    let mut area = 0.0;
    for i in 0..ring.len() {
        let j = (i + 1) % ring.len();
        area += ring[i].x * ring[j].y - ring[j].x * ring[i].y;
    }
    area * 0.5
}

fn compute_bounds(ring: &[Vec2]) -> (Vec2, Vec2) {
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for &point in ring {
        min = min.min(point);
        max = max.max(point);
    }
    (min, max - min)
}

fn compute_ring_uvs(ring: &[Vec2]) -> Vec<f32> {
    let mut lengths = Vec::with_capacity(ring.len());
    let mut total = 0.0;
    for i in 0..ring.len() {
        let next = ring[(i + 1) % ring.len()];
        let len = (next - ring[i]).length();
        lengths.push(len);
        total += len;
    }

    let mut cumulative = 0.0;
    let mut u_values = Vec::with_capacity(ring.len());
    for &len in &lengths {
        if total > EPSILON {
            u_values.push((cumulative / total).clamp(0.0, 1.0));
        } else {
            u_values.push(0.0);
        }
        cumulative += len;
    }

    if let Some(last) = u_values.last_mut() {
        *last = 1.0;
    }

    u_values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extrude_triangle_counts() {
        let polygon = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, 1.0),
        ];
        let (vertices, indices, normals, uvs) = extrude_polygon(&polygon, 1.0);

        assert_eq!(vertices.len(), 18);
        assert_eq!(indices.len(), 24);
        assert_eq!(normals.len(), vertices.len());
        assert_eq!(uvs.len(), vertices.len());
    }

    #[test]
    fn test_side_normals_direction() {
        let polygon = vec![
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
        ];
        let (_, _, normals, _) = extrude_polygon(&polygon, 2.0);

        // Bottom face normals first half, side normals at the end
        let side_normals = &normals[polygon.len() * 2..];
        for chunk in side_normals.chunks_exact(4) {
            let n = chunk[0];
            assert!(n.y.abs() < 1e-4);
            let len = n.length();
            if len > 0.0 {
                assert!((len - 1.0).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn test_uvs_are_in_unit_interval() {
        let polygon = vec![
            Vec2::new(-2.0, 0.0),
            Vec2::new(0.0, 3.0),
            Vec2::new(2.0, 0.0),
        ];
        let (_, _, _, uvs) = extrude_polygon(&polygon, 1.5);

        for uv in uvs {
            assert!(uv.x >= -1e-4 && uv.x <= 1.0 + 1e-4);
            assert!(uv.y >= -1e-4 && uv.y <= 1.0 + 1e-4);
        }
    }
}

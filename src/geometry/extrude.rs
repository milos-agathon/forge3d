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

/// Extrude an outer ring and its holes in the XZ plane, retaining courtyard walls.
#[cfg(any(test, feature = "extension-module"))]
pub(crate) fn extrude_rings(rings: &[Vec<[f32; 2]>], height: f32) -> GeometryResult<MeshBuffers> {
    use lyon_path::{math::point, Path};
    use lyon_tessellation::{
        BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, VertexBuffers,
    };
    if rings.is_empty() || !height.is_finite() || height <= 0.0 {
        return Err(GeometryError::new(
            "Building extrusion requires rings and a positive finite height",
        ));
    }
    let mut normalized = Vec::new();
    let mut builder = Path::builder();
    for (index, source) in rings.iter().enumerate() {
        let mut ring = source.clone();
        if ring.first() == ring.last() {
            ring.pop();
        }
        if ring.len() < 3 || ring.iter().flatten().any(|v| !v.is_finite()) {
            return Err(GeometryError::new(
                "Building ring requires at least three finite positions",
            ));
        }
        let area: f64 = ring
            .iter()
            .zip(ring.iter().cycle().skip(1))
            .take(ring.len())
            .map(|(a, b)| f64::from(a[0]) * f64::from(b[1]) - f64::from(b[0]) * f64::from(a[1]))
            .sum();
        if area == 0.0 {
            return Err(GeometryError::new("Building ring is degenerate"));
        }
        if (area > 0.0) != (index == 0) {
            ring.reverse();
        }
        builder.begin(point(ring[0][0], ring[0][1]));
        for p in ring.iter().skip(1) {
            builder.line_to(point(p[0], p[1]));
        }
        builder.close();
        normalized.push(ring);
    }
    let mut cap: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    FillTessellator::new()
        .tessellate_path(
            &builder.build(),
            &FillOptions::default().with_fill_rule(FillRule::EvenOdd),
            &mut BuffersBuilder::new(&mut cap, |v: FillVertex| [v.position().x, v.position().y]),
        )
        .map_err(|e| GeometryError::new(format!("Building tessellation failed: {e:?}")))?;
    let mut mesh = MeshBuffers::new();
    for (y, normal) in [(0.0, -1.0), (height, 1.0)] {
        let base = mesh.positions.len() as u32;
        for p in &cap.vertices {
            mesh.positions.push([p[0], y, p[1]]);
            mesh.normals.push([0.0, normal, 0.0]);
            mesh.uvs.push(*p);
        }
        for tri in cap.indices.chunks_exact(3) {
            let [a, b, c] = [
                cap.vertices[tri[0] as usize],
                cap.vertices[tri[1] as usize],
                cap.vertices[tri[2] as usize],
            ];
            let cross_y = (b[1] - a[1]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[1] - a[1]);
            let ordered = if cross_y * normal > 0.0 {
                [tri[0], tri[1], tri[2]]
            } else {
                [tri[0], tri[2], tri[1]]
            };
            mesh.indices.extend(ordered.map(|i| base + i));
        }
    }
    for ring in normalized {
        for (a, b) in ring
            .iter()
            .zip(ring.iter().cycle().skip(1))
            .take(ring.len())
        {
            let edge = Vec2::new(b[0] - a[0], b[1] - a[1]);
            if edge.length_squared() == 0.0 {
                continue;
            }
            let n = Vec2::new(edge.y, -edge.x).normalize();
            let base = mesh.positions.len() as u32;
            mesh.positions.extend([
                [a[0], 0.0, a[1]],
                [b[0], 0.0, b[1]],
                [a[0], height, a[1]],
                [b[0], height, b[1]],
            ]);
            mesh.normals.extend([[n.x, 0.0, n.y]; 4]);
            mesh.uvs
                .extend([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
            mesh.indices
                .extend([base, base + 2, base + 1, base + 1, base + 2, base + 3]);
        }
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn courtyard_caps_and_walls_are_preserved() {
        let rings = vec![
            vec![[0., 0.], [10., 0.], [10., 10.], [0., 10.]],
            vec![[3., 3.], [7., 3.], [7., 7.], [3., 7.]],
        ];
        let mesh = extrude_rings(&rings, 2.).unwrap();
        let mut area = 0.;
        for tri in mesh.indices.chunks_exact(3) {
            let a = glam::Vec3::from(mesh.positions[tri[0] as usize]);
            let b = glam::Vec3::from(mesh.positions[tri[1] as usize]);
            let c = glam::Vec3::from(mesh.positions[tri[2] as usize]);
            if a.y == 2. && b.y == 2. && c.y == 2. {
                area += (b - a).cross(c - a).length() / 2.;
            }
        }
        assert_eq!(area, 84.);
        assert!(mesh
            .positions
            .iter()
            .zip(&mesh.normals)
            .any(|(p, n)| p[0] == 3. && n[0] == 1.));
    }

    #[test]
    fn extrude_simple_square() {
        let square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let mesh = extrude_polygon(&square, 2.0).expect("extrusion should succeed");
        assert_eq!(mesh.vertex_count(), 24);
        assert_eq!(mesh.triangle_count(), 12);
        assert!(!mesh.is_empty());
    }

    #[test]
    fn rejects_short_polygons() {
        let line = [[0.0, 0.0], [1.0, 0.0]];
        assert!(extrude_polygon(&line, 1.0).is_err());
    }
}

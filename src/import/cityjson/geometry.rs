use serde_json::Value as JsonValue;

use super::{BuildingGeom, CityJsonError, CityJsonResult};

pub(super) fn parse_geometry(
    building: &mut BuildingGeom,
    geom: &JsonValue,
    vertices: &[[f64; 3]],
) -> CityJsonResult<()> {
    let geom_type = geom.get("type").and_then(|t| t.as_str()).unwrap_or("");
    let boundaries = geom
        .get("boundaries")
        .ok_or_else(|| CityJsonError::new("Geometry missing 'boundaries'"))?;

    match geom_type {
        "Solid" => parse_solid(building, boundaries, vertices)?,
        "MultiSurface" | "CompositeSurface" => parse_multi_surface(building, boundaries, vertices)?,
        _ => {}
    }

    Ok(())
}

pub(super) fn compute_normals(positions: &[f32], indices: &[u32]) -> Vec<f32> {
    let vertex_count = positions.len() / 3;
    let mut normals = vec![0.0f32; positions.len()];

    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }

        let i0 = tri[0] as usize * 3;
        let i1 = tri[1] as usize * 3;
        let i2 = tri[2] as usize * 3;
        if i0 + 2 >= positions.len() || i1 + 2 >= positions.len() || i2 + 2 >= positions.len() {
            continue;
        }

        let v0 = [positions[i0], positions[i0 + 1], positions[i0 + 2]];
        let v1 = [positions[i1], positions[i1 + 1], positions[i1 + 2]];
        let v2 = [positions[i2], positions[i2 + 1], positions[i2 + 2]];
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        for &idx in tri {
            let base = idx as usize * 3;
            if base + 2 < normals.len() {
                normals[base] += n[0];
                normals[base + 1] += n[1];
                normals[base + 2] += n[2];
            }
        }
    }

    for i in 0..vertex_count {
        let base = i * 3;
        let len =
            (normals[base].powi(2) + normals[base + 1].powi(2) + normals[base + 2].powi(2)).sqrt();
        if len > 1e-6 {
            normals[base] /= len;
            normals[base + 1] /= len;
            normals[base + 2] /= len;
        } else {
            normals[base] = 0.0;
            normals[base + 1] = 0.0;
            normals[base + 2] = 1.0;
        }
    }

    normals
}

fn parse_solid(
    building: &mut BuildingGeom,
    boundaries: &JsonValue,
    vertices: &[[f64; 3]],
) -> CityJsonResult<()> {
    let shells = boundaries
        .as_array()
        .ok_or_else(|| CityJsonError::new("Solid boundaries not an array"))?;

    if let Some(outer_shell) = shells.first() {
        if let Some(surfaces) = outer_shell.as_array() {
            for surface in surfaces {
                parse_surface(building, surface, vertices)?;
            }
        }
    }

    Ok(())
}

fn parse_multi_surface(
    building: &mut BuildingGeom,
    boundaries: &JsonValue,
    vertices: &[[f64; 3]],
) -> CityJsonResult<()> {
    let surfaces = boundaries
        .as_array()
        .ok_or_else(|| CityJsonError::new("MultiSurface boundaries not an array"))?;

    for surface in surfaces {
        parse_surface(building, surface, vertices)?;
    }

    Ok(())
}

fn parse_surface(
    building: &mut BuildingGeom,
    surface: &JsonValue,
    vertices: &[[f64; 3]],
) -> CityJsonResult<()> {
    let rings = surface
        .as_array()
        .ok_or_else(|| CityJsonError::new("Surface is not an array"))?;
    if rings.is_empty() {
        return Ok(());
    }

    let mut surface_rings = Vec::with_capacity(rings.len());
    for ring in rings {
        let parsed = parse_ring_vertices(ring, vertices)?;
        if parsed.len() >= 3 {
            surface_rings.push(parsed);
        }
    }
    if surface_rings.is_empty() {
        return Ok(());
    }

    tessellate_surface(building, &surface_rings)
}

fn parse_ring_vertices(ring: &JsonValue, vertices: &[[f64; 3]]) -> CityJsonResult<Vec<[f64; 3]>> {
    let values = ring
        .as_array()
        .ok_or_else(|| CityJsonError::new("Ring is not an array"))?;
    let mut ring_verts = Vec::with_capacity(values.len());
    for idx_val in values {
        let idx = idx_val
            .as_u64()
            .ok_or_else(|| CityJsonError::new("Vertex index is not a number"))?
            as usize;
        if idx >= vertices.len() {
            return Err(CityJsonError::new(format!(
                "Vertex index {idx} out of bounds"
            )));
        }
        ring_verts.push(vertices[idx]);
    }
    if ring_verts.len() > 1 && ring_verts.first() == ring_verts.last() {
        ring_verts.pop();
    }
    Ok(ring_verts)
}

fn tessellate_surface(building: &mut BuildingGeom, rings: &[Vec<[f64; 3]>]) -> CityJsonResult<()> {
    use lyon_path::Path;
    use lyon_tessellation::{
        BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, VertexBuffers,
    };

    let projection = SurfaceProjection::from_rings(rings)
        .ok_or_else(|| CityJsonError::new("Surface rings are degenerate"))?;
    let mut builder = Path::builder();
    for ring in rings {
        let first = projection.project(ring[0]);
        builder.begin(lyon_path::math::Point::new(first[0], first[1]));
        for vertex in ring.iter().skip(1) {
            let projected = projection.project(*vertex);
            builder.line_to(lyon_path::math::Point::new(projected[0], projected[1]));
        }
        builder.close();
    }
    let path = builder.build();
    let mut tessellator = FillTessellator::new();
    let mut buffers: VertexBuffers<[f32; 3], u32> = VertexBuffers::new();
    tessellator
        .tessellate_path(
            &path,
            &FillOptions::default().with_fill_rule(FillRule::EvenOdd),
            &mut BuffersBuilder::new(&mut buffers, |vertex: FillVertex| {
                let position = vertex.position();
                projection.unproject([position.x, position.y])
            }),
        )
        .map_err(|err| CityJsonError::new(format!("Surface tessellation failed: {err:?}")))?;

    let base_idx = building.positions.len() as u32 / 3;
    for vertex in &buffers.vertices {
        building.positions.extend_from_slice(vertex);
    }
    for index in buffers.indices {
        building.indices.push(base_idx + index);
    }

    Ok(())
}

#[derive(Clone, Copy)]
struct SurfaceProjection {
    drop_axis: usize,
    origin: [f64; 3],
    normal: [f64; 3],
}

impl SurfaceProjection {
    fn from_rings(rings: &[Vec<[f64; 3]>]) -> Option<Self> {
        let origin = *rings.iter().find_map(|ring| ring.first())?;
        let mut normal = None;
        for ring in rings {
            for i in 1..ring.len() {
                for j in (i + 1)..ring.len() {
                    let a = sub(ring[i], origin);
                    let b = sub(ring[j], origin);
                    let n = cross(a, b);
                    if length_sq(n) > 1.0e-18 {
                        normal = Some(n);
                        break;
                    }
                }
                if normal.is_some() {
                    break;
                }
            }
            if normal.is_some() {
                break;
            }
        }
        let normal = normal?;
        let abs = [normal[0].abs(), normal[1].abs(), normal[2].abs()];
        let drop_axis = if abs[0] >= abs[1] && abs[0] >= abs[2] {
            0
        } else if abs[1] >= abs[2] {
            1
        } else {
            2
        };
        Some(Self {
            drop_axis,
            origin,
            normal,
        })
    }

    fn project(&self, point: [f64; 3]) -> [f32; 2] {
        match self.drop_axis {
            0 => [point[1] as f32, point[2] as f32],
            1 => [point[0] as f32, point[2] as f32],
            _ => [point[0] as f32, point[1] as f32],
        }
    }

    fn unproject(&self, point: [f32; 2]) -> [f32; 3] {
        let u = point[0] as f64;
        let v = point[1] as f64;
        let mut p = self.origin;
        match self.drop_axis {
            0 => {
                p[1] = u;
                p[2] = v;
                p[0] = self.origin[0]
                    - (self.normal[1] * (p[1] - self.origin[1])
                        + self.normal[2] * (p[2] - self.origin[2]))
                        / self.normal[0];
            }
            1 => {
                p[0] = u;
                p[2] = v;
                p[1] = self.origin[1]
                    - (self.normal[0] * (p[0] - self.origin[0])
                        + self.normal[2] * (p[2] - self.origin[2]))
                        / self.normal[1];
            }
            _ => {
                p[0] = u;
                p[1] = v;
                p[2] = self.origin[2]
                    - (self.normal[0] * (p[0] - self.origin[0])
                        + self.normal[1] * (p[1] - self.origin[1]))
                        / self.normal[2];
            }
        }
        [p[0] as f32, p[1] as f32, p[2] as f32]
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length_sq(v: [f64; 3]) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

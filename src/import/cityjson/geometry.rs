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

    let outer_ring = rings[0]
        .as_array()
        .ok_or_else(|| CityJsonError::new("Ring is not an array"))?;
    if outer_ring.len() < 3 {
        return Ok(());
    }

    let mut ring_verts = Vec::with_capacity(outer_ring.len());
    for idx_val in outer_ring {
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

    let base_idx = building.positions.len() as u32 / 3;
    for vertex in &ring_verts {
        building.positions.push(vertex[0] as f32);
        building.positions.push(vertex[1] as f32);
        building.positions.push(vertex[2] as f32);
    }

    for i in 1..(ring_verts.len() as u32 - 1) {
        building.indices.push(base_idx);
        building.indices.push(base_idx + i);
        building.indices.push(base_idx + i + 1);
    }

    Ok(())
}

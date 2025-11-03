// src/geometry/weld.rs
// Vertex welding and normal recomputation for Forge3D meshes
// Exists to guarantee consistent topology and shading prior to IO/export pipelines
// RELEVANT FILES:src/geometry/mod.rs,src/geometry/validate.rs,tests/test_f8_weld.py,python/forge3d/geometry.py

use std::collections::BTreeMap;

use glam::Vec3;

use super::MeshBuffers;

/// Options controlling the mesh weld process.
#[derive(Debug, Clone, Copy)]
pub struct WeldOptions {
    pub position_epsilon: f32,
    pub uv_epsilon: f32,
}

impl Default for WeldOptions {
    fn default() -> Self {
        Self {
            position_epsilon: 1e-5,
            uv_epsilon: 1e-4,
        }
    }
}

/// Result from a weld operation including remapping information.
#[derive(Debug, Clone)]
pub struct WeldResult {
    pub mesh: MeshBuffers,
    pub remap: Vec<u32>,
    pub collapsed: usize,
}

/// Weld a mesh based on positional tolerance and recompute vertex normals.
pub fn weld_mesh(mesh: &MeshBuffers, options: WeldOptions) -> WeldResult {
    if mesh.positions.is_empty() {
        return WeldResult {
            mesh: MeshBuffers::default(),
            remap: Vec::new(),
            collapsed: 0,
        };
    }

    let vertex_count = mesh.vertex_count();
    let has_uvs = mesh.uvs.len() == vertex_count;

    let mut key_map: BTreeMap<(i64, i64, i64), Vec<usize>> = BTreeMap::new();
    let mut new_mesh = MeshBuffers::with_capacity(vertex_count, mesh.indices.len());
    let mut remap = vec![0u32; vertex_count];
    let mut accum_uv: Vec<[f32; 2]> = Vec::new();
    let mut counts: Vec<f32> = Vec::new();

    for (idx, position) in mesh.positions.iter().enumerate() {
        let key = quantize_position(*position, options.position_epsilon);
        let uv_value = if has_uvs { Some(mesh.uvs[idx]) } else { None };

        let mut matched: Option<usize> = None;
        if let Some(candidates) = key_map.get_mut(&key) {
            if has_uvs {
                if let Some(uv) = uv_value {
                    for &candidate in candidates.iter() {
                        let ref_uv = new_mesh.uvs[candidate];
                        if (ref_uv[0] - uv[0]).abs() <= options.uv_epsilon
                            && (ref_uv[1] - uv[1]).abs() <= options.uv_epsilon
                        {
                            matched = Some(candidate);
                            break;
                        }
                    }
                }
            } else {
                matched = candidates.first().copied();
            }

            if let Some(existing) = matched {
                remap[idx] = existing as u32;
                if let Some(uv) = uv_value {
                    accum_uv[existing][0] += uv[0];
                    accum_uv[existing][1] += uv[1];
                }
                counts[existing] += 1.0;
                continue;
            } else {
                let (new_index, uv_record) = append_vertex(&mut new_mesh, *position, uv_value);
                candidates.push(new_index);
                remap[idx] = new_index as u32;
                accum_uv.push(uv_record);
                counts.push(1.0);
                continue;
            }
        }

        let (new_index, uv_record) = append_vertex(&mut new_mesh, *position, uv_value);
        key_map.insert(key, vec![new_index]);
        remap[idx] = new_index as u32;
        accum_uv.push(uv_record);
        counts.push(1.0);
    }

    for (i, uv) in new_mesh.uvs.iter_mut().enumerate() {
        let count = counts[i];
        if count > 0.0 {
            uv[0] = accum_uv[i][0] / count;
            uv[1] = accum_uv[i][1] / count;
        }
    }

    let mut remapped_indices = Vec::with_capacity(mesh.indices.len());
    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let a = remap[tri[0] as usize];
        let b = remap[tri[1] as usize];
        let c = remap[tri[2] as usize];
        if a == b || b == c || a == c {
            continue;
        }
        remapped_indices.extend_from_slice(&[a, b, c]);
    }
    new_mesh.indices = remapped_indices;

    new_mesh.normals = vec![[0.0, 0.0, 0.0]; new_mesh.positions.len()];
    for tri in new_mesh.indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        let p0 = Vec3::from(new_mesh.positions[i0]);
        let p1 = Vec3::from(new_mesh.positions[i1]);
        let p2 = Vec3::from(new_mesh.positions[i2]);
        let normal = (p1 - p0).cross(p2 - p0);
        if normal.length_squared() <= 1e-12 {
            continue;
        }
        let normal = normal.normalize();
        accumulate_normal(&mut new_mesh.normals[i0], normal);
        accumulate_normal(&mut new_mesh.normals[i1], normal);
        accumulate_normal(&mut new_mesh.normals[i2], normal);
    }

    for normal in &mut new_mesh.normals {
        let vec = Vec3::from(*normal);
        let len = vec.length();
        if len > 1e-6 {
            let n = vec / len;
            *normal = [n.x, n.y, n.z];
        } else {
            *normal = [0.0, 1.0, 0.0];
        }
    }

    let collapsed = vertex_count.saturating_sub(new_mesh.vertex_count());

    WeldResult {
        mesh: new_mesh,
        remap,
        collapsed,
    }
}

fn append_vertex(
    mesh: &mut MeshBuffers,
    position: [f32; 3],
    uv: Option<[f32; 2]>,
) -> (usize, [f32; 2]) {
    let index = mesh.positions.len();
    mesh.positions.push(position);
    let uv_value = uv.unwrap_or([0.0, 0.0]);
    mesh.uvs.push(uv_value);
    (index, uv_value)
}

fn quantize_position(position: [f32; 3], eps: f32) -> (i64, i64, i64) {
    (
        quantize_scalar(position[0], eps),
        quantize_scalar(position[1], eps),
        quantize_scalar(position[2], eps),
    )
}

fn quantize_scalar(value: f32, eps: f32) -> i64 {
    (value / eps).round() as i64
}

fn accumulate_normal(target: &mut [f32; 3], normal: Vec3) {
    target[0] += normal.x;
    target[1] += normal.y;
    target[2] += normal.z;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weld_deduplicates_vertices() {
        let mesh = MeshBuffers {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0 + 1e-6, 0.0, 0.0],
            ],
            normals: vec![],
            uvs: vec![],
            tangents: vec![],
            indices: vec![0, 1, 2, 3, 2, 1],
        };
        let result = weld_mesh(&mesh, WeldOptions::default());
        assert_eq!(result.mesh.vertex_count(), 3);
        assert!(result.collapsed >= 1);
    }

    #[test]
    fn weld_respects_uv_epsilon() {
        let mesh = MeshBuffers {
            positions: vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            normals: vec![],
            uvs: vec![[0.0, 0.0], [1.0, 0.0]],
            tangents: vec![],
            indices: vec![0, 1, 1],
        };
        let result = weld_mesh(&mesh, WeldOptions::default());
        assert_eq!(result.mesh.vertex_count(), 2);
    }
}

//! Wavefront OBJ reader for Workstream F (F4).
//!
//! Minimal streaming parser for common OBJ constructs (v, vt, vn, f, usemtl, mtllib).
//! Triangulates polygon faces with a fan.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::RenderError;
use crate::geometry::MeshBuffers;

/// Material definition parsed from an accompanying MTL file.
#[derive(Debug, Clone, Default)]
pub struct ObjMaterial {
    pub name: String,
    pub diffuse_color: [f32; 3],
    pub specular_color: [f32; 3],
    pub ambient_color: [f32; 3],
    pub diffuse_texture: Option<String>,
}

/// Result of importing an OBJ file.
#[derive(Debug, Clone)]
pub struct ObjImport {
    pub mesh: MeshBuffers,
    pub materials: Vec<ObjMaterial>,
    pub groups: HashMap<String, Vec<u32>>, // material name -> triangle ordinals
    pub g_groups: HashMap<String, Vec<u32>>, // 'g' group name -> triangle ordinals
    pub o_groups: HashMap<String, Vec<u32>>, // 'o' object name -> triangle ordinals
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct VertexKey {
    vi: i32,
    vti: i32,
    vni: i32,
}

fn parse_vertex_triplet(tok: &str) -> VertexKey {
    let mut parts = tok.split('/')
        .map(|s| if s.is_empty() { None } else { s.parse::<i32>().ok() });
    let vi = parts.next().flatten().unwrap_or(0);
    let vti = parts.next().flatten().unwrap_or(0);
    let vni = parts.next().flatten().unwrap_or(0);
    VertexKey { vi, vti, vni }
}

fn index_fix(idx: i32, len: usize) -> usize {
    if idx > 0 { (idx as usize) - 1 } else { (len as i32 + idx) as usize }
}

fn parse_mtl_file(path: &Path) -> Result<HashMap<String, ObjMaterial>, RenderError> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(HashMap::new()), // lenient: missing MTL is not fatal
    };
    let reader = BufReader::new(file);

    let mut materials: HashMap<String, ObjMaterial> = HashMap::new();
    let mut current: Option<ObjMaterial> = None;

    for line in reader.lines() {
        let line = line.map_err(|e| RenderError::io(format!("{}", e)))?;
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') { continue; }
        let mut it = s.split_whitespace();
        let tag = it.next().unwrap_or("");
        match tag {
            "newmtl" => {
                if let Some(prev) = current.take() {
                    materials.insert(prev.name.clone(), prev);
                }
                let name = it.next().unwrap_or("").to_string();
                current = Some(ObjMaterial { name, ..Default::default() });
            }
            "Kd" => {
                if let Some(m) = current.as_mut() {
                    let r = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.8);
                    let g = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.8);
                    let b = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.8);
                    m.diffuse_color = [r, g, b];
                }
            }
            "Ka" => {
                if let Some(m) = current.as_mut() {
                    let r = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.0);
                    let g = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.0);
                    let b = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.0);
                    m.ambient_color = [r, g, b];
                }
            }
            "Ks" => {
                if let Some(m) = current.as_mut() {
                    let r = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.0);
                    let g = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.0);
                    let b = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.0);
                    m.specular_color = [r, g, b];
                }
            }
            "map_Kd" => {
                if let Some(m) = current.as_mut() {
                    if let Some(tex) = it.next() {
                        m.diffuse_texture = Some(tex.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    if let Some(prev) = current.take() {
        materials.insert(prev.name.clone(), prev);
    }

    Ok(materials)
}

/// Parse the OBJ file and emit a merged vertex stream with indices.
pub fn import_obj<P: AsRef<Path>>(path: P) -> Result<ObjImport, RenderError> {
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);

    let mut pos: Vec<[f32; 3]> = Vec::new();
    let mut tex: Vec<[f32; 2]> = Vec::new();
    let mut nor: Vec<[f32; 3]> = Vec::new();

    let mut mesh = MeshBuffers::new();
    let mut map: HashMap<VertexKey, u32> = HashMap::new();
    let mut next_index: u32 = 0;

    let mut current_mtl: Option<String> = None;
    let mut current_g: Option<String> = None;
    let mut current_o: Option<String> = None;
    let mut groups: HashMap<String, Vec<u32>> = HashMap::new();
    let mut g_groups: HashMap<String, Vec<u32>> = HashMap::new();
    let mut o_groups: HashMap<String, Vec<u32>> = HashMap::new();
    let mut tri_count: u32 = 0;
    let mut materials: Vec<ObjMaterial> = Vec::new();
    let base_dir = path.as_ref().parent().map(|p| p.to_path_buf());

    let mut line_no: usize = 0;
    for line in reader.lines() {
        line_no += 1;
        let line = line.map_err(|e| RenderError::io(format!("{}", e)))?;
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        let mut it = s.split_whitespace();
        let tag = it.next().unwrap_or("");
        match tag {
            "v" => {
                let x = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.0);
                let y = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.0);
                let z = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.0);
                pos.push([x, y, z]);
            }
            "vt" => {
                let u = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.0);
                let v = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.0);
                tex.push([u, v]);
            }
            "vn" => {
                let x = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.0);
                let y = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.0);
                let z = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.0);
                nor.push([x, y, z]);
            }
            "usemtl" => {
                if let Some(name) = it.next() {
                    current_mtl = Some(name.to_string());
                    groups.entry(name.to_string()).or_default();
                }
            }
            "g" => {
                if let Some(name) = it.next() {
                    current_g = Some(name.to_string());
                    g_groups.entry(name.to_string()).or_default();
                } else {
                    current_g = None;
                }
            }
            "o" => {
                if let Some(name) = it.next() {
                    current_o = Some(name.to_string());
                    o_groups.entry(name.to_string()).or_default();
                } else {
                    current_o = None;
                }
            }
            // mtllib: parse material library (minimal attributes)
            "mtllib" => {
                if let Some(fname) = it.next() {
                    if let Some(ref dir) = base_dir {
                        let mtl_path = dir.join(fname);
                        if let Ok(map) = parse_mtl_file(&mtl_path) {
                            // Keep stable order by inserting into materials vec
                            for (_k, v) in map {
                                materials.push(v);
                            }
                        }
                    }
                }
            }
            "f" => {
                let verts: Vec<VertexKey> = it.map(parse_vertex_triplet).collect();
                if verts.len() < 3 {
                    return Err(RenderError::Render(format!(
                        "OBJ parse error at line {}: face has fewer than 3 vertices",
                        line_no
                    )));
                }
                if verts.iter().any(|vk| vk.vi == 0) {
                    return Err(RenderError::Render(format!(
                        "OBJ parse error at line {}: face vertex missing position index",
                        line_no
                    )));
                }

                // Triangulate using fan
                for t in 1..(verts.len() - 1) {
                    let tri = [verts[0], verts[t], verts[t + 1]];
                    for vk in tri.iter() {
                        let idx = if let Some(&i) = map.get(vk) { i } else {
                            // create new vertex
                            let vi = index_fix(vk.vi, pos.len());
                            if vi >= pos.len() {
                                return Err(RenderError::Render(format!(
                                    "OBJ parse error at line {}: position index {} out of bounds (1..={})",
                                    line_no, vk.vi, pos.len()
                                )));
                            }
                            let p = pos[vi];
                            mesh.positions.push(p);

                            if vk.vti != 0 {
                                let vti = index_fix(vk.vti, tex.len());
                                if vti >= tex.len() {
                                    return Err(RenderError::Render(format!(
                                        "OBJ parse error at line {}: texcoord index {} out of bounds (1..={})",
                                        line_no, vk.vti, tex.len()
                                    )));
                                }
                                if mesh.uvs.len() < mesh.positions.len() - 1 { mesh.uvs.resize(mesh.positions.len() - 1, [0.0, 0.0]); }
                                mesh.uvs.push(tex[vti]);
                            } else if !mesh.uvs.is_empty() {
                                mesh.uvs.push([0.0, 0.0]);
                            }

                            if vk.vni != 0 {
                                let vni = index_fix(vk.vni, nor.len());
                                if vni >= nor.len() {
                                    return Err(RenderError::Render(format!(
                                        "OBJ parse error at line {}: normal index {} out of bounds (1..={})",
                                        line_no, vk.vni, nor.len()
                                    )));
                                }
                                if mesh.normals.len() < mesh.positions.len() - 1 { mesh.normals.resize(mesh.positions.len() - 1, [0.0, 0.0, 0.0]); }
                                mesh.normals.push(nor[vni]);
                            } else if !mesh.normals.is_empty() {
                                mesh.normals.push([0.0, 0.0, 1.0]);
                            }

                            map.insert(*vk, next_index);
                            let ret = next_index;
                            next_index += 1;
                            ret
                        };
                        mesh.indices.push(idx);
                    }

                    // track triangle ordinal per active groups (0-based)
                    if let Some(ref mname) = current_mtl {
                        let tri_list = groups.entry(mname.clone()).or_default();
                        tri_list.push(tri_count);
                    }
                    if let Some(ref gname) = current_g {
                        let tri_list = g_groups.entry(gname.clone()).or_default();
                        tri_list.push(tri_count);
                    }
                    if let Some(ref oname) = current_o {
                        let tri_list = o_groups.entry(oname.clone()).or_default();
                        tri_list.push(tri_count);
                    }
                    tri_count += 1;
                }
            }
            _ => {}
        }
    }

    Ok(ObjImport { mesh, materials, groups, g_groups, o_groups })
}

// ---------------- PyO3 bridge -----------------

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use crate::geometry::mesh_to_python;

#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn io_import_obj_py(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let result = import_obj(path).map_err(|e| e.to_py_err())?;

    let out = pyo3::types::PyDict::new_bound(py);
    let mesh_obj = mesh_to_python(py, &result.mesh)?;
    out.set_item("mesh", mesh_obj)?;

    // materials -> list[dict]
    let materials = pyo3::types::PyList::empty_bound(py);
    for m in &result.materials {
        let md = pyo3::types::PyDict::new_bound(py);
        md.set_item("name", m.name.as_str())?;
        md.set_item("diffuse_color", (m.diffuse_color[0], m.diffuse_color[1], m.diffuse_color[2]))?;
        md.set_item("ambient_color", (m.ambient_color[0], m.ambient_color[1], m.ambient_color[2]))?;
        md.set_item("specular_color", (m.specular_color[0], m.specular_color[1], m.specular_color[2]))?;
        if let Some(tex) = &m.diffuse_texture {
            md.set_item("diffuse_texture", tex.as_str())?;
        } else {
            md.set_item("diffuse_texture", py.None())?;
        }
        materials.append(md)?;
    }
    out.set_item("materials", materials)?;

    // groups -> dict[str, np.ndarray]
    let groups = pyo3::types::PyDict::new_bound(py);
    for (name, tris) in &result.groups {
        let arr = numpy::PyArray1::<u32>::from_vec_bound(py, tris.clone());
        groups.set_item(name.as_str(), arr)?;
    }
    out.set_item("groups", &groups)?; // backward-compat: material groups
    out.set_item("material_groups", &groups)?;

    // g_groups -> dict[str, np.ndarray]
    let g_groups = pyo3::types::PyDict::new_bound(py);
    for (name, tris) in &result.g_groups {
        let arr = numpy::PyArray1::<u32>::from_vec_bound(py, tris.clone());
        g_groups.set_item(name.as_str(), arr)?;
    }
    out.set_item("g_groups", g_groups)?;

    // o_groups -> dict[str, np.ndarray]
    let o_groups = pyo3::types::PyDict::new_bound(py);
    for (name, tris) in &result.o_groups {
        let arr = numpy::PyArray1::<u32>::from_vec_bound(py, tris.clone());
        o_groups.set_item(name.as_str(), arr)?;
    }
    out.set_item("o_groups", o_groups)?;

    Ok(out.into_py(py))
}

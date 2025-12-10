//! Wavefront OBJ writer for Workstream F (F5).
//!
//! Emits a minimal OBJ with v/vt/vn/f records and optional material library support.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::core::error::RenderError;
use crate::io::obj_read::ObjMaterial;

#[cfg(feature = "extension-module")]
use crate::geometry::mesh_from_python;

#[cfg_attr(feature = "extension-module", allow(unused_imports))]
use crate::geometry::MeshBuffers;

fn export_mtl_to_path(path: &Path, materials: &[ObjMaterial]) -> Result<(), RenderError> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    for m in materials {
        writeln!(w, "newmtl {}", m.name).map_err(|e| RenderError::io(format!("{}", e)))?;
        let kd = m.diffuse_color;
        let ka = m.ambient_color;
        let ks = m.specular_color;
        writeln!(w, "Kd {} {} {}", kd[0], kd[1], kd[2])
            .map_err(|e| RenderError::io(format!("{}", e)))?;
        writeln!(w, "Ka {} {} {}", ka[0], ka[1], ka[2])
            .map_err(|e| RenderError::io(format!("{}", e)))?;
        writeln!(w, "Ks {} {} {}", ks[0], ks[1], ks[2])
            .map_err(|e| RenderError::io(format!("{}", e)))?;
        if let Some(tex) = &m.diffuse_texture {
            writeln!(w, "map_Kd {}", tex).map_err(|e| RenderError::io(format!("{}", e)))?;
        }
        writeln!(w).map_err(|e| RenderError::io(format!("{}", e)))?;
    }
    Ok(())
}

pub fn export_obj_to_path<P: AsRef<Path>>(path: P, mesh: &MeshBuffers) -> Result<(), RenderError> {
    export_obj_with_metadata(path, mesh, None, None, None, None)
}

pub fn export_obj_with_metadata<P: AsRef<Path>>(
    path: P,
    mesh: &MeshBuffers,
    materials: Option<&[ObjMaterial]>,
    material_groups: Option<&HashMap<String, Vec<u32>>>,
    g_groups: Option<&HashMap<String, Vec<u32>>>,
    o_groups: Option<&HashMap<String, Vec<u32>>>,
) -> Result<(), RenderError> {
    let obj_path = path.as_ref();
    let file = File::create(obj_path)?;
    let mut w = BufWriter::new(file);

    // Optional MTL
    let mut wrote_mtllib = false;
    if let Some(mats) = materials {
        if !mats.is_empty() {
            let mtl_filename = obj_path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| format!("{}.mtl", s))
                .unwrap_or_else(|| "materials.mtl".to_string());
            let mtl_path: PathBuf = obj_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(&mtl_filename);
            export_mtl_to_path(&mtl_path, mats)?;
            writeln!(w, "mtllib {}", mtl_filename)
                .map_err(|e| RenderError::io(format!("{}", e)))?;
            wrote_mtllib = true;
        }
    }

    // Write vertices
    for p in &mesh.positions {
        writeln!(w, "v {} {} {}", p[0], p[1], p[2])
            .map_err(|e| RenderError::io(format!("{}", e)))?;
    }

    // Write texture coordinates (optional)
    if !mesh.uvs.is_empty() {
        for t in &mesh.uvs {
            writeln!(w, "vt {} {}", t[0], t[1]).map_err(|e| RenderError::io(format!("{}", e)))?;
        }
    }

    // Write normals (optional)
    if !mesh.normals.is_empty() {
        for n in &mesh.normals {
            writeln!(w, "vn {} {} {}", n[0], n[1], n[2])
                .map_err(|e| RenderError::io(format!("{}", e)))?;
        }
    }

    // Faces (assume triangle list)
    let has_uv = !mesh.uvs.is_empty();
    let has_n = !mesh.normals.is_empty();

    // Build per-triangle metadata maps from provided groups
    let tri_count = mesh.indices.len() / 3;
    let mut tri_mtl: Vec<Option<String>> = vec![None; tri_count];
    if let Some(mg) = material_groups {
        for (name, tris) in mg.iter() {
            for &t in tris {
                let ti = t as usize;
                if ti >= tri_count {
                    return Err(RenderError::Render(format!(
                        "export: material group '{}' references triangle {} (out of {})",
                        name, ti, tri_count
                    )));
                }
                if tri_mtl[ti].is_none() {
                    tri_mtl[ti] = Some(name.clone());
                }
            }
        }
    }
    let mut tri_g: Vec<Option<String>> = vec![None; tri_count];
    if let Some(gg) = g_groups {
        for (name, tris) in gg.iter() {
            for &t in tris {
                let ti = t as usize;
                if ti >= tri_count {
                    return Err(RenderError::Render(format!(
                        "export: g group '{}' references triangle {} (out of {})",
                        name, ti, tri_count
                    )));
                }
                if tri_g[ti].is_none() {
                    tri_g[ti] = Some(name.clone());
                }
            }
        }
    }
    let mut tri_o: Vec<Option<String>> = vec![None; tri_count];
    if let Some(og) = o_groups {
        for (name, tris) in og.iter() {
            for &t in tris {
                let ti = t as usize;
                if ti >= tri_count {
                    return Err(RenderError::Render(format!(
                        "export: o group '{}' references triangle {} (out of {})",
                        name, ti, tri_count
                    )));
                }
                if tri_o[ti].is_none() {
                    tri_o[ti] = Some(name.clone());
                }
            }
        }
    }

    let mut cur_mtl: Option<String> = None;
    let mut cur_g: Option<String> = None;
    let mut cur_o: Option<String> = None;

    for (t_idx, tri) in mesh.indices.chunks_exact(3).enumerate() {
        // Emit group changes before the face if needed
        if let Some(ref name) = tri_g[t_idx] {
            if cur_g.as_deref() != Some(name.as_str()) {
                writeln!(w, "g {}", name).map_err(|e| RenderError::io(format!("{}", e)))?;
                cur_g = Some(name.clone());
            }
        }
        if let Some(ref name) = tri_o[t_idx] {
            if cur_o.as_deref() != Some(name.as_str()) {
                writeln!(w, "o {}", name).map_err(|e| RenderError::io(format!("{}", e)))?;
                cur_o = Some(name.clone());
            }
        }
        if let Some(ref name) = tri_mtl[t_idx] {
            if cur_mtl.as_deref() != Some(name.as_str()) {
                // If we didn't write mtllib yet but we have a material usage, emit a warning in Render error? Keep lenient: just write usemtl.
                if !wrote_mtllib && materials.is_some() {
                    // no-op: mtllib might not be needed if materials were empty
                }
                writeln!(w, "usemtl {}", name).map_err(|e| RenderError::io(format!("{}", e)))?;
                cur_mtl = Some(name.clone());
            }
        }

        let to_one = |i: u32| (i + 1) as usize; // 1-based
        let v0 = to_one(tri[0]);
        let v1 = to_one(tri[1]);
        let v2 = to_one(tri[2]);

        let face = if has_uv && has_n {
            format!(
                "f {}/{}/{} {}/{}/{} {}/{}/{}",
                v0, v0, v0, v1, v1, v1, v2, v2, v2
            )
        } else if has_uv {
            format!("f {}/{} {}/{} {}/{}", v0, v0, v1, v1, v2, v2)
        } else if has_n {
            format!("f {}//{} {}//{} {}//{}", v0, v0, v1, v1, v2, v2)
        } else {
            format!("f {} {} {}", v0, v1, v2)
        };

        writeln!(w, "{}", face).map_err(|e| RenderError::io(format!("{}", e)))?;
    }

    Ok(())
}

// ---------------- PyO3 bridge -----------------

#[cfg(feature = "extension-module")]
use numpy::PyReadonlyArray1;
#[cfg(feature = "extension-module")]
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};

#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn io_export_obj_py(
    path: &str,
    mesh: &Bound<'_, PyDict>,
    materials: Option<&Bound<'_, PyList>>,
    material_groups: Option<&Bound<'_, PyDict>>,
    g_groups: Option<&Bound<'_, PyDict>>,
    o_groups: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let mesh_buf = mesh_from_python(mesh)?;

    // Parse materials if provided
    let mats_rust: Option<Vec<ObjMaterial>> = if let Some(list) = materials {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            let d = item.downcast::<PyDict>()?;
            let name: String = if let Some(v) = d.get_item("name")? {
                v.extract()?
            } else {
                String::new()
            };
            let kd: (f32, f32, f32) = if let Some(v) = d.get_item("diffuse_color")? {
                v.extract()?
            } else {
                (0.8, 0.8, 0.8)
            };
            let ka: (f32, f32, f32) = if let Some(v) = d.get_item("ambient_color")? {
                v.extract()?
            } else {
                (0.0, 0.0, 0.0)
            };
            let ks: (f32, f32, f32) = if let Some(v) = d.get_item("specular_color")? {
                v.extract()?
            } else {
                (0.0, 0.0, 0.0)
            };
            let dt: Option<String> = if let Some(v) = d.get_item("diffuse_texture")? {
                v.extract::<Option<String>>()?
            } else {
                None
            };
            out.push(ObjMaterial {
                name,
                diffuse_color: [kd.0, kd.1, kd.2],
                ambient_color: [ka.0, ka.1, ka.2],
                specular_color: [ks.0, ks.1, ks.2],
                diffuse_texture: dt,
            });
        }
        Some(out)
    } else {
        None
    };

    // Helper to parse dict[str] -> Vec<u32>
    fn parse_groups_dict<'py>(
        _py: Python<'py>,
        d: &Bound<'py, PyDict>,
    ) -> PyResult<HashMap<String, Vec<u32>>> {
        let mut map: HashMap<String, Vec<u32>> = HashMap::new();
        for (k, v) in d.iter() {
            let name: String = k.extract()?;
            let arr: PyReadonlyArray1<u32> = v.extract()?;
            map.insert(name, arr.as_slice()?.to_vec());
        }
        Ok(map)
    }

    let mg_rust = if let Some(d) = material_groups {
        Some(parse_groups_dict(d.py(), d)?)
    } else {
        None
    };
    let gg_rust = if let Some(d) = g_groups {
        Some(parse_groups_dict(d.py(), d)?)
    } else {
        None
    };
    let og_rust = if let Some(d) = o_groups {
        Some(parse_groups_dict(d.py(), d)?)
    } else {
        None
    };

    export_obj_with_metadata(
        path,
        &mesh_buf,
        mats_rust.as_deref(),
        mg_rust.as_ref(),
        gg_rust.as_ref(),
        og_rust.as_ref(),
    )
    .map_err(|e| e.to_py_err())
}

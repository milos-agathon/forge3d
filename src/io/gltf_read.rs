//! Minimal glTF 2.0 mesh importer.
//!
//! Loads mesh primitives as [`MeshBuffers`]. Supports both embedded
//! (data URI) and external buffer references via `gltf::import`.

use crate::core::error::RenderError;
use crate::geometry::MeshBuffers;

/// Imported glTF mesh plus per-primitive material metadata.
pub struct GltfImport {
    pub mesh: MeshBuffers,
    pub materials: Vec<GltfMaterialSummary>,
    pub primitive_materials: Vec<Option<usize>>,
}

/// Minimal PBR material fields preserved from glTF.
pub struct GltfMaterialSummary {
    pub index: usize,
    pub name: Option<String>,
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub alpha_mode: &'static str,
    pub alpha_cutoff: Option<f32>,
    pub double_sided: bool,
    pub emissive_factor: [f32; 3],
    pub has_base_color_texture: bool,
    pub has_metallic_roughness_texture: bool,
    pub has_normal_texture: bool,
    pub has_occlusion_texture: bool,
    pub has_emissive_texture: bool,
    pub unlit: bool,
    pub base_color_texture: Option<GltfTextureSlot>,
    pub metallic_roughness_texture: Option<GltfTextureSlot>,
    pub normal_texture: Option<GltfTextureSlot>,
    pub occlusion_texture: Option<GltfTextureSlot>,
    pub emissive_texture: Option<GltfTextureSlot>,
}

/// Texture slot metadata for a material map.
pub struct GltfTextureSlot {
    pub texture_index: usize,
    pub image_index: usize,
    pub tex_coord: u32,
    pub transform: Option<GltfTextureTransform>,
}

/// KHR_texture_transform fields for material texture coordinates.
pub struct GltfTextureTransform {
    pub offset: [f32; 2],
    pub rotation: f32,
    pub scale: [f32; 2],
    pub tex_coord: Option<u32>,
}

/// Import a glTF file and extract mesh primitives.
///
/// # Arguments
/// - `path`: Path to a `.gltf` or `.glb` file.
///
/// # Returns
/// A [`MeshBuffers`] containing positions, normals, UVs, tangents, and indices.
///
/// # Errors
/// Returns an error if the file cannot be read or contains no mesh primitives.
pub fn import_gltf_to_mesh(path: &str) -> Result<MeshBuffers, RenderError> {
    Ok(import_gltf_with_metadata(path)?.mesh)
}

/// Import a glTF file and preserve mesh/material primitive bindings.
pub fn import_gltf_with_metadata(path: &str) -> Result<GltfImport, RenderError> {
    let (doc, buffers, _images) = gltf::import(path).map_err(|e| RenderError::io(e.to_string()))?;

    let mut out = MeshBuffers::new();
    let mut primitive_materials = Vec::new();
    let identity = identity_matrix();
    let mut read_any_scene_node = false;

    if let Some(scene) = doc.default_scene().or_else(|| doc.scenes().next()) {
        for node in scene.nodes() {
            read_any_scene_node |= read_node(
                node,
                &identity,
                &buffers,
                &mut out,
                &mut primitive_materials,
            );
        }
    }

    if !read_any_scene_node {
        for prim in doc.meshes().flat_map(|mesh| mesh.primitives()) {
            read_primitive(
                prim,
                &identity,
                &buffers,
                &mut out,
                &mut primitive_materials,
            );
        }
    }

    if out.positions.is_empty() || out.indices.is_empty() {
        return Err(RenderError::Render(
            "glTF contains no mesh primitives".into(),
        ));
    }

    Ok(GltfImport {
        mesh: out,
        materials: collect_materials(&doc),
        primitive_materials,
    })
}

fn collect_materials(doc: &gltf::Document) -> Vec<GltfMaterialSummary> {
    doc.materials()
        .filter_map(|material| {
            let index = material.index()?;
            let pbr = material.pbr_metallic_roughness();
            let base_color_texture = pbr.base_color_texture().map(texture_slot_from_info);
            let metallic_roughness_texture =
                pbr.metallic_roughness_texture().map(texture_slot_from_info);
            let normal_texture = material.normal_texture().map(|info| GltfTextureSlot {
                texture_index: info.texture().index(),
                image_index: info.texture().source().index(),
                tex_coord: info.tex_coord(),
                transform: None,
            });
            let occlusion_texture = material.occlusion_texture().map(|info| GltfTextureSlot {
                texture_index: info.texture().index(),
                image_index: info.texture().source().index(),
                tex_coord: info.tex_coord(),
                transform: None,
            });
            let emissive_texture = material.emissive_texture().map(texture_slot_from_info);
            Some(GltfMaterialSummary {
                index,
                name: material.name().map(str::to_owned),
                base_color_factor: pbr.base_color_factor(),
                metallic_factor: pbr.metallic_factor(),
                roughness_factor: pbr.roughness_factor(),
                alpha_mode: match material.alpha_mode() {
                    gltf::material::AlphaMode::Opaque => "OPAQUE",
                    gltf::material::AlphaMode::Mask => "MASK",
                    gltf::material::AlphaMode::Blend => "BLEND",
                },
                alpha_cutoff: material.alpha_cutoff(),
                double_sided: material.double_sided(),
                emissive_factor: material.emissive_factor(),
                has_base_color_texture: base_color_texture.is_some(),
                has_metallic_roughness_texture: metallic_roughness_texture.is_some(),
                has_normal_texture: normal_texture.is_some(),
                has_occlusion_texture: occlusion_texture.is_some(),
                has_emissive_texture: emissive_texture.is_some(),
                unlit: material.unlit(),
                base_color_texture,
                metallic_roughness_texture,
                normal_texture,
                occlusion_texture,
                emissive_texture,
            })
        })
        .collect()
}

fn texture_slot_from_info(info: gltf::texture::Info<'_>) -> GltfTextureSlot {
    let texture = info.texture();
    GltfTextureSlot {
        texture_index: texture.index(),
        image_index: texture.source().index(),
        tex_coord: info.tex_coord(),
        transform: info
            .texture_transform()
            .map(|transform| GltfTextureTransform {
                offset: transform.offset(),
                rotation: transform.rotation(),
                scale: transform.scale(),
                tex_coord: transform.tex_coord(),
            }),
    }
}

fn read_node(
    node: gltf::Node<'_>,
    parent: &[[f32; 4]; 4],
    buffers: &[gltf::buffer::Data],
    out: &mut MeshBuffers,
    primitive_materials: &mut Vec<Option<usize>>,
) -> bool {
    let transform = multiply_matrices(parent, &node.transform().matrix());
    let mut read_any = false;
    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            read_any |= read_primitive(prim, &transform, buffers, out, primitive_materials);
        }
    }
    for child in node.children() {
        read_any |= read_node(child, &transform, buffers, out, primitive_materials);
    }
    read_any
}

fn read_primitive(
    prim: gltf::Primitive<'_>,
    transform: &[[f32; 4]; 4],
    buffers: &[gltf::buffer::Data],
    out: &mut MeshBuffers,
    primitive_materials: &mut Vec<Option<usize>>,
) -> bool {
    if prim.mode() != gltf::mesh::Mode::Triangles {
        return false;
    }
    let material_index = prim.material().index();
    let reader = prim.reader(|buffer| buffers.get(buffer.index()).map(|d| d.0.as_slice()));
    let mut part = MeshBuffers::new();

    read_positions(&reader, &mut part);
    if part.positions.is_empty() {
        return false;
    }
    read_normals(&reader, &mut part);
    read_uvs(&reader, &mut part);
    read_tangents(&reader, &mut part);
    read_or_generate_indices(&reader, &mut part);
    apply_transform(&mut part, transform);
    append_mesh(out, part);
    primitive_materials.push(material_index);
    true
}

/// Extract position data from a primitive reader.
fn read_positions<'a, 's, F>(reader: &gltf::mesh::Reader<'a, 's, F>, out: &mut MeshBuffers)
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    if let Some(positions) = reader.read_positions() {
        out.positions = positions.collect();
    }
}

/// Extract normal data from a primitive reader.
fn read_normals<'a, 's, F>(reader: &gltf::mesh::Reader<'a, 's, F>, out: &mut MeshBuffers)
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    if let Some(normals) = reader.read_normals() {
        out.normals = normals.collect();
    }
}

/// Extract UV coordinates from a primitive reader.
fn read_uvs<'a, 's, F>(reader: &gltf::mesh::Reader<'a, 's, F>, out: &mut MeshBuffers)
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    if let Some(tex0) = reader.read_tex_coords(0) {
        out.uvs = tex0.into_f32().collect();
    }
}

/// Extract tangent coordinates from a primitive reader.
fn read_tangents<'a, 's, F>(reader: &gltf::mesh::Reader<'a, 's, F>, out: &mut MeshBuffers)
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    if let Some(tangents) = reader.read_tangents() {
        out.tangents = tangents.collect();
    }
}

/// Extract indices or generate trivial indices if not present.
fn read_or_generate_indices<'a, 's, F>(
    reader: &gltf::mesh::Reader<'a, 's, F>,
    out: &mut MeshBuffers,
) where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    if let Some(indices) = reader.read_indices() {
        out.indices = indices.into_u32().collect();
    } else {
        let vertex_count = out.positions.len();
        if vertex_count.is_multiple_of(3) {
            out.indices = (0u32..(vertex_count as u32)).collect();
        }
    }
}

fn identity_matrix() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn multiply_matrices(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = (0..4).map(|k| a[k][row] * b[col][k]).sum();
        }
    }
    out
}

fn apply_transform(mesh: &mut MeshBuffers, matrix: &[[f32; 4]; 4]) {
    for position in &mut mesh.positions {
        *position = transform_point(*position, matrix);
    }
    for normal in &mut mesh.normals {
        *normal = normalize(transform_vector(*normal, matrix));
    }
    for tangent in &mut mesh.tangents {
        let xyz = normalize(transform_vector(
            [tangent[0], tangent[1], tangent[2]],
            matrix,
        ));
        tangent[0] = xyz[0];
        tangent[1] = xyz[1];
        tangent[2] = xyz[2];
    }
}

fn transform_point(p: [f32; 3], m: &[[f32; 4]; 4]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

fn transform_vector(v: [f32; 3], m: &[[f32; 4]; 4]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        v
    }
}

fn append_mesh(out: &mut MeshBuffers, part: MeshBuffers) {
    let base = out.positions.len() as u32;
    let old_vertex_count = out.positions.len();
    let part_vertex_count = part.positions.len();

    if part.normals.len() == part_vertex_count && out.normals.len() == old_vertex_count {
        out.normals.extend(part.normals);
    } else {
        out.normals.clear();
    }
    if part.uvs.len() == part_vertex_count && out.uvs.len() == old_vertex_count {
        out.uvs.extend(part.uvs);
    } else {
        out.uvs.clear();
    }
    if part.tangents.len() == part_vertex_count && out.tangents.len() == old_vertex_count {
        out.tangents.extend(part.tangents);
    } else {
        out.tangents.clear();
    }

    out.positions.extend(part.positions);
    out.indices
        .extend(part.indices.into_iter().map(|index| index + base));
}

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn io_import_gltf_py(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let mesh = import_gltf_to_mesh(path).map_err(|e| e.to_py_err())?;
    crate::geometry::mesh_to_python(py, &mesh)
}

#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn io_import_gltf_with_materials_py(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let result = import_gltf_with_metadata(path).map_err(|e| e.to_py_err())?;
    let out = pyo3::types::PyDict::new_bound(py);
    out.set_item("mesh", crate::geometry::mesh_to_python(py, &result.mesh)?)?;

    let materials = pyo3::types::PyList::empty_bound(py);
    for material in &result.materials {
        materials.append(material_to_python(py, material)?)?;
    }
    out.set_item("materials", materials)?;

    let primitive_materials = pyo3::types::PyList::empty_bound(py);
    for material_index in &result.primitive_materials {
        match material_index {
            Some(index) => primitive_materials.append(*index)?,
            None => primitive_materials.append(py.None())?,
        }
    }
    out.set_item("primitive_materials", primitive_materials)?;
    Ok(out.into_py(py))
}

#[cfg(feature = "extension-module")]
fn material_to_python<'py>(
    py: Python<'py>,
    material: &GltfMaterialSummary,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let out = pyo3::types::PyDict::new_bound(py);
    out.set_item("index", material.index)?;
    match &material.name {
        Some(name) => out.set_item("name", name)?,
        None => out.set_item("name", py.None())?,
    }
    out.set_item(
        "base_color_factor",
        (
            material.base_color_factor[0],
            material.base_color_factor[1],
            material.base_color_factor[2],
            material.base_color_factor[3],
        ),
    )?;
    out.set_item("metallic_factor", material.metallic_factor)?;
    out.set_item("roughness_factor", material.roughness_factor)?;
    out.set_item("alpha_mode", material.alpha_mode)?;
    match material.alpha_cutoff {
        Some(cutoff) => out.set_item("alpha_cutoff", cutoff)?,
        None => out.set_item("alpha_cutoff", py.None())?,
    }
    out.set_item("double_sided", material.double_sided)?;
    out.set_item(
        "emissive_factor",
        (
            material.emissive_factor[0],
            material.emissive_factor[1],
            material.emissive_factor[2],
        ),
    )?;
    out.set_item("has_base_color_texture", material.has_base_color_texture)?;
    out.set_item(
        "has_metallic_roughness_texture",
        material.has_metallic_roughness_texture,
    )?;
    out.set_item("has_normal_texture", material.has_normal_texture)?;
    out.set_item("has_occlusion_texture", material.has_occlusion_texture)?;
    out.set_item("has_emissive_texture", material.has_emissive_texture)?;
    out.set_item("unlit", material.unlit)?;
    out.set_item(
        "base_color_texture",
        texture_slot_to_python(py, material.base_color_texture.as_ref())?,
    )?;
    out.set_item(
        "metallic_roughness_texture",
        texture_slot_to_python(py, material.metallic_roughness_texture.as_ref())?,
    )?;
    out.set_item(
        "normal_texture",
        texture_slot_to_python(py, material.normal_texture.as_ref())?,
    )?;
    out.set_item(
        "occlusion_texture",
        texture_slot_to_python(py, material.occlusion_texture.as_ref())?,
    )?;
    out.set_item(
        "emissive_texture",
        texture_slot_to_python(py, material.emissive_texture.as_ref())?,
    )?;
    Ok(out)
}

#[cfg(feature = "extension-module")]
fn texture_slot_to_python<'py>(
    py: Python<'py>,
    slot: Option<&GltfTextureSlot>,
) -> PyResult<PyObject> {
    let Some(slot) = slot else {
        return Ok(py.None());
    };
    let out = pyo3::types::PyDict::new_bound(py);
    out.set_item("texture_index", slot.texture_index)?;
    out.set_item("image_index", slot.image_index)?;
    out.set_item("tex_coord", slot.tex_coord)?;
    match &slot.transform {
        Some(transform) => {
            let td = pyo3::types::PyDict::new_bound(py);
            td.set_item("offset", (transform.offset[0], transform.offset[1]))?;
            td.set_item("rotation", transform.rotation)?;
            td.set_item("scale", (transform.scale[0], transform.scale[1]))?;
            match transform.tex_coord {
                Some(tex_coord) => td.set_item("tex_coord", tex_coord)?,
                None => td.set_item("tex_coord", py.None())?,
            }
            out.set_item("transform", td)?;
        }
        None => out.set_item("transform", py.None())?,
    }
    Ok(out.into_py(py))
}

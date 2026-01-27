// src/import/cityjson.rs
// P4.3: CityJSON 1.1 parser for 3D building import
//
// Parses CityJSON format (https://www.cityjson.org/) which is a JSON-based
// encoding of CityGML. Supports Building and BuildingPart objects with
// LOD1-LOD3 geometries.

use serde_json::Value as JsonValue;
use std::collections::HashMap;

use super::building_materials::{material_from_tags, BuildingMaterial};
use super::osm_buildings::{infer_roof_type_from_json, RoofType};

/// Error type for CityJSON parsing
#[derive(Debug, Clone)]
pub struct CityJsonError {
    pub message: String,
}

impl std::fmt::Display for CityJsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CityJSON error: {}", self.message)
    }
}

impl std::error::Error for CityJsonError {}

impl CityJsonError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self { message: msg.into() }
    }
}

pub type CityJsonResult<T> = Result<T, CityJsonError>;

/// A parsed CityJSON building with geometry and attributes
#[derive(Debug, Clone)]
pub struct BuildingGeom {
    /// Unique identifier
    pub id: String,
    /// Vertex positions as flat [x, y, z, x, y, z, ...] in transformed coordinates
    pub positions: Vec<f32>,
    /// Triangle indices into positions (each group of 3 = one triangle)
    pub indices: Vec<u32>,
    /// Normal vectors per vertex (optional)
    pub normals: Option<Vec<f32>>,
    /// Level of detail (1-3, 0 = unknown)
    pub lod: u8,
    /// Building height in meters (if available)
    pub height: Option<f32>,
    /// Ground elevation in meters (if available)
    pub ground_height: Option<f32>,
    /// Roof type inferred from attributes
    pub roof_type: RoofType,
    /// Material properties
    pub material: BuildingMaterial,
    /// Original attributes from CityJSON
    pub attributes: HashMap<String, JsonValue>,
}

impl BuildingGeom {
    /// Create a new empty building geometry
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            positions: Vec::new(),
            indices: Vec::new(),
            normals: None,
            lod: 0,
            height: None,
            ground_height: None,
            roof_type: RoofType::Flat,
            material: BuildingMaterial::default(),
            attributes: HashMap::new(),
        }
    }

    /// Get vertex count
    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }

    /// Get triangle count
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Check if geometry is empty
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty() || self.indices.is_empty()
    }
}

/// CityJSON file metadata
#[derive(Debug, Clone)]
pub struct CityJsonMeta {
    /// CityJSON version (e.g., "1.1")
    pub version: String,
    /// Coordinate reference system EPSG code
    pub crs_epsg: Option<u32>,
    /// Transform scale factors
    pub scale: [f64; 3],
    /// Transform translation offsets
    pub translate: [f64; 3],
    /// Geographic extent [minx, miny, minz, maxx, maxy, maxz]
    pub extent: Option<[f64; 6]>,
}

impl Default for CityJsonMeta {
    fn default() -> Self {
        Self {
            version: "1.1".to_string(),
            crs_epsg: None,
            scale: [1.0, 1.0, 1.0],
            translate: [0.0, 0.0, 0.0],
            extent: None,
        }
    }
}

/// Parse a CityJSON file and extract building geometries.
///
/// Supports CityJSON 1.0 and 1.1 formats.
///
/// # Arguments
/// * `data` - Raw JSON bytes
///
/// # Returns
/// A tuple of (buildings, metadata)
pub fn parse_cityjson(data: &[u8]) -> CityJsonResult<(Vec<BuildingGeom>, CityJsonMeta)> {
    let root: JsonValue = serde_json::from_slice(data)
        .map_err(|e| CityJsonError::new(format!("JSON parse error: {e}")))?;

    // Validate type
    let doc_type = root.get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if doc_type != "CityJSON" {
        return Err(CityJsonError::new("Not a CityJSON file (missing type: CityJSON)"));
    }

    // Parse version
    let version = root.get("version")
        .and_then(|v| v.as_str())
        .unwrap_or("1.0")
        .to_string();

    // Parse transform
    let (scale, translate) = parse_transform(&root);

    // Parse CRS
    let crs_epsg = parse_crs(&root);

    // Parse extent
    let extent = parse_extent(&root);

    let meta = CityJsonMeta {
        version,
        crs_epsg,
        scale,
        translate,
        extent,
    };

    // Parse vertices
    let vertices = parse_vertices(&root, &meta)?;

    // Parse city objects (buildings)
    let buildings = parse_city_objects(&root, &vertices)?;

    Ok((buildings, meta))
}

fn parse_transform(root: &JsonValue) -> ([f64; 3], [f64; 3]) {
    let transform = root.get("transform");

    let scale = transform
        .and_then(|t| t.get("scale"))
        .and_then(|s| s.as_array())
        .map(|arr| {
            [
                arr.get(0).and_then(|v| v.as_f64()).unwrap_or(1.0),
                arr.get(1).and_then(|v| v.as_f64()).unwrap_or(1.0),
                arr.get(2).and_then(|v| v.as_f64()).unwrap_or(1.0),
            ]
        })
        .unwrap_or([1.0, 1.0, 1.0]);

    let translate = transform
        .and_then(|t| t.get("translate"))
        .and_then(|s| s.as_array())
        .map(|arr| {
            [
                arr.get(0).and_then(|v| v.as_f64()).unwrap_or(0.0),
                arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0),
                arr.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0),
            ]
        })
        .unwrap_or([0.0, 0.0, 0.0]);

    (scale, translate)
}

fn parse_crs(root: &JsonValue) -> Option<u32> {
    // CityJSON 1.1 format
    root.get("metadata")
        .and_then(|m| m.get("referenceSystem"))
        .and_then(|rs| rs.as_str())
        .and_then(|s| {
            // Extract EPSG code from URN like "urn:ogc:def:crs:EPSG::7415"
            if let Some(idx) = s.rfind("::") {
                s[idx + 2..].parse().ok()
            } else if let Some(idx) = s.rfind(':') {
                s[idx + 1..].parse().ok()
            } else {
                None
            }
        })
}

fn parse_extent(root: &JsonValue) -> Option<[f64; 6]> {
    root.get("metadata")
        .and_then(|m| m.get("geographicalExtent"))
        .and_then(|e| e.as_array())
        .and_then(|arr| {
            if arr.len() >= 6 {
                Some([
                    arr[0].as_f64()?,
                    arr[1].as_f64()?,
                    arr[2].as_f64()?,
                    arr[3].as_f64()?,
                    arr[4].as_f64()?,
                    arr[5].as_f64()?,
                ])
            } else {
                None
            }
        })
}

fn parse_vertices(root: &JsonValue, meta: &CityJsonMeta) -> CityJsonResult<Vec<[f64; 3]>> {
    let verts_json = root.get("vertices")
        .and_then(|v| v.as_array())
        .ok_or_else(|| CityJsonError::new("Missing 'vertices' array"))?;

    let mut vertices = Vec::with_capacity(verts_json.len());

    for (i, v) in verts_json.iter().enumerate() {
        let arr = v.as_array()
            .ok_or_else(|| CityJsonError::new(format!("Vertex {i} is not an array")))?;

        if arr.len() < 3 {
            return Err(CityJsonError::new(format!("Vertex {i} has fewer than 3 components")));
        }

        let x = arr[0].as_f64().or_else(|| arr[0].as_i64().map(|i| i as f64))
            .ok_or_else(|| CityJsonError::new(format!("Vertex {i} X is not a number")))?;
        let y = arr[1].as_f64().or_else(|| arr[1].as_i64().map(|i| i as f64))
            .ok_or_else(|| CityJsonError::new(format!("Vertex {i} Y is not a number")))?;
        let z = arr[2].as_f64().or_else(|| arr[2].as_i64().map(|i| i as f64))
            .ok_or_else(|| CityJsonError::new(format!("Vertex {i} Z is not a number")))?;

        // Apply transform: real_coord = (vertex_int * scale) + translate
        vertices.push([
            x * meta.scale[0] + meta.translate[0],
            y * meta.scale[1] + meta.translate[1],
            z * meta.scale[2] + meta.translate[2],
        ]);
    }

    Ok(vertices)
}

fn parse_city_objects(root: &JsonValue, vertices: &[[f64; 3]]) -> CityJsonResult<Vec<BuildingGeom>> {
    let objects = root.get("CityObjects")
        .and_then(|o| o.as_object())
        .ok_or_else(|| CityJsonError::new("Missing 'CityObjects' object"))?;

    let mut buildings = Vec::new();

    for (id, obj) in objects.iter() {
        let obj_type = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");

        // Only process Building and BuildingPart objects
        if !obj_type.starts_with("Building") {
            continue;
        }

        if let Some(building) = parse_building(id, obj, vertices)? {
            buildings.push(building);
        }
    }

    Ok(buildings)
}

fn parse_building(
    id: &str,
    obj: &JsonValue,
    vertices: &[[f64; 3]],
) -> CityJsonResult<Option<BuildingGeom>> {
    let mut building = BuildingGeom::new(id);

    // Parse attributes
    if let Some(attrs) = obj.get("attributes").and_then(|a| a.as_object()) {
        for (k, v) in attrs.iter() {
            building.attributes.insert(k.clone(), v.clone());

            // Extract height if available
            if k == "measuredHeight" || k == "height" || k == "h_dak" || k == "h_max" {
                if let Some(h) = v.as_f64() {
                    building.height = Some(h as f32);
                }
            }
            if k == "groundHeight" || k == "h_maaiveld" || k == "h_min" {
                if let Some(h) = v.as_f64() {
                    building.ground_height = Some(h as f32);
                }
            }
        }
    }

    // Infer roof type and material from attributes
    let attrs_json = obj.get("attributes");
    building.roof_type = infer_roof_type_from_json(attrs_json);

    // Build HashMap for material inference
    let tags: HashMap<String, String> = building.attributes.iter()
        .filter_map(|(k, v)| {
            let val = match v {
                JsonValue::String(s) => s.clone(),
                JsonValue::Number(n) => n.to_string(),
                JsonValue::Bool(b) => b.to_string(),
                _ => return None,
            };
            Some((k.clone(), val))
        })
        .collect();
    building.material = material_from_tags(&tags);

    // Parse geometry
    let geoms = obj.get("geometry")
        .and_then(|g| g.as_array());

    if geoms.is_none() {
        return Ok(None);
    }

    let geoms = geoms.unwrap();
    if geoms.is_empty() {
        return Ok(None);
    }

    // Find best LOD (prefer highest available)
    let mut best_geom = None;
    let mut best_lod = 0u8;

    for geom in geoms {
        let lod_str = geom.get("lod")
            .and_then(|l| l.as_str().or_else(|| l.as_f64().map(|_| "")))
            .unwrap_or("1");
        let lod: u8 = lod_str.chars().next()
            .and_then(|c| c.to_digit(10))
            .map(|d| d as u8)
            .unwrap_or(1);

        if lod >= best_lod {
            best_lod = lod;
            best_geom = Some(geom);
        }
    }

    if let Some(geom) = best_geom {
        building.lod = best_lod;
        parse_geometry(&mut building, geom, vertices)?;
    }

    if building.is_empty() {
        return Ok(None);
    }

    // Generate normals if not present
    if building.normals.is_none() {
        building.normals = Some(compute_normals(&building.positions, &building.indices));
    }

    Ok(Some(building))
}

fn parse_geometry(
    building: &mut BuildingGeom,
    geom: &JsonValue,
    vertices: &[[f64; 3]],
) -> CityJsonResult<()> {
    let geom_type = geom.get("type").and_then(|t| t.as_str()).unwrap_or("");

    let boundaries = geom.get("boundaries")
        .ok_or_else(|| CityJsonError::new("Geometry missing 'boundaries'"))?;

    match geom_type {
        "Solid" => parse_solid(building, boundaries, vertices)?,
        "MultiSurface" | "CompositeSurface" => parse_multi_surface(building, boundaries, vertices)?,
        _ => {} // Skip unknown geometry types
    }

    Ok(())
}

fn parse_solid(
    building: &mut BuildingGeom,
    boundaries: &JsonValue,
    vertices: &[[f64; 3]],
) -> CityJsonResult<()> {
    // Solid: boundaries is [ shell1, shell2, ... ] where shell = [ surface1, surface2, ... ]
    let shells = boundaries.as_array()
        .ok_or_else(|| CityJsonError::new("Solid boundaries not an array"))?;

    // Process outer shell (first shell)
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
    // MultiSurface: boundaries is [ surface1, surface2, ... ]
    let surfaces = boundaries.as_array()
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
    // Surface: [ ring1, ring2, ... ] where ring1 = outer ring, others = holes
    let rings = surface.as_array()
        .ok_or_else(|| CityJsonError::new("Surface is not an array"))?;

    if rings.is_empty() {
        return Ok(());
    }

    // Get outer ring (first ring)
    let outer_ring = rings[0].as_array()
        .ok_or_else(|| CityJsonError::new("Ring is not an array"))?;

    if outer_ring.len() < 3 {
        return Ok(()); // Need at least 3 vertices for a polygon
    }

    // Collect ring vertices
    let mut ring_verts: Vec<[f64; 3]> = Vec::with_capacity(outer_ring.len());
    for idx_val in outer_ring {
        let idx = idx_val.as_u64()
            .ok_or_else(|| CityJsonError::new("Vertex index is not a number"))?
            as usize;

        if idx >= vertices.len() {
            return Err(CityJsonError::new(format!("Vertex index {idx} out of bounds")));
        }

        ring_verts.push(vertices[idx]);
    }

    // Triangulate the polygon using ear clipping (simple fan for convex polygons)
    // For more complex polygons, a proper ear clipping algorithm would be needed
    let base_idx = building.positions.len() as u32 / 3;

    // Add vertices to building
    for v in &ring_verts {
        building.positions.push(v[0] as f32);
        building.positions.push(v[1] as f32);
        building.positions.push(v[2] as f32);
    }

    // Simple fan triangulation (works for convex polygons)
    // For concave polygons, this may produce incorrect results
    for i in 1..(ring_verts.len() as u32 - 1) {
        building.indices.push(base_idx);
        building.indices.push(base_idx + i);
        building.indices.push(base_idx + i + 1);
    }

    Ok(())
}

fn compute_normals(positions: &[f32], indices: &[u32]) -> Vec<f32> {
    let vertex_count = positions.len() / 3;
    let mut normals = vec![0.0f32; positions.len()];

    // Accumulate face normals
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

        // Edge vectors
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        // Cross product
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        // Accumulate to each vertex
        for &idx in tri {
            let base = idx as usize * 3;
            if base + 2 < normals.len() {
                normals[base] += n[0];
                normals[base + 1] += n[1];
                normals[base + 2] += n[2];
            }
        }
    }

    // Normalize
    for i in 0..vertex_count {
        let base = i * 3;
        let len = (normals[base].powi(2) + normals[base + 1].powi(2) + normals[base + 2].powi(2)).sqrt();
        if len > 1e-6 {
            normals[base] /= len;
            normals[base + 1] /= len;
            normals[base + 2] /= len;
        } else {
            // Default to up
            normals[base] = 0.0;
            normals[base + 1] = 0.0;
            normals[base + 2] = 1.0;
        }
    }

    normals
}

// ============================================================================
// Python Bindings
// ============================================================================

#[cfg(feature = "extension-module")]
use pyo3::{prelude::*, exceptions::PyValueError, types::PyBytes};

/// P4.3: Python binding for CityJSON parsing
#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn parse_cityjson_py(data: &Bound<'_, PyBytes>) -> PyResult<PyObject> {
    let bytes = data.as_bytes();
    let (buildings, meta) = parse_cityjson(bytes)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Python::with_gil(|py| {
        let result = pyo3::types::PyDict::new_bound(py);

        // Metadata
        let meta_dict = pyo3::types::PyDict::new_bound(py);
        meta_dict.set_item("version", &meta.version)?;
        meta_dict.set_item("crs_epsg", meta.crs_epsg)?;
        meta_dict.set_item("scale", (meta.scale[0], meta.scale[1], meta.scale[2]))?;
        meta_dict.set_item("translate", (meta.translate[0], meta.translate[1], meta.translate[2]))?;
        if let Some(ext) = &meta.extent {
            meta_dict.set_item("extent", (ext[0], ext[1], ext[2], ext[3], ext[4], ext[5]))?;
        }
        result.set_item("metadata", meta_dict)?;

        // Buildings list
        let buildings_list = pyo3::types::PyList::empty_bound(py);
        for b in &buildings {
            let bdict = pyo3::types::PyDict::new_bound(py);
            bdict.set_item("id", &b.id)?;
            bdict.set_item("vertex_count", b.vertex_count())?;
            bdict.set_item("triangle_count", b.triangle_count())?;
            bdict.set_item("lod", b.lod)?;
            bdict.set_item("height", b.height)?;
            bdict.set_item("ground_height", b.ground_height)?;
            bdict.set_item("roof_type", format!("{:?}", b.roof_type).to_lowercase())?;

            // Material
            let mat_dict = pyo3::types::PyDict::new_bound(py);
            mat_dict.set_item("albedo", (b.material.albedo[0], b.material.albedo[1], b.material.albedo[2]))?;
            mat_dict.set_item("roughness", b.material.roughness)?;
            mat_dict.set_item("metallic", b.material.metallic)?;
            bdict.set_item("material", mat_dict)?;

            // Positions as flat list (caller can reshape)
            bdict.set_item("positions", b.positions.clone())?;
            bdict.set_item("indices", b.indices.clone())?;
            if let Some(ref normals) = b.normals {
                bdict.set_item("normals", normals.clone())?;
            }

            buildings_list.append(bdict)?;
        }
        result.set_item("buildings", buildings_list)?;
        result.set_item("building_count", buildings.len())?;

        Ok(result.into())
    })
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cityjson() -> &'static [u8] {
        br#"{
            "type": "CityJSON",
            "version": "1.1",
            "transform": {
                "scale": [0.001, 0.001, 0.001],
                "translate": [0.0, 0.0, 0.0]
            },
            "vertices": [
                [0, 0, 0],
                [10000, 0, 0],
                [10000, 10000, 0],
                [0, 10000, 0],
                [0, 0, 5000],
                [10000, 0, 5000],
                [10000, 10000, 5000],
                [0, 10000, 5000]
            ],
            "CityObjects": {
                "building1": {
                    "type": "Building",
                    "attributes": {
                        "measuredHeight": 5.0
                    },
                    "geometry": [{
                        "type": "Solid",
                        "lod": "1",
                        "boundaries": [
                            [
                                [[0, 1, 2, 3]],
                                [[4, 5, 6, 7]],
                                [[0, 1, 5, 4]],
                                [[1, 2, 6, 5]],
                                [[2, 3, 7, 6]],
                                [[3, 0, 4, 7]]
                            ]
                        ]
                    }]
                }
            }
        }"#
    }

    #[test]
    fn test_parse_simple_cityjson() {
        let (buildings, meta) = parse_cityjson(sample_cityjson()).unwrap();

        assert_eq!(meta.version, "1.1");
        assert_eq!(meta.scale, [0.001, 0.001, 0.001]);

        assert_eq!(buildings.len(), 1);
        let b = &buildings[0];
        assert_eq!(b.id, "building1");
        assert_eq!(b.lod, 1);
        assert_eq!(b.height, Some(5.0));
        assert!(b.vertex_count() > 0);
        assert!(b.triangle_count() > 0);
    }

    #[test]
    fn test_invalid_cityjson() {
        let result = parse_cityjson(b"not json");
        assert!(result.is_err());

        let result = parse_cityjson(br#"{"type": "NotCityJSON"}"#);
        assert!(result.is_err());
    }
}

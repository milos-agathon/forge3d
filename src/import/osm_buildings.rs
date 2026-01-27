// src/import/osm_buildings.rs
// OSM buildings ingest helper with extended support for roof types, materials, and LOD.
// Accepts a Python list of features or GeoJSON FeatureCollection.
// Each feature: {"coords": np.ndarray (N,2) float32 in XY, "height": float}
// Returns a merged MeshBuffers extruded with given or default height.
//
// P4.1: Added RoofType inference from OSM tags

use std::collections::HashMap;

#[cfg(feature = "extension-module")]
use crate::geometry::{extrude_polygon_with_options, ExtrudeOptions, MeshBuffers};
use serde_json::Value as JsonValue;

// ============================================================================
// P4.1: Roof Type Inference
// ============================================================================

/// Roof shape categories supported by the building pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoofType {
    /// Flat roof (default for most commercial/industrial buildings)
    #[default]
    Flat,
    /// Gabled roof (pitched with two slopes meeting at a ridge)
    Gabled,
    /// Hipped roof (pitched with slopes on all sides)
    Hipped,
    /// Pyramidal roof (four triangular faces meeting at apex)
    Pyramidal,
    /// Dome roof (curved)
    Dome,
    /// Mansard roof (four-sided with double slope)
    Mansard,
    /// Shed/lean-to roof (single slope)
    Shed,
    /// Gambrel roof (barn-style with two slopes per side)
    Gambrel,
    /// Onion dome (bulbous, typical of Orthodox churches)
    Onion,
    /// Skillion roof (single slanting surface)
    Skillion,
}

impl RoofType {
    /// Parse from OSM roof:shape tag value
    pub fn from_osm_tag(value: &str) -> Self {
        match value.to_lowercase().trim() {
            "flat" => RoofType::Flat,
            "gabled" => RoofType::Gabled,
            "hipped" => RoofType::Hipped,
            "pyramidal" => RoofType::Pyramidal,
            "dome" => RoofType::Dome,
            "mansard" => RoofType::Mansard,
            "shed" | "lean_to" | "lean-to" => RoofType::Shed,
            "gambrel" => RoofType::Gambrel,
            "onion" => RoofType::Onion,
            "skillion" => RoofType::Skillion,
            _ => RoofType::Flat,
        }
    }

    /// Estimate roof height multiplier based on type (relative to base height)
    pub fn height_multiplier(&self) -> f32 {
        match self {
            RoofType::Flat => 0.0,
            RoofType::Gabled | RoofType::Shed | RoofType::Skillion => 0.25,
            RoofType::Hipped => 0.2,
            RoofType::Pyramidal => 0.4,
            RoofType::Dome | RoofType::Onion => 0.35,
            RoofType::Mansard => 0.3,
            RoofType::Gambrel => 0.35,
        }
    }
}

/// Infer roof type from OSM tags.
///
/// Checks common OSM tags in priority order:
/// 1. `building:roof:shape` (explicit roof shape)
/// 2. `roof:shape` (alternative tag)
/// 3. `building` type (heuristic inference)
///
/// # Example
/// ```
/// use forge3d::import::osm_buildings::{infer_roof_type, RoofType};
/// use std::collections::HashMap;
///
/// let mut tags = HashMap::new();
/// tags.insert("building:roof:shape".to_string(), "gabled".to_string());
/// assert_eq!(infer_roof_type(&tags), RoofType::Gabled);
/// ```
pub fn infer_roof_type(tags: &HashMap<String, String>) -> RoofType {
    // Priority 1: Explicit roof shape tags
    if let Some(shape) = tags.get("building:roof:shape")
        .or_else(|| tags.get("roof:shape"))
        .or_else(|| tags.get("roof_shape"))
    {
        return RoofType::from_osm_tag(shape);
    }

    // Priority 2: Infer from building type
    if let Some(building_type) = tags.get("building") {
        return match building_type.to_lowercase().as_str() {
            // Typically flat roofs
            "industrial" | "warehouse" | "retail" | "commercial" | "office"
            | "parking" | "garages" | "hangar" | "bunker" => RoofType::Flat,

            // Typically gabled roofs
            "house" | "detached" | "semidetached_house" | "terrace"
            | "residential" | "bungalow" | "cabin" | "farm" | "barn" => RoofType::Gabled,

            // Hipped roofs common
            "apartments" | "dormitory" | "hotel" => RoofType::Hipped,

            // Special building types
            "church" | "cathedral" | "chapel" | "mosque" => {
                // Check for onion dome (Orthodox churches)
                if tags.get("religion").map(|r| r == "christian") == Some(true)
                    && tags.get("denomination").map(|d| d.contains("orthodox")).unwrap_or(false)
                {
                    RoofType::Onion
                } else {
                    RoofType::Gabled
                }
            }

            "temple" | "shrine" => RoofType::Hipped,
            "greenhouse" | "shed" | "carport" => RoofType::Shed,

            // Default to flat for unknown
            _ => RoofType::Flat,
        };
    }

    RoofType::Flat
}

/// Infer roof type from GeoJSON properties
pub fn infer_roof_type_from_json(properties: Option<&JsonValue>) -> RoofType {
    let props = match properties {
        Some(JsonValue::Object(map)) => map,
        _ => return RoofType::Flat,
    };

    // Build HashMap from JSON properties
    let tags: HashMap<String, String> = props
        .iter()
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

    infer_roof_type(&tags)
}

#[cfg(feature = "extension-module")]
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
#[cfg(feature = "extension-module")]
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

#[cfg(feature = "extension-module")]
fn merge_meshes(meshes: &[MeshBuffers]) -> MeshBuffers {
    let mut out = MeshBuffers::new();
    let mut base: u32 = 0;
    for m in meshes {
        out.positions.extend(m.positions.iter().copied());
        out.normals.extend(m.normals.iter().copied());
        out.uvs.extend(m.uvs.iter().copied());
        out.indices
            .extend(m.indices.iter().copied().map(|i| i + base));
        base += m.positions.len() as u32;
    }
    out
}

#[cfg(feature = "extension-module")]
#[pyfunction(signature = (features, default_height=10.0, height_key=None))]
pub fn import_osm_buildings_extrude_py(
    features: &Bound<'_, PyAny>,
    default_height: f32,
    height_key: Option<&str>,
) -> PyResult<PyObject> {
    // Parse a Python iterable of dicts with keys: coords (Nx2 float32), height (float, optional)
    let mut meshes: Vec<MeshBuffers> = Vec::new();

    let iter = features.iter()?;
    for item in iter {
        let obj = item?;
        let d = obj.downcast::<PyDict>()?;
        let coords_obj = d
            .get_item("coords")?
            .ok_or_else(|| PyValueError::new_err("feature missing 'coords'"))?;
        let coords: PyReadonlyArray2<f32> = coords_obj.extract()?;
        if coords.shape()[1] != 2 {
            return Err(PyValueError::new_err("coords must have shape (N, 2)"));
        }
        let mut ring: Vec<[f32; 2]> = Vec::with_capacity(coords.shape()[0]);
        for row in coords.as_array().outer_iter() {
            ring.push([row[0], row[1]]);
        }
        let h = if let Some(key) = height_key {
            if let Some(v) = d.get_item(key)? {
                v.extract::<f32>().unwrap_or(default_height)
            } else {
                default_height
            }
        } else if let Some(v) = d.get_item("height")? {
            v.extract::<f32>().unwrap_or(default_height)
        } else {
            default_height
        };
        let mut opts = ExtrudeOptions::default();
        opts.height = h;
        let mesh = extrude_polygon_with_options(&ring, opts)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        meshes.push(mesh);
    }

    let merged = merge_meshes(&meshes);
    Python::with_gil(|py| crate::geometry::mesh_to_python(py, &merged))
}

#[cfg(feature = "extension-module")]
#[pyfunction(signature = (geojson, default_height=10.0, height_key=None))]
pub fn import_osm_buildings_from_geojson_py(
    geojson: &str,
    default_height: f32,
    height_key: Option<&str>,
) -> PyResult<PyObject> {
    // Parse GeoJSON FeatureCollection
    let root: JsonValue = serde_json::from_str(geojson)
        .map_err(|e| PyValueError::new_err(format!("invalid JSON: {e}")))?;
    let t = root.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if t != "FeatureCollection" {
        return Err(PyValueError::new_err("GeoJSON must be a FeatureCollection"));
    }
    let feats = root
        .get("features")
        .and_then(|v| v.as_array())
        .ok_or_else(|| PyValueError::new_err("FeatureCollection missing 'features' array"))?;

    let mut meshes: Vec<MeshBuffers> = Vec::new();

    for feat in feats {
        let props = feat.get("properties");
        let mut h = default_height;
        if let Some(key) = height_key {
            if let Some(v) = props.and_then(|p| p.get(key)) {
                if let Some(f) = v.as_f64() {
                    h = f as f32;
                } else if let Some(i) = v.as_i64() {
                    h = i as f32;
                } else if let Some(s) = v.as_str() {
                    if let Ok(parsed) = s.parse::<f32>() {
                        h = parsed;
                    }
                }
            }
        } else if let Some(v) = props.and_then(|p| p.get("height")) {
            if let Some(f) = v.as_f64() {
                h = f as f32;
            } else if let Some(i) = v.as_i64() {
                h = i as f32;
            } else if let Some(s) = v.as_str() {
                if let Ok(parsed) = s.parse::<f32>() {
                    h = parsed;
                }
            }
        }

        let geom = feat
            .get("geometry")
            .ok_or_else(|| PyValueError::new_err("feature missing geometry"))?;
        let gtype = geom.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let coords = geom
            .get("coordinates")
            .ok_or_else(|| PyValueError::new_err("geometry missing coordinates"))?;

        let mut push_ring = |ring_coords: &JsonValue| -> PyResult<()> {
            // ring_coords: array of positions, we take as exterior ring
            let arr = ring_coords
                .as_array()
                .ok_or_else(|| PyValueError::new_err("ring is not an array"))?;
            let mut ring: Vec<[f32; 2]> = Vec::with_capacity(arr.len());
            for pos in arr {
                let p = pos
                    .as_array()
                    .ok_or_else(|| PyValueError::new_err("position must be array"))?;
                if p.len() < 2 {
                    continue;
                }
                let x = p[0].as_f64().unwrap_or(0.0) as f32;
                let y = p[1].as_f64().unwrap_or(0.0) as f32;
                ring.push([x, y]);
            }
            if ring.len() >= 3 {
                let mut opts = ExtrudeOptions::default();
                opts.height = h;
                let mesh = extrude_polygon_with_options(&ring, opts)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                meshes.push(mesh);
            }
            Ok(())
        };

        match gtype {
            "Polygon" => {
                // coordinates: [ [ [x,y], ... ] , [hole] ... ]
                if let Some(rings) = coords.as_array() {
                    if let Some(outer) = rings.first() {
                        push_ring(outer)?;
                    }
                }
            }
            "MultiPolygon" => {
                // coordinates: [ [ [ [x,y], ... ], [hole] ... ], [ ... ] ]
                if let Some(polys) = coords.as_array() {
                    for poly in polys {
                        if let Some(rings) = poly.as_array() {
                            if let Some(outer) = rings.first() {
                                push_ring(outer)?;
                            }
                        }
                    }
                }
            }
            _ => { /* skip other geometry types */ }
        }
    }

    let merged = merge_meshes(&meshes);
    Python::with_gil(|py| crate::geometry::mesh_to_python(py, &merged))
}

/// P4.1: Python binding for roof type inference from GeoJSON properties
#[cfg(feature = "extension-module")]
#[pyfunction(signature = (properties_json))]
pub fn infer_roof_type_py(properties_json: &str) -> PyResult<String> {
    let props: JsonValue = serde_json::from_str(properties_json)
        .map_err(|e| PyValueError::new_err(format!("invalid JSON: {e}")))?;
    let roof_type = infer_roof_type_from_json(Some(&props));
    Ok(format!("{:?}", roof_type).to_lowercase())
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roof_type_from_osm_tag() {
        assert_eq!(RoofType::from_osm_tag("gabled"), RoofType::Gabled);
        assert_eq!(RoofType::from_osm_tag("HIPPED"), RoofType::Hipped);
        assert_eq!(RoofType::from_osm_tag("flat"), RoofType::Flat);
        assert_eq!(RoofType::from_osm_tag("unknown"), RoofType::Flat);
    }

    #[test]
    fn test_infer_roof_type_explicit() {
        let mut tags = HashMap::new();
        tags.insert("building:roof:shape".to_string(), "gabled".to_string());
        assert_eq!(infer_roof_type(&tags), RoofType::Gabled);
    }

    #[test]
    fn test_infer_roof_type_from_building_type() {
        let mut tags = HashMap::new();
        tags.insert("building".to_string(), "house".to_string());
        assert_eq!(infer_roof_type(&tags), RoofType::Gabled);

        tags.clear();
        tags.insert("building".to_string(), "warehouse".to_string());
        assert_eq!(infer_roof_type(&tags), RoofType::Flat);
    }

    #[test]
    fn test_roof_height_multiplier() {
        assert_eq!(RoofType::Flat.height_multiplier(), 0.0);
        assert!(RoofType::Gabled.height_multiplier() > 0.0);
        assert!(RoofType::Pyramidal.height_multiplier() > RoofType::Gabled.height_multiplier());
    }
}

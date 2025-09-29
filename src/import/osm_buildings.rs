// src/import/osm_buildings.rs
// Minimal OSM buildings ingest helper (Phase 5 F2): accepts a Python list of features
// Each feature: {"coords": np.ndarray (N,2) float32 in XY, "height": float}
// Returns a merged MeshBuffers extruded with given or default height.

use crate::geometry::{ExtrudeOptions, extrude_polygon_with_options, MeshBuffers};
use serde_json::Value as JsonValue;

#[cfg(feature = "extension-module")]
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
#[cfg(feature = "extension-module")]
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

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
        let mut ring: Vec<[f32;2]> = Vec::with_capacity(coords.shape()[0]);
        for row in coords.as_array().outer_iter() {
            ring.push([row[0], row[1]]);
        }
        let h = if let Some(key) = height_key {
            if let Some(v) = d.get_item(key)? { v.extract::<f32>().unwrap_or(default_height) } else { default_height }
        } else if let Some(v) = d.get_item("height")? {
            v.extract::<f32>().unwrap_or(default_height)
        } else { default_height };
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
    let feats = root.get("features")
        .and_then(|v| v.as_array())
        .ok_or_else(|| PyValueError::new_err("FeatureCollection missing 'features' array"))?;

    let mut meshes: Vec<MeshBuffers> = Vec::new();

    for feat in feats {
        let props = feat.get("properties");
        let mut h = default_height;
        if let Some(key) = height_key {
            if let Some(v) = props.and_then(|p| p.get(key)) {
                if let Some(f) = v.as_f64() { h = f as f32; }
                else if let Some(i) = v.as_i64() { h = i as f32; }
                else if let Some(s) = v.as_str() { if let Ok(parsed) = s.parse::<f32>() { h = parsed; } }
            }
        } else if let Some(v) = props.and_then(|p| p.get("height")) {
            if let Some(f) = v.as_f64() { h = f as f32; }
            else if let Some(i) = v.as_i64() { h = i as f32; }
            else if let Some(s) = v.as_str() { if let Ok(parsed) = s.parse::<f32>() { h = parsed; } }
        }

        let geom = feat.get("geometry")
            .ok_or_else(|| PyValueError::new_err("feature missing geometry"))?;
        let gtype = geom.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let coords = geom.get("coordinates").ok_or_else(|| PyValueError::new_err("geometry missing coordinates"))?;

        let mut push_ring = |ring_coords: &JsonValue| -> PyResult<()> {
            // ring_coords: array of positions, we take as exterior ring
            let arr = ring_coords.as_array().ok_or_else(|| PyValueError::new_err("ring is not an array"))?;
            let mut ring: Vec<[f32;2]> = Vec::with_capacity(arr.len());
            for pos in arr {
                let p = pos.as_array().ok_or_else(|| PyValueError::new_err("position must be array"))?;
                if p.len() < 2 { continue; }
                let x = p[0].as_f64().unwrap_or(0.0) as f32;
                let y = p[1].as_f64().unwrap_or(0.0) as f32;
                ring.push([x,y]);
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
                    if let Some(outer) = rings.first() { push_ring(outer)?; }
                }
            }
            "MultiPolygon" => {
                // coordinates: [ [ [ [x,y], ... ], [hole] ... ], [ ... ] ]
                if let Some(polys) = coords.as_array() {
                    for poly in polys {
                        if let Some(rings) = poly.as_array() {
                            if let Some(outer) = rings.first() { push_ring(outer)?; }
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

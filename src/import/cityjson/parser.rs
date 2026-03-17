use serde_json::Value as JsonValue;
use std::collections::HashMap;

use super::geometry::{compute_normals, parse_geometry};
use super::{BuildingGeom, CityJsonError, CityJsonMeta, CityJsonResult};
use crate::import::building_materials::material_from_tags;
use crate::import::osm_buildings::infer_roof_type_from_json;

/// Parse a CityJSON file and extract building geometries.
pub fn parse_cityjson(data: &[u8]) -> CityJsonResult<(Vec<BuildingGeom>, CityJsonMeta)> {
    let root: JsonValue = serde_json::from_slice(data)
        .map_err(|e| CityJsonError::new(format!("JSON parse error: {e}")))?;

    let doc_type = root.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if doc_type != "CityJSON" {
        return Err(CityJsonError::new(
            "Not a CityJSON file (missing type: CityJSON)",
        ));
    }

    let meta = CityJsonMeta {
        version: root
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("1.0")
            .to_string(),
        crs_epsg: parse_crs(&root),
        scale: parse_transform(&root).0,
        translate: parse_transform(&root).1,
        extent: parse_extent(&root),
    };
    let vertices = parse_vertices(&root, &meta)?;
    let buildings = parse_city_objects(&root, &vertices)?;

    Ok((buildings, meta))
}

fn parse_transform(root: &JsonValue) -> ([f64; 3], [f64; 3]) {
    let transform = root.get("transform");

    let scale = transform
        .and_then(|t| t.get("scale"))
        .and_then(|s| s.as_array())
        .map(|arr| parse_vec3_or_default(arr))
        .unwrap_or([1.0, 1.0, 1.0]);
    let translate = transform
        .and_then(|t| t.get("translate"))
        .and_then(|s| s.as_array())
        .map(|arr| parse_vec3_or_zero(arr))
        .unwrap_or([0.0, 0.0, 0.0]);

    (scale, translate)
}

fn parse_crs(root: &JsonValue) -> Option<u32> {
    root.get("metadata")
        .and_then(|m| m.get("referenceSystem"))
        .and_then(|rs| rs.as_str())
        .and_then(|s| {
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
            if arr.len() < 6 {
                return None;
            }
            Some([
                arr[0].as_f64()?,
                arr[1].as_f64()?,
                arr[2].as_f64()?,
                arr[3].as_f64()?,
                arr[4].as_f64()?,
                arr[5].as_f64()?,
            ])
        })
}

fn parse_vertices(root: &JsonValue, meta: &CityJsonMeta) -> CityJsonResult<Vec<[f64; 3]>> {
    let verts_json = root
        .get("vertices")
        .and_then(|v| v.as_array())
        .ok_or_else(|| CityJsonError::new("Missing 'vertices' array"))?;

    let mut vertices = Vec::with_capacity(verts_json.len());
    for (i, vertex) in verts_json.iter().enumerate() {
        let arr = vertex
            .as_array()
            .ok_or_else(|| CityJsonError::new(format!("Vertex {i} is not an array")))?;
        if arr.len() < 3 {
            return Err(CityJsonError::new(format!(
                "Vertex {i} has fewer than 3 components"
            )));
        }

        let x = parse_number(&arr[0], format!("Vertex {i} X is not a number"))?;
        let y = parse_number(&arr[1], format!("Vertex {i} Y is not a number"))?;
        let z = parse_number(&arr[2], format!("Vertex {i} Z is not a number"))?;

        vertices.push([
            x * meta.scale[0] + meta.translate[0],
            y * meta.scale[1] + meta.translate[1],
            z * meta.scale[2] + meta.translate[2],
        ]);
    }

    Ok(vertices)
}

fn parse_city_objects(
    root: &JsonValue,
    vertices: &[[f64; 3]],
) -> CityJsonResult<Vec<BuildingGeom>> {
    let objects = root
        .get("CityObjects")
        .and_then(|o| o.as_object())
        .ok_or_else(|| CityJsonError::new("Missing 'CityObjects' object"))?;

    let mut buildings = Vec::new();
    for (id, obj) in objects.iter() {
        let obj_type = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
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

    if let Some(attrs) = obj.get("attributes").and_then(|a| a.as_object()) {
        for (key, value) in attrs.iter() {
            building.attributes.insert(key.clone(), value.clone());
            update_height_fields(&mut building, key, value);
        }
    }

    building.roof_type = infer_roof_type_from_json(obj.get("attributes"));
    let tags: HashMap<String, String> = building
        .attributes
        .iter()
        .filter_map(|(key, value)| {
            let val = match value {
                JsonValue::String(s) => s.clone(),
                JsonValue::Number(n) => n.to_string(),
                JsonValue::Bool(b) => b.to_string(),
                _ => return None,
            };
            Some((key.clone(), val))
        })
        .collect();
    building.material = material_from_tags(&tags);

    let geoms = obj.get("geometry").and_then(|g| g.as_array());
    let Some(geoms) = geoms else {
        return Ok(None);
    };
    if geoms.is_empty() {
        return Ok(None);
    }

    let (best_lod, best_geom) = select_best_geometry(geoms);
    if let Some(geom) = best_geom {
        building.lod = best_lod;
        parse_geometry(&mut building, geom, vertices)?;
    }

    if building.is_empty() {
        return Ok(None);
    }
    if building.normals.is_none() {
        building.normals = Some(compute_normals(&building.positions, &building.indices));
    }

    Ok(Some(building))
}

fn parse_vec3_or_default(arr: &[JsonValue]) -> [f64; 3] {
    [
        arr.get(0).and_then(|v| v.as_f64()).unwrap_or(1.0),
        arr.get(1).and_then(|v| v.as_f64()).unwrap_or(1.0),
        arr.get(2).and_then(|v| v.as_f64()).unwrap_or(1.0),
    ]
}

fn parse_vec3_or_zero(arr: &[JsonValue]) -> [f64; 3] {
    [
        arr.get(0).and_then(|v| v.as_f64()).unwrap_or(0.0),
        arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0),
        arr.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0),
    ]
}

fn parse_number(value: &JsonValue, message: String) -> CityJsonResult<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|i| i as f64))
        .ok_or_else(|| CityJsonError::new(message))
}

fn update_height_fields(building: &mut BuildingGeom, key: &str, value: &JsonValue) {
    if (key == "measuredHeight" || key == "height" || key == "h_dak" || key == "h_max")
        && value.as_f64().is_some()
    {
        building.height = value.as_f64().map(|h| h as f32);
    }
    if (key == "groundHeight" || key == "h_maaiveld" || key == "h_min") && value.as_f64().is_some()
    {
        building.ground_height = value.as_f64().map(|h| h as f32);
    }
}

fn select_best_geometry(geoms: &[JsonValue]) -> (u8, Option<&JsonValue>) {
    let mut best_geom = None;
    let mut best_lod = 0u8;

    for geom in geoms {
        let lod_str = geom
            .get("lod")
            .and_then(|l| l.as_str().or_else(|| l.as_f64().map(|_| "")))
            .unwrap_or("1");
        let lod = lod_str
            .chars()
            .next()
            .and_then(|c| c.to_digit(10))
            .map(|d| d as u8)
            .unwrap_or(1);

        if lod >= best_lod {
            best_lod = lod;
            best_geom = Some(geom);
        }
    }

    (best_lod, best_geom)
}

use std::collections::HashMap;

use serde_json::{json, Map, Value};

use crate::gis::error::{GisError, GisResult};
use crate::gis::types::{RasterBounds, RasterWarning};

pub fn parse_osm_features_value(osm_json: &Value, tags: Option<&Value>) -> GisResult<Value> {
    let elements = osm_json
        .get("elements")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            GisError::InvalidArgument(
                "malformed_payload: OSM JSON payload must include an elements array".to_string(),
            )
        })?;
    let mut nodes = HashMap::<i64, (f64, f64, Option<Map<String, Value>>)>::new();
    for element in elements {
        if element.get("type").and_then(Value::as_str) == Some("node") {
            let id = element.get("id").and_then(Value::as_i64).ok_or_else(|| {
                GisError::InvalidArgument("malformed_payload: OSM node is missing id".to_string())
            })?;
            let lat = element.get("lat").and_then(Value::as_f64).ok_or_else(|| {
                GisError::InvalidArgument("malformed_payload: OSM node is missing lat".to_string())
            })?;
            let lon = element.get("lon").and_then(Value::as_f64).ok_or_else(|| {
                GisError::InvalidArgument("malformed_payload: OSM node is missing lon".to_string())
            })?;
            let tags = element.get("tags").and_then(Value::as_object).cloned();
            nodes.insert(id, (lon, lat, tags));
        }
    }

    let mut features = Vec::new();
    let mut skipped = Map::new();
    let mut warnings = Vec::new();
    for element in elements {
        match element.get("type").and_then(Value::as_str) {
            Some("node") => {
                let Some(element_tags) = element.get("tags").and_then(Value::as_object) else {
                    continue;
                };
                if !tags_match(element_tags, tags) {
                    continue;
                }
                let lon = element.get("lon").and_then(Value::as_f64).unwrap_or(0.0);
                let lat = element.get("lat").and_then(Value::as_f64).unwrap_or(0.0);
                features.push(feature(
                    element_tags.clone(),
                    json!({"type": "Point", "coordinates": [lon, lat]}),
                ));
            }
            Some("way") => {
                let element_tags = element
                    .get("tags")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                if element_tags.is_empty() || !tags_match(&element_tags, tags) {
                    continue;
                }
                let node_ids = element
                    .get("nodes")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        GisError::InvalidArgument(
                            "malformed_payload: OSM way is missing nodes".to_string(),
                        )
                    })?;
                let mut coords = Vec::with_capacity(node_ids.len());
                let mut incomplete = false;
                for node_id in node_ids {
                    let Some(node_id) = node_id.as_i64() else {
                        incomplete = true;
                        break;
                    };
                    if let Some((lon, lat, _)) = nodes.get(&node_id) {
                        coords.push(json!([lon, lat]));
                    } else {
                        incomplete = true;
                    }
                }
                if incomplete || coords.len() < 2 {
                    increment(&mut skipped, "incomplete_way");
                    push_warning_once(
                        &mut warnings,
                        "incomplete_way",
                        "incomplete_way: skipped way with missing node coordinates",
                    );
                    continue;
                }
                let closed = coords.len() >= 4 && coords.first() == coords.last();
                let geometry = if closed {
                    json!({"type": "Polygon", "coordinates": [coords]})
                } else {
                    json!({"type": "LineString", "coordinates": coords})
                };
                features.push(feature(element_tags, geometry));
            }
            Some("relation") => {
                increment(&mut skipped, "unsupported_relation");
                push_warning_once(
                    &mut warnings,
                    "unsupported_relation",
                    "unsupported_relation: OSM relations are not parsed by the first-pass backend",
                );
            }
            _ => {}
        }
    }
    if features.is_empty() {
        push_warning_once(
            &mut warnings,
            "empty_feature_set",
            "empty_feature_set: OSM payload parsed to zero features",
        );
    }
    let bounds = bounds_for_features(&features).map(RasterBounds::tuple);
    let mut result = Map::new();
    result.insert(
        "type".to_string(),
        Value::String("FeatureCollection".to_string()),
    );
    result.insert("features".to_string(), Value::Array(features));
    result.insert("crs".to_string(), json!({"name": "EPSG", "code": "4326"}));
    result.insert(
        "bounds".to_string(),
        bounds.map_or(Value::Null, |bounds| json!(bounds)),
    );
    result.insert("skipped".to_string(), Value::Object(skipped));
    result.insert("warnings".to_string(), warnings_to_json(&warnings));
    Ok(Value::Object(result))
}

pub fn query_osm_features_value(
    aoi: RasterBounds,
    tags: &Value,
    cache: Option<&Value>,
) -> GisResult<Value> {
    let query = overpass_query(aoi, tags)?;
    if let Some(payload) = cache.and_then(|cache| cache.get("osm_json")) {
        let osm_json = if let Some(text) = payload.as_str() {
            serde_json::from_str::<Value>(text).map_err(|err| {
                GisError::InvalidArgument(format!("malformed_payload: invalid OSM JSON: {err}"))
            })?
        } else {
            payload.clone()
        };
        let mut out = Map::new();
        out.insert("osm_json".to_string(), osm_json);
        out.insert("query".to_string(), Value::String(query));
        out.insert("bounds".to_string(), json!(aoi.tuple()));
        out.insert(
            "remote".to_string(),
            json!({"status": "mocked", "from_cache": true}),
        );
        out.insert("warnings".to_string(), Value::Array(Vec::new()));
        return Ok(Value::Object(out));
    }
    Err(GisError::BackendUnavailable(
        "backend_unavailable: explicit Overpass endpoint or cache payload required; no hidden default downloads".to_string(),
    ))
}

fn overpass_query(aoi: RasterBounds, tags: &Value) -> GisResult<String> {
    let tag_filters = tags.as_object().ok_or_else(|| {
        GisError::InvalidArgument("invalid_argument: tags must be a dict".to_string())
    })?;
    let mut filters = String::new();
    for (key, value) in tag_filters {
        if value.as_bool() == Some(true) {
            filters.push_str(&format!("[\"{key}\"]"));
        } else if let Some(value) = value.as_str() {
            filters.push_str(&format!("[\"{key}\"=\"{value}\"]"));
        }
    }
    Ok(format!(
        "[out:json];(node{filters}({},{},{},{});way{filters}({},{},{},{}););out body;>;out skel qt;",
        aoi.bottom, aoi.left, aoi.top, aoi.right, aoi.bottom, aoi.left, aoi.top, aoi.right
    ))
}

fn tags_match(element_tags: &Map<String, Value>, filter: Option<&Value>) -> bool {
    let Some(filter) = filter.and_then(Value::as_object) else {
        return true;
    };
    filter.iter().all(|(key, expected)| {
        let Some(actual) = element_tags.get(key) else {
            return false;
        };
        if expected.as_bool() == Some(true) {
            return true;
        }
        expected
            .as_str()
            .map(|expected| actual.as_str() == Some(expected))
            .unwrap_or(true)
    })
}

fn feature(properties: Map<String, Value>, geometry: Value) -> Value {
    let mut feature = Map::new();
    feature.insert("type".to_string(), Value::String("Feature".to_string()));
    feature.insert("properties".to_string(), Value::Object(properties));
    feature.insert("geometry".to_string(), geometry);
    Value::Object(feature)
}

fn bounds_for_features(features: &[Value]) -> Option<RasterBounds> {
    let mut bounds: Option<RasterBounds> = None;
    for feature in features {
        visit_coords(feature.get("geometry")?, &mut |x, y| {
            bounds = Some(match bounds {
                Some(mut b) => {
                    b.left = b.left.min(x);
                    b.right = b.right.max(x);
                    b.bottom = b.bottom.min(y);
                    b.top = b.top.max(y);
                    b
                }
                None => RasterBounds {
                    left: x,
                    bottom: y,
                    right: x,
                    top: y,
                },
            });
        });
    }
    bounds
}

fn visit_coords(geometry: &Value, f: &mut impl FnMut(f64, f64)) {
    match geometry.get("type").and_then(Value::as_str) {
        Some("Point") => {
            if let Some(coords) = geometry.get("coordinates").and_then(Value::as_array) {
                if let (Some(x), Some(y)) = (
                    coords.first().and_then(Value::as_f64),
                    coords.get(1).and_then(Value::as_f64),
                ) {
                    f(x, y);
                }
            }
        }
        Some("LineString") => visit_coord_array(geometry.get("coordinates"), f),
        Some("Polygon") => {
            if let Some(rings) = geometry.get("coordinates").and_then(Value::as_array) {
                for ring in rings {
                    visit_coord_array(Some(ring), f);
                }
            }
        }
        _ => {}
    }
}

fn visit_coord_array(value: Option<&Value>, f: &mut impl FnMut(f64, f64)) {
    let Some(items) = value.and_then(Value::as_array) else {
        return;
    };
    for coord in items {
        let Some(coord) = coord.as_array() else {
            continue;
        };
        if let (Some(x), Some(y)) = (
            coord.first().and_then(Value::as_f64),
            coord.get(1).and_then(Value::as_f64),
        ) {
            f(x, y);
        }
    }
}

fn increment(map: &mut Map<String, Value>, key: &str) {
    let next = map.get(key).and_then(Value::as_u64).unwrap_or(0) + 1;
    map.insert(key.to_string(), Value::Number(next.into()));
}

fn push_warning_once(warnings: &mut Vec<RasterWarning>, code: &'static str, message: &'static str) {
    if warnings.iter().all(|warning| warning.code != code) {
        warnings.push(RasterWarning::new(code, message, None));
    }
}

fn warnings_to_json(warnings: &[RasterWarning]) -> Value {
    Value::Array(
        warnings
            .iter()
            .map(|warning| {
                json!({
                    "code": warning.code,
                    "message": warning.message,
                    "field": warning.field,
                })
            })
            .collect(),
    )
}

#[cfg(feature = "extension-module")]
pub use py::{parse_osm_features_py, query_osm_features_py};

#[cfg(feature = "extension-module")]
mod py {
    use pyo3::prelude::*;
    use pyo3::types::{PyAny, PyDict};

    use crate::gis::affine::validate_bounds_tuple;
    use crate::gis::error::GisError;
    use crate::gis::osm::{parse_osm_features_value, query_osm_features_value};
    use crate::gis::py_json::{json_to_py, py_to_json};

    #[pyfunction(name = "parse_osm_features", signature = (osm_json, tags = None))]
    pub fn parse_osm_features_py(
        py: Python<'_>,
        osm_json: &Bound<'_, PyAny>,
        tags: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let value = if let Ok(text) = osm_json.extract::<String>() {
            serde_json::from_str(&text).map_err(|err| {
                GisError::InvalidArgument(format!("malformed_payload: invalid OSM JSON: {err}"))
            })?
        } else {
            py_to_json(osm_json)?
        };
        let tags = tags.map(py_to_json).transpose()?;
        let result = parse_osm_features_value(&value, tags.as_ref())?;
        json_to_py(py, &result)
    }

    #[pyfunction(name = "query_osm_features", signature = (aoi, tags, cache = None))]
    pub fn query_osm_features_py(
        py: Python<'_>,
        aoi: (f64, f64, f64, f64),
        tags: &Bound<'_, PyAny>,
        cache: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let aoi = validate_bounds_tuple(aoi, true)?;
        let tags = py_to_json(tags)?;
        if !tags.is_object() {
            return Err(GisError::InvalidArgument(
                "invalid_argument: tags must be a dict".to_string(),
            )
            .into());
        }
        let cache = cache
            .map(|cache| {
                if cache.is_none() {
                    Ok(None)
                } else {
                    let _ = cache.downcast::<PyDict>().map_err(|_| {
                        GisError::InvalidArgument(
                            "invalid_argument: cache must be None or a dict".to_string(),
                        )
                    })?;
                    py_to_json(cache).map(Some)
                }
            })
            .transpose()?
            .flatten();
        let result = query_osm_features_value(aoi, &tags, cache.as_ref())?;
        json_to_py(py, &result)
    }
}

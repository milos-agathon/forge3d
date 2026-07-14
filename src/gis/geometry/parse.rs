use serde_json::{json, Map, Value};

use crate::gis::error::{GisError, GisResult};

use super::model::{Coord, Geometry, NormalizedInput, INVALID_GEOMETRY, UNSUPPORTED_GEOMETRY_TYPE};

pub(super) fn normalize_input(
    source: &Value,
    allow_feature_collection: bool,
) -> GisResult<NormalizedInput> {
    let object = source.as_object().ok_or_else(|| {
        GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: geometry input must be an object"
        ))
    })?;
    let source_type = object
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| object.get("features").map(|_| "FeatureCollection"));
    match source_type {
        Some("FeatureCollection") => {
            if !allow_feature_collection {
                return Err(GisError::InvalidGeometry(format!(
                    "{UNSUPPORTED_GEOMETRY_TYPE}: FeatureCollection is not supported for this operation"
                )));
            }
            let features = object
                .get("features")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    GisError::InvalidGeometry(format!(
                        "{INVALID_GEOMETRY}: FeatureCollection requires a features array"
                    ))
                })?;
            let mut geometries = Vec::with_capacity(features.len());
            for feature in features {
                geometries.push(parse_feature_geometry(feature)?);
            }
            // Feature-level CRS metadata participates in resolution: every
            // declared CRS (collection-level and per-feature) must agree, so a
            // mixed-CRS FeatureCollection raises `crs_mismatch` instead of
            // operating under the collection-level declaration alone.
            let crs = super::crs_resolve::merged_feature_collection_crs(
                extract_crs_metadata(source),
                features.iter().filter_map(extract_crs_metadata),
            )?;
            Ok(NormalizedInput {
                geometries,
                input_geometry_type: "FeatureCollection".to_string(),
                input_count: features.len(),
                crs,
            })
        }
        Some("Feature") => {
            let geometry = parse_feature_geometry(source)?;
            let input_count = geometry_member_count(&geometry);
            Ok(NormalizedInput {
                input_geometry_type: geometry.geometry_type().to_string(),
                geometries: vec![geometry],
                input_count,
                crs: extract_crs_metadata(source),
            })
        }
        Some(_) => {
            let geometry = parse_geometry(source)?;
            let input_count = geometry_member_count(&geometry);
            Ok(NormalizedInput {
                input_geometry_type: geometry.geometry_type().to_string(),
                geometries: vec![geometry],
                input_count,
                crs: extract_crs_metadata(source),
            })
        }
        None => Err(GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: GeoJSON object requires a type string"
        ))),
    }
}

pub(super) fn raw_geometry_type(source: &Value) -> String {
    if let Some(kind) = source.get("type").and_then(Value::as_str) {
        if kind == "Feature" {
            return source
                .get("geometry")
                .map(raw_geometry_type)
                .unwrap_or_else(|| "Unknown".to_string());
        }
        return kind.to_string();
    }
    if source.get("features").is_some() {
        return "FeatureCollection".to_string();
    }
    "Unknown".to_string()
}

fn parse_feature_geometry(feature: &Value) -> GisResult<Geometry> {
    let object = feature.as_object().ok_or_else(|| {
        GisError::InvalidGeometry(format!("{INVALID_GEOMETRY}: feature must be an object"))
    })?;
    if object.get("type").and_then(Value::as_str) != Some("Feature") {
        return Err(GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: features array entries must be Feature objects"
        )));
    }
    let geometry = object.get("geometry").ok_or_else(|| {
        GisError::InvalidGeometry(format!("{INVALID_GEOMETRY}: Feature requires geometry"))
    })?;
    parse_geometry(geometry)
}

fn parse_geometry(value: &Value) -> GisResult<Geometry> {
    if value.is_null() {
        return Ok(Geometry::Empty);
    }
    let object = value.as_object().ok_or_else(|| {
        GisError::InvalidGeometry(format!("{INVALID_GEOMETRY}: geometry must be an object"))
    })?;
    let geometry_type = object.get("type").and_then(Value::as_str).ok_or_else(|| {
        GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: geometry requires a type string"
        ))
    })?;
    match geometry_type {
        "Point" => parse_point_geometry(object),
        "LineString" => parse_line_string_geometry(object),
        "Polygon" => parse_polygon_geometry(object),
        "MultiPoint" => parse_multi_point_geometry(object),
        "MultiLineString" => parse_multi_line_string_geometry(object),
        "MultiPolygon" => parse_multi_polygon_geometry(object),
        "GeometryCollection" => parse_geometry_collection(object),
        other => Err(GisError::InvalidGeometry(format!(
            "{UNSUPPORTED_GEOMETRY_TYPE}: unsupported geometry type {other:?}"
        ))),
    }
}

fn parse_point_geometry(object: &Map<String, Value>) -> GisResult<Geometry> {
    let coordinates = required_coordinates(object)?;
    if coordinates.as_array().is_some_and(Vec::is_empty) {
        return Ok(Geometry::Empty);
    }
    Ok(Geometry::Point(parse_position(coordinates)?))
}

fn parse_line_string_geometry(object: &Map<String, Value>) -> GisResult<Geometry> {
    let coordinates = required_coordinates(object)?;
    if coordinates.as_array().is_some_and(Vec::is_empty) {
        return Ok(Geometry::Empty);
    }
    Ok(Geometry::LineString(parse_position_array(coordinates)?))
}

fn parse_polygon_geometry(object: &Map<String, Value>) -> GisResult<Geometry> {
    let coordinates = required_coordinates(object)?;
    let rings = coordinates.as_array().ok_or_else(|| {
        GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: Polygon coordinates must be an array"
        ))
    })?;
    if rings.is_empty() {
        return Ok(Geometry::Empty);
    }
    rings
        .iter()
        .map(parse_position_array)
        .collect::<GisResult<Vec<_>>>()
        .map(Geometry::Polygon)
}

fn parse_multi_point_geometry(object: &Map<String, Value>) -> GisResult<Geometry> {
    let coordinates = required_coordinates(object)?;
    if coordinates.as_array().is_some_and(Vec::is_empty) {
        return Ok(Geometry::Empty);
    }
    Ok(Geometry::MultiPoint(parse_position_array(coordinates)?))
}

fn parse_multi_line_string_geometry(object: &Map<String, Value>) -> GisResult<Geometry> {
    let coordinates = required_coordinates(object)?;
    let lines = coordinates.as_array().ok_or_else(|| {
        GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: MultiLineString coordinates must be an array"
        ))
    })?;
    if lines.is_empty() {
        return Ok(Geometry::Empty);
    }
    lines
        .iter()
        .map(parse_position_array)
        .collect::<GisResult<Vec<_>>>()
        .map(Geometry::MultiLineString)
}

fn parse_multi_polygon_geometry(object: &Map<String, Value>) -> GisResult<Geometry> {
    let coordinates = required_coordinates(object)?;
    let polygons = coordinates.as_array().ok_or_else(|| {
        GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: MultiPolygon coordinates must be an array"
        ))
    })?;
    if polygons.is_empty() {
        return Ok(Geometry::Empty);
    }
    let mut out = Vec::with_capacity(polygons.len());
    for polygon in polygons {
        let rings = polygon.as_array().ok_or_else(|| {
            GisError::InvalidGeometry(format!(
                "{INVALID_GEOMETRY}: MultiPolygon polygon coordinates must be arrays"
            ))
        })?;
        out.push(
            rings
                .iter()
                .map(parse_position_array)
                .collect::<GisResult<Vec<_>>>()?,
        );
    }
    Ok(Geometry::MultiPolygon(out))
}

fn parse_geometry_collection(object: &Map<String, Value>) -> GisResult<Geometry> {
    let geometries = object
        .get("geometries")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            GisError::InvalidGeometry(format!(
                "{INVALID_GEOMETRY}: GeometryCollection requires a geometries array"
            ))
        })?;
    geometries
        .iter()
        .map(parse_geometry)
        .collect::<GisResult<Vec<_>>>()
        .map(Geometry::Collection)
}

fn required_coordinates(object: &Map<String, Value>) -> GisResult<&Value> {
    object.get("coordinates").ok_or_else(|| {
        GisError::InvalidGeometry(format!("{INVALID_GEOMETRY}: geometry requires coordinates"))
    })
}

fn parse_position_array(value: &Value) -> GisResult<Vec<Coord>> {
    let coordinates = value.as_array().ok_or_else(|| {
        GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: coordinate sequence must be an array"
        ))
    })?;
    coordinates.iter().map(parse_position).collect()
}

fn parse_position(value: &Value) -> GisResult<Coord> {
    let items = value.as_array().ok_or_else(|| {
        GisError::InvalidGeometry(format!("{INVALID_GEOMETRY}: position must be an array"))
    })?;
    if items.len() < 2 {
        return Err(GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: position requires at least x and y"
        )));
    }
    let x = finite_coordinate(&items[0], "x")?;
    let y = finite_coordinate(&items[1], "y")?;
    Ok(Coord { x, y })
}

fn finite_coordinate(value: &Value, axis: &str) -> GisResult<f64> {
    let number = value.as_f64().ok_or_else(|| {
        GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: coordinate {axis} must be numeric"
        ))
    })?;
    if !number.is_finite() {
        return Err(GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: coordinate {axis} must be finite"
        )));
    }
    Ok(number)
}

fn extract_crs_metadata(source: &Value) -> Option<Value> {
    let info = source.get("info")?.as_object()?;
    let wkt = info.get("crs_wkt").cloned().unwrap_or(Value::Null);
    let authority = info.get("crs_authority").cloned().unwrap_or(Value::Null);
    if wkt.is_null() && authority.is_null() {
        return None;
    }
    Some(json!({
        "source_kind": "vector",
        "wkt": wkt,
        "authority": authority,
    }))
}

fn geometry_member_count(geometry: &Geometry) -> usize {
    match geometry {
        Geometry::Collection(geometries) => geometries.len(),
        _ => 1,
    }
}

use serde_json::{json, Map, Value};

use crate::gis::error::{GisError, GisResult};

mod centroid;
mod line_ops;
mod math;
mod measure;
mod model;
mod parse;
#[cfg(feature = "extension-module")]
mod py;
mod validate;

use centroid::centroid_for_geometries;
use line_ops::{
    interpolate_lines, lines_for_interpolation, normalized_target_distance,
    representative_for_geometry, total_line_length, validate_distance,
};
use measure::{measure_geometries, validate_metric_names};
use model::{
    finite_value, operation_value, point_value, BACKEND_UNAVAILABLE, EPSILON,
    UNSUPPORTED_GEOMETRY_TYPE, UNSUPPORTED_OPTION,
};
use parse::normalize_input;
use validate::{validate_geometry_value, validate_input_or_error};

pub fn validate_geometry(source: &Value) -> GisResult<Value> {
    Ok(validate_geometry_value(source))
}

pub fn repair_geometry(source: &Value, method: &str) -> GisResult<Value> {
    if method != "make_valid" {
        return Err(GisError::InvalidArgument(format!(
            "{UNSUPPORTED_OPTION}: unsupported repair method {method:?}"
        )));
    }
    let input = normalize_input(source, false)?;
    if input.input_geometry_type == "FeatureCollection" {
        return Err(GisError::InvalidGeometry(format!(
            "{UNSUPPORTED_GEOMETRY_TYPE}: repair_geometry does not accept FeatureCollection"
        )));
    }
    Err(GisError::BackendUnavailable(format!(
        "{BACKEND_UNAVAILABLE}: geos-topology feature required for make_valid"
    )))
}

pub fn geometry_measure(source: &Value, metrics: &[String]) -> GisResult<Value> {
    let metric_flags = validate_metric_names(metrics)?;
    let input = normalize_input(source, true)?;
    validate_input_or_error(&input)?;
    let stats = measure_geometries(&input.geometries)?;

    let operation = operation_value(
        "geometry_measure",
        &input.input_geometry_type,
        None,
        input.input_count,
        1,
        false,
        input.crs,
        Vec::new(),
    );
    let mut out = Map::new();
    out.insert(
        "area".to_string(),
        if metric_flags.0 && stats.has_area {
            finite_value(stats.area)?
        } else {
            Value::Null
        },
    );
    out.insert(
        "length".to_string(),
        if metric_flags.1 && stats.has_length {
            finite_value(stats.length)?
        } else {
            Value::Null
        },
    );
    out.insert(
        "units".to_string(),
        Value::String("source_crs_planar".to_string()),
    );
    out.insert("operation".to_string(), operation);
    Ok(Value::Object(out))
}

pub fn geometry_centroid(source: &Value) -> GisResult<Value> {
    let input = normalize_input(source, true)?;
    validate_input_or_error(&input)?;
    let centroid = centroid_for_geometries(&input.geometries)?;
    let operation = operation_value(
        "geometry_centroid",
        &input.input_geometry_type,
        Some("Point"),
        input.input_count,
        1,
        false,
        input.crs,
        Vec::new(),
    );
    Ok(json!({
        "geometry": point_value(centroid)?,
        "operation": operation,
    }))
}

pub fn representative_point(source: &Value) -> GisResult<Value> {
    let input = normalize_input(source, false)?;
    if input.input_geometry_type == "FeatureCollection" {
        return Err(GisError::InvalidGeometry(format!(
            "{UNSUPPORTED_GEOMETRY_TYPE}: representative_point does not accept FeatureCollection"
        )));
    }
    let geometry = input
        .geometries
        .first()
        .ok_or_else(model::empty_geometry_error)?;
    let point = representative_for_geometry(geometry)?;
    let operation = operation_value(
        "representative_point",
        &input.input_geometry_type,
        Some("Point"),
        input.input_count,
        1,
        false,
        input.crs,
        Vec::new(),
    );
    Ok(json!({
        "geometry": point_value(point)?,
        "operation": operation,
    }))
}

pub fn interpolate_line(source: &Value, distance: f64, normalized: bool) -> GisResult<Value> {
    validate_distance(distance)?;
    let input = normalize_input(source, false)?;
    if input.input_geometry_type == "FeatureCollection" {
        return Err(GisError::InvalidGeometry(format!(
            "{UNSUPPORTED_GEOMETRY_TYPE}: interpolate_line does not accept FeatureCollection"
        )));
    }
    let geometry = input
        .geometries
        .first()
        .ok_or_else(model::empty_geometry_error)?;
    validate::validate_geometry_or_error(geometry)?;
    let lines = lines_for_interpolation(geometry)?;
    let total = total_line_length(&lines);
    if total <= EPSILON {
        return Err(GisError::InvalidGeometry(format!(
            "{}: line length must be positive",
            model::INVALID_GEOMETRY
        )));
    }
    let target = normalized_target_distance(distance, normalized, total)?;
    let point = interpolate_lines(&lines, target)?;
    let operation = operation_value(
        "interpolate_line",
        &input.input_geometry_type,
        Some("Point"),
        input.input_count,
        1,
        false,
        input.crs,
        Vec::new(),
    );
    Ok(json!({
        "geometry": point_value(point)?,
        "distance": distance,
        "normalized": normalized,
        "operation": operation,
    }))
}

#[cfg(feature = "extension-module")]
pub use py::{
    geometry_centroid_py, geometry_measure_py, interpolate_line_py, repair_geometry_py,
    representative_point_py, validate_geometry_py,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polygon_area_length_and_centroid_are_planar() {
        let geometry = json!({
            "type": "Polygon",
            "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]
        });

        let measure = geometry_measure(&geometry, &["area".to_string(), "length".to_string()])
            .expect("measurement succeeds");
        let centroid = geometry_centroid(&geometry).expect("centroid succeeds");

        assert_eq!(measure["area"], json!(1.0));
        assert_eq!(measure["length"], json!(4.0));
        assert_eq!(centroid["geometry"]["coordinates"], json!([0.5, 0.5]));
    }

    #[test]
    fn bowtie_polygon_is_invalid() {
        let geometry = json!({
            "type": "Polygon",
            "coordinates": [[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]]
        });

        let result = validate_geometry(&geometry).expect("validation returns a result");

        assert_eq!(result["valid"], json!(false));
        assert!(result["reason"]
            .as_str()
            .unwrap()
            .contains(model::INVALID_GEOMETRY));
    }

    #[test]
    fn multiline_interpolation_is_cumulative() {
        let geometry = json!({
            "type": "MultiLineString",
            "coordinates": [
                [[0.0, 0.0], [2.0, 0.0]],
                [[2.0, 0.0], [2.0, 2.0]]
            ]
        });

        let result = interpolate_line(&geometry, 3.0, false).expect("interpolation succeeds");

        assert_eq!(result["geometry"]["coordinates"], json!([2.0, 1.0]));
    }
}

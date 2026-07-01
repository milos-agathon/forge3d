use serde_json::{json, Value};

use crate::gis::error::{GisError, GisResult};
use crate::gis::types::RasterWarning;

use super::math::{ring_self_intersects, ring_signed_area_and_centroid};
use super::model::{
    empty_geometry_error, warnings_value, Geometry, NormalizedInput, ValidationState,
    EMPTY_GEOMETRY, INVALID_GEOMETRY, UNSUPPORTED_GEOMETRY_TYPE,
};
use super::parse::{normalize_input, raw_geometry_type};

pub(super) fn validate_geometry_value(source: &Value) -> Value {
    let raw_type = raw_geometry_type(source);
    match normalize_input(source, true) {
        Ok(input) => {
            let mut warnings = Vec::new();
            let mut state = if input.geometries.is_empty() {
                ValidationState {
                    valid: false,
                    reason: Some(format!("{EMPTY_GEOMETRY}: geometry collection is empty")),
                }
            } else {
                ValidationState {
                    valid: true,
                    reason: None,
                }
            };
            for geometry in &input.geometries {
                let item_state = validate_parsed_geometry(geometry);
                if !item_state.valid {
                    state = item_state;
                    break;
                }
            }
            if let Some(reason) = state.reason.as_deref() {
                warnings.push(warning_for_reason(reason));
            }
            validation_result_value(
                state.valid,
                state.reason,
                input.input_geometry_type,
                warnings,
            )
        }
        Err(err) => {
            let reason = err.message();
            validation_result_value(
                false,
                Some(reason.clone()),
                raw_type,
                vec![warning_for_reason(&reason)],
            )
        }
    }
}

pub(super) fn validate_input_or_error(input: &NormalizedInput) -> GisResult<()> {
    if input.geometries.is_empty() {
        return Err(empty_geometry_error());
    }
    for geometry in &input.geometries {
        validate_geometry_or_error(geometry)?;
    }
    Ok(())
}

pub(super) fn validate_geometry_or_error(geometry: &Geometry) -> GisResult<()> {
    let state = validate_parsed_geometry(geometry);
    if state.valid {
        Ok(())
    } else {
        Err(GisError::InvalidGeometry(state.reason.unwrap_or_else(
            || format!("{INVALID_GEOMETRY}: invalid geometry"),
        )))
    }
}

fn validate_parsed_geometry(geometry: &Geometry) -> ValidationState {
    match geometry {
        Geometry::Empty => invalid_state(EMPTY_GEOMETRY, "geometry is empty"),
        Geometry::Point(_) => valid_state(),
        Geometry::LineString(points) => validate_line(points, "LineString"),
        Geometry::Polygon(rings) => validate_polygon(rings),
        Geometry::MultiPoint(points) => {
            if points.is_empty() {
                invalid_state(EMPTY_GEOMETRY, "MultiPoint is empty")
            } else {
                valid_state()
            }
        }
        Geometry::MultiLineString(lines) => {
            if lines.is_empty() {
                return invalid_state(EMPTY_GEOMETRY, "MultiLineString is empty");
            }
            for line in lines {
                let state = validate_line(line, "MultiLineString member");
                if !state.valid {
                    return state;
                }
            }
            valid_state()
        }
        Geometry::MultiPolygon(polygons) => {
            if polygons.is_empty() {
                return invalid_state(EMPTY_GEOMETRY, "MultiPolygon is empty");
            }
            for polygon in polygons {
                let state = validate_polygon(polygon);
                if !state.valid {
                    return state;
                }
            }
            valid_state()
        }
        Geometry::Collection(geometries) => {
            if geometries.is_empty() {
                return invalid_state(EMPTY_GEOMETRY, "GeometryCollection is empty");
            }
            for item in geometries {
                let state = validate_parsed_geometry(item);
                if !state.valid {
                    return state;
                }
            }
            valid_state()
        }
    }
}

fn validate_line(points: &[super::model::Coord], label: &str) -> ValidationState {
    if points.is_empty() {
        invalid_state(EMPTY_GEOMETRY, format!("{label} is empty"))
    } else if points.len() < 2 {
        invalid_state(
            INVALID_GEOMETRY,
            format!("{label} requires at least two positions"),
        )
    } else {
        valid_state()
    }
}

fn validate_polygon(rings: &[Vec<super::model::Coord>]) -> ValidationState {
    if rings.is_empty() {
        return invalid_state(EMPTY_GEOMETRY, "Polygon is empty");
    }
    for (index, ring) in rings.iter().enumerate() {
        if ring.len() < 4 {
            return invalid_state(
                INVALID_GEOMETRY,
                format!("Polygon ring {index} requires at least four positions"),
            );
        }
        if ring.first() != ring.last() {
            return invalid_state(
                INVALID_GEOMETRY,
                format!("Polygon ring {index} is not closed"),
            );
        }
        if ring_self_intersects(ring) {
            return invalid_state(
                INVALID_GEOMETRY,
                format!("Polygon ring {index} has an obvious self-intersection"),
            );
        }
        if ring_signed_area_and_centroid(ring).is_none() {
            return invalid_state(
                INVALID_GEOMETRY,
                format!("Polygon ring {index} has zero area"),
            );
        }
    }
    valid_state()
}

fn valid_state() -> ValidationState {
    ValidationState {
        valid: true,
        reason: None,
    }
}

fn invalid_state(code: &str, message: impl Into<String>) -> ValidationState {
    ValidationState {
        valid: false,
        reason: Some(format!("{code}: {}", message.into())),
    }
}

fn warning_for_reason(reason: &str) -> RasterWarning {
    let code = if reason.contains(UNSUPPORTED_GEOMETRY_TYPE) {
        UNSUPPORTED_GEOMETRY_TYPE
    } else if reason.contains(EMPTY_GEOMETRY) {
        EMPTY_GEOMETRY
    } else {
        INVALID_GEOMETRY
    };
    RasterWarning::new(code, reason.to_string(), Some("geometry"))
}

fn validation_result_value(
    valid: bool,
    reason: Option<String>,
    geometry_type: String,
    warnings: Vec<RasterWarning>,
) -> Value {
    json!({
        "valid": valid,
        "reason": reason,
        "geometry_type": geometry_type,
        "warnings": warnings_value(&warnings),
    })
}

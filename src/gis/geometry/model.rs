use serde_json::{json, Map, Number, Value};

use crate::gis::error::{GisError, GisResult};
use crate::gis::types::RasterWarning;

pub(super) const EMPTY_GEOMETRY: &str = "empty_geometry";
pub(super) const INVALID_GEOMETRY: &str = "invalid_geometry";
pub(super) const UNSUPPORTED_GEOMETRY_TYPE: &str = "unsupported_geometry_type";
pub(super) const UNSUPPORTED_OPTION: &str = "unsupported_option";
pub(super) const INVALID_ARGUMENT: &str = "invalid_argument";
pub(super) const BACKEND_UNAVAILABLE: &str = "backend_unavailable";

pub(super) const EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Coord {
    pub(super) x: f64,
    pub(super) y: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Geometry {
    Empty,
    Point(Coord),
    LineString(Vec<Coord>),
    Polygon(Vec<Vec<Coord>>),
    MultiPoint(Vec<Coord>),
    MultiLineString(Vec<Vec<Coord>>),
    MultiPolygon(Vec<Vec<Vec<Coord>>>),
    GeometryCollection(Vec<Geometry>),
}

impl Geometry {
    pub(super) fn geometry_type(&self) -> &'static str {
        match self {
            Geometry::Empty => "Empty",
            Geometry::Point(_) => "Point",
            Geometry::LineString(_) => "LineString",
            Geometry::Polygon(_) => "Polygon",
            Geometry::MultiPoint(_) => "MultiPoint",
            Geometry::MultiLineString(_) => "MultiLineString",
            Geometry::MultiPolygon(_) => "MultiPolygon",
            Geometry::GeometryCollection(_) => "GeometryCollection",
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        match self {
            Geometry::Empty => true,
            Geometry::Point(_) => false,
            Geometry::LineString(points) => points.is_empty(),
            Geometry::Polygon(rings) => rings.is_empty(),
            Geometry::MultiPoint(points) => points.is_empty(),
            Geometry::MultiLineString(lines) => lines.is_empty(),
            Geometry::MultiPolygon(polygons) => polygons.is_empty(),
            Geometry::GeometryCollection(geometries) => geometries.is_empty(),
        }
    }

    pub(super) fn is_polygonal(&self) -> bool {
        match self {
            Geometry::Polygon(_) | Geometry::MultiPolygon(_) => true,
            Geometry::GeometryCollection(geometries) => {
                geometries.iter().any(Geometry::is_polygonal)
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct NormalizedInput {
    pub(super) geometries: Vec<Geometry>,
    pub(super) input_geometry_type: String,
    pub(super) input_count: usize,
    pub(super) crs: Option<Value>,
}

#[derive(Debug, Clone)]
pub(super) struct ValidationState {
    pub(super) valid: bool,
    pub(super) reason: Option<String>,
}

#[derive(Default)]
pub(super) struct MeasureStats {
    pub(super) area: f64,
    pub(super) length: f64,
    pub(super) has_area: bool,
    pub(super) has_length: bool,
}

#[derive(Default)]
pub(super) struct CentroidStats {
    pub(super) polygon_weight: f64,
    pub(super) polygon_x: f64,
    pub(super) polygon_y: f64,
    pub(super) line_weight: f64,
    pub(super) line_x: f64,
    pub(super) line_y: f64,
    pub(super) point_count: usize,
    pub(super) point_x: f64,
    pub(super) point_y: f64,
}

pub(super) fn operation_value(
    name: &str,
    input_geometry_type: &str,
    output_geometry_type: Option<&str>,
    input_count: usize,
    output_count: usize,
    changed: bool,
    crs: Option<Value>,
    warnings: Vec<RasterWarning>,
) -> Value {
    let mut out = Map::new();
    out.insert("name".to_string(), Value::String(name.to_string()));
    out.insert(
        "input_geometry_type".to_string(),
        Value::String(input_geometry_type.to_string()),
    );
    out.insert(
        "output_geometry_type".to_string(),
        output_geometry_type
            .map(|value| Value::String(value.to_string()))
            .unwrap_or(Value::Null),
    );
    out.insert("input_count".to_string(), json!(input_count));
    out.insert("output_count".to_string(), json!(output_count));
    out.insert("changed".to_string(), Value::Bool(changed));
    out.insert("crs".to_string(), crs.unwrap_or(Value::Null));
    out.insert("warnings".to_string(), warnings_value(&warnings));
    Value::Object(out)
}

pub(super) fn point_value(point: Coord) -> GisResult<Value> {
    Ok(json!({
        "type": "Point",
        "coordinates": [finite_value(point.x)?, finite_value(point.y)?],
    }))
}

pub(super) fn finite_value(value: f64) -> GisResult<Value> {
    Number::from_f64(value).map(Value::Number).ok_or_else(|| {
        GisError::InvalidGeometry(format!("{INVALID_GEOMETRY}: numeric result is not finite"))
    })
}

pub(super) fn warnings_value(warnings: &[RasterWarning]) -> Value {
    Value::Array(
        warnings
            .iter()
            .map(|warning| {
                json!({
                    "code": warning.code.clone(),
                    "message": warning.message.clone(),
                    "field": warning.field.clone(),
                })
            })
            .collect(),
    )
}

pub(super) fn empty_geometry_error() -> GisError {
    GisError::InvalidGeometry(format!("{EMPTY_GEOMETRY}: geometry is empty"))
}

pub(super) fn polygon_topology_error(operation: &str) -> GisError {
    GisError::BackendUnavailable(format!(
        "{BACKEND_UNAVAILABLE}: geos-topology feature required for {operation}"
    ))
}

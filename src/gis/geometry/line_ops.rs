use crate::gis::error::{GisError, GisResult};

use super::math::{distance, line_length};
use super::model::{
    empty_geometry_error, polygon_topology_error, Coord, Geometry, EPSILON, INVALID_ARGUMENT,
    INVALID_GEOMETRY, UNSUPPORTED_GEOMETRY_TYPE,
};
use super::validate::validate_geometry_or_error;

pub(super) fn validate_distance(distance: f64) -> GisResult<()> {
    if !distance.is_finite() {
        return Err(GisError::InvalidArgument(format!(
            "{INVALID_ARGUMENT}: distance must be finite"
        )));
    }
    if distance < 0.0 {
        return Err(GisError::InvalidArgument(format!(
            "{INVALID_ARGUMENT}: distance must be non-negative"
        )));
    }
    Ok(())
}

pub(super) fn representative_for_geometry(geometry: &Geometry) -> GisResult<Coord> {
    match geometry {
        Geometry::Empty => Err(empty_geometry_error()),
        Geometry::Point(point) => Ok(*point),
        Geometry::MultiPoint(points) => points.first().copied().ok_or_else(empty_geometry_error),
        Geometry::LineString(points) => {
            validate_geometry_or_error(geometry)?;
            representative_for_lines(std::slice::from_ref(points))
        }
        Geometry::MultiLineString(lines) => {
            validate_geometry_or_error(geometry)?;
            representative_for_lines(lines)
        }
        Geometry::Polygon(_) | Geometry::MultiPolygon(_) => {
            Err(polygon_topology_error("polygon representative_point"))
        }
        Geometry::Collection(geometries) => representative_for_collection(geometries),
    }
}

fn representative_for_collection(geometries: &[Geometry]) -> GisResult<Coord> {
    if geometries
        .iter()
        .any(|geometry| !geometry.is_empty() && geometry.is_polygonal())
    {
        return Err(polygon_topology_error("polygon representative_point"));
    }
    for geometry in geometries {
        if geometry.is_empty() {
            continue;
        }
        if matches!(
            geometry,
            Geometry::Point(_)
                | Geometry::MultiPoint(_)
                | Geometry::LineString(_)
                | Geometry::MultiLineString(_)
        ) {
            return representative_for_geometry(geometry);
        }
        if let Geometry::Collection(_) = geometry {
            return representative_for_geometry(geometry);
        }
    }
    Err(empty_geometry_error())
}

fn representative_for_lines(lines: &[Vec<Coord>]) -> GisResult<Coord> {
    let total = total_line_length(lines);
    if total <= EPSILON {
        return Err(GisError::InvalidGeometry(format!(
            "{INVALID_GEOMETRY}: line length must be positive"
        )));
    }
    interpolate_lines(lines, total * 0.5)
}

pub(super) fn lines_for_interpolation(geometry: &Geometry) -> GisResult<Vec<Vec<Coord>>> {
    match geometry {
        Geometry::LineString(points) => Ok(vec![points.clone()]),
        Geometry::MultiLineString(lines) => Ok(lines.clone()),
        Geometry::Empty => Err(empty_geometry_error()),
        other => Err(GisError::InvalidGeometry(format!(
            "{UNSUPPORTED_GEOMETRY_TYPE}: interpolate_line requires LineString or MultiLineString, got {}",
            other.geometry_type()
        ))),
    }
}

pub(super) fn normalized_target_distance(
    distance: f64,
    normalized: bool,
    total: f64,
) -> GisResult<f64> {
    if normalized {
        if distance > 1.0 {
            return Err(GisError::InvalidArgument(format!(
                "{INVALID_ARGUMENT}: normalized distance must be in [0.0, 1.0]"
            )));
        }
        Ok(distance * total)
    } else {
        if distance > total + EPSILON {
            return Err(GisError::InvalidArgument(format!(
                "{INVALID_ARGUMENT}: distance exceeds line length"
            )));
        }
        Ok(distance.min(total))
    }
}

pub(super) fn total_line_length(lines: &[Vec<Coord>]) -> f64 {
    lines.iter().map(|line| line_length(line)).sum()
}

pub(super) fn interpolate_lines(lines: &[Vec<Coord>], distance_along: f64) -> GisResult<Coord> {
    let mut first = None;
    let mut last = None;
    for line in lines {
        if let Some(point) = line.first().copied() {
            first.get_or_insert(point);
        }
        if let Some(point) = line.last().copied() {
            last = Some(point);
        }
    }
    if distance_along <= EPSILON {
        return first.ok_or_else(empty_geometry_error);
    }
    let total = total_line_length(lines);
    if distance_along >= total - EPSILON {
        return last.ok_or_else(empty_geometry_error);
    }
    let mut traversed = 0.0;
    for line in lines {
        for segment in line.windows(2) {
            let segment_length = distance(segment[0], segment[1]);
            if segment_length <= EPSILON {
                continue;
            }
            if traversed + segment_length >= distance_along {
                let t = (distance_along - traversed) / segment_length;
                return Ok(Coord {
                    x: segment[0].x + (segment[1].x - segment[0].x) * t,
                    y: segment[0].y + (segment[1].y - segment[0].y) * t,
                });
            }
            traversed += segment_length;
        }
    }
    last.ok_or_else(empty_geometry_error)
}

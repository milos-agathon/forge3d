//! Polygonal topology seams.
//!
//! Union, intersection, difference, symmetric difference, validity, and
//! guarded simplification use the in-tree EUCLIDEA core unconditionally.
//! Buffering remains behind `geos-topology`: offset geometry is a separate
//! problem and is deliberately not disguised as a boolean-overlay fallback.

use crate::geometry::overlay::{
    is_valid_polygonal, overlay, BooleanOp, MultiPolygon, Point, Polygon,
};
use crate::gis::error::{GisError, GisResult};

use super::model::{
    Coord, Geometry, BACKEND_UNAVAILABLE, EMPTY_GEOMETRY, INVALID_GEOMETRY,
    UNSUPPORTED_GEOMETRY_TYPE,
};

pub(super) use super::topology_buffer::buffer_topology;
pub(super) use super::topology_simplify::simplify_topology;

pub(crate) fn require_topology_backend(operation: &str) -> GisResult<()> {
    if operation == "buffer_geometry" && !cfg!(feature = "geos-topology") {
        return Err(GisError::BackendUnavailable(format!(
            "{BACKEND_UNAVAILABLE}: geos-topology feature required for buffer_geometry"
        )));
    }
    if operation == "make_valid" {
        return Err(GisError::BackendUnavailable(format!(
            "{BACKEND_UNAVAILABLE}: make_valid is not implemented"
        )));
    }
    Ok(())
}

fn ring_to_overlay(ring: &[Coord]) -> Vec<Point> {
    ring.iter()
        .map(|coord| Point::new(coord.x, coord.y))
        .collect()
}

fn polygon_to_overlay(rings: &[Vec<Coord>]) -> GisResult<Polygon> {
    let exterior = rings
        .first()
        .ok_or_else(|| GisError::InvalidGeometry(format!("{EMPTY_GEOMETRY}: Polygon is empty")))?;
    Ok(Polygon {
        exterior: ring_to_overlay(exterior),
        holes: rings[1..]
            .iter()
            .map(|ring| ring_to_overlay(ring))
            .collect(),
    })
}

fn polygonal_to_overlay(geometry: &Geometry, operation: &str) -> GisResult<MultiPolygon> {
    match geometry {
        Geometry::Empty => Ok(MultiPolygon::default()),
        Geometry::Polygon(rings) => Ok(MultiPolygon(vec![polygon_to_overlay(rings)?])),
        Geometry::MultiPolygon(polygons) => polygons
            .iter()
            .map(|rings| polygon_to_overlay(rings))
            .collect::<GisResult<Vec<_>>>()
            .map(MultiPolygon),
        other => Err(GisError::InvalidGeometry(format!(
            "{UNSUPPORTED_GEOMETRY_TYPE}: {operation} supports Polygon and MultiPolygon, got {}",
            other.geometry_type()
        ))),
    }
}

fn ring_from_overlay(ring: &[Point]) -> Vec<Coord> {
    ring.iter()
        .map(|point| Coord {
            x: point.x,
            y: point.y,
        })
        .collect()
}

fn polygonal_from_overlay(value: MultiPolygon) -> Geometry {
    let mut polygons = value
        .0
        .into_iter()
        .map(|polygon| {
            std::iter::once(ring_from_overlay(&polygon.exterior))
                .chain(polygon.holes.iter().map(|hole| ring_from_overlay(hole)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    match polygons.len() {
        0 => Geometry::Empty,
        1 => Geometry::Polygon(polygons.pop().unwrap()),
        _ => Geometry::MultiPolygon(polygons),
    }
}

fn exact_overlay(
    left: &Geometry,
    right: &Geometry,
    operation_name: &str,
    operation: BooleanOp,
) -> GisResult<Geometry> {
    let left = polygonal_to_overlay(left, operation_name)?;
    let right = polygonal_to_overlay(right, operation_name)?;
    overlay(&left, &right, operation)
        .map(|result| polygonal_from_overlay(result.geometry))
        .map_err(|error| {
            GisError::InvalidGeometry(format!("{INVALID_GEOMETRY}: {operation_name}: {error}"))
        })
}

pub(super) fn union_polygonal(geometries: &[Geometry]) -> GisResult<Geometry> {
    let Some(first) = geometries.first() else {
        return Ok(Geometry::Empty);
    };
    let mut output = first.clone();
    // Validate a singleton too; otherwise an unsupported geometry could pass
    // through without reaching the overlay adapter.
    polygonal_to_overlay(&output, "union_geometries")?;
    for geometry in &geometries[1..] {
        output = exact_overlay(&output, geometry, "union_geometries", BooleanOp::Union)?;
    }
    Ok(output)
}

pub(super) fn intersection_polygonal(
    left: &Geometry,
    right: &Geometry,
    operation: &str,
) -> GisResult<Geometry> {
    exact_overlay(left, right, operation, BooleanOp::Intersection)
}

pub(super) fn difference_polygonal(
    left: &Geometry,
    right: &Geometry,
    operation: &str,
) -> GisResult<Geometry> {
    exact_overlay(left, right, operation, BooleanOp::Difference)
}

pub(super) fn symmetric_difference_polygonal(
    left: &Geometry,
    right: &Geometry,
    operation: &str,
) -> GisResult<Geometry> {
    exact_overlay(left, right, operation, BooleanOp::SymmetricDifference)
}

pub(super) fn polygonal_validity(
    geometry: &Geometry,
) -> GisResult<crate::geometry::overlay::ValidityReport> {
    polygonal_to_overlay(geometry, "is_valid").map(|value| is_valid_polygonal(&value))
}

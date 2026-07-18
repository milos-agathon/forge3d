//! Deterministic simplification followed by exact polygon validity refusal.

use crate::gis::error::{GisError, GisResult};

use super::model::{Coord, Geometry, INVALID_GEOMETRY, UNSUPPORTED_GEOMETRY_TYPE};
use super::topology::polygonal_validity;

fn point_segment_distance2(point: Coord, start: Coord, end: Coord) -> f64 {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let length2 = dx * dx + dy * dy;
    if length2 == 0.0 {
        return (point.x - start.x).powi(2) + (point.y - start.y).powi(2);
    }
    let position =
        (((point.x - start.x) * dx + (point.y - start.y) * dy) / length2).clamp(0.0, 1.0);
    let x = start.x + position * dx;
    let y = start.y + position * dy;
    (point.x - x).powi(2) + (point.y - y).powi(2)
}

fn simplify_line(points: &[Coord], tolerance: f64) -> Vec<Coord> {
    if points.len() <= 2 || tolerance == 0.0 {
        return points.to_vec();
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    let mut ranges = vec![(0usize, points.len() - 1)];
    while let Some((start, end)) = ranges.pop() {
        let mut farthest = None;
        for index in (start + 1)..end {
            let distance2 = point_segment_distance2(points[index], points[start], points[end]);
            if farthest.is_none_or(|(_, best)| distance2 > best) {
                farthest = Some((index, distance2));
            }
        }
        if let Some((index, distance2)) = farthest {
            if distance2 > tolerance * tolerance {
                keep[index] = true;
                ranges.push((start, index));
                ranges.push((index, end));
            }
        }
    }
    points
        .iter()
        .copied()
        .zip(keep)
        .filter_map(|(point, keep)| keep.then_some(point))
        .collect()
}

fn simplify_ring(ring: &[Coord], tolerance: f64) -> Vec<Coord> {
    if ring.len() <= 4 || tolerance == 0.0 {
        return ring.to_vec();
    }
    let mut points = ring[..ring.len() - 1].to_vec();
    loop {
        let candidate = (0..points.len())
            .map(|index| {
                let previous = points[(index + points.len() - 1) % points.len()];
                let current = points[index];
                let next = points[(index + 1) % points.len()];
                let area2 = crate::geometry::exact::orient2d(
                    [previous.x, previous.y],
                    [current.x, current.y],
                    [next.x, next.y],
                )
                .abs();
                (index, area2)
            })
            .min_by(|left, right| left.1.total_cmp(&right.1));
        let Some((index, area2)) = candidate else {
            break;
        };
        if points.len() <= 3 || area2 > tolerance * tolerance {
            break;
        }
        points.remove(index);
    }
    points.push(points[0]);
    points
}

pub(super) fn simplify_topology(
    geometry: &Geometry,
    tolerance: f64,
    _preserve_topology: bool,
) -> GisResult<Geometry> {
    let simplified = match geometry {
        Geometry::LineString(points) => Geometry::LineString(simplify_line(points, tolerance)),
        Geometry::MultiLineString(lines) => Geometry::MultiLineString(
            lines
                .iter()
                .map(|line| simplify_line(line, tolerance))
                .collect(),
        ),
        Geometry::Polygon(rings) => Geometry::Polygon(
            rings
                .iter()
                .map(|ring| simplify_ring(ring, tolerance))
                .collect(),
        ),
        Geometry::MultiPolygon(polygons) => Geometry::MultiPolygon(
            polygons
                .iter()
                .map(|rings| {
                    rings
                        .iter()
                        .map(|ring| simplify_ring(ring, tolerance))
                        .collect()
                })
                .collect(),
        ),
        other => {
            return Err(GisError::InvalidGeometry(format!(
                "{UNSUPPORTED_GEOMETRY_TYPE}: simplify_geometry supports LineString, MultiLineString, Polygon, and MultiPolygon, got {}",
                other.geometry_type()
            )))
        }
    };
    if matches!(simplified, Geometry::Polygon(_) | Geometry::MultiPolygon(_)) {
        let report = polygonal_validity(&simplified)?;
        if !report.valid {
            return Err(GisError::InvalidGeometry(format!(
                "{INVALID_GEOMETRY}: simplify tolerance breaks polygon validity: {}",
                report.reasons.join("; ")
            )));
        }
    }
    Ok(simplified)
}

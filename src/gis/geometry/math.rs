use once_cell::sync::Lazy;

use crate::geo::geodesic::{polygon_perimeter_area, polyline_length, Geodesic};
use crate::gis::error::{GisError, GisResult};

use super::model::{Coord, Geometry, EPSILON, INVALID_GEOMETRY};

static WGS84_GEODESIC: Lazy<Geodesic> = Lazy::new(Geodesic::wgs84);

/// Karney geodesic length of a polyline whose coordinates are WGS84 lon/lat
/// degrees. Returns metres.
pub(super) fn geodesic_line_length(points: &[Coord]) -> f64 {
    let pts: Vec<(f64, f64)> = points.iter().map(|c| (c.x, c.y)).collect();
    polyline_length(&WGS84_GEODESIC, &pts)
}

/// Geodesic (authalic) polygon area on the WGS84 ellipsoid, in m². Outer ring
/// first, holes subtracted — mirrors the planar `polygon_area` contract.
/// Handles the antimeridian via geodesic crossing counting.
pub(super) fn geodesic_polygon_area(rings: &[Vec<Coord>]) -> GisResult<f64> {
    let mut total = 0.0;
    for (index, ring) in rings.iter().enumerate() {
        let pts: Vec<(f64, f64)> = ring.iter().map(|c| (c.x, c.y)).collect();
        let (_, area) = polygon_perimeter_area(&WGS84_GEODESIC, &pts);
        if area <= EPSILON {
            return Err(GisError::InvalidGeometry(format!(
                "{INVALID_GEOMETRY}: polygon ring has zero area"
            )));
        }
        if index == 0 {
            total += area;
        } else {
            total -= area;
        }
    }
    Ok(total.max(0.0))
}

/// Geodesic perimeter of a polygon (sum of closed-ring lengths), metres.
pub(super) fn geodesic_polygon_perimeter(rings: &[Vec<Coord>]) -> f64 {
    rings.iter().map(|ring| geodesic_line_length(ring)).sum()
}

/// True when every coordinate is plausibly geographic lon/lat degrees.
pub(super) fn looks_geographic(geometries: &[Geometry]) -> bool {
    fn coords_ok(points: &[Coord]) -> bool {
        points
            .iter()
            .all(|c| c.x.abs() <= 180.0 + 1e-9 && c.y.abs() <= 90.0 + 1e-9)
    }
    fn geometry_ok(geometry: &Geometry) -> bool {
        match geometry {
            Geometry::Empty => true,
            Geometry::Point(p) => coords_ok(core::slice::from_ref(p)),
            Geometry::MultiPoint(ps) | Geometry::LineString(ps) => coords_ok(ps),
            Geometry::MultiLineString(lines) => lines.iter().all(|l| coords_ok(l)),
            Geometry::Polygon(rings) => rings.iter().all(|r| coords_ok(r)),
            Geometry::MultiPolygon(polys) => {
                polys.iter().all(|rings| rings.iter().all(|r| coords_ok(r)))
            }
            Geometry::Collection(items) => items.iter().all(geometry_ok),
        }
    }
    geometries.iter().all(geometry_ok)
}

/// Unwrap antimeridian-crossing sequences: whenever consecutive longitudes
/// jump by more than 180°, shift subsequent points by ±360° so the sequence
/// is continuous (179 → -179 becomes 179 → 181). Returns true if anything
/// changed. Only meaningful for geographic coordinates — gate on
/// `looks_geographic` first.
pub(super) fn unwrap_dateline(geometries: &mut [Geometry]) -> bool {
    fn unwrap_points(points: &mut [Coord]) -> bool {
        let mut offset = 0.0f64;
        let mut changed = false;
        for i in 1..points.len() {
            let prev = points[i - 1].x;
            let mut adj = points[i].x + offset;
            if adj - prev > 180.0 {
                offset -= 360.0;
                adj -= 360.0;
                changed = true;
            } else if adj - prev < -180.0 {
                offset += 360.0;
                adj += 360.0;
                changed = true;
            }
            points[i].x = adj;
        }
        changed
    }
    fn unwrap_geometry(geometry: &mut Geometry) -> bool {
        fn unwrap_all<'a>(items: impl IntoIterator<Item = &'a mut Vec<Coord>>) -> bool {
            let mut changed = false;
            for item in items {
                changed |= unwrap_points(item);
            }
            changed
        }
        match geometry {
            Geometry::Empty | Geometry::Point(_) | Geometry::MultiPoint(_) => false,
            Geometry::LineString(ps) => unwrap_points(ps),
            Geometry::MultiLineString(lines) | Geometry::Polygon(lines) => unwrap_all(lines),
            Geometry::MultiPolygon(polys) => {
                let mut changed = false;
                for rings in polys {
                    changed |= unwrap_all(rings);
                }
                changed
            }
            Geometry::Collection(items) => {
                let mut changed = false;
                for item in items {
                    changed |= unwrap_geometry(item);
                }
                changed
            }
        }
    }
    let mut changed = false;
    for geometry in geometries {
        changed |= unwrap_geometry(geometry);
    }
    changed
}

/// Shift every vertex longitude of a geometry by `delta` degrees (used to
/// align two dateline-unwrapped operands onto the same 360° sheet).
pub(super) fn shift_geometry_lons(geometry: &mut Geometry, delta: f64) {
    fn shift_points(points: &mut [Coord], delta: f64) {
        for p in points {
            p.x += delta;
        }
    }
    match geometry {
        Geometry::Empty => {}
        Geometry::Point(p) => p.x += delta,
        Geometry::MultiPoint(ps) | Geometry::LineString(ps) => shift_points(ps, delta),
        Geometry::MultiLineString(lines) => lines.iter_mut().for_each(|l| shift_points(l, delta)),
        Geometry::Polygon(rings) => rings.iter_mut().for_each(|r| shift_points(r, delta)),
        Geometry::MultiPolygon(polys) => polys
            .iter_mut()
            .for_each(|rings| rings.iter_mut().for_each(|r| shift_points(r, delta))),
        Geometry::Collection(items) => items.iter_mut().for_each(|g| shift_geometry_lons(g, delta)),
    }
}

/// First vertex longitude of a geometry, if it has one.
pub(super) fn first_lon(geometry: &Geometry) -> Option<f64> {
    match geometry {
        Geometry::Empty => None,
        Geometry::Point(p) => Some(p.x),
        Geometry::MultiPoint(ps) | Geometry::LineString(ps) => ps.first().map(|c| c.x),
        Geometry::MultiLineString(lines) => lines.iter().find_map(|l| l.first().map(|c| c.x)),
        Geometry::Polygon(rings) => rings.iter().find_map(|r| r.first().map(|c| c.x)),
        Geometry::MultiPolygon(polys) => polys
            .iter()
            .find_map(|rings| rings.iter().find_map(|r| r.first().map(|c| c.x))),
        Geometry::Collection(items) => items.iter().find_map(first_lon),
    }
}

/// Normalize a longitude back into (-180, 180].
pub(super) fn wrap_lon(x: f64) -> f64 {
    let r = (x + 180.0).rem_euclid(360.0) - 180.0;
    if r == -180.0 {
        180.0
    } else {
        r
    }
}

pub(super) fn polygon_area(rings: &[Vec<Coord>]) -> GisResult<f64> {
    let mut total = 0.0;
    for (index, ring) in rings.iter().enumerate() {
        let area = ring_signed_area_and_centroid(ring)
            .ok_or_else(|| {
                GisError::InvalidGeometry(format!("{INVALID_GEOMETRY}: polygon ring has zero area"))
            })?
            .0
            .abs();
        if index == 0 {
            total += area;
        } else {
            total -= area;
        }
    }
    Ok(total.max(0.0))
}

pub(super) fn polygon_perimeter(rings: &[Vec<Coord>]) -> f64 {
    rings.iter().map(|ring| line_length(ring)).sum()
}

pub(super) fn line_length(points: &[Coord]) -> f64 {
    points
        .windows(2)
        .map(|segment| distance(segment[0], segment[1]))
        .sum()
}

pub(super) fn distance(left: Coord, right: Coord) -> f64 {
    (right.x - left.x).hypot(right.y - left.y)
}

pub(super) fn ring_signed_area_and_centroid(ring: &[Coord]) -> Option<(f64, Coord)> {
    let mut cross_sum = 0.0;
    let mut cx_sum = 0.0;
    let mut cy_sum = 0.0;
    for segment in ring.windows(2) {
        let cross = segment[0].x * segment[1].y - segment[1].x * segment[0].y;
        cross_sum += cross;
        cx_sum += (segment[0].x + segment[1].x) * cross;
        cy_sum += (segment[0].y + segment[1].y) * cross;
    }
    let signed_area = cross_sum * 0.5;
    if signed_area.abs() <= EPSILON {
        return None;
    }
    Some((
        signed_area,
        Coord {
            x: cx_sum / (6.0 * signed_area),
            y: cy_sum / (6.0 * signed_area),
        },
    ))
}

pub(super) fn ring_self_intersects(ring: &[Coord]) -> bool {
    let segment_count = ring.len().saturating_sub(1);
    for i in 0..segment_count {
        for j in (i + 1)..segment_count {
            if j == i + 1 || (i == 0 && j + 1 == segment_count) {
                continue;
            }
            if segments_intersect(ring[i], ring[i + 1], ring[j], ring[j + 1]) {
                return true;
            }
        }
    }
    false
}

fn segments_intersect(a: Coord, b: Coord, c: Coord, d: Coord) -> bool {
    let o1 = orientation(a, b, c);
    let o2 = orientation(a, b, d);
    let o3 = orientation(c, d, a);
    let o4 = orientation(c, d, b);

    if o1 * o2 < -EPSILON && o3 * o4 < -EPSILON {
        return true;
    }
    (o1.abs() <= EPSILON && on_segment(a, c, b))
        || (o2.abs() <= EPSILON && on_segment(a, d, b))
        || (o3.abs() <= EPSILON && on_segment(c, a, d))
        || (o4.abs() <= EPSILON && on_segment(c, b, d))
}

fn orientation(a: Coord, b: Coord, c: Coord) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn on_segment(a: Coord, b: Coord, c: Coord) -> bool {
    b.x >= a.x.min(c.x) - EPSILON
        && b.x <= a.x.max(c.x) + EPSILON
        && b.y >= a.y.min(c.y) - EPSILON
        && b.y <= a.y.max(c.y) + EPSILON
}

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

/// Karney geodesic distance between two WGS84 lon/lat coordinates, metres.
pub(super) fn geodesic_distance(left: Coord, right: Coord) -> f64 {
    WGS84_GEODESIC.inverse(left.y, left.x, right.y, right.x).s12
}

/// Point `distance_m` metres along the geodesic from `from` toward `to`
/// (inverse problem for the departure azimuth, then the direct problem).
pub(super) fn geodesic_point_along(from: Coord, to: Coord, distance_m: f64) -> Coord {
    let inverse = WGS84_GEODESIC.inverse(from.y, from.x, to.y, to.x);
    let direct = WGS84_GEODESIC.direct(from.y, from.x, inverse.azi1, distance_m);
    Coord {
        x: direct.lon2,
        y: direct.lat2,
    }
}

/// Unwrap antimeridian-crossing sequences: whenever consecutive longitudes
/// jump by more than 180°, shift subsequent points by ±360° so the sequence
/// is continuous (179 → -179 becomes 179 → 181). Returns true if anything
/// changed. Only meaningful for geographic (WGS84 lon/lat) coordinates — the
/// caller decides that from an explicit CRS, never from coordinate ranges.
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

/// Apply `f` to every vertex longitude of a geometry in place. The single
/// recursive traversal shared by every whole-geometry longitude rewrite.
fn for_each_lon(geometry: &mut Geometry, f: &mut impl FnMut(&mut f64)) {
    fn points(points: &mut [Coord], f: &mut impl FnMut(&mut f64)) {
        for p in points {
            f(&mut p.x);
        }
    }
    match geometry {
        Geometry::Empty => {}
        Geometry::Point(p) => f(&mut p.x),
        Geometry::MultiPoint(ps) | Geometry::LineString(ps) => points(ps, f),
        Geometry::MultiLineString(lines) | Geometry::Polygon(lines) => {
            for line in lines {
                points(line, f);
            }
        }
        Geometry::MultiPolygon(polys) => {
            for rings in polys {
                for ring in rings {
                    points(ring, f);
                }
            }
        }
        Geometry::Collection(items) => {
            for item in items {
                for_each_lon(item, f);
            }
        }
    }
}

/// Wrap every vertex longitude of a geometry back into (-180, 180] in place.
/// Used to return a continuous-sheet union result to authored coordinates so a
/// downstream pair-unwrap can re-detect (and split) an antimeridian crossing
/// instead of receiving an already-split, multi-sheet mask.
pub(super) fn wrap_geometry_lons(geometry: &mut Geometry) {
    for_each_lon(geometry, &mut |x| *x = wrap_lon(*x));
}

/// Mean outer-ring longitude of one polygonal part.
fn part_mean_lon(rings: &[Vec<Coord>]) -> Option<f64> {
    let outer = rings.first()?;
    if outer.is_empty() {
        return None;
    }
    Some(outer.iter().map(|c| c.x).sum::<f64>() / outer.len() as f64)
}

/// Mean outer-ring longitude of the first polygonal part of a geometry — the
/// reference sheet for `align_parts_to_sheet`.
pub(super) fn first_part_mean_lon(geometry: &Geometry) -> Option<f64> {
    match geometry {
        Geometry::Polygon(rings) => part_mean_lon(rings),
        Geometry::MultiPolygon(polys) => polys.iter().find_map(|rings| part_mean_lon(rings)),
        _ => None,
    }
}

/// Shift each polygonal part of `geometry` by the 360°-multiple that brings
/// its mean outer-ring longitude within 180° of `reference`, so both operands
/// of a geographic topology op share one continuous longitude sheet. A
/// whole-geometry shift is not enough: a MultiPolygon that was previously
/// split at the antimeridian has parts on OPPOSITE sheets, and aligning only
/// by its first vertex strands the other part 360° away, silently dropping it
/// from the intersection. Returns true if any part moved.
pub(super) fn align_parts_to_sheet(geometry: &mut Geometry, reference: f64) -> bool {
    fn align_part(rings: &mut [Vec<Coord>], reference: f64) -> bool {
        let Some(mean) = part_mean_lon(rings) else {
            return false;
        };
        let delta = 360.0 * ((reference - mean) / 360.0).round();
        if delta == 0.0 {
            return false;
        }
        for ring in rings {
            for p in ring.iter_mut() {
                p.x += delta;
            }
        }
        true
    }
    match geometry {
        Geometry::Polygon(rings) => align_part(rings, reference),
        Geometry::MultiPolygon(polys) => {
            let mut changed = false;
            for rings in polys {
                changed |= align_part(rings, reference);
            }
            changed
        }
        _ => false,
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

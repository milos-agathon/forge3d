use crate::gis::error::{GisError, GisResult};

use super::model::{Coord, EPSILON, INVALID_GEOMETRY};

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

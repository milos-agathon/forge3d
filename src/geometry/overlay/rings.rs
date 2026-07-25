//! DCEL boundary traversal, canonical ring orientation, and hole assignment.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::geometry::exact::predicates::{orient2d, sign_ordering, signed_area2};

use super::faces::{locate_in_ring, Location, PointKey};
use super::{MultiPolygon, OverlayError, Point, Polygon, Ring, Segment};

pub(crate) fn stitch(edges: &[Segment]) -> Result<Vec<Ring>, OverlayError> {
    let mut outgoing: BTreeMap<PointKey, Vec<usize>> = BTreeMap::new();
    for (index, edge) in edges.iter().enumerate() {
        outgoing
            .entry(PointKey::of(edge.start))
            .or_default()
            .push(index);
    }
    for indices in outgoing.values_mut() {
        indices.sort_by_key(|index| PointKey::of(edges[*index].end));
    }
    let mut unused = (0..edges.len()).collect::<BTreeSet<_>>();
    let mut rings = Vec::new();
    while let Some(&first) = unused.iter().next() {
        let start = edges[first].start;
        let mut ring = vec![start];
        let mut current = first;
        loop {
            if !unused.remove(&current) {
                return Err(OverlayError(
                    "boundary edge reused while stitching".to_string(),
                ));
            }
            let end = edges[current].end;
            ring.push(end);
            if end == start {
                break;
            }
            let candidates = outgoing.get(&PointKey::of(end)).ok_or_else(|| {
                OverlayError(format!(
                    "open boundary after snap-rounded subdivision at {end:?}; edges={edges:?}"
                ))
            })?;
            let mut available = candidates
                .iter()
                .copied()
                .filter(|candidate| unused.contains(candidate))
                .collect::<Vec<_>>();
            if available.is_empty() {
                return Err(OverlayError(
                    "no unused outgoing boundary edge while stitching".to_string(),
                ));
            }
            // DCEL left-face traversal: choose the outgoing half-edge
            // immediately clockwise from the incoming edge's twin.  Direction
            // ordering uses exact orient2d, never a raw cross-product sign.
            let reverse = Point::new(
                edges[current].start.x - end.x,
                edges[current].start.y - end.y,
            );
            available.sort_by(|left, right| {
                direction_cmp(
                    Point::new(edges[*left].end.x - end.x, edges[*left].end.y - end.y),
                    Point::new(edges[*right].end.x - end.x, edges[*right].end.y - end.y),
                )
                .then_with(|| PointKey::of(edges[*left].end).cmp(&PointKey::of(edges[*right].end)))
            });
            current = available
                .iter()
                .copied()
                .rev()
                .find(|candidate| {
                    direction_cmp(
                        Point::new(
                            edges[*candidate].end.x - end.x,
                            edges[*candidate].end.y - end.y,
                        ),
                        reverse,
                    ) == Ordering::Less
                })
                .unwrap_or_else(|| *available.last().unwrap());
            if ring.len() > edges.len() + 1 {
                return Err(OverlayError("boundary stitching did not close".to_string()));
            }
        }
        if ring.len() >= 4 {
            rings.push(ring);
        }
    }
    Ok(rings)
}

fn upper_half(direction: Point) -> bool {
    direction.y > 0.0 || (direction.y == 0.0 && direction.x >= 0.0)
}

fn direction_cmp(left: Point, right: Point) -> Ordering {
    upper_half(left)
        .cmp(&upper_half(right))
        .reverse()
        .then_with(|| {
            sign_ordering(orient2d([0.0, 0.0], left.as_array(), right.as_array())).reverse()
        })
}

fn compare_points(left: Point, right: Point) -> Ordering {
    left.x
        .total_cmp(&right.x)
        .then_with(|| left.y.total_cmp(&right.y))
}

fn canonicalize(mut ring: Ring, want_ccw: bool) -> Ring {
    ring.pop();
    let area_positive = signed_area2(
        &ring
            .iter()
            .copied()
            .map(Point::as_array)
            .collect::<Vec<_>>(),
    ) > 0.0;
    if area_positive != want_ccw {
        ring.reverse();
    }
    let first = ring
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| compare_points(**a, **b))
        .map(|(index, _)| index)
        .unwrap_or(0);
    ring.rotate_left(first);
    ring.push(ring[0]);
    ring
}

fn ring_area_abs(ring: &Ring) -> f64 {
    let Some(anchor) = ring.first().copied() else {
        return 0.0;
    };
    let mut area2 = 0.0f64;
    for index in 1..ring.len().saturating_sub(2) {
        area2 += orient2d(
            anchor.as_array(),
            ring[index].as_array(),
            ring[index + 1].as_array(),
        );
    }
    area2.abs() * 0.5
}

pub(crate) fn build_polygonal(rings: Vec<Ring>) -> Result<MultiPolygon, OverlayError> {
    let mut shells = Vec::new();
    let mut holes = Vec::new();
    for ring in rings {
        let points = ring
            .iter()
            .copied()
            .map(Point::as_array)
            .collect::<Vec<_>>();
        match sign_ordering(signed_area2(&points)) {
            Ordering::Greater => shells.push(canonicalize(ring, true)),
            Ordering::Less => holes.push(canonicalize(ring, false)),
            Ordering::Equal => {}
        }
    }
    let mut polygons = shells
        .into_iter()
        .map(|exterior| Polygon {
            exterior,
            holes: Vec::new(),
        })
        .collect::<Vec<_>>();
    for hole in holes {
        let sample = hole[0];
        let owner = polygons
            .iter()
            .enumerate()
            .filter(|(_, polygon)| locate_in_ring(sample, &polygon.exterior) != Location::Outside)
            .min_by(|(_, a), (_, b)| {
                ring_area_abs(&a.exterior).total_cmp(&ring_area_abs(&b.exterior))
            })
            .map(|(index, _)| index)
            .ok_or_else(|| OverlayError("orphan hole after face assembly".to_string()))?;
        polygons[owner].holes.push(hole);
    }
    for polygon in &mut polygons {
        polygon.holes.sort_by(|a, b| compare_points(a[0], b[0]));
    }
    polygons.sort_by(|a, b| compare_points(a.exterior[0], b.exterior[0]));
    Ok(MultiPolygon(polygons))
}

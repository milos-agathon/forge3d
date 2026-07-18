//! Construct boundary rings and assign holes from snap-rounded atomic edges.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::geometry::exact::predicates::{orient2d, sign_ordering};

use super::snap::SnapGrid;
use super::{BooleanOp, MultiPolygon, OverlayError, Point, Ring, Segment};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PointKey(u64, u64);

impl PointKey {
    pub(crate) fn of(point: Point) -> Self {
        Self(point.x.to_bits(), point.y.to_bits())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EdgeKey(PointKey, PointKey);

fn edge_key(segment: Segment) -> EdgeKey {
    let start = PointKey::of(segment.start);
    let end = PointKey::of(segment.end);
    if start <= end {
        EdgeKey(start, end)
    } else {
        EdgeKey(end, start)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Location {
    Outside,
    Inside,
    Boundary,
}

fn point_on_segment(point: Point, segment: Segment) -> bool {
    sign_ordering(orient2d(
        segment.start.as_array(),
        segment.end.as_array(),
        point.as_array(),
    )) == Ordering::Equal
        && point.x >= segment.start.x.min(segment.end.x)
        && point.x <= segment.start.x.max(segment.end.x)
        && point.y >= segment.start.y.min(segment.end.y)
        && point.y <= segment.start.y.max(segment.end.y)
}

pub(crate) fn locate_in_ring(point: Point, ring: &Ring) -> Location {
    let mut winding = 0i32;
    for pair in ring.windows(2) {
        let segment = Segment {
            start: pair[0],
            end: pair[1],
        };
        if point_on_segment(point, segment) {
            return Location::Boundary;
        }
        if pair[0].y <= point.y && pair[1].y > point.y {
            if sign_ordering(orient2d(
                pair[0].as_array(),
                pair[1].as_array(),
                point.as_array(),
            )) == Ordering::Greater
            {
                winding += 1;
            }
        } else if pair[0].y > point.y
            && pair[1].y <= point.y
            && sign_ordering(orient2d(
                pair[0].as_array(),
                pair[1].as_array(),
                point.as_array(),
            )) == Ordering::Less
        {
            winding -= 1;
        }
    }
    if winding == 0 {
        Location::Outside
    } else {
        Location::Inside
    }
}

pub(crate) fn locate(point: Point, polygonal: &MultiPolygon) -> Location {
    for polygon in &polygonal.0 {
        match locate_in_ring(point, &polygon.exterior) {
            Location::Outside => continue,
            Location::Boundary => return Location::Boundary,
            Location::Inside => {
                let mut in_hole = false;
                for hole in &polygon.holes {
                    match locate_in_ring(point, hole) {
                        Location::Boundary => return Location::Boundary,
                        Location::Inside => {
                            in_hole = true;
                            break;
                        }
                        Location::Outside => {}
                    }
                }
                if !in_hole {
                    return Location::Inside;
                }
            }
        }
    }
    Location::Outside
}

fn membership(point: Point, polygonal: &MultiPolygon) -> bool {
    locate(point, polygonal) == Location::Inside
}

fn oriented_boundary_edges(
    atomic: &[Segment],
    left: &MultiPolygon,
    right: &MultiPolygon,
    operation: BooleanOp,
    grid: SnapGrid,
) -> Vec<Segment> {
    let mut unique = BTreeMap::new();
    for segment in atomic.iter().copied() {
        unique.entry(edge_key(segment)).or_insert(segment);
    }
    let mut boundary = Vec::new();
    for segment in unique.into_values() {
        let dx = segment.end.x - segment.start.x;
        let dy = segment.end.y - segment.start.y;
        let length = dx.hypot(dy);
        if length == 0.0 {
            continue;
        }
        let midpoint = Point::new(
            (segment.start.x + segment.end.x) * 0.5,
            (segment.start.y + segment.end.y) * 0.5,
        );
        // Stay inside the hot pixel while moving far enough that both sample
        // coordinates survive binary64 rounding even on diagonal edges.
        let offset = grid.step() * 0.45;
        let normal = Point::new(-dy / length * offset, dx / length * offset);
        let left_sample = Point::new(midpoint.x + normal.x, midpoint.y + normal.y);
        let right_sample = Point::new(midpoint.x - normal.x, midpoint.y - normal.y);
        let result_left = operation.evaluate(
            membership(left_sample, left),
            membership(left_sample, right),
        );
        let result_right = operation.evaluate(
            membership(right_sample, left),
            membership(right_sample, right),
        );
        if result_left != result_right {
            boundary.push(if result_left {
                segment
            } else {
                Segment {
                    start: segment.end,
                    end: segment.start,
                }
            });
        }
    }
    boundary.sort_by(|a, b| {
        PointKey::of(a.start)
            .cmp(&PointKey::of(b.start))
            .then_with(|| PointKey::of(a.end).cmp(&PointKey::of(b.end)))
    });
    boundary
}

pub(crate) fn assemble(
    atomic: &[Segment],
    left: &MultiPolygon,
    right: &MultiPolygon,
    operation: BooleanOp,
    grid: SnapGrid,
) -> Result<MultiPolygon, OverlayError> {
    super::rings::build_polygonal(super::rings::stitch(&oriented_boundary_edges(
        atomic, left, right, operation, grid,
    ))?)
}

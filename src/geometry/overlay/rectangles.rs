//! Exact fast path for the overwhelmingly common axis-aligned rectangle case.

use std::collections::BTreeMap;

use super::{BooleanOp, MultiPolygon, OverlayError, Point, Segment};

#[derive(Clone, Copy)]
struct Bounds {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

fn bounds(value: &MultiPolygon) -> Option<Bounds> {
    let polygon = value.0.first()?;
    if value.0.len() != 1 || !polygon.holes.is_empty() || polygon.exterior.len() != 5 {
        return None;
    }
    let ring = &polygon.exterior;
    if ring.first() != ring.last()
        || ring
            .windows(2)
            .any(|edge| edge[0] == edge[1] || (edge[0].x != edge[1].x && edge[0].y != edge[1].y))
    {
        return None;
    }
    let min_x = ring.iter().map(|point| point.x).reduce(f64::min)?;
    let max_x = ring.iter().map(|point| point.x).reduce(f64::max)?;
    let min_y = ring.iter().map(|point| point.y).reduce(f64::min)?;
    let max_y = ring.iter().map(|point| point.y).reduce(f64::max)?;
    let corners = [
        Point::new(min_x, min_y),
        Point::new(max_x, min_y),
        Point::new(max_x, max_y),
        Point::new(min_x, max_y),
    ];
    if corners.iter().all(|corner| ring[..4].contains(corner)) {
        Some(Bounds {
            min_x,
            min_y,
            max_x,
            max_y,
        })
    } else {
        None
    }
}

fn contains(bounds: Bounds, x: f64, y: f64) -> bool {
    x > bounds.min_x && x < bounds.max_x && y > bounds.min_y && y < bounds.max_y
}

fn contains_bounds(outer: Bounds, inner: Bounds) -> bool {
    outer.min_x <= inner.min_x
        && outer.min_y <= inner.min_y
        && outer.max_x >= inner.max_x
        && outer.max_y >= inner.max_y
}

fn rectangle(bounds: Bounds) -> super::Polygon {
    super::Polygon {
        exterior: vec![
            Point::new(bounds.min_x, bounds.min_y),
            Point::new(bounds.max_x, bounds.min_y),
            Point::new(bounds.max_x, bounds.max_y),
            Point::new(bounds.min_x, bounds.max_y),
            Point::new(bounds.min_x, bounds.min_y),
        ],
        holes: Vec::new(),
    }
}

fn disjoint_result(left: Bounds, right: Bounds) -> MultiPolygon {
    let mut values = vec![left, right];
    values.sort_by(|a, b| {
        a.min_x
            .total_cmp(&b.min_x)
            .then_with(|| a.min_y.total_cmp(&b.min_y))
    });
    MultiPolygon(values.into_iter().map(rectangle).collect())
}

fn coordinate_key(value: f64) -> u64 {
    if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

type PointKey = (u64, u64);
type EdgeKey = (PointKey, PointKey);

fn point_key(point: Point) -> PointKey {
    (coordinate_key(point.x), coordinate_key(point.y))
}

fn edge_key(segment: Segment) -> EdgeKey {
    let start = point_key(segment.start);
    let end = point_key(segment.end);
    if start <= end {
        (start, end)
    } else {
        (end, start)
    }
}

fn toggle(edges: &mut BTreeMap<EdgeKey, Segment>, segment: Segment) {
    let key = edge_key(segment);
    if edges.remove(&key).is_none() {
        edges.insert(key, segment);
    }
}

pub(super) fn try_overlay(
    left: &MultiPolygon,
    right: &MultiPolygon,
    operation: BooleanOp,
) -> Option<Result<MultiPolygon, OverlayError>> {
    let left = bounds(left)?;
    let right = bounds(right)?;
    let strictly_disjoint = left.max_x < right.min_x
        || right.max_x < left.min_x
        || left.max_y < right.min_y
        || right.max_y < left.min_y;
    if strictly_disjoint {
        return Some(Ok(match operation {
            BooleanOp::Union | BooleanOp::SymmetricDifference => disjoint_result(left, right),
            BooleanOp::Intersection => MultiPolygon::default(),
            BooleanOp::Difference => MultiPolygon(vec![rectangle(left)]),
        }));
    }
    if operation == BooleanOp::Union {
        if contains_bounds(left, right) {
            return Some(Ok(MultiPolygon(vec![rectangle(left)])));
        }
        if contains_bounds(right, left) {
            return Some(Ok(MultiPolygon(vec![rectangle(right)])));
        }
    }
    if operation == BooleanOp::Intersection {
        let min_x = left.min_x.max(right.min_x);
        let min_y = left.min_y.max(right.min_y);
        let max_x = left.max_x.min(right.max_x);
        let max_y = left.max_y.min(right.max_y);
        if min_x >= max_x || min_y >= max_y {
            return Some(Ok(MultiPolygon::default()));
        }
        return Some(Ok(MultiPolygon(vec![rectangle(Bounds {
            min_x,
            min_y,
            max_x,
            max_y,
        })])));
    }
    let mut xs = vec![left.min_x, left.max_x, right.min_x, right.max_x];
    let mut ys = vec![left.min_y, left.max_y, right.min_y, right.max_y];
    xs.sort_by(f64::total_cmp);
    ys.sort_by(f64::total_cmp);
    xs.dedup();
    ys.dedup();
    let mut edges = BTreeMap::new();
    for x in xs.windows(2) {
        for y in ys.windows(2) {
            let midpoint = ((x[0] + x[1]) * 0.5, (y[0] + y[1]) * 0.5);
            if !operation.evaluate(
                contains(left, midpoint.0, midpoint.1),
                contains(right, midpoint.0, midpoint.1),
            ) {
                continue;
            }
            let bottom_left = Point::new(x[0], y[0]);
            let bottom_right = Point::new(x[1], y[0]);
            let top_right = Point::new(x[1], y[1]);
            let top_left = Point::new(x[0], y[1]);
            for segment in [
                Segment {
                    start: bottom_left,
                    end: bottom_right,
                },
                Segment {
                    start: bottom_right,
                    end: top_right,
                },
                Segment {
                    start: top_right,
                    end: top_left,
                },
                Segment {
                    start: top_left,
                    end: bottom_left,
                },
            ] {
                toggle(&mut edges, segment);
            }
        }
    }
    Some(
        super::rings::stitch(&edges.into_values().collect::<Vec<_>>())
            .and_then(super::rings::build_polygonal),
    )
}

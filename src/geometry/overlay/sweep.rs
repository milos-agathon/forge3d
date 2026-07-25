//! Bentley--Ottmann ordered segment discovery and exact-rational intersections.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use crate::geometry::exact::predicates::{
    difference_expansion, expansion_diff, expansion_estimate, expansion_product, expansion_sum,
    orient2d, scale_expansion, sign_ordering,
};

use super::snap::SnapGrid;
use super::{OverlayError, Point, Segment};

#[derive(Debug, Clone)]
struct ExactRational {
    numerator: Vec<f64>,
    denominator: Vec<f64>,
}

impl ExactRational {
    fn to_f64(&self) -> f64 {
        expansion_estimate(&self.numerator) / expansion_estimate(&self.denominator)
    }
}

#[derive(Debug, Clone)]
struct ExactPoint {
    x: ExactRational,
    y: ExactRational,
}

impl ExactPoint {
    fn to_point(&self) -> Point {
        Point::new(self.x.to_f64(), self.y.to_f64())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum EventKind {
    Start,
    End,
}

#[derive(Debug, Clone, Copy)]
struct Event {
    point: Point,
    kind: EventKind,
    segment: usize,
}

fn compare_points(left: Point, right: Point) -> Ordering {
    left.x
        .total_cmp(&right.x)
        .then_with(|| left.y.total_cmp(&right.y))
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

fn exact_cross(ax: &[f64], ay: &[f64], bx: &[f64], by: &[f64]) -> Vec<f64> {
    expansion_diff(&expansion_product(ax, by), &expansion_product(ay, bx))
}

fn proper_intersection(left: Segment, right: Segment) -> ExactPoint {
    let rx = difference_expansion(left.end.x, left.start.x);
    let ry = difference_expansion(left.end.y, left.start.y);
    let sx = difference_expansion(right.end.x, right.start.x);
    let sy = difference_expansion(right.end.y, right.start.y);
    let qpx = difference_expansion(right.start.x, left.start.x);
    let qpy = difference_expansion(right.start.y, left.start.y);
    let denominator = exact_cross(&rx, &ry, &sx, &sy);
    let t_numerator = exact_cross(&qpx, &qpy, &sx, &sy);
    let x_numerator = expansion_sum(
        &scale_expansion(&denominator, left.start.x),
        &expansion_product(&rx, &t_numerator),
    );
    let y_numerator = expansion_sum(
        &scale_expansion(&denominator, left.start.y),
        &expansion_product(&ry, &t_numerator),
    );
    ExactPoint {
        x: ExactRational {
            numerator: x_numerator,
            denominator: denominator.clone(),
        },
        y: ExactRational {
            numerator: y_numerator,
            denominator,
        },
    }
}

fn intersections(left: Segment, right: Segment) -> Vec<Point> {
    let o1 = sign_ordering(orient2d(
        left.start.as_array(),
        left.end.as_array(),
        right.start.as_array(),
    ));
    let o2 = sign_ordering(orient2d(
        left.start.as_array(),
        left.end.as_array(),
        right.end.as_array(),
    ));
    let o3 = sign_ordering(orient2d(
        right.start.as_array(),
        right.end.as_array(),
        left.start.as_array(),
    ));
    let o4 = sign_ordering(orient2d(
        right.start.as_array(),
        right.end.as_array(),
        left.end.as_array(),
    ));

    let opposite = |a: Ordering, b: Ordering| {
        matches!(
            (a, b),
            (Ordering::Less, Ordering::Greater) | (Ordering::Greater, Ordering::Less)
        )
    };
    if opposite(o1, o2) && opposite(o3, o4) {
        return vec![proper_intersection(left, right).to_point()];
    }
    let mut points = Vec::new();
    for point in [left.start, left.end, right.start, right.end] {
        if point_on_segment(point, left)
            && point_on_segment(point, right)
            && !points.contains(&point)
        {
            points.push(point);
        }
    }
    points.sort_by(|a, b| compare_points(*a, *b));
    points
}

fn events(segments: &[Segment]) -> Vec<Event> {
    let mut events = Vec::with_capacity(segments.len() * 2);
    for (index, segment) in segments.iter().copied().enumerate() {
        let (start, end) = if compare_points(segment.start, segment.end) != Ordering::Greater {
            (segment.start, segment.end)
        } else {
            (segment.end, segment.start)
        };
        events.push(Event {
            point: start,
            kind: EventKind::Start,
            segment: index,
        });
        events.push(Event {
            point: end,
            kind: EventKind::End,
            segment: index,
        });
    }
    events.sort_by(|left, right| {
        compare_points(left.point, right.point)
            .then_with(|| left.kind.cmp(&right.kind))
            .then_with(|| left.segment.cmp(&right.segment))
    });
    events
}

fn compare_along(segment: Segment, left: Point, right: Point) -> Ordering {
    let dx = segment.end.x - segment.start.x;
    let dy = segment.end.y - segment.start.y;
    let ordering = if dx.abs() >= dy.abs() {
        left.x
            .total_cmp(&right.x)
            .then_with(|| left.y.total_cmp(&right.y))
    } else {
        left.y
            .total_cmp(&right.y)
            .then_with(|| left.x.total_cmp(&right.x))
    };
    if (dx.abs() >= dy.abs() && dx < 0.0) || (dx.abs() < dy.abs() && dy < 0.0) {
        ordering.reverse()
    } else {
        ordering
    }
}

/// Split every segment at exact intersections and every crossed hot pixel.
pub(crate) fn snap_rounded_atomic_segments(
    segments: &[Segment],
    grid: SnapGrid,
) -> Result<Vec<Segment>, OverlayError> {
    let mut points = segments
        .iter()
        .map(|segment| vec![segment.start, segment.end])
        .collect::<Vec<_>>();
    let mut hot_pixels = segments
        .iter()
        .flat_map(|segment| [segment.start, segment.end])
        .collect::<Vec<_>>();
    let mut active: BTreeSet<usize> = BTreeSet::new();
    for event in events(segments) {
        match event.kind {
            EventKind::Start => {
                for &other in &active {
                    for point in intersections(segments[event.segment], segments[other]) {
                        let snapped = grid.snap(point);
                        if !snapped.is_finite() {
                            return Err(OverlayError(
                                "non-finite exact-rational intersection".to_string(),
                            ));
                        }
                        points[event.segment].push(snapped);
                        points[other].push(snapped);
                        hot_pixels.push(snapped);
                    }
                }
                active.insert(event.segment);
            }
            EventKind::End => {
                active.remove(&event.segment);
            }
        }
    }
    hot_pixels.sort_by(|a, b| compare_points(*a, *b));
    hot_pixels.dedup();

    let mut atomic = Vec::new();
    for (index, segment) in segments.iter().copied().enumerate() {
        for &center in &hot_pixels {
            if grid.hot_pixel_intersects(segment, center) {
                points[index].push(center);
            }
        }
        points[index].sort_by(|a, b| compare_along(segment, *a, *b));
        points[index].dedup();
        for pair in points[index].windows(2) {
            if pair[0] != pair[1] {
                atomic.push(Segment {
                    start: pair[0],
                    end: pair[1],
                });
            }
        }
    }
    Ok(atomic)
}

pub(crate) fn segment_intersections(left: Segment, right: Segment) -> Vec<Point> {
    intersections(left, right)
}

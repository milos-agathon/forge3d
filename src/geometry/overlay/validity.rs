//! Constructive polygon validity checks using the exact sweep predicates.

use super::faces::{locate_in_ring, Location};
use super::sweep::segment_intersections;
use super::{MultiPolygon, Point, Ring, Segment};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidityReport {
    pub valid: bool,
    pub reasons: Vec<String>,
}

fn ring_segments(ring: &Ring) -> Vec<Segment> {
    ring.windows(2)
        .map(|pair| Segment {
            start: pair[0],
            end: pair[1],
        })
        .collect()
}

fn check_ring(ring: &Ring, label: &str, reasons: &mut Vec<String>) {
    if ring.len() < 4 {
        reasons.push(format!("{label} has fewer than four coordinates"));
        return;
    }
    if ring.first() != ring.last() {
        reasons.push(format!("{label} is not closed"));
    }
    if ring.iter().any(|point| !point.is_finite()) {
        reasons.push(format!("{label} contains NaN or infinity"));
        return;
    }
    if ring.windows(2).any(|pair| pair[0] == pair[1]) {
        reasons.push(format!("{label} has a zero-length edge"));
    }
    let segments = ring_segments(ring);
    for left in 0..segments.len() {
        for right in (left + 1)..segments.len() {
            let adjacent = right == left + 1 || (left == 0 && right + 1 == segments.len());
            if !adjacent && !segment_intersections(segments[left], segments[right]).is_empty() {
                reasons.push(format!("{label} self-intersects"));
                return;
            }
        }
    }
}

fn rings_intersect(left: &Ring, right: &Ring) -> bool {
    ring_segments(left).into_iter().any(|left_segment| {
        ring_segments(right)
            .into_iter()
            .any(|right_segment| !segment_intersections(left_segment, right_segment).is_empty())
    })
}

fn rings_cross_or_overlap(left: &Ring, right: &Ring) -> bool {
    ring_segments(left).into_iter().any(|left_segment| {
        ring_segments(right).into_iter().any(|right_segment| {
            let intersections = segment_intersections(left_segment, right_segment);
            if intersections.len() > 1 {
                return true;
            }
            intersections.first().is_some_and(|point| {
                let left_endpoint = *point == left_segment.start || *point == left_segment.end;
                let right_endpoint = *point == right_segment.start || *point == right_segment.end;
                !(left_endpoint && right_endpoint)
            })
        })
    })
}

fn sample(ring: &Ring) -> Option<Point> {
    ring.first().copied()
}

/// Check closure, self-intersections, shell/hole containment, and component
/// overlap using the same exact predicates as overlay.
pub fn is_valid_polygonal(value: &MultiPolygon) -> ValidityReport {
    let mut reasons = Vec::new();
    for (polygon_index, polygon) in value.0.iter().enumerate() {
        check_ring(
            &polygon.exterior,
            &format!("polygon {polygon_index} exterior"),
            &mut reasons,
        );
        for (hole_index, hole) in polygon.holes.iter().enumerate() {
            check_ring(
                hole,
                &format!("polygon {polygon_index} hole {hole_index}"),
                &mut reasons,
            );
            if rings_intersect(&polygon.exterior, hole)
                || sample(hole)
                    .map(|point| locate_in_ring(point, &polygon.exterior) != Location::Inside)
                    .unwrap_or(true)
            {
                reasons.push(format!(
                    "polygon {polygon_index} hole {hole_index} is not strictly inside its shell"
                ));
            }
        }
        for left in 0..polygon.holes.len() {
            for right in (left + 1)..polygon.holes.len() {
                if rings_intersect(&polygon.holes[left], &polygon.holes[right])
                    || sample(&polygon.holes[left])
                        .map(|point| {
                            locate_in_ring(point, &polygon.holes[right]) != Location::Outside
                        })
                        .unwrap_or(false)
                    || sample(&polygon.holes[right])
                        .map(|point| {
                            locate_in_ring(point, &polygon.holes[left]) != Location::Outside
                        })
                        .unwrap_or(false)
                {
                    reasons.push(format!(
                        "polygon {polygon_index} holes {left} and {right} overlap"
                    ));
                }
            }
        }
    }
    for left in 0..value.0.len() {
        for right in (left + 1)..value.0.len() {
            let a = &value.0[left].exterior;
            let b = &value.0[right].exterior;
            if rings_cross_or_overlap(a, b)
                || a.iter()
                    .any(|point| locate_in_ring(*point, b) == Location::Inside)
                || b.iter()
                    .any(|point| locate_in_ring(*point, a) == Location::Inside)
            {
                reasons.push(format!("polygon components {left} and {right} overlap"));
            }
        }
    }
    ValidityReport {
        valid: reasons.is_empty(),
        reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::overlay::Polygon;

    #[test]
    fn checker_rejects_bowtie_and_orphan_hole() {
        let bowtie = MultiPolygon(vec![Polygon {
            exterior: vec![
                Point::new(0.0, 0.0),
                Point::new(2.0, 2.0),
                Point::new(0.0, 2.0),
                Point::new(2.0, 0.0),
                Point::new(0.0, 0.0),
            ],
            holes: Vec::new(),
        }]);
        assert!(!is_valid_polygonal(&bowtie).valid);

        let orphan = MultiPolygon(vec![Polygon {
            exterior: vec![
                Point::new(0.0, 0.0),
                Point::new(1.0, 0.0),
                Point::new(1.0, 1.0),
                Point::new(0.0, 1.0),
                Point::new(0.0, 0.0),
            ],
            holes: vec![vec![
                Point::new(2.0, 2.0),
                Point::new(3.0, 2.0),
                Point::new(3.0, 3.0),
                Point::new(2.0, 3.0),
                Point::new(2.0, 2.0),
            ]],
        }]);
        assert!(!is_valid_polygonal(&orphan).valid);
    }
}

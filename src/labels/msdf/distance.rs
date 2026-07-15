use super::edge::{Contour, Edge};
use lyon_path::math::Point;

fn cross(left: Point, right: Point) -> f32 {
    left.x * right.y - left.y * right.x
}

#[derive(Clone, Copy, Debug)]
pub struct EdgeDistance {
    /// Euclidean distance to the finite segment, signed by contour direction.
    pub true_distance: f32,
    /// Tangent-extended pseudo-distance used after nearest-edge selection.
    pub pseudo_distance: f32,
}

pub fn edge_distance(point: Point, edge: Edge) -> EdgeDistance {
    let vector = Point::new(edge.to.x - edge.from.x, edge.to.y - edge.from.y);
    let relative = Point::new(point.x - edge.from.x, point.y - edge.from.y);
    let length = vector.x.hypot(vector.y);
    if length == 0.0 {
        let distance = relative.x.hypot(relative.y);
        return EdgeDistance {
            true_distance: distance,
            pseudo_distance: distance,
        };
    }
    let direction = Point::new(vector.x / length, vector.y / length);
    let projection = relative.x * direction.x + relative.y * direction.y;
    let t = (projection / length).clamp(0.0, 1.0);
    let nearest = Point::new(edge.from.x + vector.x * t, edge.from.y + vector.y * t);
    let true_distance = (point.x - nearest.x).hypot(point.y - nearest.y);
    let perpendicular = cross(direction, relative);
    let directional_sign = (perpendicular * edge.winding).signum();
    let directional_sign = if directional_sign == 0.0 {
        1.0
    } else {
        directional_sign
    };

    // Chlumsky-style endpoint pseudo-distance: nearest-edge selection still uses
    // the finite-segment Euclidean distance, but reconstruction extends the edge
    // tangent beyond either endpoint. This prevents radial endpoint distance from
    // rounding a corner when adjacent colored edges meet.
    let pseudo_distance = if projection < 0.0 || projection > length {
        perpendicular.abs()
    } else {
        true_distance
    };
    EdgeDistance {
        true_distance: true_distance.copysign(directional_sign),
        pseudo_distance: pseudo_distance.copysign(directional_sign),
    }
}

pub fn signed_pseudo_distance(point: Point, edge: Edge) -> f32 {
    edge_distance(point, edge).pseudo_distance
}

pub fn contains(contours: &[Contour], point: Point) -> bool {
    let mut winding = 0i32;
    for contour in contours {
        for points in contour.points.windows(2) {
            let (from, to) = (points[0], points[1]);
            let side = cross(
                Point::new(to.x - from.x, to.y - from.y),
                Point::new(point.x - from.x, point.y - from.y),
            );
            if from.y <= point.y {
                if to.y > point.y && side > 0.0 {
                    winding += 1;
                }
            } else if to.y <= point.y && side < 0.0 {
                winding -= 1;
            }
        }
    }
    winding != 0
}

pub fn median(values: [f32; 3]) -> f32 {
    values[0]
        .max(values[1].min(values[2]))
        .min(values[1].max(values[2]))
}

pub fn correct_collision(mut channels: [f32; 3], truth: f32) -> [f32; 3] {
    let tolerance = 1.0e-4;
    let mut reconstructed = median(channels);
    if (reconstructed >= 0.0) == (truth >= 0.0) && (reconstructed - truth).abs() <= tolerance {
        return channels;
    }

    // Replace the worst channel first and stop as soon as the median agrees with
    // the independently computed scalar distance. This corrects edge-color
    // collisions without flattening all three channels at legitimate corners.
    let mut indices = [0usize, 1, 2];
    indices.sort_by(|left, right| {
        (channels[*right] - truth)
            .abs()
            .total_cmp(&(channels[*left] - truth).abs())
    });
    for index in indices {
        channels[index] = truth;
        reconstructed = median(channels);
        if (reconstructed >= 0.0) == (truth >= 0.0) && (reconstructed - truth).abs() <= tolerance {
            break;
        }
    }
    channels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pseudo_distance_is_signed_by_edge_side() {
        let edge = Edge {
            from: Point::new(0.0, 0.0),
            to: Point::new(4.0, 0.0),
            color: 1,
            winding: 1.0,
        };
        assert_eq!(signed_pseudo_distance(Point::new(2.0, 2.0), edge), 2.0);
        assert_eq!(signed_pseudo_distance(Point::new(2.0, -2.0), edge), -2.0);
    }

    #[test]
    fn endpoint_pseudo_distance_extends_the_edge_tangent() {
        let edge = Edge {
            from: Point::new(0.0, 0.0),
            to: Point::new(4.0, 0.0),
            color: 1,
            winding: 1.0,
        };
        let measured = edge_distance(Point::new(-1.0, 2.0), edge);
        assert!((measured.true_distance - 5.0f32.sqrt()).abs() < 1.0e-6);
        assert_eq!(measured.pseudo_distance, 2.0);

        let reversed = Edge {
            from: edge.to,
            to: edge.from,
            winding: -1.0,
            ..edge
        };
        let reversed = edge_distance(Point::new(-1.0, 2.0), reversed);
        assert_eq!(reversed.pseudo_distance, measured.pseudo_distance);
    }

    #[test]
    fn collision_correction_restores_truth_sign_to_the_median() {
        let corrected = correct_collision([-2.0, -3.0, 1.0], 0.5);
        assert!(median(corrected) >= 0.0);
        assert!(corrected.iter().any(|value| *value != corrected[0]));
        assert_eq!(corrected.iter().filter(|value| **value == 0.5).count(), 1);
    }
}

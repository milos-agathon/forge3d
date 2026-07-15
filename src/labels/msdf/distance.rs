use super::edge::{Contour, Edge};
use lyon_path::math::Point;

fn cross(left: Point, right: Point) -> f32 {
    left.x * right.y - left.y * right.x
}

pub fn signed_pseudo_distance(point: Point, edge: Edge) -> f32 {
    let vector = Point::new(edge.to.x - edge.from.x, edge.to.y - edge.from.y);
    let relative = Point::new(point.x - edge.from.x, point.y - edge.from.y);
    let length_squared = vector.x * vector.x + vector.y * vector.y;
    if length_squared == 0.0 {
        return relative.x.hypot(relative.y);
    }
    let t = ((relative.x * vector.x + relative.y * vector.y) / length_squared).clamp(0.0, 1.0);
    let nearest = Point::new(edge.from.x + vector.x * t, edge.from.y + vector.y * t);
    let distance = (point.x - nearest.x).hypot(point.y - nearest.y);
    distance.copysign(cross(vector, relative))
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
    let wanted_positive = truth >= 0.0;
    let matching = channels
        .iter()
        .filter(|value| (**value >= 0.0) == wanted_positive)
        .count();
    if matching < 2 {
        let mut indices = [0usize, 1, 2];
        indices.sort_by(|left, right| channels[*left].abs().total_cmp(&channels[*right].abs()));
        for index in indices.into_iter().take(2) {
            channels[index] = truth;
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
        };
        assert_eq!(signed_pseudo_distance(Point::new(2.0, 2.0), edge), 2.0);
        assert_eq!(signed_pseudo_distance(Point::new(2.0, -2.0), edge), -2.0);
    }

    #[test]
    fn collision_correction_restores_truth_sign_to_the_median() {
        let corrected = correct_collision([-2.0, -3.0, 1.0], 0.5);
        assert!(median(corrected) >= 0.0);
        assert!(corrected.iter().any(|value| *value != corrected[0]));
    }
}

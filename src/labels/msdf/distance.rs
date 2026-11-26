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
    let nearest = if t == 0.0 {
        edge.from
    } else if t == 1.0 {
        edge.to
    } else {
        Point::new(edge.from.x + vector.x * t, edge.from.y + vector.y * t)
    };
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

fn bilinear(values: [[f32; 3]; 4], x: f32, y: f32) -> [f32; 3] {
    let mut output = [0.0; 3];
    for channel in 0..3 {
        output[channel] = values[0][channel] * (1.0 - x) * (1.0 - y)
            + values[1][channel] * x * (1.0 - y)
            + values[2][channel] * (1.0 - x) * y
            + values[3][channel] * x * y;
    }
    output
}

fn scalar_truth(contours: &[Contour], edges: &[Edge], point: Point) -> f32 {
    let nearest = edges
        .iter()
        .map(|edge| edge_distance(point, *edge).true_distance.abs())
        .fold(f32::INFINITY, f32::min);
    nearest.copysign(if contains(contours, point) { 1.0 } else { -1.0 })
}

/// Detect interpolation sign clashes using neighboring texels and independent
/// point-sampled containment. Only texels participating in a real off-grid
/// clash are eligible for scalar correction.
pub fn collision_mask(
    contours: &[Contour],
    edges: &[Edge],
    fields: &[[f32; 3]],
    width: usize,
    height: usize,
    origin: Point,
) -> Vec<bool> {
    let mut mask = vec![false; width.saturating_mul(height)];
    if width < 2 || height < 2 || fields.len() != mask.len() || edges.is_empty() {
        return mask;
    }
    const SUBCELL_SAMPLES: [f32; 2] = [0.25, 0.75];
    for y in 0..height - 1 {
        for x in 0..width - 1 {
            let indices = [
                y * width + x,
                y * width + x + 1,
                (y + 1) * width + x,
                (y + 1) * width + x + 1,
            ];
            let values = [
                fields[indices[0]],
                fields[indices[1]],
                fields[indices[2]],
                fields[indices[3]],
            ];
            let mut clash = false;
            for fy in SUBCELL_SAMPLES {
                for fx in SUBCELL_SAMPLES {
                    let sample = Point::new(
                        origin.x + x as f32 + 0.5 + fx,
                        origin.y + y as f32 + 0.5 + fy,
                    );
                    let reconstructed = median(bilinear(values, fx, fy));
                    let truth = scalar_truth(contours, edges, sample);
                    // A collision is not limited to a literal sign flip. Color
                    // pseudo-distances can interpolate close to zero far from
                    // the outline, producing long halo/fill rays even while
                    // every texel retains the correct sign. Correct only
                    // dangerous underestimation (the reconstructed boundary
                    // is materially closer than the independent scalar
                    // boundary), preserving genuine corner separation.
                    let sign_clash = (reconstructed >= 0.0) != (truth >= 0.0);
                    let distance_clash = reconstructed.abs() + 0.75 < truth.abs();
                    if sign_clash || distance_clash {
                        clash = true;
                    }
                }
            }
            if clash {
                for index in indices {
                    mask[index] = true;
                }
            }
        }
    }
    mask
}

fn roots(quadratic: f64, linear: f64, constant: f64) -> Vec<f64> {
    if quadratic == 0.0 {
        return if linear == 0.0 {
            Vec::new()
        } else {
            vec![-constant / linear]
        };
    }
    let discriminant = linear * linear - 4.0 * quadratic * constant;
    if discriminant < 0.0 {
        return Vec::new();
    }
    let q = -0.5 * (linear + discriminant.sqrt().copysign(linear));
    if q == 0.0 {
        vec![-linear / (2.0 * quadratic)]
    } else {
        vec![q / quadratic, constant / q]
    }
}

fn interpolation_artifact(
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    d: [f32; 3],
    diagonal: bool,
    protected: bool,
) -> bool {
    let am = f64::from(median(a));
    let dm = f64::from(median(d));
    if am.abs() < dm.abs() {
        return false;
    }
    let a = a.map(f64::from);
    let d = d.map(f64::from);
    let linear: [f64; 3] = std::array::from_fn(|i| {
        if diagonal {
            f64::from(b[i]) + f64::from(c[i]) - 2.0 * a[i]
        } else {
            d[i] - a[i]
        }
    });
    let quadratic: [f64; 3] = std::array::from_fn(|i| {
        if diagonal {
            a[i] + d[i] - f64::from(b[i]) - f64::from(c[i])
        } else {
            0.0
        }
    });
    let value = |t: f64| {
        let p: [f64; 3] = std::array::from_fn(|i| a[i] + t * (linear[i] + t * quadratic[i]));
        p[0].max(p[1].min(p[2])).min(p[1].max(p[2]))
    };
    let span = if diagonal { 2.0_f64.sqrt() } else { 1.0 };
    let invalid_range = |low: f64, high: f64, t: f64, m: f64| {
        let lm = value(low);
        let hm = value(high);
        let inversion = (lm > 0.0 && hm > 0.0 && m <= 0.0) || (lm < 0.0 && hm < 0.0 && m >= 0.0);
        let excursion = m < lm.min(hm) || m > lm.max(hm);
        // The median may legitimately bend at a protected corner. Correct
        // only false inversions there; elsewhere detect extrema outside the
        // endpoint range and the unit-distance cones from both endpoints.
        (inversion || (!protected && excursion))
            && ((m - lm).abs() > (t - low) * span || (m - hm).abs() > (high - t) * span)
    };
    for (i, j) in [(0, 1), (1, 2), (2, 0)] {
        for t in roots(
            quadratic[i] - quadratic[j],
            linear[i] - linear[j],
            a[i] - a[j],
        ) {
            if t <= 0.0 || t >= 1.0 || !t.is_finite() {
                continue;
            }
            let m = value(t);
            if invalid_range(0.0, 1.0, t, m) {
                return true;
            }
            // A diagonal bilinear restriction is quadratic. Its channel
            // extrema partition the range and can expose a hidden clash.
            for channel in [i, j] {
                if quadratic[channel] != 0.0 {
                    let extreme = -linear[channel] / (2.0 * quadratic[channel]);
                    if extreme > 0.0 && extreme < 1.0 {
                        let (low, high) = if extreme < t {
                            (extreme, 1.0)
                        } else {
                            (0.0, extreme)
                        };
                        if invalid_range(low, high, t, m) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Detect channel-exchange extrema at their analytic locations, with corner
/// protection as in msdfgen's artifact stencil. No fixed sampling grid or
/// blanket per-channel Lipschitz cleanup participates. Correction preserves
/// each texel's median; it only removes erroneous interpolation separation.
pub fn correct_interpolation_artifacts(
    fields: &mut [[f32; 3]],
    edges: &[Edge],
    width: usize,
    height: usize,
    origin: Point,
) {
    if width < 2 || height < 2 || fields.len() != width * height {
        return;
    }
    let mut protected = vec![false; fields.len()];
    for edge in edges {
        if let Some(previous) = edges.iter().find(|previous| previous.to == edge.from) {
            if (previous.color & edge.color).count_ones() <= 1 {
                let x = (edge.from.x - origin.x - 0.5).floor() as isize;
                let y = (edge.from.y - origin.y - 0.5).floor() as isize;
                for yy in [y, y + 1] {
                    for xx in [x, x + 1] {
                        if xx >= 0 && yy >= 0 && (xx as usize) < width && (yy as usize) < height {
                            protected[yy as usize * width + xx as usize] = true;
                        }
                    }
                }
            }
        }
    }
    let directions = [
        (1_isize, 0_isize),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ];
    let mut errors = vec![false; fields.len()];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            for (dx, dy) in directions {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx < 0 || ny < 0 || nx as usize >= width || ny as usize >= height {
                    continue;
                }
                let nx = nx as usize;
                let ny = ny as usize;
                if interpolation_artifact(
                    fields[index],
                    fields[y * width + nx],
                    fields[ny * width + x],
                    fields[ny * width + nx],
                    dx != 0 && dy != 0,
                    protected[index],
                ) {
                    errors[index] = true;
                    break;
                }
            }
        }
    }
    for (pixel, error) in fields.iter_mut().zip(errors) {
        if error {
            *pixel = [median(*pixel); 3];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_exchange_extrema_are_corrected_without_moving_texel_medians() {
        let a = [-3.0, -1.0, 3.0];
        let b = [3.0, -1.0, -3.0];
        assert!(interpolation_artifact(a, a, b, b, false, true));
        let mut fields = vec![a, b, a, b];
        correct_interpolation_artifacts(&mut fields, &[], 2, 2, Point::new(0.0, 0.0));
        assert_eq!(fields, vec![[-1.0; 3]; 4]);
    }

    #[test]
    fn corner_protection_preserves_legitimate_channel_separation() {
        let a = [1.0, 1.5, 6.0];
        let b = [6.0, 1.5, 1.0];
        assert!(interpolation_artifact(a, a, b, b, false, false));
        assert!(!interpolation_artifact(a, a, b, b, false, true));
    }

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

    #[test]
    fn containment_uses_closing_edge_to_reject_left_outside_points() {
        let contour = Contour {
            points: vec![
                Point::new(0.0, 0.0),
                Point::new(4.0, 0.0),
                Point::new(4.0, 4.0),
                Point::new(0.0, 4.0),
                Point::new(0.0, 0.0),
            ],
        };

        assert!(contains(
            std::slice::from_ref(&contour),
            Point::new(0.5, 2.0)
        ));
        assert!(!contains(
            std::slice::from_ref(&contour),
            Point::new(-0.5, 2.0)
        ));
    }

    #[test]
    fn spatial_collision_mask_distinguishes_clean_and_clashing_neighbors() {
        let contour = Contour {
            points: vec![
                Point::new(0.0, 0.0),
                Point::new(3.0, 0.0),
                Point::new(3.0, 3.0),
                Point::new(0.0, 3.0),
                Point::new(0.0, 0.0),
            ],
        };
        let edges = crate::labels::msdf::edge::color_edges(&contour);
        let clean = vec![[1.0; 3]; 9];
        assert!(!collision_mask(
            std::slice::from_ref(&contour),
            &edges,
            &clean,
            3,
            3,
            Point::new(0.0, 0.0),
        )
        .into_iter()
        .any(|value| value));

        let clash = vec![[-1.0; 3]; 9];
        assert!(collision_mask(
            std::slice::from_ref(&contour),
            &edges,
            &clash,
            3,
            3,
            Point::new(0.0, 0.0),
        )
        .into_iter()
        .any(|value| value));

        let far_contour = Contour {
            points: contour
                .points
                .iter()
                .map(|point| Point::new(point.x + 10.0, point.y + 10.0))
                .collect(),
        };
        let far_edges = crate::labels::msdf::edge::color_edges(&far_contour);
        let false_near_boundary = vec![[-0.05; 3]; 9];
        assert!(collision_mask(
            std::slice::from_ref(&far_contour),
            &far_edges,
            &false_near_boundary,
            3,
            3,
            Point::new(0.0, 0.0),
        )
        .into_iter()
        .any(|value| value));

        let clean_far_outside = vec![[-20.0; 3]; 9];
        assert!(!collision_mask(
            std::slice::from_ref(&far_contour),
            &far_edges,
            &clean_far_outside,
            3,
            3,
            Point::new(0.0, 0.0),
        )
        .into_iter()
        .any(|value| value));
    }
}

use lyon_geom::{CubicBezierSegment, LineSegment, QuadraticBezierSegment};
use lyon_path::{math::Point, Event, Path};

pub const CYAN: u8 = 0b110;
pub const MAGENTA: u8 = 0b101;
pub const YELLOW: u8 = 0b011;
const COLORS: [u8; 3] = [CYAN, MAGENTA, YELLOW];
const CORNER_ANGLE_RADIANS: f32 = 0.75;
const FLATTENING_TOLERANCE: f32 = 0.125;

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub from: Point,
    pub to: Point,
    pub color: u8,
    /// Signed contour orientation (+1 counter-clockwise, -1 clockwise).
    pub winding: f32,
}

#[derive(Clone, Debug)]
pub struct Contour {
    pub points: Vec<Point>,
}

pub fn contour_orientation(contour: &Contour) -> f32 {
    contour
        .points
        .windows(2)
        .map(|edge| edge[0].x * edge[1].y - edge[1].x * edge[0].y)
        .sum::<f32>()
        * 0.5
}

fn normalized(vector: Point) -> Point {
    let length = vector.x.hypot(vector.y);
    if length == 0.0 {
        Point::new(0.0, 0.0)
    } else {
        Point::new(vector.x / length, vector.y / length)
    }
}

pub fn sharp_corners(contour: &Contour) -> Vec<usize> {
    let count = contour.points.len().saturating_sub(1);
    if count < 2 {
        return Vec::new();
    }
    let threshold = CORNER_ANGLE_RADIANS.cos();
    (0..count)
        .filter(|&index| {
            let previous = contour.points[(index + count - 1) % count];
            let vertex = contour.points[index];
            let next = contour.points[(index + 1) % count];
            let incoming = normalized(Point::new(vertex.x - previous.x, vertex.y - previous.y));
            let outgoing = normalized(Point::new(next.x - vertex.x, next.y - vertex.y));
            incoming.x * outgoing.x + incoming.y * outgoing.y < threshold
        })
        .collect()
}

fn compare_points(left: Point, right: Point) -> std::cmp::Ordering {
    left.x
        .total_cmp(&right.x)
        .then_with(|| left.y.total_cmp(&right.y))
}

fn canonical_index(contour: &Contour, candidates: &[usize]) -> usize {
    let count = contour.points.len().saturating_sub(1);
    candidates
        .iter()
        .copied()
        .min_by(|&left, &right| {
            compare_points(contour.points[left], contour.points[right]).then_with(|| {
                compare_points(
                    contour.points[(left + 1) % count],
                    contour.points[(right + 1) % count],
                )
            })
        })
        .unwrap_or(0)
}

fn span_colors(count: usize, phase: usize) -> Vec<u8> {
    let mut colors = (0..count)
        .map(|index| COLORS[(phase + index) % COLORS.len()])
        .collect::<Vec<_>>();
    if count > 1 && colors.first() == colors.last() {
        let first = colors[0];
        let previous = colors[count - 2];
        colors[count - 1] = COLORS
            .iter()
            .copied()
            .find(|color| *color != first && *color != previous)
            .unwrap();
    }
    colors
}

fn color_edges_with_phase(contour: &Contour, phase: usize) -> Vec<Edge> {
    let corners = sharp_corners(contour);
    let count = contour.points.len().saturating_sub(1);
    if count == 0 {
        return Vec::new();
    }
    let winding = contour_orientation(contour).signum();
    let winding = if winding == 0.0 { 1.0 } else { winding };
    if corners.is_empty() {
        return contour
            .points
            .windows(2)
            .map(|points| Edge {
                from: points[0],
                to: points[1],
                color: CYAN | MAGENTA | YELLOW,
                winding,
            })
            .collect();
    }

    let start = canonical_index(contour, &corners);
    let colors = if corners.len() == 1 {
        // A one-corner contour (the classic teardrop case) needs three
        // non-degenerate spline spans. Splitting the flattened loop into three
        // balanced spans prevents every smooth edge from sharing one channel.
        span_colors(3.min(count), phase)
    } else {
        span_colors(corners.len(), phase)
    };
    let mut corner_offsets = corners
        .iter()
        .map(|&corner| (corner + count - start) % count)
        .collect::<Vec<_>>();
    corner_offsets.sort_unstable();

    contour
        .points
        .windows(2)
        .enumerate()
        .map(|(index, points)| {
            let offset = (index + count - start) % count;
            let span = if corners.len() == 1 {
                (offset * colors.len() / count).min(colors.len() - 1)
            } else {
                corner_offsets
                    .partition_point(|&corner_offset| corner_offset <= offset)
                    .saturating_sub(1)
            };
            Edge {
                from: points[0],
                to: points[1],
                color: colors[span],
                winding,
            }
        })
        .collect()
}

pub fn color_edges(contour: &Contour) -> Vec<Edge> {
    color_edges_with_phase(contour, 0)
}

/// Color every contour with a deterministic cross-contour phase.
pub fn color_contours(contours: &[Contour]) -> Vec<Edge> {
    let mut ordered = contours.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        let left_index = canonical_index(
            left,
            &(0..left.points.len().saturating_sub(1)).collect::<Vec<_>>(),
        );
        let right_index = canonical_index(
            right,
            &(0..right.points.len().saturating_sub(1)).collect::<Vec<_>>(),
        );
        compare_points(left.points[left_index], right.points[right_index])
    });
    ordered
        .into_iter()
        .enumerate()
        .flat_map(|(index, contour)| color_edges_with_phase(contour, index % COLORS.len()))
        .collect()
}

pub fn flatten_path(path: &Path, scale: f32, offset: Point) -> Vec<Contour> {
    let transform =
        |point: Point| Point::new(offset.x + point.x * scale, offset.y + point.y * scale);
    let mut contours = Vec::new();
    let mut points = Vec::new();
    for event in path.iter() {
        match event {
            Event::Begin { at } => {
                if !points.is_empty() {
                    contours.push(Contour {
                        points: std::mem::take(&mut points),
                    });
                }
                points.push(transform(at));
            }
            Event::Line { to, .. } => points.push(transform(to)),
            Event::Quadratic { from, ctrl, to } => {
                QuadraticBezierSegment {
                    from: transform(from),
                    ctrl: transform(ctrl),
                    to: transform(to),
                }
                .for_each_flattened(FLATTENING_TOLERANCE, &mut |line: &LineSegment<f32>| {
                    points.push(line.to)
                });
            }
            Event::Cubic {
                from,
                ctrl1,
                ctrl2,
                to,
            } => {
                CubicBezierSegment {
                    from: transform(from),
                    ctrl1: transform(ctrl1),
                    ctrl2: transform(ctrl2),
                    to: transform(to),
                }
                .for_each_flattened(FLATTENING_TOLERANCE, &mut |line: &LineSegment<f32>| {
                    points.push(line.to)
                })
            }
            Event::End { first, close, .. } => {
                let first = transform(first);
                if close && points.last().copied() != Some(first) {
                    points.push(first);
                }
                if !points.is_empty() {
                    contours.push(Contour {
                        points: std::mem::take(&mut points),
                    });
                }
            }
        }
    }
    contours
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn square() -> Contour {
        Contour {
            points: vec![
                Point::new(0.0, 0.0),
                Point::new(4.0, 0.0),
                Point::new(4.0, 4.0),
                Point::new(0.0, 4.0),
                Point::new(0.0, 0.0),
            ],
        }
    }

    fn teardrop() -> Contour {
        let steps = 32;
        let mut points = (0..steps)
            .map(|index| {
                let angle = std::f32::consts::TAU * index as f32 / steps as f32;
                let radius = 1.0 - angle.cos();
                Point::new(radius, angle.sin() * radius)
            })
            .collect::<Vec<_>>();
        points.push(points[0]);
        Contour { points }
    }

    fn color_map(contour: &Contour) -> BTreeMap<(u32, u32, u32, u32), u8> {
        color_edges(contour)
            .into_iter()
            .map(|edge| {
                let mut points = [edge.from, edge.to];
                points.sort_by(|left, right| compare_points(*left, *right));
                (
                    (
                        points[0].x.to_bits(),
                        points[0].y.to_bits(),
                        points[1].x.to_bits(),
                        points[1].y.to_bits(),
                    ),
                    edge.color,
                )
            })
            .collect()
    }

    #[test]
    fn orientation_and_corners_are_geometric() {
        let contour = square();
        assert!(contour_orientation(&contour) > 0.0);
        assert_eq!(sharp_corners(&contour), vec![0, 1, 2, 3]);
    }

    #[test]
    fn adjacent_sharp_edges_receive_three_colors() {
        let colors: Vec<_> = color_edges(&square())
            .iter()
            .map(|edge| edge.color)
            .collect();
        assert_eq!(colors, vec![CYAN, MAGENTA, YELLOW, MAGENTA]);
        assert!(colors
            .iter()
            .zip(colors.iter().cycle().skip(1))
            .take(colors.len())
            .all(|(left, right)| left != right));
    }

    #[test]
    fn closing_edge_keeps_orientation_and_a_distinct_seam_color() {
        let edges = color_edges(&square());
        assert_eq!(edges.len(), 4);
        assert!(edges.iter().all(|edge| edge.winding > 0.0));
        assert_ne!(edges.first().unwrap().color, edges.last().unwrap().color);

        let mut reversed = square();
        reversed.points.reverse();
        let reversed = color_edges(&reversed);
        assert!(reversed.iter().all(|edge| edge.winding < 0.0));
        assert_ne!(
            reversed.first().unwrap().color,
            reversed.last().unwrap().color
        );
    }

    #[test]
    fn one_corner_teardrop_uses_three_non_degenerate_spans() {
        let contour = teardrop();
        assert_eq!(sharp_corners(&contour).len(), 1);
        let colors = color_edges(&contour)
            .into_iter()
            .map(|edge| edge.color)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(colors.len(), 3);
    }

    #[test]
    fn cyclic_start_rotation_preserves_geometric_edge_colors() {
        let contour = teardrop();
        let mut unique = contour.points[..contour.points.len() - 1].to_vec();
        unique.rotate_left(11);
        unique.push(unique[0]);
        let rotated = Contour { points: unique };
        assert_eq!(color_map(&contour), color_map(&rotated));
    }

    #[test]
    fn multiple_contours_do_not_restart_the_same_color_phase() {
        let left = square();
        let right = Contour {
            points: left
                .points
                .iter()
                .map(|point| Point::new(point.x + 10.0, point.y))
                .collect(),
        };
        let colors = color_contours(&[left, right]);
        assert_eq!(colors.len(), 8);
        assert_ne!(colors[0].color, colors[4].color);
    }
}

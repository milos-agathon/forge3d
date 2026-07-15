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

pub fn color_edges(contour: &Contour) -> Vec<Edge> {
    let corners = sharp_corners(contour);
    let mut color_index = 0usize;
    let mut edges: Vec<_> = contour
        .points
        .windows(2)
        .enumerate()
        .map(|(index, points)| {
            if index != 0 && corners.contains(&index) {
                color_index = (color_index + 1) % COLORS.len();
            }
            Edge {
                from: points[0],
                to: points[1],
                color: if corners.is_empty() {
                    CYAN | MAGENTA | YELLOW
                } else {
                    COLORS[color_index]
                },
            }
        })
        .collect();
    if corners.contains(&0)
        && edges.len() > 1
        && edges.first().map(|edge| edge.color) == edges.last().map(|edge| edge.color)
    {
        let first = edges[0].color;
        let previous = edges[edges.len() - 2].color;
        edges.last_mut().unwrap().color = COLORS
            .iter()
            .copied()
            .find(|color| *color != first && *color != previous)
            .unwrap();
    }
    edges
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
                QuadraticBezierSegment { from, ctrl, to }.for_each_flattened(
                    FLATTENING_TOLERANCE / scale.max(1.0e-6),
                    &mut |line: &LineSegment<f32>| points.push(transform(line.to)),
                );
            }
            Event::Cubic {
                from,
                ctrl1,
                ctrl2,
                to,
            } => CubicBezierSegment {
                from,
                ctrl1,
                ctrl2,
                to,
            }
            .for_each_flattened(
                FLATTENING_TOLERANCE / scale.max(1.0e-6),
                &mut |line: &LineSegment<f32>| points.push(transform(line.to)),
            ),
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
}

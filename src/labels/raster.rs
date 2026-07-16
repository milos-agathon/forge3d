use crate::labels::positioned::PositionedOutline;
use lyon_geom::{CubicBezierSegment, LineSegment, QuadraticBezierSegment};
use lyon_path::{math::Point, Event};
use ndarray::Array2;

const FLATTENING_TOLERANCE: f32 = 0.125;
const SUBPIXELS: usize = 8;

#[derive(Clone)]
struct FlatOutline {
    contours: Vec<Vec<Point>>,
    bounds: [f32; 4],
}

fn flatten(outline: &PositionedOutline, origin: (f32, f32)) -> FlatOutline {
    let offset = |point: Point| Point::new(point.x + origin.0, point.y + origin.1);
    let mut contours = Vec::new();
    let mut contour = Vec::new();
    let mut bounds = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    let push = |point: Point, contour: &mut Vec<Point>, bounds: &mut [f32; 4]| {
        let point = offset(point);
        bounds[0] = bounds[0].min(point.x);
        bounds[1] = bounds[1].min(point.y);
        bounds[2] = bounds[2].max(point.x);
        bounds[3] = bounds[3].max(point.y);
        contour.push(point);
    };
    for event in outline.path.iter() {
        match event {
            Event::Begin { at } => {
                if !contour.is_empty() {
                    contours.push(std::mem::take(&mut contour));
                }
                push(at, &mut contour, &mut bounds);
            }
            Event::Line { to, .. } => push(to, &mut contour, &mut bounds),
            Event::Quadratic { from, ctrl, to } => {
                QuadraticBezierSegment { from, ctrl, to }
                    .for_each_flattened(FLATTENING_TOLERANCE, &mut |segment: &LineSegment<f32>| {
                        push(segment.to, &mut contour, &mut bounds)
                    });
            }
            Event::Cubic {
                from,
                ctrl1,
                ctrl2,
                to,
            } => {
                CubicBezierSegment {
                    from,
                    ctrl1,
                    ctrl2,
                    to,
                }
                .for_each_flattened(FLATTENING_TOLERANCE, &mut |segment: &LineSegment<f32>| {
                    push(segment.to, &mut contour, &mut bounds)
                });
            }
            Event::End { first, close, .. } => {
                if close && contour.last().copied() != Some(offset(first)) {
                    push(first, &mut contour, &mut bounds);
                }
                if !contour.is_empty() {
                    contours.push(std::mem::take(&mut contour));
                }
            }
        }
    }
    FlatOutline { contours, bounds }
}

fn contains(outline: &FlatOutline, point: Point) -> bool {
    let mut winding = 0i32;
    for contour in &outline.contours {
        for edge in contour.windows(2) {
            let (a, b) = (edge[0], edge[1]);
            if a.y <= point.y {
                if b.y > point.y
                    && (b.x - a.x) * (point.y - a.y) - (point.x - a.x) * (b.y - a.y) > 0.0
                {
                    winding += 1;
                }
            } else if b.y <= point.y
                && (b.x - a.x) * (point.y - a.y) - (point.x - a.x) * (b.y - a.y) < 0.0
            {
                winding -= 1;
            }
        }
    }
    winding != 0
}

pub fn rasterize(
    outlines: &[PositionedOutline],
    width: usize,
    height: usize,
    origin: (f32, f32),
) -> Array2<f32> {
    let outlines: Vec<_> = outlines
        .iter()
        .map(|outline| flatten(outline, origin))
        .collect();
    let mut output = Array2::zeros((height, width));
    if outlines.is_empty() || width == 0 || height == 0 {
        return output;
    }
    let bounds = outlines.iter().fold(
        [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ],
        |mut bounds, outline| {
            bounds[0] = bounds[0].min(outline.bounds[0]);
            bounds[1] = bounds[1].min(outline.bounds[1]);
            bounds[2] = bounds[2].max(outline.bounds[2]);
            bounds[3] = bounds[3].max(outline.bounds[3]);
            bounds
        },
    );
    let x0 = bounds[0].floor().max(0.0) as usize;
    let y0 = bounds[1].floor().max(0.0) as usize;
    let x1 = bounds[2].ceil().min(width as f32) as usize;
    let y1 = bounds[3].ceil().min(height as f32) as usize;
    for y in y0..y1 {
        for x in x0..x1 {
            let mut covered = 0usize;
            for sy in 0..SUBPIXELS {
                for sx in 0..SUBPIXELS {
                    let point = Point::new(
                        x as f32 + (sx as f32 + 0.5) / SUBPIXELS as f32,
                        y as f32 + (sy as f32 + 0.5) / SUBPIXELS as f32,
                    );
                    if outlines.iter().any(|outline| {
                        point.x >= outline.bounds[0]
                            && point.y >= outline.bounds[1]
                            && point.x <= outline.bounds[2]
                            && point.y <= outline.bounds[3]
                            && contains(outline, point)
                    }) {
                        covered += 1;
                    }
                }
            }
            output[(y, x)] = covered as f32 / (SUBPIXELS * SUBPIXELS) as f32;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::rasterize;
    use crate::labels::positioned::PositionedOutline;
    use lyon_path::{math::point, Path};

    #[test]
    fn analytic_coverage_is_deterministic_and_subpixel() {
        let mut builder = Path::builder();
        builder.begin(point(0.25, 0.25));
        builder.line_to(point(1.75, 0.25));
        builder.line_to(point(1.75, 1.75));
        builder.line_to(point(0.25, 1.75));
        builder.close();
        let outlines = [PositionedOutline {
            glyph_id: 1,
            font_index: 0,
            cluster: 0,
            line_index: 0,
            path: builder.build(),
        }];

        let first = rasterize(&outlines, 3, 3, (0.0, 0.0));
        let second = rasterize(&outlines, 3, 3, (0.0, 0.0));
        assert_eq!(first, second);
        assert!(first[(0, 0)] > 0.0 && first[(0, 0)] < 1.0);
        assert_eq!(first[(1, 1)], 0.5625);
        assert_eq!(first[(2, 2)], 0.0);
    }
}

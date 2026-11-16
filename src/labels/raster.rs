//! Outline coverage from lyon non-zero fill tessellation. Pixel-square clipping
//! integrates triangle area analytically; no sample grid participates.
use crate::labels::positioned::PositionedOutline;
use lyon_path::{math::Point, Event};
use lyon_tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers};
use ndarray::Array2;
use std::collections::BTreeMap;

type Position = [f64; 2];

fn clip(polygon: &[Position], axis: usize, boundary: f64, greater: bool) -> Vec<Position> {
    clip_impl(polygon, axis, boundary, greater, false)
}

fn clip_impl(
    polygon: &[Position],
    axis: usize,
    boundary: f64,
    greater: bool,
    stable_edges: bool,
) -> Vec<Position> {
    let mut output = Vec::with_capacity(polygon.len() + 1);
    let Some(mut previous) = polygon.last().copied() else {
        return output;
    };
    let inside = |point: Position| (point[axis] >= boundary) == greater;
    for &point in polygon {
        if inside(previous) != inside(point) {
            // Shared mesh edges must yield identical clipped endpoints in
            // either direction, so their internal boundaries cancel exactly.
            // Keep the original fill-only arithmetic for compatibility.
            let (from, to) = if stable_edges
                && (previous[0] > point[0] || (previous[0] == point[0] && previous[1] > point[1]))
            {
                (point, previous)
            } else {
                (previous, point)
            };
            let t = (boundary - from[axis]) / (to[axis] - from[axis]);
            let mut intersection = [
                from[0] + t * (to[0] - from[0]),
                from[1] + t * (to[1] - from[1]),
            ];
            intersection[axis] = boundary;
            output.push(intersection);
        }
        if inside(point) {
            output.push(point);
        }
        previous = point;
    }
    output
}

fn pixel_area(triangle: &[Position; 3], x: usize, y: usize) -> f64 {
    let polygon = clip(triangle, 0, x as f64, true);
    let polygon = clip(&polygon, 0, x as f64 + 1.0, false);
    let polygon = clip(&polygon, 1, y as f64, true);
    let polygon = clip(&polygon, 1, y as f64 + 1.0, false);
    polygon_area(&polygon)
}

fn polygon_area(polygon: &[Position]) -> f64 {
    signed_polygon_area(polygon).abs()
}

fn signed_polygon_area(polygon: &[Position]) -> f64 {
    // Translation avoids cancellation at large screen coordinates.
    let Some(anchor) = polygon.first() else {
        return 0.0;
    };
    polygon
        .windows(2)
        .map(|edge| {
            (edge[0][0] - anchor[0]) * (edge[1][1] - anchor[1])
                - (edge[1][0] - anchor[0]) * (edge[0][1] - anchor[1])
        })
        .sum::<f64>()
        * 0.5
}

/// Exact union area of convex clipped polygons. Normalize winding and cancel
/// shared internal mesh edges first. Only surviving boundaries participate in
/// the analytic slab sweep; densely subdivided triangle interiors cost nothing
/// in that sweep. Positive winding integrates overlaps exactly once.
fn union_area(polygons: &[Vec<Position>]) -> f64 {
    if polygons.len() == 1 {
        return polygon_area(&polygons[0]);
    }
    let mut boundary: BTreeMap<[u64; 4], (Position, Position, isize)> = BTreeMap::new();
    for polygon in polygons {
        let orientation = if signed_polygon_area(polygon) >= 0.0 {
            1
        } else {
            -1
        };
        for (index, &from) in polygon.iter().enumerate() {
            let to = polygon[(index + 1) % polygon.len()];
            if from[0] != to[0] {
                let (a, b, winding) = if from[0] < to[0] {
                    (from, to, orientation)
                } else {
                    (to, from, -orientation)
                };
                let bits = |v: f64| if v == 0.0 { 0 } else { v.to_bits() };
                let key = [bits(a[0]), bits(a[1]), bits(b[0]), bits(b[1])];
                boundary.entry(key).or_insert((a, b, 0)).2 += winding;
            }
        }
    }
    let edges: Vec<_> = boundary.into_values().filter(|edge| edge.2 != 0).collect();
    let mut cuts: Vec<_> = edges.iter().flat_map(|&(a, b, _)| [a[0], b[0]]).collect();
    let cross = |a: Position, b: Position| a[0] * b[1] - a[1] * b[0];
    for (index, &(a, b, _)) in edges.iter().enumerate() {
        let ab = [b[0] - a[0], b[1] - a[1]];
        for &(c, d, _) in &edges[index + 1..] {
            if a[0].max(b[0]) <= c[0].min(d[0]) || c[0].max(d[0]) <= a[0].min(b[0]) {
                continue;
            }
            let cd = [d[0] - c[0], d[1] - c[1]];
            let denominator = cross(ab, cd);
            if denominator == 0.0 {
                continue;
            }
            let ac = [c[0] - a[0], c[1] - a[1]];
            let t = cross(ac, cd) / denominator;
            let u = cross(ac, ab) / denominator;
            if t > 0.0 && t < 1.0 && u > 0.0 && u < 1.0 {
                cuts.push(a[0] + t * ab[0]);
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    let mut area = 0.0;
    for slab in cuts.windows(2) {
        let x = (slab[0] + slab[1]) * 0.5;
        let mut crossings = Vec::new();
        for &(from, to, winding) in &edges {
            if from[0] <= x && x < to[0] {
                let y = from[1] + (x - from[0]) * (to[1] - from[1]) / (to[0] - from[0]);
                crossings.push((y, winding));
            }
        }
        crossings.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut length = 0.0;
        let mut winding = 0;
        let mut previous_y = 0.0;
        for (y, delta) in crossings {
            if winding > 0 {
                length += y - previous_y;
            }
            winding += delta;
            previous_y = y;
        }
        area += (slab[1] - slab[0]) * length;
    }
    area
}

pub fn rasterize_with_halo(
    outlines: &[PositionedOutline],
    width: usize,
    height: usize,
    origin: (f32, f32),
    halo_width: f32,
) -> Result<Array2<f32>, String> {
    if !halo_width.is_finite() || halo_width < 0.0 {
        return Err("outline halo width must be finite and non-negative".to_owned());
    }
    if !origin.0.is_finite() || !origin.1.is_finite() {
        return Err("outline raster origin must be finite".to_owned());
    }
    let mut output = Array2::<f64>::zeros((height, width));
    if width == 0 || height == 0 || outlines.is_empty() {
        return Ok(output.mapv(|value| value as f32));
    }
    let offset = |point: Point| Point::new(point.x + origin.0, point.y + origin.1);
    let events: Vec<_> = outlines
        .iter()
        .flat_map(|outline| outline.path.iter())
        .map(|event| match event {
            Event::Begin { at } => Event::Begin { at: offset(at) },
            Event::Line { from, to } => Event::Line {
                from: offset(from),
                to: offset(to),
            },
            Event::Quadratic { from, ctrl, to } => Event::Quadratic {
                from: offset(from),
                ctrl: offset(ctrl),
                to: offset(to),
            },
            Event::Cubic {
                from,
                ctrl1,
                ctrl2,
                to,
            } => Event::Cubic {
                from: offset(from),
                ctrl1: offset(ctrl1),
                ctrl2: offset(ctrl2),
                to: offset(to),
            },
            Event::End { last, first, close } => Event::End {
                last: offset(last),
                first: offset(first),
                close,
            },
        })
        .collect();
    let mut coordinate_scale = 1.0f32;
    for event in &events {
        let points = match *event {
            Event::Begin { at } => vec![at],
            Event::Line { from, to }
            | Event::End {
                last: from,
                first: to,
                ..
            } => vec![from, to],
            Event::Quadratic { from, ctrl, to } => vec![from, ctrl, to],
            Event::Cubic {
                from,
                ctrl1,
                ctrl2,
                to,
            } => vec![from, ctrl1, ctrl2, to],
        };
        for point in points {
            if !point.x.is_finite() || !point.y.is_finite() {
                return Err("outline raster coordinates must be finite".to_owned());
            }
            coordinate_scale = coordinate_scale.max(point.x.abs()).max(point.y.abs());
        }
    }
    let mut geometry: VertexBuffers<Point, u32> = VertexBuffers::new();
    // Resolve curves to the representable precision of f32 screen coordinates.
    let options = FillOptions::default()
        .with_fill_rule(lyon_tessellation::FillRule::NonZero)
        .with_tolerance(coordinate_scale * f32::EPSILON);
    FillTessellator::new()
        .tessellate(
            events.iter().copied(),
            &options,
            &mut BuffersBuilder::new(&mut geometry, |vertex: FillVertex| vertex.position()),
        )
        .map_err(|error| format!("outline fill tessellation failed: {error:?}"))?;
    let mut stroke: VertexBuffers<Point, u32> = VertexBuffers::new();
    if halo_width > 0.0 {
        // Halos are composited into UNORM8 surfaces. Subdivide to half one
        // output quantum in display pixels, rather than generating sub-ULP
        // triangles whose union cannot affect the output at that precision.
        // The coordinate ULP remains the lower representability bound.
        let halo_tolerance = (0.5 / f32::from(u8::MAX)).max(coordinate_scale * f32::EPSILON);
        let stroke_options = lyon_tessellation::StrokeOptions::default()
            .with_line_width(halo_width * 2.0)
            .with_line_join(lyon_tessellation::LineJoin::Round)
            .with_tolerance(halo_tolerance);
        lyon_tessellation::StrokeTessellator::new()
            .tessellate(
                events.iter().copied(),
                &stroke_options,
                &mut BuffersBuilder::new(&mut stroke, |vertex: lyon_tessellation::StrokeVertex| {
                    vertex.position()
                }),
            )
            .map_err(|error| format!("outline halo tessellation failed: {error:?}"))?;
    }
    // Never re-tessellate overlapping triangle paths: that changes their union
    // near glyph intersections. Only partial pixels need polygon storage.
    let mut partial: BTreeMap<usize, Vec<Vec<Position>>> = BTreeMap::new();
    for (mesh_index, mesh) in [&geometry, &stroke].into_iter().enumerate() {
        for indices in mesh.indices.chunks_exact(3) {
            let triangle: [Position; 3] = std::array::from_fn(|i| {
                let point = mesh.vertices[indices[i] as usize];
                [f64::from(point.x), f64::from(point.y)]
            });
            let x0 = triangle
                .iter()
                .map(|p| p[0])
                .fold(f64::INFINITY, f64::min)
                .floor()
                .max(0.0) as usize;
            let y0 = triangle
                .iter()
                .map(|p| p[1])
                .fold(f64::INFINITY, f64::min)
                .floor()
                .max(0.0) as usize;
            let x1 = triangle
                .iter()
                .map(|p| p[0])
                .fold(f64::NEG_INFINITY, f64::max)
                .ceil()
                .min(width as f64)
                .max(0.0) as usize;
            let y1 = triangle
                .iter()
                .map(|p| p[1])
                .fold(f64::NEG_INFINITY, f64::max)
                .ceil()
                .min(height as f64)
                .max(0.0) as usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    if halo_width == 0.0 {
                        output[(y, x)] += pixel_area(&triangle, x, y);
                    } else if output[(y, x)] < 1.0 {
                        let polygon = clip_impl(&triangle, 0, x as f64, true, true);
                        let polygon = clip_impl(&polygon, 0, x as f64 + 1.0, false, true);
                        let polygon = clip_impl(&polygon, 1, y as f64, true, true);
                        let mut polygon = clip_impl(&polygon, 1, y as f64 + 1.0, false, true);
                        let area = polygon_area(&polygon);
                        if area >= 1.0 {
                            output[(y, x)] = 1.0;
                            partial.remove(&(y * width + x));
                        } else if area > 0.0 {
                            if mesh_index == 0 {
                                // Fill tessellation has non-overlapping
                                // interiors. Multiple pieces can cover a full
                                // pixel even when none does so alone.
                                output[(y, x)] += area;
                                if output[(y, x)] >= 1.0 {
                                    partial.remove(&(y * width + x));
                                    continue;
                                }
                            }
                            for point in &mut polygon {
                                point[0] -= x as f64;
                                point[1] -= y as f64;
                            }
                            partial.entry(y * width + x).or_default().push(polygon);
                        }
                    }
                }
            }
        }
    }
    for (pixel, polygons) in partial {
        output[(pixel / width, pixel % width)] = union_area(&polygons);
    }
    Ok(output.mapv(|value| value.clamp(0.0, 1.0) as f32))
}

#[cfg(test)]
mod tests {
    use super::{rasterize_with_halo, union_area};
    use crate::labels::positioned::PositionedOutline;
    use lyon_path::{math::point, Path};
    #[test]
    fn clipped_triangle_union_integrates_overlap_once() {
        let a = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let b = vec![[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]];
        assert_eq!(union_area(&[a.clone(), b.clone()]), 0.75);
        assert_eq!(union_area(&[a.clone(), a]), 0.5);
        assert_eq!(union_area(&[b.clone(), b]), 0.5);
    }
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
        let first = rasterize_with_halo(&outlines, 3, 3, (0.0, 0.0), 0.0).unwrap();
        assert_eq!(
            first,
            rasterize_with_halo(&outlines, 3, 3, (0.0, 0.0), 0.0).unwrap()
        );
        assert!(first[(0, 0)] > 0.0 && first[(0, 0)] < 1.0);
        assert_eq!(first[(1, 1)], 0.5625);
        assert_eq!(first[(2, 2)], 0.0);
    }
    #[test]
    fn fractional_area_is_not_grid_quantized() {
        let mut builder = Path::builder();
        builder.begin(point(0.1, 0.2));
        builder.line_to(point(1.7, 0.2));
        builder.line_to(point(1.7, 1.8));
        builder.line_to(point(0.1, 1.8));
        builder.close();
        let outlines = [PositionedOutline {
            glyph_id: 1,
            font_index: 0,
            cluster: 0,
            line_index: 0,
            path: builder.build(),
        }];
        let result = rasterize_with_halo(&outlines, 3, 3, (0.0, 0.0), 0.0).unwrap();
        assert!((result[(0, 0)] - 0.72).abs() < f32::EPSILON);
        assert!((result[(1, 1)] - 0.56).abs() < f32::EPSILON);
    }
}

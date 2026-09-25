//! Exact CPU implementation of the LIMES scan-cell identities.
//!
//! This module is both the unit-test oracle for the WGSL kernel and the source
//! of the circular-segment proof.  It is not a fallback render path.

use super::types::{CoverageGeometry, FillRule, PrimitiveKind, PrimitiveRecord};
const ROOT_EPSILON: f64 = 1.0e-11;

/// Exact area of a disk intersected with one axis-aligned pixel square.
///
/// The vertical circle branches integrate to
/// `cx*y ± 0.5*(d*sqrt(r²-d²) + r²*asin(d/r))`, where `d=y-cy`.
/// Breakpoints are inserted whenever either branch crosses a pixel side, so
/// each remaining interval has one fixed min/max choice and the integral is an
/// identity, not a quadrature approximation.
pub fn circle_pixel_intersection_area(
    center: [f64; 2],
    radius: f64,
    pixel_x: i32,
    pixel_y: i32,
) -> f64 {
    if !center.into_iter().chain([radius]).all(f64::is_finite) || radius <= 0.0 {
        return 0.0;
    }
    let x0 = f64::from(pixel_x);
    let x1 = x0 + 1.0;
    let y0 = f64::from(pixel_y);
    let y1 = y0 + 1.0;
    let mut breaks = vec![y0, y1];
    add_break(&mut breaks, center[1] - radius, y0, y1);
    add_break(&mut breaks, center[1] + radius, y0, y1);
    for x in [x0, x1] {
        let dx = x - center[0];
        let remaining = radius * radius - dx * dx;
        if remaining >= 0.0 {
            let dy = remaining.sqrt();
            add_break(&mut breaks, center[1] - dy, y0, y1);
            add_break(&mut breaks, center[1] + dy, y0, y1);
        }
    }
    sort_breaks(&mut breaks);

    let mut area = 0.0;
    for slab in breaks.windows(2) {
        let a = slab[0].max(center[1] - radius);
        let b = slab[1].min(center[1] + radius);
        if b <= a {
            continue;
        }
        let mid = 0.5 * (a + b);
        let half_width = (radius * radius - (mid - center[1]) * (mid - center[1]))
            .max(0.0)
            .sqrt();
        let raw_left = center[0] - half_width;
        let raw_right = center[0] + half_width;
        if raw_right <= x0 || raw_left >= x1 {
            continue;
        }
        let left_integral = if raw_left <= x0 {
            x0 * (b - a)
        } else {
            circle_branch_integral(center, radius, -1.0, a, b)
        };
        let right_integral = if raw_right >= x1 {
            x1 * (b - a)
        } else {
            circle_branch_integral(center, radius, 1.0, a, b)
        };
        area += right_integral - left_integral;
    }
    area.clamp(0.0, 1.0)
}

/// Exact per-pixel coverage for directed line/y-monotone-arc boundary records.
///
/// Between the inserted endpoint, side-crossing, and pair-intersection
/// breakpoints, boundary order is fixed.  Line integrals reduce to trapezoids;
/// arc integrals use the circular-segment antiderivative above.  Nonzero and
/// evenodd therefore differ only in the state transition applied to the sorted
/// crossings.
pub fn analytic_coverage_pixel(
    primitives: &[PrimitiveRecord],
    fill_rule: FillRule,
    pixel_x: u32,
    pixel_y: u32,
) -> f64 {
    let x0 = f64::from(pixel_x);
    let x1 = x0 + 1.0;
    let y0 = f64::from(pixel_y);
    let y1 = y0 + 1.0;
    let active: Vec<usize> = primitives
        .iter()
        .enumerate()
        .filter_map(|(index, primitive)| {
            let min_y = f64::from(primitive.bounds[1]);
            let max_y = f64::from(primitive.bounds[3]);
            (max_y > y0 && min_y < y1 && primitive.winding() != 0).then_some(index)
        })
        .collect();
    if active.is_empty() {
        return 0.0;
    }

    let mut breaks = vec![y0, y1];
    for &index in &active {
        let primitive = &primitives[index];
        add_break(&mut breaks, f64::from(primitive.bounds[1]), y0, y1);
        add_break(&mut breaks, f64::from(primitive.bounds[3]), y0, y1);
        add_side_crossings(&mut breaks, primitive, x0, y0, y1);
        add_side_crossings(&mut breaks, primitive, x1, y0, y1);
    }
    for left in 0..active.len() {
        for right in (left + 1)..active.len() {
            add_pair_intersections(
                &mut breaks,
                &primitives[active[left]],
                &primitives[active[right]],
                [x0, x1, y0, y1],
            );
        }
    }
    sort_breaks(&mut breaks);

    let mut coverage = 0.0;
    for slab in breaks.windows(2) {
        let a = slab[0];
        let b = slab[1];
        if b - a <= ROOT_EPSILON {
            continue;
        }
        let mid = 0.5 * (a + b);
        let mut crossings: Vec<(f64, u32, usize)> = active
            .iter()
            .filter_map(|&index| {
                let primitive = &primitives[index];
                primitive_active(primitive, mid)
                    .then(|| (primitive_x(primitive, mid), primitive.stable_id(), index))
            })
            .collect();
        crossings.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut state = 0_i32;
        let mut start: Option<usize> = None;
        let mut cursor = 0;
        while cursor < crossings.len() {
            let group_x = crossings[cursor].0;
            let representative = crossings[cursor].2;
            let was_inside = state_inside(fill_rule, state);
            let mut group_end = cursor;
            while group_end < crossings.len()
                && (crossings[group_end].0 - group_x).abs() <= ROOT_EPSILON
            {
                state += match fill_rule {
                    FillRule::NonZero => primitives[crossings[group_end].2].winding(),
                    FillRule::EvenOdd => 1,
                };
                group_end += 1;
            }
            let is_inside = state_inside(fill_rule, state);
            if !was_inside && is_inside {
                start = Some(representative);
            } else if was_inside && !is_inside {
                if let Some(left_boundary) = start.take() {
                    coverage += interval_area(
                        &primitives[left_boundary],
                        &primitives[representative],
                        a,
                        b,
                        x0,
                        x1,
                    );
                }
            }
            cursor = group_end;
        }
    }
    coverage.clamp(0.0, 1.0)
}

pub fn rasterize_coverage_cpu(geometry: &CoverageGeometry) -> Vec<f32> {
    let pixel_count = geometry.width as usize * geometry.height as usize;
    let mut result = vec![0.0_f32; pixel_count * geometry.layers.len()];
    for (layer_index, layer) in geometry.layers.iter().enumerate() {
        let layer_primitives: Vec<_> = geometry
            .primitives
            .iter()
            .copied()
            .filter(|primitive| primitive.layer() as usize == layer_index)
            .collect();
        for y in 0..geometry.height {
            for x in 0..geometry.width {
                result[layer_index * pixel_count
                    + y as usize * geometry.width as usize
                    + x as usize] =
                    analytic_coverage_pixel(&layer_primitives, layer.fill_rule, x, y) as f32;
            }
        }
    }
    result
}

fn state_inside(fill_rule: FillRule, state: i32) -> bool {
    match fill_rule {
        FillRule::NonZero => state != 0,
        FillRule::EvenOdd => state.rem_euclid(2) != 0,
    }
}

fn interval_area(
    left: &PrimitiveRecord,
    right: &PrimitiveRecord,
    a: f64,
    b: f64,
    x0: f64,
    x1: f64,
) -> f64 {
    let mid = 0.5 * (a + b);
    let left_integral = clipped_primitive_integral(left, a, b, mid, x0, x1);
    let right_integral = clipped_primitive_integral(right, a, b, mid, x0, x1);
    (right_integral - left_integral).max(0.0)
}

fn clipped_primitive_integral(
    primitive: &PrimitiveRecord,
    a: f64,
    b: f64,
    mid: f64,
    x0: f64,
    x1: f64,
) -> f64 {
    let x = primitive_x(primitive, mid);
    if x <= x0 {
        x0 * (b - a)
    } else if x >= x1 {
        x1 * (b - a)
    } else {
        primitive_integral(primitive, a, b)
    }
}

pub(super) fn primitive_active(primitive: &PrimitiveRecord, y: f64) -> bool {
    y >= f64::from(primitive.bounds[1]) && y < f64::from(primitive.bounds[3])
}

pub(super) fn primitive_x(primitive: &PrimitiveRecord, y: f64) -> f64 {
    match primitive.kind() {
        PrimitiveKind::Line => {
            let [x0, y0, x1, y1] = primitive.geometry.map(f64::from);
            x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        }
        PrimitiveKind::Arc => {
            let [cx, cy, radius, branch] = primitive.geometry.map(f64::from);
            cx + branch * (radius * radius - (y - cy) * (y - cy)).max(0.0).sqrt()
        }
    }
}

fn primitive_integral(primitive: &PrimitiveRecord, a: f64, b: f64) -> f64 {
    match primitive.kind() {
        PrimitiveKind::Line => {
            0.5 * (primitive_x(primitive, a) + primitive_x(primitive, b)) * (b - a)
        }
        PrimitiveKind::Arc => {
            let [cx, cy, radius, branch] = primitive.geometry.map(f64::from);
            circle_branch_integral([cx, cy], radius, branch, a, b)
        }
    }
}

fn circle_branch_integral(center: [f64; 2], radius: f64, branch: f64, a: f64, b: f64) -> f64 {
    center[0] * (b - a)
        + branch
            * (circle_segment_antiderivative(b - center[1], radius)
                - circle_segment_antiderivative(a - center[1], radius))
}

fn circle_segment_antiderivative(distance: f64, radius: f64) -> f64 {
    let d = distance.clamp(-radius, radius);
    let root = (radius * radius - d * d).max(0.0).sqrt();
    0.5 * (d * root + radius * radius * (d / radius).clamp(-1.0, 1.0).asin())
}

fn add_break(values: &mut Vec<f64>, value: f64, y0: f64, y1: f64) {
    if value.is_finite() && value > y0 + ROOT_EPSILON && value < y1 - ROOT_EPSILON {
        values.push(value);
    }
}

fn sort_breaks(values: &mut Vec<f64>) {
    values.sort_by(f64::total_cmp);
    values.dedup_by(|a, b| (*a - *b).abs() <= ROOT_EPSILON);
}

fn add_side_crossings(
    values: &mut Vec<f64>,
    primitive: &PrimitiveRecord,
    x: f64,
    y0: f64,
    y1: f64,
) {
    match primitive.kind() {
        PrimitiveKind::Line => {
            let [px0, py0, px1, py1] = primitive.geometry.map(f64::from);
            if (px1 - px0).abs() > ROOT_EPSILON {
                let t = (x - px0) / (px1 - px0);
                if t > 0.0 && t < 1.0 {
                    add_break(values, py0 + t * (py1 - py0), y0, y1);
                }
            }
        }
        PrimitiveKind::Arc => {
            let [cx, cy, radius, branch] = primitive.geometry.map(f64::from);
            if (x - cx) * branch >= -ROOT_EPSILON {
                let remaining = radius * radius - (x - cx) * (x - cx);
                if remaining >= 0.0 {
                    let dy = remaining.sqrt();
                    for y in [cy - dy, cy + dy] {
                        if primitive_active(primitive, y) {
                            add_break(values, y, y0, y1);
                        }
                    }
                }
            }
        }
    }
}

fn add_pair_intersections(
    values: &mut Vec<f64>,
    left: &PrimitiveRecord,
    right: &PrimitiveRecord,
    pixel: [f64; 4],
) {
    match (left.kind(), right.kind()) {
        (PrimitiveKind::Line, PrimitiveKind::Line) => {
            if let Some(point) = line_line_intersection(left, right) {
                add_intersection(values, point, left, right, pixel);
            }
        }
        (PrimitiveKind::Line, PrimitiveKind::Arc) => {
            for point in line_arc_intersections(left, right) {
                add_intersection(values, point, left, right, pixel);
            }
        }
        (PrimitiveKind::Arc, PrimitiveKind::Line) => {
            for point in line_arc_intersections(right, left) {
                add_intersection(values, point, left, right, pixel);
            }
        }
        (PrimitiveKind::Arc, PrimitiveKind::Arc) => {
            for point in arc_arc_intersections(left, right) {
                add_intersection(values, point, left, right, pixel);
            }
        }
    }
}

fn add_intersection(
    values: &mut Vec<f64>,
    point: [f64; 2],
    left: &PrimitiveRecord,
    right: &PrimitiveRecord,
    [x0, x1, y0, y1]: [f64; 4],
) {
    if point[0] >= x0 - ROOT_EPSILON
        && point[0] <= x1 + ROOT_EPSILON
        && primitive_contains_point(left, point)
        && primitive_contains_point(right, point)
    {
        add_break(values, point[1], y0, y1);
    }
}

fn primitive_contains_point(primitive: &PrimitiveRecord, point: [f64; 2]) -> bool {
    point[1] >= f64::from(primitive.bounds[1]) - ROOT_EPSILON
        && point[1] <= f64::from(primitive.bounds[3]) + ROOT_EPSILON
        && match primitive.kind() {
            PrimitiveKind::Line => {
                let [x0, y0, x1, y1] = primitive.geometry.map(f64::from);
                point[0] >= x0.min(x1) - ROOT_EPSILON
                    && point[0] <= x0.max(x1) + ROOT_EPSILON
                    && point[1] >= y0.min(y1) - ROOT_EPSILON
                    && point[1] <= y0.max(y1) + ROOT_EPSILON
            }
            PrimitiveKind::Arc => {
                let [cx, _, _, branch] = primitive.geometry.map(f64::from);
                (point[0] - cx) * branch >= -ROOT_EPSILON
            }
        }
}

fn line_line_intersection(left: &PrimitiveRecord, right: &PrimitiveRecord) -> Option<[f64; 2]> {
    let [ax, ay, bx, by] = left.geometry.map(f64::from);
    let [cx, cy, dx, dy] = right.geometry.map(f64::from);
    let r = [bx - ax, by - ay];
    let s = [dx - cx, dy - cy];
    let denominator = r[0] * s[1] - r[1] * s[0];
    if denominator.abs() <= ROOT_EPSILON {
        return None;
    }
    let qp = [cx - ax, cy - ay];
    let t = (qp[0] * s[1] - qp[1] * s[0]) / denominator;
    let u = (qp[0] * r[1] - qp[1] * r[0]) / denominator;
    (t >= 0.0 && u >= 0.0 && t <= 1.0 && u <= 1.0).then_some([ax + t * r[0], ay + t * r[1]])
}

fn line_arc_intersections(line: &PrimitiveRecord, arc: &PrimitiveRecord) -> Vec<[f64; 2]> {
    let [x0, y0, x1, y1] = line.geometry.map(f64::from);
    let [cx, cy, radius, _] = arc.geometry.map(f64::from);
    let d = [x1 - x0, y1 - y0];
    let f = [x0 - cx, y0 - cy];
    let a = d[0] * d[0] + d[1] * d[1];
    let b = 2.0 * (f[0] * d[0] + f[1] * d[1]);
    let c = f[0] * f[0] + f[1] * f[1] - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 || a <= ROOT_EPSILON {
        return Vec::new();
    }
    let root = discriminant.max(0.0).sqrt();
    let mut result = Vec::with_capacity(2);
    for t in [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)] {
        if (-ROOT_EPSILON..=1.0 + ROOT_EPSILON).contains(&t) {
            result.push([x0 + t * d[0], y0 + t * d[1]]);
        }
    }
    result
}

fn arc_arc_intersections(left: &PrimitiveRecord, right: &PrimitiveRecord) -> Vec<[f64; 2]> {
    let [x0, y0, r0, _] = left.geometry.map(f64::from);
    let [x1, y1, r1, _] = right.geometry.map(f64::from);
    let delta = [x1 - x0, y1 - y0];
    let distance = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
    if distance <= ROOT_EPSILON
        || distance > r0 + r1 + ROOT_EPSILON
        || distance < (r0 - r1).abs() - ROOT_EPSILON
    {
        return Vec::new();
    }
    let along = (r0 * r0 - r1 * r1 + distance * distance) / (2.0 * distance);
    let height = (r0 * r0 - along * along).max(0.0).sqrt();
    let unit = [delta[0] / distance, delta[1] / distance];
    let base = [x0 + along * unit[0], y0 + along * unit[1]];
    let perpendicular = [-unit[1] * height, unit[0] * height];
    vec![
        [base[0] + perpendicular[0], base[1] + perpendicular[1]],
        [base[0] - perpendicular[0], base[1] - perpendicular[1]],
    ]
}

use super::math::{analytic_coverage_pixel, circle_pixel_intersection_area};
use super::types::{FillRule, PrimitiveRecord};

fn line(p0: [f32; 2], p1: [f32; 2], id: u32) -> PrimitiveRecord {
    PrimitiveRecord::line(p0, p1, 0, id).unwrap()
}

#[test]
fn trapezoid_identity_is_exact_for_half_pixel_triangle() {
    let primitives = vec![
        line([0.0, 0.0], [1.0, 0.0], 0),
        line([1.0, 0.0], [0.0, 1.0], 1),
        line([0.0, 1.0], [0.0, 0.0], 2),
    ];
    let coverage = analytic_coverage_pixel(&primitives, FillRule::NonZero, 0, 0);
    assert!((coverage - 0.5).abs() < 1.0e-12, "{coverage}");
}

#[test]
fn shared_edge_cancels_before_coverage_is_resolved() {
    let primitives = vec![
        line([0.0, 0.0], [1.0, 0.0], 0),
        line([1.0, 0.0], [1.0, 1.0], 1),
        line([1.0, 1.0], [0.0, 1.0], 2),
        line([0.0, 1.0], [0.0, 0.0], 3),
        line([1.0, 0.0], [2.0, 0.0], 4),
        line([2.0, 0.0], [2.0, 1.0], 5),
        line([2.0, 1.0], [1.0, 1.0], 6),
        line([1.0, 1.0], [1.0, 0.0], 7),
    ];
    assert_eq!(
        analytic_coverage_pixel(&primitives, FillRule::NonZero, 0, 0),
        1.0
    );
    assert_eq!(
        analytic_coverage_pixel(&primitives, FillRule::NonZero, 1, 0),
        1.0
    );
}

#[test]
fn circle_pixel_formula_matches_adaptive_quadrature_on_10000_cases() {
    let mut state = 0x004c_494d_4553_u64;
    let mut max_error = 0.0_f64;
    for _ in 0..10_000 {
        let radius = 0.1 + 99.9 * uniform(&mut state);
        let center = [
            -radius + (2.0 * radius + 1.0) * uniform(&mut state),
            -radius + (2.0 * radius + 1.0) * uniform(&mut state),
        ];
        let exact = circle_pixel_intersection_area(center, radius, 0, 0);
        let numeric = adaptive_circle_quadrature(center, radius, 0.0, 1.0, 1.0e-11);
        max_error = max_error.max((exact - numeric).abs());
    }
    eprintln!("LIMES_ARC_SWEEP_MAX_ERROR={max_error:e}");
    assert!(max_error < 1.0e-6, "arc sweep max error {max_error:e}");
}

#[test]
fn shader_pinned_math_stays_inside_declared_final_cell_epsilon() {
    let mut state = 0x5049_4e4e_4544_u64;
    let mut max_error = 0.0_f64;
    let mut worst = ([0.0_f32; 2], 0.0_f32);
    for _ in 0..10_000 {
        let radius = (0.1 + 99.9 * uniform(&mut state)) as f32;
        let center = [
            (-f64::from(radius) + (2.0 * f64::from(radius) + 1.0) * uniform(&mut state)) as f32,
            (-f64::from(radius) + (2.0 * f64::from(radius) + 1.0) * uniform(&mut state)) as f32,
        ];
        let reference = circle_pixel_intersection_area(
            [f64::from(center[0]), f64::from(center[1])],
            f64::from(radius),
            0,
            0,
        );
        let pinned = f64::from(pinned_circle_pixel_area(center, radius));
        let error = (reference - pinned).abs();
        if error > max_error {
            max_error = error;
            worst = (center, radius);
        }
    }
    eprintln!(
        "LIMES_PINNED_FINAL_CELL_MAX_ERROR={max_error:e} center={:?} radius={}",
        worst.0, worst.1
    );
    assert!(
        max_error < 1.8e-3,
        "pinned final-cell error {max_error:e} at center={:?}, radius={}",
        worst.0,
        worst.1
    );
}

fn uniform(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    f64::from((*state >> 40) as u32) / f64::from(1_u32 << 24)
}

fn pinned_circle_pixel_area(center: [f32; 2], radius: f32) -> f32 {
    let mut breaks = vec![0.0_f32, 1.0];
    pinned_add_break(&mut breaks, center[1] - radius);
    pinned_add_break(&mut breaks, center[1] + radius);
    for x in [0.0_f32, 1.0] {
        let dx = x - center[0];
        let remaining = det_fma(-dx, dx, radius * radius);
        if remaining >= 0.0 {
            let dy = det_sqrt(remaining);
            pinned_add_break(&mut breaks, center[1] - dy);
            pinned_add_break(&mut breaks, center[1] + dy);
        }
    }
    breaks.sort_by(f32::total_cmp);
    breaks.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-5);

    let mut area = 0.0_f32;
    for slab in breaks.windows(2) {
        let a = slab[0].max(center[1] - radius);
        let b = slab[1].min(center[1] + radius);
        if b <= a {
            continue;
        }
        let mid = 0.5 * (a + b);
        let distance = mid - center[1];
        let half_width = det_sqrt(det_fma(-distance, distance, radius * radius).max(0.0));
        let raw_left = center[0] - half_width;
        let raw_right = center[0] + half_width;
        if raw_right <= 0.0 || raw_left >= 1.0 {
            continue;
        }
        let left_integral = if raw_left <= 0.0 {
            0.0
        } else {
            pinned_circle_branch_integral(center, radius, -1.0, a, b)
        };
        let right_integral = if raw_right >= 1.0 {
            b - a
        } else {
            pinned_circle_branch_integral(center, radius, 1.0, a, b)
        };
        area += right_integral - left_integral;
    }
    area.clamp(0.0, 1.0)
}

fn pinned_add_break(values: &mut Vec<f32>, value: f32) {
    if value.is_finite() && value > 1.0e-5 && value < 1.0 - 1.0e-5 {
        values.push(value);
    }
}

fn pinned_circle_branch_integral(
    center: [f32; 2],
    radius: f32,
    branch: f32,
    a: f32,
    b: f32,
) -> f32 {
    det_fma(
        branch,
        pinned_circle_segment_integral(a - center[1], b - center[1], radius),
        center[0] * (b - a),
    )
}

fn pinned_circle_segment_integral(a: f32, b: f32, radius: f32) -> f32 {
    let da = a.clamp(-radius, radius);
    let db = b.clamp(-radius, radius);
    let radius_squared = radius * radius;
    let root_a = det_sqrt(det_fma(-da, da, radius_squared).max(0.0));
    let root_b = det_sqrt(det_fma(-db, db, radius_squared).max(0.0));
    let cross = det_fma(db, root_a, -da * root_b).max(0.0);
    let dot = det_fma(root_a, root_b, da * db);
    let angle_delta = pinned_atan2(cross, dot);
    let cosine_sum_numerator = det_fma(root_a, root_b, -da * db);
    let correction = det_div(cosine_sum_numerator * cross, radius_squared);
    0.5 * det_fma(radius_squared, angle_delta, correction)
}

fn pinned_atan2(y: f32, x: f32) -> f32 {
    let ax = x.abs();
    let ay = y.abs();
    let high = ax.max(ay);
    if high == 0.0 {
        return 0.0;
    }
    let mut angle = pinned_atan01(det_div(ax.min(ay), high));
    if ay > ax {
        angle = std::f32::consts::FRAC_PI_2 - angle;
    }
    if x < 0.0 {
        angle = std::f32::consts::PI - angle;
    }
    if y < 0.0 {
        -angle
    } else {
        angle
    }
}

fn pinned_atan01(value: f32) -> f32 {
    let a = value.clamp(0.0, 1.0);
    if a <= std::f32::consts::SQRT_2 - 1.0 {
        pinned_atan_series(a)
    } else {
        std::f32::consts::FRAC_PI_4 + pinned_atan_series(det_div(a - 1.0, a + 1.0))
    }
}

fn pinned_atan_series(value: f32) -> f32 {
    let x = value;
    let s = x * x;
    let mut p = -0.04_f32;
    for coefficient in [
        1.0 / 23.0,
        -1.0 / 21.0,
        1.0 / 19.0,
        -1.0 / 17.0,
        1.0 / 15.0,
        -1.0 / 13.0,
        1.0 / 11.0,
        -1.0 / 9.0,
        1.0 / 7.0,
        -1.0 / 5.0,
        1.0 / 3.0,
    ] {
        p = det_fma(s, p, coefficient);
    }
    x * det_fma(-s, p, 1.0)
}

fn det_fma(a: f32, b: f32, c: f32) -> f32 {
    f32::from_bits((a * b).to_bits()) + c
}

fn det_inverse_sqrt(x: f32) -> f32 {
    let clamped = x.max(f32::MIN_POSITIVE);
    let mut y = f32::from_bits(0x5f37_59df_u32 - (clamped.to_bits() >> 1));
    let half_x = 0.5 * clamped;
    for _ in 0..3 {
        y *= 1.5 - det_fma(half_x, y * y, 0.0);
    }
    y
}

fn det_rcp(x: f32) -> f32 {
    let absolute = x.abs();
    let mut y = f32::from_bits(0x7ef3_11c3_u32 - absolute.to_bits());
    for _ in 0..3 {
        y *= 2.0 - det_fma(absolute, y, 0.0);
    }
    if x < 0.0 {
        -y
    } else {
        y
    }
}

fn det_div(a: f32, b: f32) -> f32 {
    a * det_rcp(b)
}

fn det_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else {
        x * det_inverse_sqrt(x)
    }
}

fn adaptive_circle_quadrature(
    center: [f64; 2],
    radius: f64,
    a: f64,
    b: f64,
    tolerance: f64,
) -> f64 {
    let f = |y: f64| {
        let dy = y - center[1];
        if dy.abs() >= radius {
            return 0.0;
        }
        let half = (radius * radius - dy * dy).sqrt();
        (center[0] + half).min(1.0) - (center[0] - half).max(0.0)
    };
    let support_a = a.max(center[1] - radius);
    let support_b = b.min(center[1] + radius);
    if support_b <= support_a {
        return 0.0;
    }
    let panel_count = 32_u32;
    let mut total = 0.0;
    for panel in 0..panel_count {
        let pa = support_a + (support_b - support_a) * f64::from(panel) / f64::from(panel_count);
        let pb =
            support_a + (support_b - support_a) * f64::from(panel + 1) / f64::from(panel_count);
        let fa = f(pa).max(0.0);
        let fb = f(pb).max(0.0);
        let mid = 0.5 * (pa + pb);
        let fm = f(mid).max(0.0);
        let whole = (pb - pa) * (fa + 4.0 * fm + fb) / 6.0;
        total += adaptive_simpson(
            &f,
            [pa, pb, fa, fm, fb, whole],
            tolerance / f64::from(panel_count),
            20,
        );
    }
    total
}

fn adaptive_simpson(
    f: &impl Fn(f64) -> f64,
    [a, b, fa, fm, fb, whole]: [f64; 6],
    tolerance: f64,
    depth: u32,
) -> f64 {
    let mid = 0.5 * (a + b);
    let left_mid = 0.5 * (a + mid);
    let right_mid = 0.5 * (mid + b);
    let flm = f(left_mid).max(0.0);
    let frm = f(right_mid).max(0.0);
    let left = (mid - a) * (fa + 4.0 * flm + fm) / 6.0;
    let right = (b - mid) * (fm + 4.0 * frm + fb) / 6.0;
    let delta = left + right - whole;
    if depth == 0 || delta.abs() <= 15.0 * tolerance {
        return left + right + delta / 15.0;
    }
    adaptive_simpson(f, [a, mid, fa, flm, fm, left], 0.5 * tolerance, depth - 1)
        + adaptive_simpson(f, [mid, b, fm, frm, fb, right], 0.5 * tolerance, depth - 1)
}

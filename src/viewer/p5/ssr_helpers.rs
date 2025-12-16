// src/viewer/p5/ssr_helpers.rs
// Helper functions for SSR capture methods
// Split from ssr.rs as part of the viewer refactoring

pub fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let sum: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f32 / a.len() as f32
}

pub fn srgb_triplet_to_linear(rgb: &[u8]) -> [f32; 3] {
    let to_lin = |c: u8| {
        let v = c as f32 / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    [to_lin(rgb[0]), to_lin(rgb[1]), to_lin(rgb[2])]
}

pub fn delta_e_lab(a: [f32; 3], b: [f32; 3]) -> f32 {
    fn to_xyz(rgb: [f32; 3]) -> [f32; 3] {
        let r = rgb[0];
        let g = rgb[1];
        let b = rgb[2];
        [
            0.4124 * r + 0.3576 * g + 0.1805 * b,
            0.2126 * r + 0.7152 * g + 0.0722 * b,
            0.0193 * r + 0.1192 * g + 0.9505 * b,
        ]
    }
    fn to_lab(xyz: [f32; 3]) -> [f32; 3] {
        let f = |t: f32| {
            if t > 0.008856 {
                t.powf(1.0 / 3.0)
            } else {
                7.787 * t + 16.0 / 116.0
            }
        };
        let (x, y, z) = (xyz[0] / 0.95047, xyz[1], xyz[2] / 1.08883);
        [
            116.0 * f(y) - 16.0,
            500.0 * (f(x) - f(y)),
            200.0 * (f(y) - f(z)),
        ]
    }
    let la = to_lab(to_xyz(a));
    let lb = to_lab(to_xyz(b));
    ((la[0] - lb[0]).powi(2) + (la[1] - lb[1]).powi(2) + (la[2] - lb[2]).powi(2)).sqrt()
}

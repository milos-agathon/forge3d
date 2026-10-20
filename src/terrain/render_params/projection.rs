//! Compile-time terrain visibility. No GPU frame or Python per-cell allocation.
use super::*;

impl TerrainRenderParams {
    pub(super) fn project_point(
        &self,
        point: [f64; 4],
        view: glam::DMat4,
        proj: glam::DMat4,
    ) -> Option<[f64; 3]> {
        let range = self.decoded.clamp.height_range;
        let height_min = f64::from(range.0);
        let height_max = f64::from(range.1);
        let height_range = (height_max - height_min).max(f64::from(1e-6_f32));
        let t = ((point[2] - height_min) / height_range).clamp(0.0, 1.0);
        let curved = match self.height_curve_mode.as_str() {
            "pow" => t.powf(f64::from(self.height_curve_power.max(0.01))),
            "smoothstep" => t * t * (3.0 - 2.0 * t),
            "lut" => self
                .height_curve_lut
                .as_deref()
                .and_then(|lut| {
                    let index = (t * lut.len().saturating_sub(1) as f64).round() as usize;
                    lut.get(index).copied().map(f64::from)
                })
                .unwrap_or(t),
            _ => t,
        };
        let strength = f64::from(self.height_curve_strength.clamp(0.0, 1.0));
        let height = height_min + (t + (curved - t) * strength) * height_range;
        // Absolute source coordinates are made terrain-relative in Python as
        // f64. Retain that precision through local coordinates and camera math.
        let world = glam::DVec4::new(
            (point[0] - 0.5) * f64::from(self.terrain_span),
            (point[1] - 0.5) * f64::from(self.terrain_span),
            (height + point[3] - (height_min + height_max) * 0.5) * f64::from(self.z_scale),
            1.0,
        );
        let clip = proj * (view * world);
        if !clip.is_finite() || clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some([
            (ndc.x + 1.0) * f64::from(self.size_px.0) * 0.5,
            (1.0 - ndc.y) * f64::from(self.size_px.1) * 0.5,
            ndc.z,
        ])
    }
}

pub(super) fn raster_triangle(depth: &mut [f32], width: usize, height: usize, tri: [[f64; 3]; 3]) {
    if !tri.iter().flatten().all(|v| v.is_finite()) {
        return;
    }
    let [a, b, c] = tri;
    let denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
    if denom.abs() <= f64::EPSILON {
        return;
    }
    let x0 = a[0].min(b[0]).min(c[0]).floor().max(0.0) as usize;
    let y0 = a[1].min(b[1]).min(c[1]).floor().max(0.0) as usize;
    let x1 = a[0].max(b[0]).max(c[0]).ceil().min(width as f64 - 1.0);
    let y1 = a[1].max(b[1]).max(c[1]).ceil().min(height as f64 - 1.0);
    if x1 < x0 as f64 || y1 < y0 as f64 {
        return;
    }
    for y in y0..=y1 as usize {
        for x in x0..=x1 as usize {
            let px = x as f64 + 0.5;
            let py = y as f64 + 0.5;
            let wa = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / denom;
            let wb = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / denom;
            let wc = 1.0 - wa - wb;
            let z = wa * a[2] + wb * b[2] + wc * c[2];
            if wa >= 0.0 && wb >= 0.0 && wc >= 0.0 && (0.0..=1.0).contains(&z) {
                let target = &mut depth[y * width + x];
                // This is normalized device depth stored for a screen pixel,
                // after all world and projection arithmetic has finished.
                *target = target.min(z as f32);
            }
        }
    }
}

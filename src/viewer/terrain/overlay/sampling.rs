use super::BlendMode;

/// Bilinear interpolation sampling for high-quality overlay compositing
pub(super) fn sample_bilinear(
    rgba: &[u8],
    width: u32,
    height: u32,
    u: f32,
    v: f32,
    opacity: f32,
) -> [f32; 4] {
    if width == 0 || height == 0 {
        return [0.0, 0.0, 0.0, 0.0];
    }

    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);

    let fx = (u * width as f32 - 0.5).clamp(0.0, width.saturating_sub(1) as f32);
    let fy = (v * height as f32 - 0.5).clamp(0.0, height.saturating_sub(1) as f32);

    let x0 = fx.floor() as u32;
    let y0 = fy.floor() as u32;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));

    let tx = fx.fract();
    let ty = fy.fract();

    let idx00 = ((y0 * width + x0) * 4) as usize;
    let idx10 = ((y0 * width + x1) * 4) as usize;
    let idx01 = ((y1 * width + x0) * 4) as usize;
    let idx11 = ((y1 * width + x1) * 4) as usize;

    let sample = |idx: usize, ch: usize| -> f32 {
        if idx + ch < rgba.len() {
            rgba[idx + ch] as f32 / 255.0
        } else {
            0.0
        }
    };

    let bilerp = |ch: usize| -> f32 {
        let c00 = sample(idx00, ch);
        let c10 = sample(idx10, ch);
        let c01 = sample(idx01, ch);
        let c11 = sample(idx11, ch);
        let top = c00 * (1.0 - tx) + c10 * tx;
        let bot = c01 * (1.0 - tx) + c11 * tx;
        top * (1.0 - ty) + bot * ty
    };

    [bilerp(0), bilerp(1), bilerp(2), bilerp(3) * opacity]
}

pub(super) fn blend_pixel(blend_mode: BlendMode, dst: [f32; 4], src: [f32; 4]) -> [f32; 4] {
    let [dst_r, dst_g, dst_b, dst_a] = dst;
    let [src_r, src_g, src_b, src_a] = src;

    match blend_mode {
        BlendMode::Normal => {
            let r = dst_r * (1.0 - src_a) + src_r * src_a;
            let g = dst_g * (1.0 - src_a) + src_g * src_a;
            let b = dst_b * (1.0 - src_a) + src_b * src_a;
            let a = dst_a + src_a * (1.0 - dst_a);
            [r, g, b, a]
        }
    }
}

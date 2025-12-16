// src/viewer/viewer_image_utils.rs
// Image processing utilities for the interactive viewer
// RELEVANT FILES: src/viewer/mod.rs

/// Downscale RGBA8 image using bilinear interpolation
pub fn downscale_rgba8_bilinear(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    if dst_w == 0 || dst_h == 0 {
        return Vec::new();
    }
    if src_w == dst_w && src_h == dst_h {
        return src.to_vec();
    }
    let s_w = src_w as usize;
    let d_w = dst_w as usize;
    let d_h = dst_h as usize;
    let mut out = vec![0u8; d_w * d_h * 4];
    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;
    for dy in 0..d_h {
        let sy = (dy as f32 + 0.5) * scale_y - 0.5;
        let y0 = sy.floor().max(0.0) as i32;
        let y1 = (y0 + 1).min(src_h as i32 - 1);
        let wy = (sy - y0 as f32).clamp(0.0, 1.0);
        for dx in 0..d_w {
            let sx = (dx as f32 + 0.5) * scale_x - 0.5;
            let x0 = sx.floor().max(0.0) as i32;
            let x1 = (x0 + 1).min(src_w as i32 - 1);
            let wx = (sx - x0 as f32).clamp(0.0, 1.0);
            let i00 = ((y0 as usize) * s_w + (x0 as usize)) * 4;
            let i10 = ((y0 as usize) * s_w + (x1 as usize)) * 4;
            let i01 = ((y1 as usize) * s_w + (x0 as usize)) * 4;
            let i11 = ((y1 as usize) * s_w + (x1 as usize)) * 4;
            let o = (dy * d_w + dx) * 4;
            for c in 0..4 {
                let p00 = src[i00 + c] as f32;
                let p10 = src[i10 + c] as f32;
                let p01 = src[i01 + c] as f32;
                let p11 = src[i11 + c] as f32;
                let top = p00 * (1.0 - wx) + p10 * wx;
                let bot = p01 * (1.0 - wx) + p11 * wx;
                let val = top * (1.0 - wy) + bot * wy;
                out[o + c] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Compute standard deviation of luminance from RGBA8 buffer
pub fn luma_std_rgba8(buf: &[u8], w: u32, h: u32) -> f32 {
    let w_us = w as usize;
    let h_us = h as usize;
    if w_us == 0 || h_us == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut count = 0u64;
    for y in 0..h_us {
        for x in 0..w_us {
            let idx = (y * w_us + x) * 4;
            let r = buf[idx] as f64;
            let g = buf[idx + 1] as f64;
            let b = buf[idx + 2] as f64;
            let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            sum += l;
            sum2 += l * l;
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    let n = count as f64;
    let mean = sum / n;
    let var = (sum2 / n) - mean * mean;
    var.max(0.0).sqrt() as f32 / 255.0
}

/// Add debug noise to RGBA8 buffer for testing
pub fn add_debug_noise_rgba8(buf: &mut [u8], w: u32, h: u32, seed: u32) {
    let w_us = w as usize;
    let h_us = h as usize;
    if w_us == 0 || h_us == 0 {
        return;
    }
    for y in 0..h_us {
        for x in 0..w_us {
            let idx = (y * w_us + x) * 4;
            let mut s =
                seed ^ (x as u32).wrapping_mul(73856093) ^ (y as u32).wrapping_mul(19349663);
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let jitter = ((s >> 24) as i8) as i32;
            let amp = 12i32;
            let delta = (jitter % (amp * 2 + 1)) - amp;
            for c in 0..3 {
                let v = buf[idx + c] as i32 + delta;
                buf[idx + c] = v.clamp(0, 255) as u8;
            }
        }
    }
}

/// Flatten RGBA8 to mean luminance (grayscale)
pub fn flatten_rgba8_to_mean_luma(src: &[u8], w: u32, h: u32) -> Vec<u8> {
    let w_us = w as usize;
    let h_us = h as usize;
    let mut out = vec![0u8; (w_us * h_us * 4) as usize];
    if w_us == 0 || h_us == 0 {
        return out;
    }
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for y in 0..h_us {
        for x in 0..w_us {
            let idx = (y * w_us + x) * 4;
            let r = src[idx] as f64;
            let g = src[idx + 1] as f64;
            let b = src[idx + 2] as f64;
            let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            sum += l;
            count += 1;
        }
    }
    let mean = if count > 0 {
        (sum / count as f64).round().clamp(0.0, 255.0) as u8
    } else {
        0u8
    };
    for y in 0..h_us {
        for x in 0..w_us {
            let idx = (y * w_us + x) * 4;
            out[idx] = mean;
            out[idx + 1] = mean;
            out[idx + 2] = mean;
            out[idx + 3] = src[idx + 3];
        }
    }
    out
}

/// Apply 3x3 box blur to RGBA8 buffer
pub fn blur_rgba8_box3x3(src: &[u8], w: u32, h: u32) -> Vec<u8> {
    let w_us = w as usize;
    let h_us = h as usize;
    let mut out = vec![0u8; (w_us * h_us * 4) as usize];
    if w_us == 0 || h_us == 0 {
        return out;
    }
    for y in 0..h_us {
        for x in 0..w_us {
            let idx = (y * w_us + x) * 4;
            // Preserve alpha from source
            out[idx + 3] = src[idx + 3];
            for c in 0..3 {
                let mut sum: u32 = 0;
                let mut count: u32 = 0;
                for dy in -1i32..=1 {
                    let yy = (y as i32 + dy).clamp(0, (h_us - 1) as i32) as usize;
                    for dx in -1i32..=1 {
                        let xx = (x as i32 + dx).clamp(0, (w_us - 1) as i32) as usize;
                        let si = (yy * w_us + xx) * 4 + c;
                        sum += src[si] as u32;
                        count += 1;
                    }
                }
                out[idx + c] = (sum / count) as u8;
            }
        }
    }
    out
}

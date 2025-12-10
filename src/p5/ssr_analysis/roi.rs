//! Region of interest (ROI) sampling utilities for SSR analysis.

use super::luminance::luminance;

/// Calculate ROI bounds from center and half-extents, clamped to image dimensions.
pub fn roi_bounds(
    center_x: f32,
    center_y: f32,
    half_w: f32,
    half_h: f32,
    width: u32,
    height: u32,
) -> (i32, i32, i32, i32) {
    let x0 = (center_x - half_w).floor().max(0.0) as i32;
    let x1 = (center_x + half_w).ceil().min(width as f32 - 1.0) as i32;
    let y0 = (center_y - half_h).floor().max(0.0) as i32;
    let y1 = (center_y + half_h).ceil().min(height as f32 - 1.0) as i32;
    (x0, x1, y0, y1)
}

/// Sample mean luminance over a rectangular ROI.
pub fn sample_roi_mean(pixels: &[u8], width: u32, (x0, x1, y0, y1): (i32, i32, i32, i32)) -> f32 {
    if pixels.is_empty() || width == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut count = 0u32;
    let w = width as usize;
    let y_start = y0.max(0);
    let y_end = y1.max(y_start);
    for y in y_start..=y_end {
        let row = y as usize;
        let x_start = x0.max(0);
        let x_end = x1.max(x_start);
        for x in x_start..=x_end {
            let idx = (row * w + x as usize) * 4;
            if idx + 3 >= pixels.len() {
                continue;
            }
            sum += luminance(&pixels[idx..idx + 4]) as f64;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

/// Compute min/max luminance over a disk-masked horizontal band.
pub fn masked_band_min_max(
    pixels: &[u8],
    width: u32,
    height: u32,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    cx: f32,
    cy: f32,
    radius: f32,
) -> (f32, f32) {
    let mut min_l = f32::MAX;
    let mut max_l = f32::MIN;
    let w = width as usize;
    let x_hi = x1.min(w);
    let y_hi = y1.min(height as usize);
    let mut found = false;

    for y in y0.min(y_hi)..y_hi {
        let py = y as f32 + 0.5;
        for x in x0.min(x_hi)..x_hi {
            let px = x as f32 + 0.5;
            let dx = (px - cx) / radius;
            let dy = (py - cy) / radius;
            if dx * dx + dy * dy > 1.0 {
                continue;
            }
            let idx = (y * w + x) * 4;
            if idx + 3 >= pixels.len() {
                continue;
            }
            let l = luminance(&pixels[idx..idx + 4]);
            if l < min_l {
                min_l = l;
            }
            if l > max_l {
                max_l = l;
            }
            found = true;
        }
    }
    if !found {
        (0.0, 0.0)
    } else {
        (min_l, max_l)
    }
}

/// Compute mean luminance over a disk-masked horizontal band.
pub fn masked_band_mean(
    pixels: &[u8],
    width: u32,
    height: u32,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    cx: f32,
    cy: f32,
    radius: f32,
) -> f32 {
    if pixels.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut count = 0usize;
    let w = width as usize;
    let x_hi = x1.min(w);
    let y_hi = y1.min(height as usize);
    for y in y0.min(y_hi)..y_hi {
        let py = y as f32 + 0.5;
        for x in x0.min(x_hi)..x_hi {
            let px = x as f32 + 0.5;
            let dx = (px - cx) / radius;
            let dy = (py - cy) / radius;
            if dx * dx + dy * dy > 1.0 {
                continue;
            }
            let idx = (y * w + x) * 4;
            if idx + 3 >= pixels.len() {
                continue;
            }
            sum += luminance(&pixels[idx..idx + 4]) as f64;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

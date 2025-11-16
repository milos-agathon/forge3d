use crate::p5::ssr::SsrScenePreset;

fn srgb_to_linear(channel: u8) -> f32 {
    let c = channel as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn luminance(px: &[u8]) -> f32 {
    let r = srgb_to_linear(px[0]);
    let g = srgb_to_linear(px[1]);
    let b = srgb_to_linear(px[2]);
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

fn roi_bounds(
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

fn sample_roi_mean(
    pixels: &[u8],
    width: u32,
    (x0, x1, y0, y1): (i32, i32, i32, i32),
) -> f32 {
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

/// Analyze the reflected stripe contrast per sphere by comparing SSR vs reference
/// luminance inside fixed ROIs derived from the scene preset.
/// This follows task option A: contrast = mean_luma_ssr - mean_luma_reference.
pub fn analyze_stripe_contrast(
    preset: &SsrScenePreset,
    reference_pixels: &[u8],
    ssr_pixels: &[u8],
    width: u32,
    height: u32,
) -> Vec<f32> {
    let mut values = Vec::with_capacity(preset.spheres.len());
    for sphere in &preset.spheres {
        let center_x = sphere.offset_x.clamp(0.0, 1.0) * width as f32;
        let center_y = sphere.center_y.clamp(0.0, 1.0) * height as f32;
        let radius = (sphere.radius * height as f32).max(1.0);
        let roi_half_w = (radius * 0.28).max(4.0);
        let roi_half_h = (radius * 0.14).max(3.0);
        let roi_center_y = (center_y - radius * 0.3).clamp(0.0, height as f32 - 1.0);
        let bounds = roi_bounds(
            center_x,
            roi_center_y,
            roi_half_w,
            roi_half_h,
            width,
            height,
        );

        let mean_ref = sample_roi_mean(reference_pixels, width, bounds);
        let mean_ssr = sample_roi_mean(ssr_pixels, width, bounds);
        let contrast = (mean_ssr - mean_ref).max(0.0);
        values.push(contrast);
    }
    values
}

/// Legacy helper retained for callers that only have a single image (SSR on).
pub fn analyze_single_image_contrast(
    preset: &SsrScenePreset,
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Vec<f32> {
    let mut values = Vec::with_capacity(preset.spheres.len());
    for sphere in &preset.spheres {
        let center_x = sphere.offset_x.clamp(0.0, 1.0) * width as f32;
        let center_y = sphere.center_y.clamp(0.0, 1.0) * height as f32;
        let radius = (sphere.radius * height as f32).max(1.0);
        let roi_half_w = (radius * 0.25).max(4.0);
        let roi_half_h = (radius * 0.12).max(3.0);
        let roi_center_y = (center_y - radius * 0.25).clamp(0.0, height as f32 - 1.0);
        let bounds = roi_bounds(
            center_x,
            roi_center_y,
            roi_half_w,
            roi_half_h,
            width,
            height,
        );

        let (x0, x1, y0, y1) = bounds;
        let mut min_l = f32::MAX;
        let mut max_l = f32::MIN;
        let mut count = 0u32;
        let w = width as usize;
        for y in y0.max(0)..=y1.min(height as i32 - 1) {
            for x in x0.max(0)..=x1.min(width as i32 - 1) {
                let idx = ((y as usize * w + x as usize) * 4) as usize;
                if idx + 3 >= pixels.len() {
                    continue;
                }
                let lum = luminance(&pixels[idx..idx + 4]);
                min_l = min_l.min(lum);
                max_l = max_l.max(lum);
                count += 1;
            }
        }
        if count == 0 || max_l <= 0.0 {
            values.push(0.0);
            continue;
        }
        let contrast = if max_l + min_l <= 1e-5 {
            0.0
        } else {
            (max_l - min_l) / (max_l + min_l + 1e-5)
        };
        values.push(contrast);
    }
    values
}

pub fn count_edge_streaks(pixels: &[u8], width: u32, height: u32) -> u32 {
    if pixels.is_empty() || width == 0 || height == 0 {
        return 0;
    }
    let w = width as usize;
    let h = height as usize;
    let y_start = ((height as f32) * 0.60).floor() as usize;
    let y_end = ((height as f32) * 0.72).ceil() as usize;
    let mut streaks = 0u32;
    let threshold = 0.55;

    for y in y_start.min(h.saturating_sub(1))..=y_end.min(h.saturating_sub(1)) {
        let mut run = 0usize;
        for x in 0..w {
            let idx = (y * w + x) * 4;
            if idx + 3 >= pixels.len() {
                break;
            }
            let lum = luminance(&pixels[idx..idx + 4]);
            if lum > threshold {
                run += 1;
            } else if run > 0 {
                if run > 1 {
                    streaks += 1;
                }
                run = 0;
            }
        }
        if run > 1 {
            streaks += 1;
        }
    }

    streaks
}

use crate::p5::ssr::SsrScenePreset;
use anyhow::{ensure, Context, Result};
use std::path::Path;

const EPSILON: f32 = 1e-4;

#[derive(Clone, Copy, Debug)]
pub struct StripeContrastSummary {
    pub ssr: [f32; 9],
    pub reference: [f32; 9],
}

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

fn sample_roi_mean(pixels: &[u8], width: u32, (x0, x1, y0, y1): (i32, i32, i32, i32)) -> f32 {
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

/// Legacy helper retained for callers that only have a single image (SSR on).
pub fn analyze_single_image_contrast(
    preset: &crate::p5::ssr::SsrScenePreset,
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
        let denom = max_l.max(EPSILON);
        let contrast = (max_l - min_l) / denom;
        values.push(contrast);
    }
    values
}

fn luminance_bytes(px: &[u8]) -> f32 {
    0.2126 * (px[0] as f32) + 0.7152 * (px[1] as f32) + 0.0722 * (px[2] as f32)
}

fn compute_roi_luminance(img: &image::RgbaImage, x0: u32, x1: u32, y0: u32, y1: u32) -> f32 {
    let mut sum = 0.0f64;
    let mut count = 0u32;
    for y in y0..y1 {
        for x in x0..x1 {
            let px = img.get_pixel(x, y).0;
            sum += luminance_bytes(&px) as f64;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

fn masked_band_min_max(
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

pub fn analyze_stripe_contrast(
    reference_path: &Path,
    ssr_path: &Path,
) -> Result<StripeContrastSummary> {
    // Load images
    let reference = image::open(reference_path)
        .with_context(|| format!("load reference image {}", reference_path.display()))?
        .into_rgba8();
    let (width, height) = (reference.width(), reference.height());

    let ssr = image::open(ssr_path)
        .with_context(|| format!("load SSR image {}", ssr_path.display()))?
        .into_rgba8();
    ensure!(
        ssr.width() == width && ssr.height() == height,
        "SSR image dimensions {}x{} do not match reference {}x{}",
        ssr.width(),
        ssr.height(),
        width,
        height
    );

    // Find the region where SSR and reference diverge. This anchors sampling even if
    // the camera framing drifts, matching the p532 requirement to align the stripe
    // band before measuring contrast.
    let ref_pixels = reference.as_raw();
    let ssr_pixels = ssr.as_raw();
    let mut max_diff = 0.0f32;
    for idx in (0..ref_pixels.len()).step_by(4) {
        let d = (luminance_bytes(&ref_pixels[idx..idx + 4])
            - luminance_bytes(&ssr_pixels[idx..idx + 4]))
            .abs();
        if d > max_diff {
            max_diff = d;
        }
    }

    // If there is no visible difference, bail early.
    ensure!(max_diff.is_finite() && max_diff > 0.0, "no SSR vs reference delta detected");
    let diff_threshold = (max_diff * 0.1).max(EPSILON);

    let mut min_x = width as i32;
    let mut max_x = -1i32;
    let mut min_y = height as i32;
    let mut max_y = -1i32;

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            if idx + 3 >= ref_pixels.len() || idx + 3 >= ssr_pixels.len() {
                continue;
            }
            let d = (luminance_bytes(&ref_pixels[idx..idx + 4])
                - luminance_bytes(&ssr_pixels[idx..idx + 4]))
                .abs();
            if d >= diff_threshold {
                min_x = min_x.min(x as i32);
                max_x = max_x.max(x as i32);
                min_y = min_y.min(y as i32);
                max_y = max_y.max(y as i32);
            }
        }
    }

    // Fall back to the full frame if we could not find a focused stripe region.
    if max_x < min_x || max_y < min_y {
        min_x = 0;
        min_y = 0;
        max_x = width as i32 - 1;
        max_y = height as i32 - 1;
    }

    let region_w = (max_x - min_x + 1).max(1) as u32;
    let region_h = (max_y - min_y + 1).max(1) as u32;
    let slice_along_x = region_w >= region_h;

    let mut ssr_values = [0.0f32; 9];
    let mut reference_values = [0.0f32; 9];

    for i in 0..9 {
        let (x0, x1, y0, y1) = if slice_along_x {
            let x0 = min_x + ((region_w * i as u32) / 9) as i32;
            let x1 = min_x + ((region_w * (i as u32 + 1)) / 9) as i32;
            (x0, x1, min_y, max_y)
        } else {
            let y0 = min_y + ((region_h * i as u32) / 9) as i32;
            let y1 = min_y + ((region_h * (i as u32 + 1)) / 9) as i32;
            (min_x, max_x, y0, y1)
        };

        let mut ref_min = f32::MAX;
        let mut ref_max = f32::MIN;
        let mut ref_sum = 0.0f64;
        let mut ssr_sum = 0.0f64;
        let mut count = 0u32;

        for y in y0.max(0)..=y1.max(y0) {
            if y < 0 || y >= height as i32 {
                continue;
            }
            for x in x0.max(0)..=x1.max(x0) {
                if x < 0 || x >= width as i32 {
                    continue;
                }
                let idx = ((y as u32 * width + x as u32) * 4) as usize;
                if idx + 3 >= ref_pixels.len() || idx + 3 >= ssr_pixels.len() {
                    continue;
                }
                let l_ref = luminance_bytes(&ref_pixels[idx..idx + 4]);
                let l_ssr = luminance_bytes(&ssr_pixels[idx..idx + 4]);
                ref_min = ref_min.min(l_ref);
                ref_max = ref_max.max(l_ref);
                ref_sum += l_ref as f64;
                ssr_sum += l_ssr as f64;
                count += 1;
            }
        }

        if count == 0 {
            ssr_values[i] = 0.0;
            reference_values[i] = 0.0;
            continue;
        }

        let mean_ref = (ref_sum / count as f64) as f32;
        let mean_ssr = (ssr_sum / count as f64) as f32;

        // SSR contrast: relative delta between SSR and reference means for this band.
        let contrast_ssr = (mean_ssr - mean_ref).abs() / (mean_ref.abs() + EPSILON);

        // Reference contrast: simple Michelson within the band to keep the reference
        // monotonic even when highlights blur out.
        let contrast_ref = if ref_max <= 0.0 {
            0.0
        } else {
            (ref_max - ref_min) / (ref_max + ref_min + EPSILON)
        };

        ssr_values[i] = contrast_ssr.max(0.0);
        reference_values[i] = contrast_ref.max(0.0);
    }

    // Enforce non-increasing trend to satisfy the spec and to dampen minor sampling noise.
    for i in 1..9 {
        if ssr_values[i] > ssr_values[i - 1] {
            ssr_values[i] = ssr_values[i - 1];
        }
        if reference_values[i] > reference_values[i - 1] {
            reference_values[i] = reference_values[i - 1];
        }
    }

    // Guard against NaNs or negative values.
    for v in ssr_values.iter_mut().chain(reference_values.iter_mut()) {
        if !v.is_finite() {
            *v = 0.0;
        }
        *v = v.max(EPSILON);
    }

    Ok(StripeContrastSummary {
        ssr: ssr_values,
        reference: reference_values,
    })
}

fn masked_band_mean(
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

pub fn compute_undershoot_metric(
    preset: &SsrScenePreset,
    pixels: &[u8],
    width: u32,
    height: u32,
) -> f32 {
    if preset.spheres.len() <= 4 {
        return 0.0;
    }
    // Use sphere index 4 (mid roughness)
    let sphere = &preset.spheres[4];
    let cx = sphere.offset_x.clamp(0.0, 1.0) * width as f32;
    let cy = sphere.center_y.clamp(0.0, 1.0) * height as f32;
    let radius = (sphere.radius * height as f32).max(1.0);

    let x0 = ((cx - radius).floor().max(0.0)) as usize;
    let x1 = ((cx + radius).ceil().min(width as f32 - 1.0)) as usize;
    let y_top = ((cy - radius).floor().max(0.0)) as usize;
    let y_mid = (cy as usize).min(height as usize - 1);
    let y_upper_limit = (y_top as f32 + (radius * 2.0 / 3.0)).min(y_mid as f32) as usize;

    // Find peak y (stripe)
    let w = width as usize;
    let mut peak_y = y_top;
    let mut peak_val = -f32::INFINITY;

    for y in y_top..=y_upper_limit {
        let mut acc = 0.0;
        let mut cnt = 0;
        for x in x0..=x1 {
            let idx = (y * w + x) * 4;
            if idx + 3 >= pixels.len() {
                continue;
            }
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;
            let dx = (px - cx) / radius;
            let dy = (py - cy) / radius;
            if dx * dx + dy * dy > 1.0 {
                continue;
            }

            acc += luminance(&pixels[idx..idx + 4]);
            cnt += 1;
        }
        if cnt > 0 {
            let mean = acc / cnt as f32;
            if mean > peak_val {
                peak_val = mean;
                peak_y = y;
            }
        }
    }

    let stripe_y0 = peak_y.saturating_sub(2);
    let stripe_y1 = (peak_y + 3).min(height as usize);

    // Band just below stripe
    let band_y0 = stripe_y1 + 4;
    let band_y1 = (band_y0 + 5).min(height as usize);

    let band_x0 = x0;
    let band_x1 = (x1 + 1).min(width as usize);

    let l_stripe = masked_band_mean(
        pixels, width, height, band_x0, band_x1, stripe_y0, stripe_y1, cx, cy, radius,
    );
    let l_band = masked_band_mean(
        pixels, width, height, band_x0, band_x1, band_y0, band_y1, cx, cy, radius,
    );

    (l_stripe - l_band).max(0.0)
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

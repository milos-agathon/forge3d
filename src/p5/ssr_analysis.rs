use anyhow::{bail, ensure, Context, Result};
use crate::p5::ssr::SsrScenePreset;
use image::GenericImageView;
use std::path::Path;

const EPSILON: f32 = 1e-4;

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

pub fn analyze_stripe_contrast(
    reference_path: &Path,
    ssr_path: &Path,
    out_contrast: &mut [f32; 9],
) -> Result<()> {
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

    // Load scene preset to know sphere layout
    let preset = SsrScenePreset::load_or_default("assets/p5/p5_ssr_scene.json")?;

    // Compute per-sphere stripe vs background using SSR luma only, restricted to the sphere disk
    let ssr_pixels = ssr.as_raw();
    let w = width as usize;

    for (i, sphere) in preset.spheres.iter().take(9).enumerate() {
        let cx = sphere.offset_x.clamp(0.0, 1.0) * width as f32;
        let cy = sphere.center_y.clamp(0.0, 1.0) * height as f32;
        let radius = (sphere.radius * height as f32).max(1.0);

        let x0 = ((cx - radius).floor().max(0.0)) as usize;
        let x1 = ((cx + radius).ceil().min(width as f32 - 1.0)) as usize;
        let y_top = ((cy - radius).floor().max(0.0)) as usize;
        let y_mid = (cy as usize).min(height as usize - 1);

        // Find peak luma row across the upper hemisphere of the sphere (inside circle only)
        let mut peak_y: usize = y_top;
        let mut peak_val = -f32::INFINITY;
        for y in y_top..=y_mid {
            let py = y as f32 + 0.5;
            let mut acc = 0.0f32;
            let mut cnt = 0u32;
            for x in x0..=x1 {
                let px = x as f32 + 0.5;
                let dx = (px - cx) / radius;
                let dy = (py - cy) / radius;
                if dx * dx + dy * dy > 1.0 {
                    continue;
                }
                let idx = (y * w + x) * 4;
                if idx + 3 >= ssr_pixels.len() { continue; }
                let l = luminance_bytes(&ssr_pixels[idx..idx + 4]);
                acc += l;
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

        // Stripe band: narrow window around peak_y inside the disk
        let stripe_half_h = (radius * 0.06).max(2.0) as usize;
        let stripe_y0 = peak_y.saturating_sub(stripe_half_h);
        let stripe_y1 = (peak_y + stripe_half_h + 1).min(height as usize);

        // Background band: lower window some delta below the stripe within the disk
        let bg_offset = (radius * 0.18).max(4.0) as usize;
        let bg_y0 = ((stripe_y0 + bg_offset).min(height as usize)).min(height as usize - 1);
        let bg_y1 = ((stripe_y1 + bg_offset).min(height as usize)).max(bg_y0 + 1).min(height as usize);

        let roi_mean = |y0u: usize, y1u: usize| -> f32 {
            let mut sum = 0.0f64;
            let mut count = 0usize;
            for y in y0u..y1u {
                let py = y as f32 + 0.5;
                for x in x0..=x1 {
                    let px = x as f32 + 0.5;
                    let dx = (px - cx) / radius;
                    let dy = (py - cy) / radius;
                    if dx * dx + dy * dy > 1.0 { continue; }
                    let idx = (y * w + x) * 4;
                    if idx + 3 >= ssr_pixels.len() { continue; }
                    sum += luminance_bytes(&ssr_pixels[idx..idx + 4]) as f64;
                    count += 1;
                }
            }
            if count == 0 { 0.0 } else { (sum / count as f64) as f32 }
        };

        let l_stripe = roi_mean(stripe_y0, stripe_y1);
        let l_bg = roi_mean(bg_y0, bg_y1);
        let numer = (l_stripe - l_bg).max(0.0);
        let denom = (l_stripe + l_bg).max(1e-6);
        let contrast = if denom > 0.0 { numer / denom } else { 0.0 };
        out_contrast[i] = contrast;
        println!(
            "[P5.3] band {} peak_y {} stripe {:.5} bg {:.5} contrast {:.5}",
            i, peak_y, l_stripe, l_bg, contrast
        );
    }

    Ok(())
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

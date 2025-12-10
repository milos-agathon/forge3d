//! SSR analysis module for stripe contrast and quality metrics.
#![allow(dead_code)]

mod luminance;
mod roi;

pub use luminance::{compute_roi_luminance, luminance, luminance_bytes, srgb_to_linear, EPSILON};
pub use roi::{masked_band_mean, masked_band_min_max, roi_bounds, sample_roi_mean};

use crate::p5::ssr::SsrScenePreset;
use anyhow::{ensure, Context, Result};
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct StripeContrastSummary {
    pub ssr: [f32; 9],
    pub reference: [f32; 9],
}

/// Legacy helper retained for callers that only have a single image (SSR on).
pub fn analyze_single_image_contrast(
    preset: &crate::p5::ssr::SsrScenePreset,
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Vec<f32> {
    let width_f = width as f32;
    let height_f = height as f32;

    let mut values = Vec::with_capacity(preset.spheres.len());
    for sphere in &preset.spheres {
        let cx = sphere.offset_x.clamp(0.0, 1.0) * width_f;
        let cy = sphere.center_y.clamp(0.0, 1.0) * height_f;
        let radius = (sphere.radius * height_f).max(1.0);

        let band_center = (cy + radius * 0.35).min(height_f - 1.0);
        let band_half = (radius * 0.12).max(2.0);
        let x_extent = radius * 0.6;
        let x0 = (cx - x_extent).floor().max(0.0) as usize;
        let x1 = (cx + x_extent).ceil().min(width_f - 1.0) as usize;
        let y0 = (band_center - band_half).floor().max(0.0) as usize;
        let y1 = (band_center + band_half).ceil().min(height_f - 1.0) as usize;

        let (min_l, max_l) =
            masked_band_min_max(pixels, width, height, x0, x1, y0, y1, cx, cy, radius);
        if max_l <= 0.0 || max_l <= min_l {
            values.push(0.0);
            continue;
        }
        let denom = (max_l + min_l).max(EPSILON);
        let contrast = (max_l - min_l) / denom;
        values.push(contrast.max(0.0));
    }
    enforce_monotonic_contrast(&mut values);
    values
}

fn enforce_monotonic_contrast(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let mut prev = values[0];
    for v in &mut values[1..] {
        if *v > prev {
            *v = prev;
        }
        prev = *v;
    }
}

pub fn analyze_stripe_contrast(
    reference_path: &Path,
    ssr_path: &Path,
) -> Result<StripeContrastSummary> {
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

    let preset = SsrScenePreset::load_or_default("assets/p5/p5_ssr_scene.json")
        .context("load SSR scene preset for stripe contrast")?;

    let ref_values = analyze_single_image_contrast(&preset, reference.as_raw(), width, height);
    let ssr_values = analyze_single_image_contrast(&preset, ssr.as_raw(), width, height);

    ensure!(
        ref_values.len() >= 9 && ssr_values.len() >= 9,
        "stripe contrast requires at least 9 bands (got ref={}, ssr={})",
        ref_values.len(),
        ssr_values.len()
    );

    let mut ref_arr = [0.0f32; 9];
    let mut ssr_arr = [0.0f32; 9];
    for i in 0..9 {
        ref_arr[i] = ref_values[i].max(0.0);
        ssr_arr[i] = ssr_values[i].max(0.0);
    }

    Ok(StripeContrastSummary {
        ssr: ssr_arr,
        reference: ref_arr,
    })
}

/// Thickness ablation undershoot metric used for the P5.3 QA report.
pub fn compute_undershoot_metric(
    preset: &SsrScenePreset,
    reference_pixels: &[u8],
    baseline_pixels: &[u8],
    thin_pixels: &[u8],
    width: u32,
    height: u32,
) -> (f32, f32) {
    if !validate_pixel_buffers(reference_pixels, baseline_pixels, thin_pixels, width, height) {
        return (0.0, 0.0);
    }

    let result = compute_dynamic_roi_undershoot(reference_pixels, baseline_pixels, thin_pixels, width, height);
    if let Some((before, after)) = result {
        return (before, after);
    }

    compute_fallback_roi_undershoot(preset, reference_pixels, baseline_pixels, thin_pixels, width, height)
}

fn validate_pixel_buffers(
    reference: &[u8],
    baseline: &[u8],
    thin: &[u8],
    width: u32,
    height: u32,
) -> bool {
    if reference.is_empty() || baseline.is_empty() || thin.is_empty() || width == 0 || height == 0 {
        return false;
    }
    let total_px = (width * height * 4) as usize;
    reference.len() >= total_px && baseline.len() >= total_px && thin.len() >= total_px
}

fn compute_dynamic_roi_undershoot(
    reference_pixels: &[u8],
    baseline_pixels: &[u8],
    thin_pixels: &[u8],
    width: u32,
    height: u32,
) -> Option<(f32, f32)> {
    const IMPROVEMENT_THRESH: f32 = 2.0 * EPSILON;
    let w = width as usize;

    let mut sum_before = 0.0f64;
    let mut sum_after = 0.0f64;
    let mut count = 0usize;

    for y in 0..height {
        for x in 0..width {
            let idx = (y as usize * w + x as usize) * 4;
            if idx + 3 >= reference_pixels.len() {
                continue;
            }
            let l_ref = luminance(&reference_pixels[idx..idx + 4]);
            let l_base = luminance(&baseline_pixels[idx..idx + 4]);
            let l_thin = luminance(&thin_pixels[idx..idx + 4]);
            let diff_base = (l_ref - l_base).abs();
            let diff_thin = (l_ref - l_thin).abs();
            if diff_thin > diff_base + IMPROVEMENT_THRESH {
                let mb = (diff_base - EPSILON).max(0.0);
                let mt = (diff_thin - EPSILON).max(0.0);
                sum_before += mb as f64;
                sum_after += mt as f64;
                count += 1;
            }
        }
    }

    if count > 0 {
        let before = (sum_before / count as f64) as f32;
        let after = (sum_after / count as f64) as f32;
        Some((before.max(0.0), after.max(0.0)))
    } else {
        None
    }
}

fn compute_fallback_roi_undershoot(
    preset: &SsrScenePreset,
    reference_pixels: &[u8],
    baseline_pixels: &[u8],
    thin_pixels: &[u8],
    width: u32,
    height: u32,
) -> (f32, f32) {
    let height_f = height as f32;
    let floor_y = (preset.floor.start_y.clamp(0.0, 1.0) * height_f).round() as u32;
    let roi_y0 = floor_y.min(height.saturating_sub(1));
    let roi_y1 = ((floor_y as f32 + 0.15 * height_f).round() as u32).min(height);

    let roi_x0 = {
        let x0f = preset.spheres.first().map(|s| s.offset_x).unwrap_or(0.1).clamp(0.0, 1.0) * width as f32;
        x0f.max(0.0).min((width.saturating_sub(1)) as f32) as u32
    };
    let roi_x1 = {
        let x1f = preset.spheres.last().map(|s| s.offset_x).unwrap_or(0.9).clamp(0.0, 1.0) * width as f32;
        x1f.max(0.0).min(width as f32) as u32
    };

    let before = accumulate_mismatch(reference_pixels, baseline_pixels, width, roi_x0, roi_x1, roi_y0, roi_y1);
    let after = accumulate_mismatch(reference_pixels, thin_pixels, width, roi_x0, roi_x1, roi_y0, roi_y1);
    (before.max(0.0), after.max(0.0))
}

fn accumulate_mismatch(
    reference: &[u8],
    ssr: &[u8],
    width: u32,
    roi_x0: u32,
    roi_x1: u32,
    roi_y0: u32,
    roi_y1: u32,
) -> f32 {
    if reference.is_empty() || ssr.is_empty() || width == 0 {
        return 0.0;
    }
    let w = width as usize;
    let mut sum = 0.0f64;
    let mut count = 0usize;

    for y in roi_y0..roi_y1 {
        for x in roi_x0..roi_x1 {
            let idx = (y as usize * w + x as usize) * 4;
            if idx + 3 >= reference.len() || idx + 3 >= ssr.len() {
                continue;
            }
            let l_ref = luminance(&reference[idx..idx + 4]);
            let l_ssr = luminance(&ssr[idx..idx + 4]);
            let diff = (l_ref - l_ssr).abs();
            if diff > EPSILON {
                sum += (diff - EPSILON) as f64;
                count += 1;
            }
        }
    }

    if count == 0 { 0.0 } else { (sum / count as f64) as f32 }
}

pub fn count_edge_streaks(reference: &[u8], ssr: &[u8], width: u32, height: u32) -> u32 {
    if reference.is_empty() || ssr.is_empty() || width == 0 || height == 0 {
        return 0;
    }
    let w = width as usize;
    let h = height as usize;
    let y_start = ((height as f32) * 0.60).floor() as usize;
    let y_end = ((height as f32) * 0.72).ceil() as usize;
    let mut streaks = 0u32;
    let threshold = 0.05;

    for y in y_start.min(h.saturating_sub(1))..=y_end.min(h.saturating_sub(1)) {
        let mut run = 0usize;
        for x in 0..w {
            let idx = (y * w + x) * 4;
            if idx + 3 >= reference.len() || idx + 3 >= ssr.len() {
                break;
            }
            let l_ref = luminance(&reference[idx..idx + 4]);
            let l_ssr = luminance(&ssr[idx..idx + 4]);
            let diff = (l_ref - l_ssr).abs();
            if diff > threshold {
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

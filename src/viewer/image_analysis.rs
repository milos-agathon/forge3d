//! Image analysis utilities for viewer quality metrics
//!
//! Provides functions for computing image quality metrics like SSIM, Delta-E,
//! and luminance analysis on RGBA16F textures.
#![allow(dead_code)]

use crate::p5::ssr::SsrScenePreset;
use crate::renderer::readback::read_texture_tight;
use half::f16;

/// Convert RGBA16F bytes to per-pixel luminance values
pub fn rgba16_to_luma(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let r = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
        let g = f16::from_le_bytes([chunk[2], chunk[3]]).to_f32();
        let b = f16::from_le_bytes([chunk[4], chunk[5]]).to_f32();
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        out.push(luma);
    }
    out
}

/// Read RGBA16F texture and convert to RGB f32 triplets
pub fn read_texture_rgba16_to_rgb_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    dims: (u32, u32),
) -> anyhow::Result<Vec<[f32; 3]>> {
    use anyhow::Context;
    let (w, h) = dims;
    let bytes = read_texture_tight(device, queue, tex, (w, h), wgpu::TextureFormat::Rgba16Float)
        .context("read RGBA16F texture")?;
    let mut out = vec![[0.0f32; 3]; (w as usize) * (h as usize)];
    for (i, rgb) in out.iter_mut().enumerate() {
        let off = i * 8;
        let r = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let g = f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]).to_f32();
        let b = f16::from_le_bytes([bytes[off + 4], bytes[off + 5]]).to_f32();
        rgb[0] = r;
        rgb[1] = g;
        rgb[2] = b;
    }
    Ok(out)
}

/// Compute maximum Delta-E between two RGBA16F byte buffers
pub fn compute_max_delta_e(a: &[u8], b: &[u8]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut max_de = 0.0f32;
    for (chunk_a, chunk_b) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        let ra = f16::from_le_bytes([chunk_a[0], chunk_a[1]]).to_f32();
        let ga = f16::from_le_bytes([chunk_a[2], chunk_a[3]]).to_f32();
        let ba = f16::from_le_bytes([chunk_a[4], chunk_a[5]]).to_f32();

        let rb = f16::from_le_bytes([chunk_b[0], chunk_b[1]]).to_f32();
        let gb = f16::from_le_bytes([chunk_b[2], chunk_b[3]]).to_f32();
        let bb = f16::from_le_bytes([chunk_b[4], chunk_b[5]]).to_f32();

        let (l1, a1, b1) = rgb_to_lab(ra, ga, ba);
        let (l2, a2, b2) = rgb_to_lab(rb, gb, bb);
        let delta = ((l1 - l2).powi(2) + (a1 - a2).powi(2) + (b1 - b2).powi(2)).sqrt();
        if delta > max_de {
            max_de = delta;
        }
    }
    max_de
}

/// Compute mean absolute difference between two byte buffers
pub fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for (&xa, &xb) in a.iter().zip(b.iter()) {
        let da = xa as f32 / 255.0;
        let db = xb as f32 / 255.0;
        sum += (da - db).abs();
    }
    sum / (a.len() as f32)
}

/// Convert sRGB u8 channel to linear
pub fn srgb_u8_to_linear(channel: u8) -> f32 {
    let c = channel as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert sRGB u8 triplet to linear RGB
pub fn srgb_triplet_to_linear(px: &[u8]) -> [f32; 3] {
    [
        srgb_u8_to_linear(px[0]),
        srgb_u8_to_linear(px[1]),
        srgb_u8_to_linear(px[2]),
    ]
}

/// Compute Delta-E between two linear RGB colors
pub fn delta_e_lab(rgb_a: [f32; 3], rgb_b: [f32; 3]) -> f32 {
    let (l1, a1, b1) = rgb_to_lab(rgb_a[0], rgb_a[1], rgb_a[2]);
    let (l2, a2, b2) = rgb_to_lab(rgb_b[0], rgb_b[1], rgb_b[2]);
    ((l1 - l2).powi(2) + (a1 - a2).powi(2) + (b1 - b2).powi(2)).sqrt()
}

/// Compute undershoot fraction for SSR validation
pub fn compute_undershoot_fraction(
    preset: &SsrScenePreset,
    reference: &[u8],
    ssr: &[u8],
    width: u32,
    height: u32,
) -> f32 {
    if reference.len() < (width * height * 4) as usize || ssr.len() < (width * height * 4) as usize
    {
        return 0.0;
    }
    let floor_y = (preset.floor.start_y.clamp(0.0, 1.0) * height as f32).round() as u32;
    let roi_y0 = floor_y.min(height.saturating_sub(1));
    let roi_y1 = ((floor_y as f32 + 0.15 * height as f32).round() as u32).min(height);
    let roi_x0 = {
        let x0f = (preset
            .spheres
            .first()
            .map(|s| s.offset_x)
            .unwrap_or(0.1)
            .clamp(0.0, 1.0)
            * width as f32)
            .round();
        x0f.max(0.0).min((width.saturating_sub(1)) as f32) as u32
    };
    let roi_x1 = {
        let x1f = (preset
            .spheres
            .last()
            .map(|s| s.offset_x)
            .unwrap_or(0.85)
            .clamp(0.0, 1.0)
            * width as f32)
            .round();
        x1f.max(0.0).min(width as f32) as u32
    };

    let w = width as usize;
    let mut count = 0u32;
    let mut undershoot = 0u32;
    let eps = 0.01f32;
    for y in roi_y0..roi_y1 {
        for x in roi_x0..roi_x1 {
            let idx = ((y as usize * w + x as usize) * 4) as usize;
            let rl = srgb_u8_to_linear(reference[idx]) * 0.2126
                + srgb_u8_to_linear(reference[idx + 1]) * 0.7152
                + srgb_u8_to_linear(reference[idx + 2]) * 0.0722;
            let sl = srgb_u8_to_linear(ssr[idx]) * 0.2126
                + srgb_u8_to_linear(ssr[idx + 1]) * 0.7152
                + srgb_u8_to_linear(ssr[idx + 2]) * 0.0722;
            if sl > rl + eps {
                undershoot += 1;
            }
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (undershoot as f32 / count as f32).clamp(0.0, 1.0)
    }
}

/// Convert linear RGB to CIELAB color space
pub fn rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Assume linear sRGB inputs in [0,1]
    let x = 0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b;
    let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175 * b;
    let z = 0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b;

    let xn = 0.950_47;
    let yn = 1.0;
    let zn = 1.088_83;

    let fx = lab_pivot(x / xn);
    let fy = lab_pivot(y / yn);
    let fz = lab_pivot(z / zn);

    let l = (116.0 * fy - 16.0).clamp(0.0, 100.0);
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);
    (l, a, b)
}

/// Helper function for LAB conversion
fn lab_pivot(t: f32) -> f32 {
    const EPSILON: f32 = 0.008856;
    const KAPPA: f32 = 903.3;
    if t > EPSILON {
        t.cbrt()
    } else {
        (KAPPA * t + 16.0) / 116.0
    }
}

/// Compute SSIM (Structural Similarity Index) between two luminance buffers
pub fn compute_ssim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 1.0;
    }
    let n = a.len() as f32;
    let mean_a = a.iter().copied().sum::<f32>() / n;
    let mean_b = b.iter().copied().sum::<f32>() / n;

    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    let mut cov = 0.0f32;
    for (&xa, &xb) in a.iter().zip(b.iter()) {
        var_a += (xa - mean_a).powi(2);
        var_b += (xb - mean_b).powi(2);
        cov += (xa - mean_a) * (xb - mean_b);
    }
    if n > 1.0 {
        var_a /= n - 1.0;
        var_b /= n - 1.0;
        cov /= n - 1.0;
    }

    const L: f32 = 1.0;
    const C1: f32 = 0.0001; // (0.01 * L)^2 with L=1
    const C2: f32 = 0.0009; // (0.03 * L)^2 with L=1

    let numerator = (2.0 * mean_a * mean_b + C1) * (2.0 * cov + C2);
    let denominator = (mean_a.powi(2) + mean_b.powi(2) + C1) * (var_a + var_b + C2);
    if denominator.abs() < f32::EPSILON {
        1.0
    } else {
        (numerator / denominator).clamp(-1.0, 1.0)
    }
}

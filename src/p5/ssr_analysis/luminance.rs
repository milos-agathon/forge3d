//! Color space conversion and luminance utilities for SSR analysis.

/// Epsilon constant for numerical stability.
pub const EPSILON: f32 = 1e-4;

/// Convert sRGB channel to linear space.
pub fn srgb_to_linear(channel: u8) -> f32 {
    let c = channel as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Compute Rec.709 luminance from RGBA pixel slice in linear space.
pub fn luminance(px: &[u8]) -> f32 {
    let r = srgb_to_linear(px[0]);
    let g = srgb_to_linear(px[1]);
    let b = srgb_to_linear(px[2]);
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Compute luminance directly from byte values (no gamma correction).
pub fn luminance_bytes(px: &[u8]) -> f32 {
    0.2126 * (px[0] as f32) + 0.7152 * (px[1] as f32) + 0.0722 * (px[2] as f32)
}

/// Compute luminance from an image::RgbaImage over a rectangular region.
pub fn compute_roi_luminance(img: &image::RgbaImage, x0: u32, x1: u32, y0: u32, y1: u32) -> f32 {
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

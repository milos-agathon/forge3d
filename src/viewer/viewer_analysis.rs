// src/viewer/viewer_analysis.rs
// Image analysis utilities for the interactive viewer
// RELEVANT FILES: src/viewer/mod.rs

/// Compute mean luminance over a region of an RGBA8 image
pub fn mean_luma_region(buf: &[u8], w: u32, h: u32, x0: u32, y0: u32, rw: u32, rh: u32) -> f32 {
    let (w, h) = (w as usize, h as usize);
    let (x0, y0, rw, rh) = (x0 as usize, y0 as usize, rw as usize, rh as usize);
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for y in y0..(y0 + rh).min(h) {
        for x in x0..(x0 + rw).min(w) {
            let i = (y * w + x) * 4;
            let r = buf[i] as f32;
            let g = buf[i + 1] as f32;
            let b = buf[i + 2] as f32;
            let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            sum += (l / 255.0) as f64;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

/// Compute variance of luminance over entire RGBA8 image
pub fn variance_luma(buf: &[u8], w: u32, h: u32) -> f32 {
    let (w, h) = (w as usize, h as usize);
    let n = (w * h).max(1);
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    for i in (0..(n * 4)).step_by(4) {
        let r = buf[i] as f32;
        let g = buf[i + 1] as f32;
        let b = buf[i + 2] as f32;
        let l = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
        sum += l as f64;
        sum2 += (l * l) as f64;
    }
    let mean = sum / n as f64;
    let var = (sum2 / n as f64) - mean * mean;
    var.max(0.0) as f32
}

/// Sobel-like gradient energy (used for blur effectiveness metric)
pub fn gradient_energy(buf: &[u8], w: u32, h: u32) -> f32 {
    if w < 2 || h < 2 {
        return 0.0;
    }
    let (w_usize, h_usize) = (w as usize, h as usize);
    let mut energy = 0.0f64;
    let mut samples = 0usize;
    for y in 0..(h_usize.saturating_sub(1)) {
        for x in 0..(w_usize.saturating_sub(1)) {
            let idx = (y * w_usize + x) * 4;
            let l = (0.2126 * buf[idx] as f32
                + 0.7152 * buf[idx + 1] as f32
                + 0.0722 * buf[idx + 2] as f32)
                / 255.0;
            let idx_x = (y * w_usize + (x + 1)) * 4;
            let lx = (0.2126 * buf[idx_x] as f32
                + 0.7152 * buf[idx_x + 1] as f32
                + 0.0722 * buf[idx_x + 2] as f32)
                / 255.0;
            let idx_y = ((y + 1) * w_usize + x) * 4;
            let ly = (0.2126 * buf[idx_y] as f32
                + 0.7152 * buf[idx_y + 1] as f32
                + 0.0722 * buf[idx_y + 2] as f32)
                / 255.0;
            let dx = lx - l;
            let dy = ly - l;
            energy += (dx * dx + dy * dy) as f64;
            samples += 1;
        }
    }
    if samples == 0 {
        0.0
    } else {
        (energy / samples as f64) as f32
    }
}

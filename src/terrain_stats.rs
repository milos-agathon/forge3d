// T02-BEGIN:terrain_stats
//! Minimal helpers for DEM min / max with optional percentile clamp.
//! Works on a borrowed `&[f32]` to avoid copies.
//!
//! Percentile is computed with a coarse reservoir sample when `len > 65_536`
//! to keep O(N) memory and deterministic output.

use std::cmp::Ordering;

/// Compute `(min, max)` or a clamped 1–99 percentile range (`clamp=true`).
pub fn min_max(data: &[f32], clamp: bool) -> (f32, f32) {
    assert!(!data.is_empty(), "heightmap slice empty");
    if !clamp {
        // Fast path single sweep
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in data {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        return (lo, hi);
    }
    // Clamp percentile: sample if huge, else full sort.
    const SAMPLE: usize = 65_536;
    let mut buf: Vec<f32> = if data.len() > SAMPLE {
        // Simple stride sampling for determinism
        let step = data.len() / SAMPLE;
        data.iter().step_by(step).cloned().collect()
    } else {
        data.to_vec()
    };
    buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p1 = buf[(buf.len() as f32 * 0.01) as usize];
    let p99 = buf[(buf.len() as f32 * 0.99) as usize];
    (p1, p99)
}
// T02-END:terrain_stats

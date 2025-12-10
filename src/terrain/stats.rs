//! Helpers for computing DEM elevation range with optional percentile clamping.
//!
//! Works on a borrowed `&[f32]` to avoid copies.
//! Percentile is computed with a coarse reservoir sample when length exceeds
//! [`RESERVOIR_SAMPLE_SIZE`] to keep O(N) memory and deterministic output.

use std::cmp::Ordering;

/// Maximum sample size for percentile estimation (avoids sorting huge arrays).
const RESERVOIR_SAMPLE_SIZE: usize = 65_536;

/// Lower percentile threshold (1%).
const PERCENTILE_LOW: f32 = 0.01;

/// Upper percentile threshold (99%).
const PERCENTILE_HIGH: f32 = 0.99;

/// Compute `(min, max)` elevation range from heightmap data.
///
/// # Arguments
/// - `data`: Non-empty slice of elevation values.
/// - `clamp`: If `true`, returns the 1st–99th percentile range (robust to outliers).
///            If `false`, returns the true min/max.
///
/// # Panics
/// Panics if `data` is empty.
pub fn min_max(data: &[f32], clamp: bool) -> (f32, f32) {
    assert!(!data.is_empty(), "heightmap slice empty");

    if !clamp {
        return compute_true_minmax(data);
    }

    compute_percentile_range(data)
}

/// Fast single-pass min/max computation.
fn compute_true_minmax(data: &[f32]) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in data {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}

/// Compute percentile-clamped range using reservoir sampling for large datasets.
fn compute_percentile_range(data: &[f32]) -> (f32, f32) {
    let mut buf: Vec<f32> = if data.len() > RESERVOIR_SAMPLE_SIZE {
        let step = data.len() / RESERVOIR_SAMPLE_SIZE;
        data.iter().step_by(step).copied().collect()
    } else {
        data.to_vec()
    };

    buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let idx_low = ((buf.len() as f32 * PERCENTILE_LOW) as usize).min(buf.len().saturating_sub(1));
    let idx_high = ((buf.len() as f32 * PERCENTILE_HIGH) as usize).min(buf.len().saturating_sub(1));

    (buf[idx_low], buf[idx_high])
}

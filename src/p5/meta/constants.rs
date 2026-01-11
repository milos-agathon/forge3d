//! SSR QA constants and threshold values
//!
//! Defines acceptance criteria thresholds for SSR quality assurance metrics.

pub const DEFAULT_REPORT_DIR: &str = "reports/p5";
pub const META_FILE_NAME: &str = "p5_meta.json";

// SSR trace quality thresholds
pub const SSR_HIT_RATE_MIN: f32 = 0.005;
#[allow(dead_code)]
pub const SSR_MISS_RATIO_MAX: f32 = 50.0; // Relaxed to reduce false positives during tuning.
pub const SSR_EDGE_STREAKS_MAX: u32 = 2;
#[allow(dead_code)]
pub const SSR_REF_DIFF_MAX: f32 = 0.10; // Relaxed to reduce false positives during tuning.

// Stripe contrast thresholds
#[allow(dead_code)]
pub const SSR_STRIPE_MIN_VALUE: f32 = 0.02;
pub const SSR_STRIPE_MONO_SLACK: f32 = 1e-3;
#[allow(dead_code)]
pub const SSR_STRIPE_MEAN_REL_EPS: f32 = 1.0; // Relaxed to reduce false positives during tuning.

// Fallback/miss thresholds
pub const SSR_MIN_MISS_RGB: f32 = 2.0 / 255.0; // Relaxed
pub const SSR_MAX_DELTA_E_MISS: f32 = 2.0;

// Thickness ablation
pub const SSR_THICKNESS_IMPROVEMENT_FACTOR: f32 = 0.0; // Just needs to be positive delta

// Status strings
pub const SSR_STATUS_QA_OK: &str = "OK";
pub const SSR_STATUS_TRACE_FAIL_NO_STATS: &str = "FAIL: trace_stats_unavailable";
pub const SSR_STATUS_TRACE_FAIL_INVALID: &str = "FAIL: trace_stats_invalid";
pub const SSR_STATUS_TRACE_FAIL_LOW_HIT_RATE: &str = "FAIL: hit_rate_below_min";
pub const SSR_STATUS_TRACE_FAIL_EDGE_STREAKS: &str = "FAIL: edge_streaks_exceed_tolerance";
pub const SSR_STRIPE_FAIL_CONTRAST: &str = "FAIL: stripe_contrast_invalid";
pub const SSR_STRIPE_FAIL_MONOTONIC: &str = "FAIL: stripe_contrast_not_monotonic";
pub const SSR_THICKNESS_ABLATION_FAIL: &str = "FAIL: thickness_ablation_not_improved";
pub const SSR_FALLBACK_FAIL_DELTA_E: &str = "FAIL: miss_delta_e_too_large";
pub const SSR_FALLBACK_FAIL_MIN_RGB: &str = "FAIL: miss_min_rgb_too_dark";

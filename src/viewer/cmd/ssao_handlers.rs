// src/viewer/cmd/ssao_handlers.rs
// SSAO command handlers for the interactive viewer

use crate::core::screen_space_effects::ScreenSpaceEffectsManager;
use wgpu::Queue;

/// Set SSAO samples
pub fn set_ssao_samples(gi: &mut Option<ScreenSpaceEffectsManager>, queue: &Queue, n: u32) {
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.update_ssao_settings(queue, |s| {
            s.num_samples = n.max(1);
        });
    }
}

/// Set SSAO radius
pub fn set_ssao_radius(gi: &mut Option<ScreenSpaceEffectsManager>, queue: &Queue, r: f32) {
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.update_ssao_settings(queue, |s| {
            s.radius = r.max(0.0);
        });
    }
}

/// Set SSAO intensity (updates both settings and composite multiplier)
pub fn set_ssao_intensity(
    gi: &mut Option<ScreenSpaceEffectsManager>,
    queue: &Queue,
    ssao_composite_mul: &mut f32,
    v: f32,
) {
    *ssao_composite_mul = v.max(0.0);
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.update_ssao_settings(queue, |s| {
            s.intensity = v.max(0.0);
        });
        gi_mgr.set_ssao_composite_multiplier(queue, v);
    }
}

/// Set SSAO bias
pub fn set_ssao_bias(gi: &mut Option<ScreenSpaceEffectsManager>, queue: &Queue, b: f32) {
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.set_ssao_bias(queue, b);
    }
}

/// Set SSAO directions (maps to num_samples = dirs * 4)
pub fn set_ssao_directions(gi: &mut Option<ScreenSpaceEffectsManager>, queue: &Queue, dirs: u32) {
    if let Some(ref mut gi_mgr) = gi {
        let ns = dirs.saturating_mul(4).max(8);
        gi_mgr.update_ssao_settings(queue, |s| {
            s.num_samples = ns;
        });
    }
}

/// Set SSAO temporal alpha
pub fn set_ssao_temporal_alpha(gi: &mut Option<ScreenSpaceEffectsManager>, queue: &Queue, a: f32) {
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.set_ssao_temporal_alpha(queue, a);
    }
}

/// Set SSAO temporal enabled
pub fn set_ssao_temporal_enabled(gi: &mut Option<ScreenSpaceEffectsManager>, on: bool) {
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.set_ssao_temporal(on);
    }
}

/// Set SSAO technique (0 = SSAO, 1 = GTAO)
pub fn set_ssao_technique(gi: &mut Option<ScreenSpaceEffectsManager>, queue: &Queue, tech: u32) {
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.update_ssao_settings(queue, |s| {
            s.technique = if tech != 0 { 1 } else { 0 };
        });
    }
}

/// Set SSAO blur
pub fn set_ssao_blur(
    gi: &mut Option<ScreenSpaceEffectsManager>,
    ssao_blur_enabled: &mut bool,
    on: bool,
) {
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.set_ssao_blur(on);
    }
    *ssao_blur_enabled = on;
}

/// Set SSAO composite multiplier
pub fn set_ssao_composite_mul(
    gi: &mut Option<ScreenSpaceEffectsManager>,
    queue: &Queue,
    ssao_composite_mul: &mut f32,
    v: f32,
) {
    *ssao_composite_mul = v.max(0.0);
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.set_ssao_composite_multiplier(queue, v);
    }
}

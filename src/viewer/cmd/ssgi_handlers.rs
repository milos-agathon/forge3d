// src/viewer/cmd/ssgi_handlers.rs
// SSGI command handlers for the interactive viewer

use crate::core::screen_space_effects::ScreenSpaceEffectsManager;
use std::sync::Arc;
use wgpu::{Device, Queue};

/// Set SSGI steps
pub fn set_ssgi_steps(gi: &mut Option<ScreenSpaceEffectsManager>, queue: &Queue, n: u32) {
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.update_ssgi_settings(queue, |s| {
            s.num_steps = n.max(0);
        });
    }
}

/// Set SSGI radius
pub fn set_ssgi_radius(gi: &mut Option<ScreenSpaceEffectsManager>, queue: &Queue, r: f32) {
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.update_ssgi_settings(queue, |s| {
            s.radius = r.max(0.0);
        });
    }
}

/// Set SSGI half resolution
pub fn set_ssgi_half(
    gi: &mut Option<ScreenSpaceEffectsManager>,
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    on: bool,
) {
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.set_ssgi_half_res_with_queue(device, queue, on);
    }
}

/// Set SSGI temporal alpha
pub fn set_ssgi_temporal_alpha(gi: &mut Option<ScreenSpaceEffectsManager>, queue: &Queue, a: f32) {
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.update_ssgi_settings(queue, |s| {
            s.temporal_alpha = a.clamp(0.0, 1.0);
        });
    }
}

/// Set SSGI temporal enabled
pub fn set_ssgi_temporal_enabled(
    gi: &mut Option<ScreenSpaceEffectsManager>,
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    on: bool,
) {
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.update_ssgi_settings(queue, |s| {
            s.temporal_enabled = if on { 1 } else { 0 };
        });
        let _ = gi_mgr.ssgi_reset_history(device, queue);
    }
}

/// Set SSGI edge-aware filtering
pub fn set_ssgi_edges(gi: &mut Option<ScreenSpaceEffectsManager>, queue: &Queue, on: bool) {
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.update_ssgi_settings(queue, |s| {
            s.use_edge_aware = if on { 1 } else { 0 };
        });
    }
}

/// Set SSGI upsample sigma depth
pub fn set_ssgi_upsample_sigma_depth(
    gi: &mut Option<ScreenSpaceEffectsManager>,
    queue: &Queue,
    sig: f32,
) {
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.update_ssgi_settings(queue, |s| {
            s.upsample_depth_sigma = sig.max(1e-4);
        });
    }
}

/// Set SSGI upsample sigma normal
pub fn set_ssgi_upsample_sigma_normal(
    gi: &mut Option<ScreenSpaceEffectsManager>,
    queue: &Queue,
    sig: f32,
) {
    if let Some(ref mut gi_mgr) = gi {
        gi_mgr.update_ssgi_settings(queue, |s| {
            s.upsample_normal_sigma = sig.max(1e-4);
        });
    }
}

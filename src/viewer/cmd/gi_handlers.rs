// src/viewer/cmd/gi_handlers.rs
// GI/SSAO/SSR/SSGI command handlers for the interactive viewer

use crate::core::screen_space_effects::ScreenSpaceEffectsManager;
use crate::render::params::SsrParams;
use wgpu::{Device, Queue};

/// Print GI status to stdout
pub fn handle_gi_status(
    gi: &Option<ScreenSpaceEffectsManager>,
    ssr_params: &SsrParams,
    gi_ao_weight: f32,
    gi_ssgi_weight: f32,
    gi_ssr_weight: f32,
) {
    use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;
    if let Some(ref gi_mgr) = gi {
        let ssao_on = gi_mgr.is_enabled(SSE::SSAO);
        let ssgi_on = gi_mgr.is_enabled(SSE::SSGI);
        let ssr_on = gi_mgr.is_enabled(SSE::SSR) && ssr_params.ssr_enable;

        let ssao = gi_mgr.ssao_settings();
        println!(
            "GI: ssao={} radius={:.6} intensity={:.6}",
            if ssao_on { "on" } else { "off" },
            ssao.radius,
            ssao.intensity
        );

        if let Some(ssgi) = gi_mgr.ssgi_settings() {
            println!(
                "GI: ssgi={} steps={} radius={:.6}",
                if ssgi_on { "on" } else { "off" },
                ssgi.num_steps,
                ssgi.radius
            );
        } else {
            println!("GI: ssgi=<unavailable>");
        }

        println!(
            "GI: ssr={} max_steps={} thickness={:.6}",
            if ssr_on { "on" } else { "off" },
            ssr_params.ssr_max_steps,
            ssr_params.ssr_thickness
        );

        println!(
            "GI: weights ao={:.6} ssgi={:.6} ssr={:.6}",
            gi_ao_weight, gi_ssgi_weight, gi_ssr_weight
        );
    } else {
        println!("GI: <unavailable: GI manager not initialized>");
    }
}

/// Set GI random seed
pub fn handle_set_gi_seed(
    gi: &mut Option<ScreenSpaceEffectsManager>,
    device: &Device,
    queue: &Queue,
    seed: u32,
) -> Option<u32> {
    if let Some(ref mut gi_mgr) = gi {
        if let Err(e) = gi_mgr.set_gi_seed(device, queue, seed) {
            eprintln!("Failed to set GI seed {}: {}", seed, e);
            None
        } else {
            println!("GI seed set to {}", seed);
            Some(seed)
        }
    } else {
        eprintln!("GI manager not available");
        None
    }
}

/// Query SSAO radius
pub fn query_ssao_radius(gi: &Option<ScreenSpaceEffectsManager>) {
    if let Some(ref gi_mgr) = gi {
        let s = gi_mgr.ssao_settings();
        println!("ssao-radius = {:.6}", s.radius);
    } else {
        println!("ssao-radius = <unavailable: GI manager not initialized>");
    }
}

/// Query SSAO intensity
pub fn query_ssao_intensity(gi: &Option<ScreenSpaceEffectsManager>) {
    if let Some(ref gi_mgr) = gi {
        let s = gi_mgr.ssao_settings();
        println!("ssao-intensity = {:.6}", s.intensity);
    } else {
        println!("ssao-intensity = <unavailable: GI manager not initialized>");
    }
}

/// Query SSAO bias
pub fn query_ssao_bias(gi: &Option<ScreenSpaceEffectsManager>) {
    if let Some(ref gi_mgr) = gi {
        let s = gi_mgr.ssao_settings();
        println!("ssao-bias = {:.6}", s.bias);
    } else {
        println!("ssao-bias = <unavailable: GI manager not initialized>");
    }
}

/// Query SSAO samples
pub fn query_ssao_samples(gi: &Option<ScreenSpaceEffectsManager>) {
    if let Some(ref gi_mgr) = gi {
        let s = gi_mgr.ssao_settings();
        println!("ssao-samples = {}", s.num_samples);
    } else {
        println!("ssao-samples = <unavailable: GI manager not initialized>");
    }
}

/// Query SSAO directions
pub fn query_ssao_directions(gi: &Option<ScreenSpaceEffectsManager>) {
    if let Some(ref gi_mgr) = gi {
        let s = gi_mgr.ssao_settings();
        let dirs = (s.num_samples / 4).max(1);
        println!("ssao-directions = {}", dirs);
    } else {
        println!("ssao-directions = <unavailable: GI manager not initialized>");
    }
}

/// Query SSAO temporal alpha
pub fn query_ssao_temporal_alpha(gi: &Option<ScreenSpaceEffectsManager>) {
    if let Some(ref gi_mgr) = gi {
        let a = gi_mgr.ssao_temporal_alpha();
        println!("ssao-temporal-alpha = {:.6}", a);
    } else {
        println!("ssao-temporal-alpha = <unavailable: GI manager not initialized>");
    }
}

/// Query SSAO temporal enabled
pub fn query_ssao_temporal_enabled(gi: &Option<ScreenSpaceEffectsManager>) {
    if let Some(ref gi_mgr) = gi {
        let on = gi_mgr.ssao_temporal_enabled();
        println!("ssao-temporal = {}", if on { "on" } else { "off" });
    } else {
        println!("ssao-temporal = <unavailable: GI manager not initialized>");
    }
}

/// Query SSAO technique
pub fn query_ssao_technique(gi: &Option<ScreenSpaceEffectsManager>) {
    if let Some(ref gi_mgr) = gi {
        let s = gi_mgr.ssao_settings();
        let name = if s.technique != 0 { "gtao" } else { "ssao" };
        println!("ssao-technique = {}", name);
    } else {
        println!("ssao-technique = <unavailable: GI manager not initialized>");
    }
}

/// Query SSGI steps
pub fn query_ssgi_steps(gi: &Option<ScreenSpaceEffectsManager>) {
    if let Some(ref gi_mgr) = gi {
        if let Some(s) = gi_mgr.ssgi_settings() {
            println!("ssgi-steps = {}", s.num_steps);
        } else {
            println!("ssgi-steps = <unavailable>");
        }
    } else {
        println!("ssgi-steps = <unavailable: GI manager not initialized>");
    }
}

/// Query SSGI radius
pub fn query_ssgi_radius(gi: &Option<ScreenSpaceEffectsManager>) {
    if let Some(ref gi_mgr) = gi {
        if let Some(s) = gi_mgr.ssgi_settings() {
            println!("ssgi-radius = {:.6}", s.radius);
        } else {
            println!("ssgi-radius = <unavailable>");
        }
    } else {
        println!("ssgi-radius = <unavailable: GI manager not initialized>");
    }
}

/// Query SSGI half-res
pub fn query_ssgi_half(gi: &Option<ScreenSpaceEffectsManager>) {
    if let Some(ref gi_mgr) = gi {
        if let Some(on) = gi_mgr.ssgi_half_res() {
            println!("ssgi-half = {}", if on { "on" } else { "off" });
        } else {
            println!("ssgi-half = <unavailable>");
        }
    } else {
        println!("ssgi-half = <unavailable: GI manager not initialized>");
    }
}

/// Query SSGI temporal alpha
pub fn query_ssgi_temporal_alpha(gi: &Option<ScreenSpaceEffectsManager>) {
    if let Some(ref gi_mgr) = gi {
        if let Some(s) = gi_mgr.ssgi_settings() {
            println!("ssgi-temporal-alpha = {:.6}", s.temporal_alpha);
        } else {
            println!("ssgi-temporal-alpha = <unavailable>");
        }
    } else {
        println!("ssgi-temporal-alpha = <unavailable: GI manager not initialized>");
    }
}

/// Query SSGI temporal enabled
pub fn query_ssgi_temporal_enabled(gi: &Option<ScreenSpaceEffectsManager>) {
    if let Some(ref gi_mgr) = gi {
        if let Some(s) = gi_mgr.ssgi_settings() {
            println!(
                "ssgi-temporal = {}",
                if s.temporal_enabled != 0 { "on" } else { "off" }
            );
        } else {
            println!("ssgi-temporal = <unavailable>");
        }
    } else {
        println!("ssgi-temporal = <unavailable: GI manager not initialized>");
    }
}

/// Query SSGI edges
pub fn query_ssgi_edges(gi: &Option<ScreenSpaceEffectsManager>) {
    if let Some(ref gi_mgr) = gi {
        if let Some(s) = gi_mgr.ssgi_settings() {
            println!(
                "ssgi-edges = {}",
                if s.use_edge_aware != 0 { "on" } else { "off" }
            );
        } else {
            println!("ssgi-edges = <unavailable>");
        }
    } else {
        println!("ssgi-edges = <unavailable: GI manager not initialized>");
    }
}

/// Query SSGI upsample sigma depth
pub fn query_ssgi_upsample_sigma_depth(gi: &Option<ScreenSpaceEffectsManager>) {
    if let Some(ref gi_mgr) = gi {
        if let Some(s) = gi_mgr.ssgi_settings() {
            println!("ssgi-upsample-sigma-depth = {:.6}", s.upsample_depth_sigma);
        } else {
            println!("ssgi-upsample-sigma-depth = <unavailable>");
        }
    } else {
        println!("ssgi-upsample-sigma-depth = <unavailable: GI manager not initialized>");
    }
}

/// Query SSGI upsample sigma normal
pub fn query_ssgi_upsample_sigma_normal(gi: &Option<ScreenSpaceEffectsManager>) {
    if let Some(ref gi_mgr) = gi {
        if let Some(s) = gi_mgr.ssgi_settings() {
            println!(
                "ssgi-upsample-sigma-normal = {:.6}",
                s.upsample_normal_sigma
            );
        } else {
            println!("ssgi-upsample-sigma-normal = <unavailable>");
        }
    } else {
        println!("ssgi-upsample-sigma-normal = <unavailable: GI manager not initialized>");
    }
}

/// Query SSR enable state
pub fn query_ssr_enable(ssr_params: &SsrParams) {
    println!(
        "ssr-enable = {}",
        if ssr_params.ssr_enable { "on" } else { "off" }
    );
}

/// Query SSR max steps
pub fn query_ssr_max_steps(ssr_params: &SsrParams) {
    println!("ssr-max-steps = {}", ssr_params.ssr_max_steps);
}

/// Query SSR thickness
pub fn query_ssr_thickness(ssr_params: &SsrParams) {
    println!("ssr-thickness = {:.6}", ssr_params.ssr_thickness);
}

/// Query SSAO blur state
pub fn query_ssao_blur(gi: &Option<ScreenSpaceEffectsManager>, blur_enabled: bool) {
    if gi.is_some() {
        println!("ssao-blur = {}", if blur_enabled { "on" } else { "off" });
    } else {
        println!("ssao-blur = <unavailable: GI manager not initialized>");
    }
}

/// Query SSAO composite state
pub fn query_ssao_composite(gi: &Option<ScreenSpaceEffectsManager>, use_composite: bool) {
    if gi.is_some() {
        println!(
            "ssao-composite = {}",
            if use_composite { "on" } else { "off" }
        );
    } else {
        println!("ssao-composite = <unavailable: GI manager not initialized>");
    }
}

/// Query SSAO composite multiplier
pub fn query_ssao_mul(gi: &Option<ScreenSpaceEffectsManager>, mul: f32) {
    if gi.is_some() {
        println!("ssao-mul = {:.6}", mul);
    } else {
        println!("ssao-mul = <unavailable: GI manager not initialized>");
    }
}

/// Query GI AO weight
pub fn query_gi_ao_weight(weight: f32) {
    println!("ao-weight = {:.6}", weight);
}

/// Query GI SSGI weight
pub fn query_gi_ssgi_weight(weight: f32) {
    println!("ssgi-weight = {:.6}", weight);
}

/// Query GI SSR weight
pub fn query_gi_ssr_weight(weight: f32) {
    println!("ssr-weight = {:.6}", weight);
}

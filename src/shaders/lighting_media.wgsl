// src/shaders/lighting_media.wgsl
// WGSL helpers for Henyey–Greenstein phase and simple transmittance
// Minimal library to support A11 participating media deliverables
// RELEVANT FILES:python/forge3d/lighting.py,tests/test_media_hg.py,tests/test_media_fog.py

let PI : f32 = 3.14159265358979323846;

fn hg_phase(cos_theta: f32, g_in: f32) -> f32 {
    var g = clamp(g_in, -0.999, 0.999);
    let one_minus_g2 = 1.0 - g * g;
    let denom = max(1e-8, 1.0 + g * g - 2.0 * g * cos_theta);
    return one_minus_g2 / (4.0 * PI * pow(denom, 1.5));
}

fn transmittance(depth: f32, sigma_t: f32) -> f32 {
    return exp(-sigma_t * max(depth, 0.0));
}

fn fog_factor(depth: f32, sigma_t: f32) -> f32 {
    return clamp(1.0 - transmittance(depth, sigma_t), 0.0, 1.0);
}


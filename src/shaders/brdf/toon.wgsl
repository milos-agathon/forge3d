// src/shaders/brdf/toon.wgsl
// Stylised toon shading BRDF with hard thresholds
// Exists to provide a cel-shaded option within the unified shading dispatch
// RELEVANT FILES: src/shaders/brdf/dispatch.wgsl, src/shaders/brdf/common.wgsl, src/shaders/lighting.wgsl, src/lighting/types.rs
//
// P2-04: Non-PBR ornamental model with safe defaults
// Usage:
//   - Creates cel-shaded/cartoon appearance with hard light/shadow transition
//   - params.roughness controls threshold: low roughness = harsher threshold (0.9), high roughness = softer threshold (0.4)
//   - params.sheen controls rim light intensity: 0.0 = no rim, 1.0 = full base_color rim
// Safety:
//   - All dot products clamped with saturate() to [0, 1]
//   - Threshold computed via mix() guarantees [0.4, 0.9] range
//   - Returns base_color in lit regions, rim light in shadow (safe for all inputs)
// Not physically-based: Use for artistic/stylized rendering only

fn brdf_toon(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, base_color: vec3<f32>, params: ShadingParamsGPU) -> vec3<f32> {
    let n_dot_l = saturate(dot(normal, light));
    let threshold = mix(0.4, 0.9, 1.0 - params.roughness);
    if (n_dot_l > threshold) {
        return base_color;
    }
    let rim = pow(1.0 - saturate(dot(view, normal)), 2.0);
    let rim_color = mix(vec3<f32>(0.0), base_color, params.sheen);
    return rim_color * rim;
}

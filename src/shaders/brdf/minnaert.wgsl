// src/shaders/brdf/minnaert.wgsl
// Minnaert diffuse BRDF for dark-backscattering surfaces
// Exists to approximate dark velvet-like response controlled by subsurface parameter
// RELEVANT FILES: src/shaders/brdf/dispatch.wgsl, src/shaders/brdf/common.wgsl, src/shaders/lighting.wgsl, src/lighting/types.rs
//
// P2-04: Non-PBR ornamental model with safe defaults
// Usage:
//   - Models dark velvet-like materials with limb darkening (darker at edges)
//   - params.subsurface controls darkening intensity: 0.0 = standard diffuse, 1.0 = maximum darkening (k=2.0)
//   - Formula: (n·l * n·v)^(k/2) where k = mix(0.0, 2.0, subsurface)
// Safety:
//   - Returns vec3(0.0) if n·l or n·v <= 0 (backfacing or grazing angles)
//   - All dot products clamped with saturate() to [0, 1]
//   - k parameter clamped via mix() to [0.0, 2.0] range (prevents negative exponents)
//   - Normalized with INV_PI for energy conservation
// Not physically-based: Empirical model for artistic effects

fn brdf_minnaert(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, base_color: vec3<f32>, params: ShadingParamsGPU) -> vec3<f32> {
    let n_dot_l = saturate(dot(normal, light));
    let n_dot_v = saturate(dot(normal, view));
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3<f32>(0.0);
    }
    let k = mix(0.0, 2.0, saturate(params.subsurface));
    let minnaert = pow(n_dot_l * n_dot_v, k * 0.5);
    return base_color * minnaert * INV_PI;
}

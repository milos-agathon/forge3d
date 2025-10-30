// src/shaders/brdf/minnaert.wgsl
// Minnaert diffuse BRDF for dark-backscattering surfaces
// Exists to approximate dark velvet-like response controlled by subsurface parameter
// RELEVANT FILES: src/shaders/brdf/dispatch.wgsl, src/shaders/brdf/common.wgsl, src/shaders/lighting.wgsl, src/lighting/types.rs

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

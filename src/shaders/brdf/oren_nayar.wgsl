// src/shaders/brdf/oren_nayar.wgsl
// Oren–Nayar rough diffuse BRDF implementation
// Exists to model diffuse retro-reflection for rough surfaces
// RELEVANT FILES: src/shaders/brdf/dispatch.wgsl, src/shaders/brdf/common.wgsl, src/shaders/lighting.wgsl, src/lighting/types.rs

fn brdf_oren_nayar(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, base_color: vec3<f32>, params: ShadingParamsGPU) -> vec3<f32> {
    let rough = max(params.roughness, 0.001);
    let sigma2 = rough * rough;
    let a = 1.0 - (sigma2 / (2.0 * (sigma2 + 0.33)));
    let b = 0.45 * sigma2 / (sigma2 + 0.09);

    let n_dot_l = saturate(dot(normal, light));
    let n_dot_v = saturate(dot(normal, view));

    let angle_v = acos(n_dot_v);
    let angle_l = acos(n_dot_l);
    let alpha = max(angle_v, angle_l);
    let beta = min(angle_v, angle_l);

    let tangent = normalize(view - normal * n_dot_v);
    let bitangent = normalize(light - normal * n_dot_l);
    let cos_phi = dot(tangent, bitangent);

    let oren = a + b * max(0.0, cos_phi) * sin(alpha) * tan(beta);
    return base_color * oren * INV_PI * n_dot_l;
}

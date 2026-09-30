// src/shaders/brdf/cook_torrance.wgsl
// Microfacet Cook–Torrance BRDF variants (GGX and Beckmann)
// Exists to provide physically-based specular responses for PBR workflows
// RELEVANT FILES: src/shaders/brdf/dispatch.wgsl, src/shaders/brdf/common.wgsl, src/shaders/lighting.wgsl, src/lighting/types.rs

fn brdf_cook_torrance_ggx(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, base_color: vec3<f32>, params: ShadingParamsGPU) -> vec3<f32> {
    let half_vec = det_normalize3(view + light);
    let f0 = det_barrier3(det_mix3(vec3<f32>(0.04), base_color, params.metallic));
    let fresnel = det_barrier3(fresnel_schlick(max(det_dot3(half_vec, view), 0.0), f0));
    let distribution = distribution_ggx(normal, half_vec, params.roughness);
    let geometry = det_barrier(geometry_smith_ggx(normal, view, light, params.roughness));
    let numerator = det_barrier(distribution * geometry) * fresnel;
    let denom = max(det_barrier(4.0 * saturate(det_dot3(normal, view))) * saturate(det_dot3(normal, light)), 1e-4);
    let specular = det_div3(numerator, vec3<f32>(denom));
    let kd = det_barrier3((vec3<f32>(1.0) - fresnel) * (1.0 - params.metallic));
    let diffuse = det_barrier3(kd * det_barrier3(brdf_lambert(base_color)));
    return diffuse + specular;
}

fn brdf_cook_torrance_beckmann(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, base_color: vec3<f32>, params: ShadingParamsGPU) -> vec3<f32> {
    let half_vec = det_normalize3(view + light);
    let f0 = det_barrier3(det_mix3(vec3<f32>(0.04), base_color, params.metallic));
    let fresnel = det_barrier3(fresnel_schlick(max(det_dot3(half_vec, view), 0.0), f0));
    let distribution = distribution_beckmann(normal, half_vec, params.roughness);
    let geometry = geometry_beckmann(normal, view, light, params.roughness);
    let numerator = det_barrier(distribution * geometry) * fresnel;
    let denom = max(det_barrier(4.0 * saturate(det_dot3(normal, view))) * saturate(det_dot3(normal, light)), 1e-4);
    let specular = det_div3(numerator, vec3<f32>(denom));
    let kd = det_barrier3((vec3<f32>(1.0) - fresnel) * (1.0 - params.metallic));
    let diffuse = det_barrier3(kd * det_barrier3(brdf_lambert(base_color)));
    return diffuse + specular;
}

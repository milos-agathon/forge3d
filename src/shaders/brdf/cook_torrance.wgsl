// src/shaders/brdf/cook_torrance.wgsl
// Microfacet Cook–Torrance BRDF variants (GGX and Beckmann)
// Exists to provide physically-based specular responses for PBR workflows
// RELEVANT FILES: src/shaders/brdf/dispatch.wgsl, src/shaders/brdf/common.wgsl, src/shaders/lighting.wgsl, src/lighting/types.rs

fn brdf_cook_torrance_ggx(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, base_color: vec3<f32>, params: ShadingParamsGPU) -> vec3<f32> {
    let half_vec = normalize(view + light);
    let f0 = mix(vec3<f32>(0.04), base_color, params.metallic);
    let fresnel = fresnel_schlick(max(dot(half_vec, view), 0.0), f0);
    let distribution = distribution_ggx(normal, half_vec, params.roughness);
    let geometry = geometry_smith_ggx(normal, view, light, params.roughness);
    let numerator = distribution * geometry * fresnel;
    let denom = max(4.0 * saturate(dot(normal, view)) * saturate(dot(normal, light)), 1e-4);
    let specular = numerator / denom;
    let kd = (vec3<f32>(1.0) - fresnel) * (1.0 - params.metallic);
    let diffuse = kd * brdf_lambert(base_color);
    return diffuse + specular;
}

fn brdf_cook_torrance_beckmann(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, base_color: vec3<f32>, params: ShadingParamsGPU) -> vec3<f32> {
    let half_vec = normalize(view + light);
    let f0 = mix(vec3<f32>(0.04), base_color, params.metallic);
    let fresnel = fresnel_schlick(max(dot(half_vec, view), 0.0), f0);
    let distribution = distribution_beckmann(normal, half_vec, params.roughness);
    let geometry = geometry_beckmann(normal, view, light, params.roughness);
    let numerator = distribution * geometry * fresnel;
    let denom = max(4.0 * saturate(dot(normal, view)) * saturate(dot(normal, light)), 1e-4);
    let specular = numerator / denom;
    let kd = (vec3<f32>(1.0) - fresnel) * (1.0 - params.metallic);
    let diffuse = kd * brdf_lambert(base_color);
    return diffuse + specular;
}

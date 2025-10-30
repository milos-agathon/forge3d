// src/shaders/brdf/lambert.wgsl
// Classic Lambertian diffuse BRDF implementation
// Exists to provide the baseline diffuse response for all materials
// RELEVANT FILES: src/shaders/brdf/dispatch.wgsl, src/shaders/brdf/common.wgsl, src/shaders/lighting.wgsl, src/lighting/types.rs

fn brdf_lambert(base_color: vec3<f32>) -> vec3<f32> {
    return base_color * INV_PI;
}

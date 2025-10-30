// src/shaders/brdf/phong.wgsl
// Blinn-Phong style specular BRDF with roughness to shininess conversion
// Exists to emulate classic Phong highlights for compatibility presets
// RELEVANT FILES: src/shaders/brdf/dispatch.wgsl, src/shaders/brdf/common.wgsl, src/shaders/lighting.wgsl, src/lighting/types.rs

fn brdf_phong(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, base_color: vec3<f32>, params: ShadingParamsGPU) -> vec3<f32> {
    let shininess = to_shininess(params.roughness);
    let reflect_dir = reflect(-light, normal);
    let spec_angle = saturate(dot(reflect_dir, view));
    let spec = pow(spec_angle, shininess);
    let diffuse = brdf_lambert(base_color);
    let spec_color = mix(vec3<f32>(0.04), base_color, params.metallic);
    return diffuse + spec_color * spec * (shininess + 2.0) * 0.125 * INV_PI;
}

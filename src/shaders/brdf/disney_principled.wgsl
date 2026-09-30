// src/shaders/brdf/disney_principled.wgsl
// Disney "Principled" BRDF approximation (subset)
// Exists to provide a flexible default shading model with metallic/roughness workflow
// RELEVANT FILES: src/shaders/brdf/dispatch.wgsl, src/shaders/brdf/common.wgsl, src/shaders/lighting.wgsl, src/lighting/types.rs

fn brdf_disney_principled(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, base_color: vec3<f32>, params: ShadingParamsGPU) -> vec3<f32> {
    let half_vec = det_normalize3(view + light);
    let n_dot_l = saturate(det_dot3(normal, light));
    let n_dot_v = saturate(det_dot3(normal, view));
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3<f32>(0.0);
    }

    let metallic = saturate(params.metallic);
    let rough = saturate(params.roughness);
    let f0 = det_mix3(vec3<f32>(0.04), base_color, metallic);

    let specular = brdf_cook_torrance_ggx(normal, view, light, base_color, params);
    let diffuse = det_barrier3(base_color * INV_PI);

    let energy_comp = 1.0 + det_barrier(det_barrier(metallic * rough) * 0.5);
    let sheen_color = det_mix3(vec3<f32>(0.0), base_color, params.sheen);
    let sheen = det_barrier3(sheen_color * det_pow(1.0 - det_barrier(det_dot3(light, view)), 5.0));

    let clearcoat_spec = det_div(det_barrier(det_barrier(params.clearcoat * fresnel_schlick_scalar(det_dot3(half_vec, view), 0.04)) * distribution_ggx(normal, half_vec, max(rough * 0.5, 0.1))) * det_barrier(geometry_smith_ggx(normal, view, light, max(rough * 0.5, 0.1))), det_barrier(det_barrier(4.0 * n_dot_l) * n_dot_v) + 1e-4);

    return det_barrier3(det_barrier3(det_barrier3(det_barrier3(diffuse * (1.0 - metallic)) * n_dot_l) + det_barrier3(specular * energy_comp)) + det_barrier3(sheen * n_dot_l)) + vec3<f32>(clearcoat_spec);
}

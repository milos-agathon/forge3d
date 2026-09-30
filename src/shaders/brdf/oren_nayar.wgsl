// src/shaders/brdf/oren_nayar.wgsl
// Oren–Nayar rough diffuse BRDF implementation
// Exists to model diffuse retro-reflection for rough surfaces
// RELEVANT FILES: src/shaders/brdf/dispatch.wgsl, src/shaders/brdf/common.wgsl, src/shaders/lighting.wgsl, src/lighting/types.rs

fn brdf_oren_nayar(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, base_color: vec3<f32>, params: ShadingParamsGPU) -> vec3<f32> {
    let rough = max(params.roughness, 0.001);
    let sigma2 = det_barrier(rough * rough);
    let a = det_barrier(1.0 - (det_div(sigma2, 2.0 * (sigma2 + 0.33))));
    let b = det_div(0.45 * sigma2, sigma2 + 0.09);

    let n_dot_l = saturate(det_dot3(normal, light));
    let n_dot_v = saturate(det_dot3(normal, view));

    let angle_v = det_acos(n_dot_v);
    let angle_l = det_acos(n_dot_l);
    let alpha = max(angle_v, angle_l);
    let beta = min(angle_v, angle_l);

    let tangent = det_normalize3(view - det_barrier3(normal * n_dot_v));
    let bitangent = det_normalize3(light - det_barrier3(normal * n_dot_l));
    let cos_phi = det_dot3(tangent, bitangent);

    let oren = a + det_barrier(det_barrier(det_barrier(b * max(0.0, cos_phi)) * det_sin(alpha)) * det_tan(beta));
    return det_barrier3(det_barrier3(base_color * oren) * INV_PI) * n_dot_l;
}

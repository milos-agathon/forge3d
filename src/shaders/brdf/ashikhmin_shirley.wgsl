// src/shaders/brdf/ashikhmin_shirley.wgsl
// Ashikhmin–Shirley anisotropic BRDF approximation
// Exists to approximate glossy highlights with anisotropy control
// RELEVANT FILES: src/shaders/brdf/dispatch.wgsl, src/shaders/brdf/common.wgsl, src/shaders/lighting.wgsl, src/lighting/types.rs

fn brdf_ashikhmin_shirley(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, base_color: vec3<f32>, params: ShadingParamsGPU) -> vec3<f32> {
    let basis = build_orthonormal_basis(normal);
    let tangent = basis[0];
    let bitangent = basis[1];
    let half_vec = det_normalize3(view + light);

    let n_dot_l = saturate(det_dot3(normal, light));
    let n_dot_v = saturate(det_dot3(normal, view));
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3<f32>(0.0);
    }

    let h_dot_n = saturate(det_dot3(half_vec, normal));
    let h_dot_t = det_dot3(half_vec, tangent);
    let h_dot_b = det_dot3(half_vec, bitangent);

    let rough = max(params.roughness, 0.05);
    let anisotropy = params.anisotropy;
    let nu = max(0.1, to_shininess(rough) * (1.0 + anisotropy));
    let nv = max(0.1, to_shininess(rough) * (1.0 - anisotropy));

    let spec = det_barrier((det_div(det_sqrt((nu + 1.0) * (nv + 1.0)), 8.0 * PI)) *
        det_pow(h_dot_n, det_div((det_barrier(det_barrier(nu * h_dot_t) * h_dot_t)) + (det_barrier(det_barrier(nv * h_dot_b) * h_dot_b)), max(1.0 - det_barrier(h_dot_n * h_dot_n), 1e-4))));

    let fresnel = det_barrier3(fresnel_schlick(saturate(det_dot3(half_vec, view)), det_barrier3(det_mix3(vec3<f32>(0.04), base_color, params.metallic))));
    let diffuse = det_barrier3(det_barrier3(det_barrier3(base_color * (1.0 - fresnel)) * INV_PI) * n_dot_l);

    return diffuse + det_div3(det_barrier3(fresnel * spec) * n_dot_l, vec3<f32>(det_barrier(n_dot_v + n_dot_l) + 1e-4));
}

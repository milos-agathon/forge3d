// src/shaders/brdf/ashikhmin_shirley.wgsl
// Ashikhmin–Shirley anisotropic BRDF approximation
// Exists to approximate glossy highlights with anisotropy control
// RELEVANT FILES: src/shaders/brdf/dispatch.wgsl, src/shaders/brdf/common.wgsl, src/shaders/lighting.wgsl, src/lighting/types.rs

fn brdf_ashikhmin_shirley(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, base_color: vec3<f32>, params: ShadingParamsGPU) -> vec3<f32> {
    let basis = build_orthonormal_basis(normal);
    let tangent = basis[0];
    let bitangent = basis[1];
    let half_vec = normalize(view + light);

    let n_dot_l = saturate(dot(normal, light));
    let n_dot_v = saturate(dot(normal, view));
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3<f32>(0.0);
    }

    let h_dot_n = saturate(dot(half_vec, normal));
    let h_dot_t = dot(half_vec, tangent);
    let h_dot_b = dot(half_vec, bitangent);

    let rough = max(params.roughness, 0.05);
    let anisotropy = params.anisotropy;
    let nu = max(0.1, to_shininess(rough) * (1.0 + anisotropy));
    let nv = max(0.1, to_shininess(rough) * (1.0 - anisotropy));

    let spec = ((sqrt((nu + 1.0) * (nv + 1.0))) / (8.0 * PI)) *
        pow(h_dot_n, ((nu * h_dot_t * h_dot_t) + (nv * h_dot_b * h_dot_b)) / max(1.0 - h_dot_n * h_dot_n, 1e-4));

    let fresnel = fresnel_schlick(saturate(dot(half_vec, view)), mix(vec3<f32>(0.04), base_color, params.metallic));
    let diffuse = base_color * (1.0 - fresnel) * INV_PI * n_dot_l;

    return diffuse + fresnel * spec * n_dot_l / (n_dot_v + n_dot_l + 1e-4);
}

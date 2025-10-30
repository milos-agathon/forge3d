// src/shaders/brdf/ward.wgsl
// Ward anisotropic microfacet BRDF
// Exists to approximate elongated highlights for brushed materials
// RELEVANT FILES: src/shaders/brdf/dispatch.wgsl, src/shaders/brdf/common.wgsl, src/shaders/lighting.wgsl, src/lighting/types.rs

fn brdf_ward(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, base_color: vec3<f32>, params: ShadingParamsGPU) -> vec3<f32> {
    let basis = build_orthonormal_basis(normal);
    let tangent = basis[0];
    let bitangent = basis[1];
    let half_vec = normalize(view + light);

    let n_dot_l = saturate(dot(normal, light));
    let n_dot_v = saturate(dot(normal, view));
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3<f32>(0.0);
    }

    let alpha_x = max(params.roughness * (1.0 + params.anisotropy), 0.05);
    let alpha_y = max(params.roughness * (1.0 - params.anisotropy), 0.05);

    let h_dot_n = saturate(dot(half_vec, normal));
    let h_dot_t = dot(half_vec, tangent);
    let h_dot_b = dot(half_vec, bitangent);

    let exponent = -((h_dot_t * h_dot_t) / (alpha_x * alpha_x) + (h_dot_b * h_dot_b) / (alpha_y * alpha_y)) / max(h_dot_n * h_dot_n, 1e-4);
    let spec = exp(exponent) / (4.0 * PI * alpha_x * alpha_y * sqrt(n_dot_l * n_dot_v) + 1e-4);

    let fresnel = fresnel_schlick(saturate(dot(half_vec, view)), mix(vec3<f32>(0.04), base_color, params.metallic));
    let diffuse = base_color * (1.0 - fresnel) * INV_PI * n_dot_l;

    return diffuse + fresnel * spec * n_dot_l;
}

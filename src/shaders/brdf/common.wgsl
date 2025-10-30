// src/shaders/brdf/common.wgsl
// Shared BRDF math utilities and helper functions
// Exists to consolidate Fresnel/geometry terms for all BRDF implementations
// RELEVANT FILES: src/shaders/brdf/dispatch.wgsl, src/shaders/brdf/cook_torrance.wgsl, src/shaders/lighting.wgsl, src/lighting/types.rs

const PI: f32 = 3.14159265359;
const INV_PI: f32 = 1.0 / PI;

fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq <= 1e-8) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return v / sqrt(len_sq);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let clamped = saturate(cos_theta);
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - clamped, 5.0);
}

fn fresnel_schlick_scalar(cos_theta: f32, f0: f32) -> f32 {
    let clamped = saturate(cos_theta);
    return f0 + (1.0 - f0) * pow(1.0 - clamped, 5.0);
}

fn distribution_ggx(normal: vec3<f32>, half_vec: vec3<f32>, roughness: f32) -> f32 {
    let a = max(roughness * roughness, 1e-4);
    let a2 = a * a;
    let n_dot_h = saturate(dot(normal, half_vec));
    let n_dot_h2 = n_dot_h * n_dot_h;
    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 1e-6);
}

fn geometry_smith_ggx(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, roughness: f32) -> f32 {
    let n_dot_v = saturate(dot(normal, view));
    let n_dot_l = saturate(dot(normal, light));
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    let ggx_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let ggx_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return ggx_v * ggx_l;
}

fn distribution_beckmann(normal: vec3<f32>, half_vec: vec3<f32>, roughness: f32) -> f32 {
    let alpha = max(roughness * roughness, 1e-4);
    let n_dot_h = saturate(dot(normal, half_vec));
    if (n_dot_h <= 0.0) {
        return 0.0;
    }
    let tan_theta = sqrt(max(1.0 - n_dot_h * n_dot_h, 0.0)) / n_dot_h;
    let exponent = -(tan_theta * tan_theta) / (alpha * alpha);
    return exp(exponent) / (PI * alpha * alpha * n_dot_h * n_dot_h * n_dot_h * n_dot_h + 1e-6);
}

fn geometry_beckmann(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, roughness: f32) -> f32 {
    let alpha = max(roughness * roughness, 1e-4);
    let n_dot_v = saturate(dot(normal, view));
    let n_dot_l = saturate(dot(normal, light));
    let lambda_v = lambda_beckmann(n_dot_v, alpha);
    let lambda_l = lambda_beckmann(n_dot_l, alpha);
    return 0.5 / (lambda_v + lambda_l + 1e-6);
}

fn lambda_beckmann(n_dot_w: f32, alpha: f32) -> f32 {
    let n_dot_w_clamped = max(n_dot_w, 1e-4);
    let tan_theta = sqrt(max(1.0 - n_dot_w_clamped * n_dot_w_clamped, 0.0)) / n_dot_w_clamped;
    if (tan_theta == 0.0) {
        return 0.0;
    }
    let a = 1.0 / (alpha * tan_theta);
    if (a >= 1.6) {
        return 0.0;
    }
    return (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a);
}

fn to_shininess(roughness: f32) -> f32 {
    let r = saturate(roughness);
    return max(1.0, pow(1.0 - r, 5.0) * 512.0);
}

fn build_orthonormal_basis(normal: vec3<f32>) -> mat3x3<f32> {
    var tangent = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(normal.y) > 0.99) {
        tangent = vec3<f32>(1.0, 0.0, 0.0);
    }
    tangent = normalize(cross(tangent, normal));
    let bitangent = cross(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
}

// src/shaders/ibl_brdf.wgsl
// BRDF integration LUT compute shader for split-sum IBL approximation
// RELEVANT FILES: src/core/ibl.rs, src/shaders/ibl_prefilter.wgsl, src/shaders/lighting.wgsl, src/shaders/terrain_pbr_pom.wgsl

const PI: f32 = 3.14159265359;

struct PrefilterUniforms {
    env_size: u32,
    src_width: u32,
    src_height: u32,
    face_count: u32,
    mip_level: u32,
    max_mip_levels: u32,
    sample_count: u32,
    brdf_size: u32,
    roughness: f32,
    intensity: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0)
var<uniform> params: PrefilterUniforms;

@group(0) @binding(1)
var brdf_target: texture_storage_2d<rgba16float, write>;

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    var bits = i;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    let radical_inverse = f32(bits) * 2.3283064365386963e-10;
    return vec2<f32>(f32(i) / f32(n), radical_inverse);
}

fn importance_sample_ggx(xi: vec2<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    return vec3<f32>(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta,
    );
}

@compute @workgroup_size(8, 8, 1)
fn cs_brdf_lut(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.z > 0u {
        return;
    }
    let size = params.brdf_size;
    if gid.x >= size || gid.y >= size {
        return;
    }

    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / f32(size);
    // Clamp to [0,1] domain as per spec
    let n_dot_v = clamp(uv.x, 0.0, 1.0);
    let roughness = clamp(uv.y, 0.0, 1.0);

    // Ensure sqrt doesn't produce NaN
    let sin_theta = sqrt(max(1.0 - n_dot_v * n_dot_v, 0.0));
    let view = vec3<f32>(sin_theta, 0.0, n_dot_v);
    let normal = vec3<f32>(0.0, 0.0, 1.0);

    var a = 0.0;
    var b = 0.0;
    let sample_count = max(params.sample_count, 1u);

    for (var i = 0u; i < sample_count; i = i + 1u) {
        let xi = hammersley(i, sample_count);
        let half_vector = importance_sample_ggx(xi, roughness);
        let light_dir = normalize(2.0 * dot(view, half_vector) * half_vector - view);

        let n_dot_l = max(light_dir.z, 0.0);
        let n_dot_h = max(half_vector.z, 0.0);
        let v_dot_h = max(dot(view, half_vector), 0.0);

        if n_dot_l > 0.0 {
            let g = (2.0 * n_dot_h * n_dot_v) / max(v_dot_h, 1e-5);
            let g_vis = g / max(n_dot_l, 1e-5);
            let fresnel = pow(1.0 - v_dot_h, 5.0);

            a += (1.0 - fresnel) * g_vis;
            b += fresnel * g_vis;
        }
    }

    a /= f32(sample_count);
    b /= f32(sample_count);

    // Clamp output to [0,1] and ensure no NaNs (spec requirement)
    a = saturate(a);
    b = saturate(b);

    textureStore(brdf_target, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(a, b, 0.0, 0.0));
}

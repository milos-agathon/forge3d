// src/shaders/ibl_prefilter.wgsl
// Compute shaders for irradiance convolution and GGX specular prefiltering
// Shared between IBL precomputation passes to build cube-map mip hierarchies
// RELEVANT FILES: src/core/ibl.rs, src/shaders/ibl_equirect.wgsl, src/shaders/ibl_brdf.wgsl

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

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
var env_cubemap: texture_cube<f32>;

@group(0) @binding(2)
var env_sampler: sampler;

@group(0) @binding(3)
var target_cube: texture_storage_2d_array<rgba16float, write>;

fn uv_to_direction(uv: vec2<f32>, face: u32) -> vec3<f32> {
    let coord = det_barrier2(uv * 2.0) - vec2<f32>(1.0, 1.0);
    switch face {
        case 0u: { return det_normalize3(vec3<f32>(1.0, -coord.y, -coord.x)); }
        case 1u: { return det_normalize3(vec3<f32>(-1.0, -coord.y, coord.x)); }
        case 2u: { return det_normalize3(vec3<f32>(coord.x, 1.0, coord.y)); }
        case 3u: { return det_normalize3(vec3<f32>(coord.x, -1.0, -coord.y)); }
        case 4u: { return det_normalize3(vec3<f32>(coord.x, -coord.y, 1.0)); }
        default: { return det_normalize3(vec3<f32>(-coord.x, -coord.y, -1.0)); }
    }
}

fn sample_environment(dir: vec3<f32>) -> vec3<f32> {
    return textureSampleLevel(env_cubemap, env_sampler, dir, 0.0).rgb;
}

fn hemisphere_sample_uniform(xi: vec2<f32>) -> vec3<f32> {
    let phi = TWO_PI * xi.x;
    let cos_theta = 1.0 - xi.y;
    let sin_theta = det_sqrt(1.0 - det_barrier(cos_theta * cos_theta));
    return vec3<f32>(
        det_cos(phi) * sin_theta,
        det_sin(phi) * sin_theta,
        cos_theta,
    );
}

fn importance_sample_ggx(xi: vec2<f32>, normal: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = det_barrier(roughness * roughness);
    let phi = TWO_PI * xi.x;
    let cos_theta = det_sqrt(det_div(1.0 - xi.y, 1.0 + det_barrier((det_barrier(a * a) - 1.0) * xi.y)));
    let sin_theta = det_sqrt(1.0 - det_barrier(cos_theta * cos_theta));

    let h = vec3<f32>(
        det_cos(phi) * sin_theta,
        det_sin(phi) * sin_theta,
        cos_theta,
    );

    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.z) < 0.999);
    let tangent = det_normalize3(det_cross3(up, normal));
    let bitangent = det_cross3(normal, tangent);
    return det_normalize3(det_barrier3(det_barrier3(tangent * det_barrier(h.x)) + det_barrier3(bitangent * det_barrier(h.y))) + det_barrier3(normal * det_barrier(h.z)));
}

fn hammersley_2d(i: u32, n: u32) -> vec2<f32> {
    var bits = i;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return vec2<f32>(det_div(f32(i), f32(n)), f32(bits) * 2.3283064365386963e-10);
}

@compute @workgroup_size(8, 8, 1)
fn cs_irradiance_convolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    det_seed(f32(gid.x));
    let face = gid.z;
    if face >= params.face_count {
        return;
    }
    let size = params.env_size;
    if gid.x >= size || gid.y >= size {
        return;
    }

    let uv = det_div2(vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5, vec2<f32>(f32(size)));
    let normal = uv_to_direction(uv, face);

    var irradiance = vec3<f32>(0.0);
    // Fixed 128 samples per texel for deterministic results (spec requirement)
    let sample_count = 128u;
    for (var i = 0u; i < sample_count; i = i + 1u) {
        // Stratified hemisphere sampling (cos-weighted)
        let xi = hammersley_2d(i, sample_count);
        let phi = TWO_PI * det_barrier(xi.x);
        let cos_theta = det_sqrt(1.0 - det_barrier(xi.y)); // cos-weighted distribution
        let sin_theta = det_sqrt(1.0 - det_barrier(cos_theta * cos_theta));
        let sample_dir_local = vec3<f32>(
            det_cos(phi) * sin_theta,
            det_sin(phi) * sin_theta,
            cos_theta,
        );

        let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.z) < 0.999);
        let tangent = det_normalize3(det_cross3(up, normal));
        let bitangent = det_cross3(normal, tangent);
        let sample_dir = det_normalize3(
            det_barrier3(det_barrier3(tangent * det_barrier(sample_dir_local.x)) +
            det_barrier3(bitangent * det_barrier(sample_dir_local.y))) +
            det_barrier3(normal * det_barrier(sample_dir_local.z))
        );

        irradiance = det_barrier3(irradiance + sample_environment(sample_dir));
    }

    // Cosine-weighted sampling (pdf = cos/pi) already integrates the n dot wi
    // factor: mean(L_i) estimates (1/pi) * integral(L * cos) — the normalized
    // irradiance convention shared with fs_irradiance_convolution and the
    // albedo * irradiance IBL consumers.
    irradiance = det_div3(irradiance, vec3<f32>(f32(sample_count)));
    // Clamp to prevent NaNs/inf and ensure no pixel > 1.0 for unit-intensity HDR (spec requirement)
    irradiance = saturate(irradiance);
    textureStore(
        target_cube,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        i32(face),
        vec4<f32>(irradiance, 1.0),
    );
}

@compute @workgroup_size(8, 8, 1)
fn cs_specular_prefilter(@builtin(global_invocation_id) gid: vec3<u32>) {
    det_seed(f32(gid.x));
    let face = gid.z;
    if face >= params.face_count {
        return;
    }
    let size = params.env_size;
    if gid.x >= size || gid.y >= size {
        return;
    }

    let uv = det_div2(vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5, vec2<f32>(f32(size)));
    let normal = uv_to_direction(uv, face);
    let view_dir = normal;

    var prefiltered = vec3<f32>(0.0);
    var total_weight = 0.0;
    let roughness = clamp(params.roughness, 0.0, 1.0);
    let sample_count = max(params.sample_count, 1u);

    for (var i = 0u; i < sample_count; i = i + 1u) {
        let xi = hammersley_2d(i, sample_count);
        let half_dir = importance_sample_ggx(xi, normal, roughness);
        let light_dir = det_normalize3(det_barrier3(det_barrier(2.0 * det_dot3(view_dir, half_dir)) * half_dir) - view_dir);

        let n_dot_l = max(det_dot3(normal, light_dir), 0.0);
        if n_dot_l > 0.0 {
            let n_dot_h = max(det_dot3(normal, half_dir), 0.0);
            let v_dot_h = max(det_dot3(view_dir, half_dir), 0.0);
            let d = det_barrier(roughness * roughness);
            let pdf = det_div(det_barrier(d * d) * n_dot_h, max(4.0 * v_dot_h, 1e-4)) + 1e-4;

            let resolution = f32(params.env_size);
            let sa_texel = det_div(4.0 * PI, det_barrier(6.0 * resolution) * resolution);
            let sa_sample = det_div(1.0, f32(sample_count) * pdf);
            let lod = 0.5 * det_log2(det_div(sa_sample, sa_texel));
            let mip = clamp(lod, 0.0, f32(params.max_mip_levels - 1u));

            let color = textureSampleLevel(env_cubemap, env_sampler, light_dir, mip);
            prefiltered = det_barrier3(prefiltered + det_barrier3(color.rgb * n_dot_l));
            total_weight = det_barrier(total_weight + n_dot_l);
        }
    }

    prefiltered = det_div3(prefiltered, vec3<f32>(max(total_weight, 1e-3)));
    // Clamp to prevent NaNs/inf and ensure no pixel > 1.0 for unit-intensity HDR (spec requirement)
    prefiltered = saturate(prefiltered);
    textureStore(
        target_cube,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        i32(face),
        vec4<f32>(prefiltered, 1.0),
    );
}

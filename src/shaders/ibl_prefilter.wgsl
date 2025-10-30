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
    let coord = uv * 2.0 - vec2<f32>(1.0, 1.0);
    switch face {
        case 0u: { return normalize(vec3<f32>(1.0, -coord.y, -coord.x)); }
        case 1u: { return normalize(vec3<f32>(-1.0, -coord.y, coord.x)); }
        case 2u: { return normalize(vec3<f32>(coord.x, 1.0, coord.y)); }
        case 3u: { return normalize(vec3<f32>(coord.x, -1.0, -coord.y)); }
        case 4u: { return normalize(vec3<f32>(coord.x, -coord.y, 1.0)); }
        default: { return normalize(vec3<f32>(-coord.x, -coord.y, -1.0)); }
    }
}

fn sample_environment(dir: vec3<f32>) -> vec3<f32> {
    return textureSampleLevel(env_cubemap, env_sampler, dir, 0.0).rgb;
}

fn hemisphere_sample_uniform(xi: vec2<f32>) -> vec3<f32> {
    let phi = TWO_PI * xi.x;
    let cos_theta = 1.0 - xi.y;
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    return vec3<f32>(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta,
    );
}

fn importance_sample_ggx(xi: vec2<f32>, normal: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = TWO_PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    let h = vec3<f32>(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta,
    );

    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.z) < 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return normalize(tangent * h.x + bitangent * h.y + normal * h.z);
}

fn hammersley_2d(i: u32, n: u32) -> vec2<f32> {
    var bits = i;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return vec2<f32>(f32(i) / f32(n), f32(bits) * 2.3283064365386963e-10);
}

@compute @workgroup_size(8, 8, 1)
fn cs_irradiance_convolution(@builtin(global_invocation_id) gid: vec3<u32>) {
    let face = gid.z;
    if face >= params.face_count {
        return;
    }
    let size = params.env_size;
    if gid.x >= size || gid.y >= size {
        return;
    }

    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / f32(size);
    let normal = uv_to_direction(uv, face);

    var irradiance = vec3<f32>(0.0);
    let sample_count = max(params.sample_count, 1u);
    for (var i = 0u; i < sample_count; i = i + 1u) {
        let sample_dir_local = hemisphere_sample_uniform(vec2<f32>(
            f32(i) / f32(sample_count),
            f32(i32(i * 16807u % sample_count)) / f32(sample_count)
        ));

        let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.z) < 0.999);
        let tangent = normalize(cross(up, normal));
        let bitangent = cross(normal, tangent);
        let sample_dir = normalize(
            tangent * sample_dir_local.x +
            bitangent * sample_dir_local.y +
            normal * sample_dir_local.z
        );

        irradiance += sample_environment(sample_dir) * sample_dir_local.z;
    }

    irradiance = PI * irradiance / f32(sample_count);
    textureStore(
        target_cube,
        vec3<i32>(i32(gid.x), i32(gid.y), i32(face)),
        vec4<f32>(irradiance, 1.0),
    );
}

@compute @workgroup_size(8, 8, 1)
fn cs_specular_prefilter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let face = gid.z;
    if face >= params.face_count {
        return;
    }
    let size = params.env_size;
    if gid.x >= size || gid.y >= size {
        return;
    }

    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / f32(size);
    let normal = uv_to_direction(uv, face);
    let view_dir = normal;

    var prefiltered = vec3<f32>(0.0);
    var total_weight = 0.0;
    let roughness = clamp(params.roughness, 0.0, 1.0);
    let sample_count = max(params.sample_count, 1u);

    for (var i = 0u; i < sample_count; i = i + 1u) {
        let xi = hammersley_2d(i, sample_count);
        let half_dir = importance_sample_ggx(xi, normal, roughness);
        let light_dir = normalize(2.0 * dot(view_dir, half_dir) * half_dir - view_dir);

        let n_dot_l = max(dot(normal, light_dir), 0.0);
        if n_dot_l > 0.0 {
            let n_dot_h = max(dot(normal, half_dir), 0.0);
            let v_dot_h = max(dot(view_dir, half_dir), 0.0);
            let d = roughness * roughness;
            let pdf = (d * d * n_dot_h) / max(4.0 * v_dot_h, 1e-4) + 1e-4;

            let resolution = f32(params.env_size);
            let sa_texel = 4.0 * PI / (6.0 * resolution * resolution);
            let sa_sample = 1.0 / (f32(sample_count) * pdf);
            let lod = 0.5 * log2(sa_sample / sa_texel);
            let mip = clamp(lod, 0.0, f32(params.max_mip_levels - 1u));

            let color = textureSampleLevel(env_cubemap, env_sampler, light_dir, mip);
            prefiltered += color.rgb * n_dot_l;
            total_weight += n_dot_l;
        }
    }

    prefiltered = prefiltered / max(total_weight, 1e-3);
    textureStore(
        target_cube,
        vec3<i32>(i32(gid.x), i32(gid.y), i32(face)),
        vec4<f32>(prefiltered, 1.0),
    );
}

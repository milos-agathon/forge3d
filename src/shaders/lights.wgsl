// src/shaders/lights.wgsl
// Light GPU buffer layout and sampling helpers
// Exists to expose multi-light data/sampling to shared lighting shaders
// RELEVANT FILES: src/lighting/types.rs, src/lighting/light_buffer.rs, src/shaders/lighting.wgsl, python/forge3d/config.py

const PI: f32 = 3.14159265359;

// Light types (matches Rust LightType enum)
const LIGHT_DIRECTIONAL: u32 = 0u;
const LIGHT_POINT: u32 = 1u;
const LIGHT_SPOT: u32 = 2u;
const LIGHT_ENVIRONMENT: u32 = 3u;
const LIGHT_AREA_RECT: u32 = 4u;
const LIGHT_AREA_DISK: u32 = 5u;
const LIGHT_AREA_SPHERE: u32 = 6u;

struct LightGPU {
    type_: u32,
    intensity: f32,
    range: f32,
    env_texture_index: u32,

    color: vec3<f32>,
    _pad0: f32,

    dir_ws: vec3<f32>,
    _pad1: f32,

    pos_ws: vec3<f32>,
    _pad2: f32,

    cone_cos: vec2<f32>,
    area_half: vec2<f32>,
};

struct LightMetadata {
    count: u32,
    frame_index: u32,
    seed_bits_x: u32,
    seed_bits_y: u32,
};

@group(0) @binding(3) var<storage, read> lights: array<LightGPU>;
@group(0) @binding(4) var<uniform> lightMeta: LightMetadata;
@group(0) @binding(5) var<uniform> environmentParams: vec4<f32>;

fn light_count() -> u32 {
    return lightMeta.count;
}

fn light_sequence_seed() -> vec2<f32> {
    return vec2<f32>(
        bitcast<f32>(lightMeta.seed_bits_x),
        bitcast<f32>(lightMeta.seed_bits_y)
    );
}

fn sample_directional(i: u32) -> vec3<f32> {
    let light = lights[i];
    return normalize(-light.dir_ws);
}

fn sample_point(i: u32, Xi: vec2<f32>) -> vec3<f32> {
    let u = Xi.x;
    let v = Xi.y;
    let phi = 2.0 * PI * u;
    let cos_theta = 1.0 - 2.0 * v;
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    return vec3<f32>(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
}

fn sample_spot(i: u32, Xi: vec2<f32>) -> vec3<f32> {
    let light = lights[i];
    let inner_cos = light.cone_cos.x;
    let outer_cos = light.cone_cos.y;
    let u = Xi.x;
    let v = Xi.y;
    let cos_theta = mix(outer_cos, inner_cos, u);
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 2.0 * PI * v;
    let local_dir = vec3<f32>(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
    // Build orthonormal basis from light direction
    let forward = normalize(light.dir_ws);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(up, forward));
    let ortho_up = cross(forward, right);
    return normalize(
        local_dir.x * right +
        local_dir.y * ortho_up +
        local_dir.z * forward
    );
}

fn sample_area_rect(i: u32, Xi: vec2<f32>) -> vec3<f32> {
    let light = lights[i];
    let half_width = light.area_half.x;
    let half_height = light.area_half.y;
    let local = vec3<f32>(
        mix(-half_width, half_width, Xi.x),
        0.0,
        mix(-half_height, half_height, Xi.y)
    );
    let normal = normalize(light.dir_ws);
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(normal.y) > 0.99) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    let world_pos = light.pos_ws + tangent * local.x + bitangent * local.z;
    return normalize(world_pos - light.pos_ws);
}

fn sample_area_disk(i: u32, Xi: vec2<f32>) -> vec3<f32> {
    let light = lights[i];
    let radius = light.area_half.x;
    let r = radius * sqrt(Xi.x);
    let theta = 2.0 * PI * Xi.y;
    let local = vec3<f32>(r * cos(theta), 0.0, r * sin(theta));
    let normal = normalize(light.dir_ws);
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(normal.y) > 0.99) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    let world_pos = light.pos_ws + tangent * local.x + bitangent * local.z;
    return normalize(world_pos - light.pos_ws);
}

fn sample_area_sphere(i: u32, Xi: vec2<f32>) -> vec3<f32> {
    let light = lights[i];
    return sample_point(i, Xi);
}

fn sample_environment(Xi: vec2<f32>) -> vec3<f32> {
    // Placeholder lat-long mapping
    let phi = 2.0 * PI * Xi.x;
    let cos_theta = 1.0 - 2.0 * Xi.y;
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    return vec3<f32>(
        sin_theta * cos(phi),
        cos_theta,
        sin_theta * sin(phi)
    );
}

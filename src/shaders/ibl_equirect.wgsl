// src/shaders/ibl_equirect.wgsl
// Compute shader converting equirectangular HDR maps into cube-map faces
// Enables GPU-side environment pre-processing before irradiance/specular integration
// RELEVANT FILES: src/core/ibl.rs, src/shaders/ibl_prefilter.wgsl, src/shaders/ibl_brdf.wgsl

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
var equirect_map: texture_2d<f32>;

@group(0) @binding(2)
var env_sampler: sampler;

@group(0) @binding(3)
var target_cube: texture_storage_2d_array<rgba16float, write>;

fn uv_to_direction(uv: vec2<f32>, face: u32) -> vec3<f32> {
    let coord = uv * 2.0 - vec2<f32>(1.0, 1.0);
    switch face {
        case 0u: { return normalize(vec3<f32>(1.0, -coord.y, -coord.x)); }   // +X
        case 1u: { return normalize(vec3<f32>(-1.0, -coord.y, coord.x)); }   // -X
        case 2u: { return normalize(vec3<f32>(coord.x, 1.0, coord.y)); }     // +Y
        case 3u: { return normalize(vec3<f32>(coord.x, -1.0, -coord.y)); }  // -Y
        case 4u: { return normalize(vec3<f32>(coord.x, -coord.y, 1.0)); }   // +Z
        case 5u: { return normalize(vec3<f32>(-coord.x, -coord.y, -1.0)); } // -Z
        default: { return vec3<f32>(0.0, 0.0, 1.0); }
    }
}

fn direction_to_equirect(dir: vec3<f32>) -> vec2<f32> {
    let d = normalize(dir);
    let u = atan2(d.z, d.x) / TWO_PI + 0.5;
    let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2<f32>(fract(u), clamp(v, 0.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_equirect_to_cubemap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let face = gid.z;
    if face >= params.face_count {
        return;
    }

    let size = params.env_size;
    if gid.x >= size || gid.y >= size {
        return;
    }

    let pixel = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5) / f32(size);
    let world_dir = uv_to_direction(pixel, face);
    let uv = direction_to_equirect(world_dir);

    let color = textureSampleLevel(equirect_map, env_sampler, uv, 0.0);
    textureStore(
        target_cube,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        i32(face),
        color,
    );
}

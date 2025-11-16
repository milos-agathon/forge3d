// src/shaders/ssr/shade.wgsl
// Convert SSR hits into specular contributions using Fresnel weighting

struct SsrSettings {
    max_steps: u32,
    thickness: f32,
    max_distance: f32,
    intensity: f32,
    inv_resolution: vec2<f32>,
    _pad: vec2<f32>,
}

struct CameraParams {
    view_matrix: mat4x4<f32>,
    inv_view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    inv_proj_matrix: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var hit_texture: texture_2d<f32>;
@group(0) @binding(3) var normal_texture: texture_2d<f32>;
@group(0) @binding(4) var material_texture: texture_2d<f32>;
@group(0) @binding(5) var depth_texture: texture_2d<f32>;
@group(0) @binding(6) var ssr_spec_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var<uniform> settings: SsrSettings;
@group(0) @binding(8) var<uniform> camera: CameraParams;

fn decode_normal(encoded: vec4<f32>) -> vec3<f32> {
    return normalize(encoded.xyz * 2.0 - 1.0);
}

fn reconstruct_view_position(uv: vec2<f32>, linear_depth: f32) -> vec3<f32> {
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let focal = vec2<f32>(camera.inv_proj_matrix[0][0], camera.inv_proj_matrix[1][1]);
    let center = vec2<f32>(camera.inv_proj_matrix[2][0], camera.inv_proj_matrix[2][1]);
    let view_xy = (ndc_xy - center) / focal;
    return vec3<f32>(view_xy * linear_depth, -linear_depth);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let clamped = clamp(1.0 - cos_theta, 0.0, 1.0);
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamped, 5.0);
}

fn edge_fade(uv: vec2<f32>) -> f32 {
    let pad = 0.04;
    let dist = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    return smoothstep(0.0, pad, dist);
}

@compute @workgroup_size(8, 8, 1)
fn cs_shade(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(hit_texture);
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    let hit = textureLoad(hit_texture, pixel, 0);
    if (hit.w < 0.5) {
        textureStore(ssr_spec_out, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let hit_uv = hit.xy;
    let sample_color = textureSampleLevel(
        scene_color,
        scene_sampler,
        clamp(hit_uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0)),
        0.0,
    )
    .rgb;

    let normal_sample = textureLoad(normal_texture, pixel, 0);
    let normal_vs = decode_normal(normal_sample);
    let roughness = clamp(normal_sample.w, 0.0, 1.0);

    let material = textureLoad(material_texture, pixel, 0);
    let albedo = material.rgb;
    let metallic = clamp(material.a, 0.0, 1.0);

    let depth = textureLoad(depth_texture, pixel, 0).r;
    if (depth <= 0.0) {
        textureStore(ssr_spec_out, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let uv = (vec2<f32>(f32(pixel.x), f32(pixel.y)) + vec2<f32>(0.5, 0.5)) * settings.inv_resolution;
    let view_pos = reconstruct_view_position(uv, depth);
    let view_dir = normalize(-view_pos);

    let f0 = mix(vec3<f32>(0.04, 0.04, 0.04), albedo, vec3<f32>(metallic));
    let fresnel = fresnel_schlick(max(dot(normal_vs, view_dir), 0.0), f0);
    let cone_weight = pow(max(1.0 - roughness, 0.0), 1.5);
    let fade = edge_fade(uv);

    let spec = sample_color * fresnel * cone_weight * fade * settings.intensity;
    textureStore(ssr_spec_out, pixel, vec4<f32>(spec, 1.0));
}

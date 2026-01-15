// src/shaders/ssr/shade.wgsl
// Convert SSR hits into specular contributions using Fresnel weighting.
// ssr_spec_out is Rgba16Float with:
//   rgb = view-space specular reflection color in linear HDR units,
//   a   = reflection weight in [0,1] used by the composite stage.

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
    // P1.1: Previous frame view-projection for motion vectors
    prev_view_proj_matrix: mat4x4<f32>,
    camera_pos: vec3<f32>,
    frame_index: u32,
    // P1.2: Sub-pixel jitter offset for TAA (pixel units, [-0.5, 0.5])
    jitter_offset: vec2<f32>,
    _pad_jitter: vec2<f32>,
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

fn sample_hit_mask(px: vec2<u32>) -> f32 {
    let h = textureLoad(hit_texture, px, 0).w;
    return step(0.5, h);
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
    // Sample the pre-GBuffer material (albedo) at the hit location; this contains the floor stripe texture
    let sample_color = textureSampleLevel(scene_color, scene_sampler, hit_uv, 0.0).rgb;

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
    // Roughness-based cone weight: strongly favor glossy surfaces while
    // allowing very rough materials to contribute only weak reflections.
    let cone_weight = pow(max(1.0 - roughness, 0.0), 2.0);
    let fade = pow(edge_fade(uv), 0.5);

    // Edge fade: reduce streaks at SSR edges using local hit-mask neighborhood
    let offx = vec2<i32>(1, 0);
    let offy = vec2<i32>(0, 1);
    let px_i = vec2<i32>(i32(pixel.x), i32(pixel.y));
    let clamp_to = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);
    let q_r = clamp(px_i + offx, vec2<i32>(0, 0), clamp_to);
    let q_l = clamp(px_i - offx, vec2<i32>(0, 0), clamp_to);
    let q_u = clamp(px_i - offy, vec2<i32>(0, 0), clamp_to);
    let q_d = clamp(px_i + offy, vec2<i32>(0, 0), clamp_to);
    let h_r = sample_hit_mask(vec2<u32>(u32(q_r.x), u32(q_r.y)));
    let h_l = sample_hit_mask(vec2<u32>(u32(q_l.x), u32(q_l.y)));
    let h_u = sample_hit_mask(vec2<u32>(u32(q_u.x), u32(q_u.y)));
    let h_d = sample_hit_mask(vec2<u32>(u32(q_d.x), u32(q_d.y)));
    let miss_near = clamp(1.0 - 0.25 * (h_r + h_l + h_u + h_d), 0.0, 1.0);
    let edge_fade_local = mix(0.35, 1.0, 1.0 - miss_near);

    // Depth discontinuity fade: reduce SSR near strong depth jumps
    let d_c = textureLoad(depth_texture, pixel, 0).r;
    let px_r = min(pixel.x + 1u, dims.x - 1u);
    let px_l = u32(max(i32(pixel.x) - 1, 0));
    let py_u = u32(max(i32(pixel.y) - 1, 0));
    let py_d = min(pixel.y + 1u, dims.y - 1u);
    let d_r = textureLoad(depth_texture, vec2<u32>(px_r, pixel.y), 0).r;
    let d_l = textureLoad(depth_texture, vec2<u32>(px_l, pixel.y), 0).r;
    let d_u = textureLoad(depth_texture, vec2<u32>(pixel.x, py_u), 0).r;
    let d_d = textureLoad(depth_texture, vec2<u32>(pixel.x, py_d), 0).r;
    let depth_grad_x = max(abs(d_c - d_r), abs(d_c - d_l));
    let depth_grad_y = max(abs(d_c - d_u), abs(d_c - d_d));
    let depth_grad = max(depth_grad_x, depth_grad_y);
    let depth_fade = 1.0 - smoothstep(0.02, 0.08, depth_grad);
    let depth_weight = mix(0.4, 1.0, depth_fade);

    let total_weight = cone_weight * fade * edge_fade_local * depth_weight;
    let spec = sample_color * fresnel * total_weight * settings.intensity;
    textureStore(ssr_spec_out, pixel, vec4<f32>(spec, total_weight));
}

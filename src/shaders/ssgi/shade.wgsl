// src/shaders/ssgi/shade.wgsl
// P5.2: Shade half-res SSGI using previous-frame color with IBL fallback.
// Produces `outRadiance` as Rgba16Float where rgb = diffuse GI radiance in linear HDR
// units and a = 1.0. This is a purely diffuse term intended to be added onto
// L_diffuse_base (never the specular component).

struct SsgiSettings {
    radius: f32,
    intensity: f32,
    num_steps: u32,
    step_size: f32,
    inv_resolution: vec2<f32>,
    temporal_alpha: f32,
    temporal_enabled: u32,
    use_half_res: u32,
    upsample_depth_sigma: f32,
    upsample_normal_sigma: f32,
    use_edge_aware: u32,
    _pad1: u32,
    frame_index: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: vec4<u32>,
    _pad7: vec3<u32>,
    _pad8: vec4<u32>,
};

struct CameraParams {
    view_matrix: mat4x4<f32>,
    inv_view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    inv_proj_matrix: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
};

@group(0) @binding(0) var tPrevColor: texture_2d<f32>;
@group(0) @binding(1) var sLinear: sampler;
@group(0) @binding(2) var tEnvDiffuse: texture_cube<f32>;
@group(0) @binding(3) var sEnv: sampler;
@group(0) @binding(4) var tHit: texture_2d<f32>;
@group(0) @binding(5) var outRadiance: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var<uniform> uSsgi: SsgiSettings;
@group(0) @binding(7) var<uniform> uCam: CameraParams;
@group(0) @binding(8) var tNormalFull: texture_2d<f32>;
@group(0) @binding(9) var tAlbedo: texture_2d<f32>;

fn decode_normal(encoded: vec4<f32>) -> vec3<f32> {
    return normalize(encoded.xyz * 2.0 - 1.0);
}

fn to_world(normal_vs: vec3<f32>) -> vec3<f32> {
    return normalize((uCam.inv_view_matrix * vec4<f32>(normal_vs, 0.0)).xyz);
}

fn sample_prev_color(uv: vec2<f32>) -> vec3<f32> {
    let clamped = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    return textureSampleLevel(tPrevColor, sLinear, clamped, 0.0).rgb;
}

fn sample_albedo(pixel: vec2<u32>) -> vec3<f32> {
    return textureLoad(tAlbedo, pixel, 0).rgb;
}

fn sample_world_normal(pixel: vec2<u32>) -> vec3<f32> {
    return to_world(decode_normal(textureLoad(tNormalFull, pixel, 0)));
}

@compute @workgroup_size(8, 8, 1)
fn cs_shade(@builtin(global_invocation_id) gid: vec3<u32>) {
    var read_xy = gid.xy;
    if (uSsgi.use_half_res == 1u) {
        read_xy *= 2u;
    }

    let hit = textureLoad(tHit, gid.xy, 0);
    let hit_uv = hit.xy;
    let hit_mask = step(0.5, hit.w);

    let albedo = sample_albedo(read_xy);
    let normal_ws = sample_world_normal(read_xy);
    // Task 2: Pure diffuse IBL fallback (same as engine's reference)
    let fallback = textureSampleLevel(tEnvDiffuse, sEnv, normal_ws, 0.0).rgb * albedo;

    // Task 2: When steps==0, output only diffuse IBL (no previous frame color)
    if (uSsgi.num_steps == 0u) {
        textureStore(outRadiance, gid.xy, vec4<f32>(fallback, 1.0));
        return;
    }

    // Normal case: mix fallback with previous frame color based on hit mask
    let prev_rgb = sample_prev_color(hit_uv);
    let indirect = mix(fallback, prev_rgb, hit_mask);
    textureStore(outRadiance, gid.xy, vec4<f32>(indirect * uSsgi.intensity, 1.0));
}

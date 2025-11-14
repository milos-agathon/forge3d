// src/shaders/ssgi/shade.wgsl
// P5.2: Shade SSGI from trace hit data using previous-frame color (fallback: diffuse IBL)

struct SsgiSettings {
    radius: f32,
    intensity: f32,
    num_steps: u32,
    step_size: f32,
    inv_resolution: vec2<f32>,
    temporal_alpha: f32,
    use_half_res: u32,
    upsample_depth_sigma: f32,
    upsample_normal_exp: f32,
    use_edge_aware: u32,
    _pad1: u32,
};

@group(0) @binding(0) var tAlbedoOrPrev: texture_2d<f32>; // previous-frame color or albedo
@group(0) @binding(1) var tEnvDiffuse: texture_cube<f32>;
@group(0) @binding(2) var sEnv: sampler;
@group(0) @binding(3) var tHit: texture_2d<f32>;
@group(0) @binding(4) var outRadiance: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> uSsgi: SsgiSettings;
@group(0) @binding(6) var tNormalFull: texture_2d<f32>;

fn unpack_normal(packed: vec4<f32>) -> vec3<f32> {
    return normalize(packed.xyz * 2.0 - 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs_shade(@builtin(global_invocation_id) gid: vec3<u32>) {
    var out_xy = gid.xy;
    var read_xy = gid.xy;
    if (uSsgi.use_half_res == 1u) {
        read_xy = gid.xy * 2u;
    }

    let dims = textureDimensions(tHit);
    if (out_xy.x >= dims.x || out_xy.y >= dims.y) { return; }

    let hit = textureLoad(tHit, out_xy, 0);
    let hit_uv = hit.xy;
    let hit_mask = step(0.5, hit.w);

    // Sample previous-frame color at hit UV (fallback: albedo)
    let dims_prev = textureDimensions(tAlbedoOrPrev);
    let uv_prev = clamp(hit_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let prev_px = vec2<u32>(uv_prev * vec2<f32>(dims_prev));
    let prev_rgb = textureLoad(tAlbedoOrPrev, prev_px, 0).rgb;

    // Local normal for IBL fallback
    let n = unpack_normal(textureLoad(tNormalFull, read_xy, 0));
    let ibl_rgb = textureSampleLevel(tEnvDiffuse, sEnv, n, 0.0).rgb;

    let gi = mix(ibl_rgb, prev_rgb, hit_mask);
    textureStore(outRadiance, out_xy, vec4<f32>(gi * uSsgi.intensity, 1.0));
}

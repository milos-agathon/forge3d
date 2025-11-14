// src/shaders/ssgi/resolve_temporal.wgsl
// P5.2: Temporal accumulation with neighborhood clamp

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

@group(0) @binding(0) var tCurrent: texture_2d<f32>;
@group(0) @binding(1) var tHistory: texture_2d<f32>;
@group(0) @binding(2) var outFiltered: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> uSsgi: SsgiSettings;

fn neighborhood_min_max(uv: vec2<u32>) -> vec2<f32> {
    let dims = textureDimensions(tCurrent);
    var mn = vec3<f32>(1e9);
    var mx = vec3<f32>(-1e9);
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let sx = clamp(i32(uv.x) + dx, 0, i32(dims.x) - 1);
            let sy = clamp(i32(uv.y) + dy, 0, i32(dims.y) - 1);
            let c = textureLoad(tCurrent, vec2<u32>(u32(sx), u32(sy)), 0).rgb;
            mn = min(mn, c);
            mx = max(mx, c);
        }
    }
    // Return luminance min/max for clamp
    let w = vec3<f32>(0.2126, 0.7152, 0.0722);
    return vec2<f32>(dot(mn, w), dot(mx, w));
}

@compute @workgroup_size(8, 8, 1)
fn cs_resolve_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let xy = gid.xy;
    let dims = textureDimensions(tCurrent);
    if (xy.x >= dims.x || xy.y >= dims.y) { return; }

    let cur = textureLoad(tCurrent, xy, 0).rgb;
    let his = textureLoad(tHistory, xy, 0).rgb;

    var out_rgb = cur;
    if (uSsgi.temporal_alpha > 0.0) {
        let a = clamp(uSsgi.temporal_alpha, 0.0, 1.0);
        // Neighborhood clamp on luminance
        let mm = neighborhood_min_max(xy);
        let w = vec3<f32>(0.2126, 0.7152, 0.0722);
        let l_cur = dot(cur, w);
        let l_his = dot(his, w);
        let l_his_clamped = clamp(l_his, mm.x, mm.y);
        var his_scale = 1.0;
        if (l_his > 0.0) {
            his_scale = l_his_clamped / max(l_his, 1e-5);
        }
        let his_rgb = his * his_scale;
        out_rgb = mix(cur, his_rgb, a);
    }

    textureStore(outFiltered, xy, vec4<f32>(out_rgb, 1.0));
}

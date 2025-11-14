// src/shaders/ssgi/resolve_temporal.wgsl
// P5.2: Temporal accumulation with luminance clamp and depth-aware confidence

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

@group(0) @binding(0) var tCurrent: texture_2d<f32>;
@group(0) @binding(1) var tHistory: texture_2d<f32>;
@group(0) @binding(2) var outFiltered: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> uSsgi: SsgiSettings;
@group(0) @binding(4) var tDepthFull: texture_2d<f32>;
@group(0) @binding(5) var tNormalFull: texture_2d<f32>;

fn luminance(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn decode_normal(encoded: vec4<f32>) -> vec3<f32> {
    return normalize(encoded.xyz * 2.0 - 1.0);
}

fn full_res_xy(xy: vec2<u32>) -> vec2<u32> {
    let half_dims = textureDimensions(tCurrent);
    let full_dims = textureDimensions(tDepthFull);
    let uv = (vec2<f32>(xy) + vec2<f32>(0.5)) / vec2<f32>(half_dims);
    let coord = clamp(uv * vec2<f32>(full_dims), vec2<f32>(0.0), vec2<f32>(full_dims) - vec2<f32>(1.0));
    return vec2<u32>(u32(coord.x), u32(coord.y));
}

fn neighborhood_bounds(xy: vec2<u32>) -> vec2<f32> {
    let dims = textureDimensions(tCurrent);
    var mn = vec3<f32>(99999.0);
    var mx = vec3<f32>(-99999.0);
    for (var j: i32 = -1; j <= 1; j = j + 1) {
        for (var i: i32 = -1; i <= 1; i = i + 1) {
            let sx = clamp(i32(xy.x) + i, 0, i32(dims.x) - 1);
            let sy = clamp(i32(xy.y) + j, 0, i32(dims.y) - 1);
            let sample_rgb = textureLoad(tCurrent, vec2<u32>(u32(sx), u32(sy)), 0).rgb;
            mn = min(mn, sample_rgb);
            mx = max(mx, sample_rgb);
        }
    }
    return vec2<f32>(luminance(mn), luminance(mx));
}

@compute @workgroup_size(8, 8, 1)
fn cs_resolve_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(tCurrent);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let cur = textureLoad(tCurrent, gid.xy, 0).rgb;
    if (uSsgi.temporal_enabled == 0u || uSsgi.temporal_alpha <= 0.0) {
        textureStore(outFiltered, gid.xy, vec4<f32>(cur, 1.0));
        return;
    }

    let full_xy = full_res_xy(gid.xy);
    let depth_cur = textureLoad(tDepthFull, full_xy, 0).r;
    let normal_ws = decode_normal(textureLoad(tNormalFull, full_xy, 0));

    var history_rgb = textureLoad(tHistory, gid.xy, 0).rgb;
    let bounds = neighborhood_bounds(gid.xy);
    let hist_luma = luminance(history_rgb);
    let clamped = clamp(hist_luma, bounds.x, bounds.y);
    if (hist_luma > 1e-4) {
        history_rgb *= clamped / hist_luma;
    } else {
        history_rgb = cur;
    }

    let stability = clamp(0.5 + 0.5 * abs(normal_ws.z), 0.0, 1.0) * clamp(depth_cur * 0.25 + 0.5, 0.0, 1.0);
    let alpha = clamp(uSsgi.temporal_alpha, 0.0, 1.0) * stability;
    let blended = mix(history_rgb, cur, alpha);
    textureStore(outFiltered, gid.xy, vec4<f32>(blended, 1.0));
}

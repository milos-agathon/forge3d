// src/shaders/filters/edge_aware_upsample.wgsl
// P5.2: Edge-aware upsample from half to full resolution with bilateral weights

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

@group(0) @binding(0) var tHalf: texture_2d<f32>;
@group(0) @binding(1) var outFull: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var sLinear: sampler;
@group(0) @binding(3) var tDepthFull: texture_2d<f32>;
@group(0) @binding(4) var tNormalFull: texture_2d<f32>;
@group(0) @binding(5) var<uniform> uSsgi: SsgiSettings;

fn decode_normal(encoded: vec4<f32>) -> vec3<f32> {
    return normalize(encoded.xyz * 2.0 - 1.0);
}

fn half_uv_from_full(full_xy: vec2<u32>) -> vec2<f32> {
    let half_dims = vec2<f32>(textureDimensions(tHalf));
    let full_dims = vec2<f32>(textureDimensions(outFull));
    let uv = (vec2<f32>(full_xy) + vec2<f32>(0.5)) / full_dims;
    let scale = full_dims / half_dims;
    return clamp(uv * scale, vec2<f32>(0.0), vec2<f32>(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_edge_aware_upsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_dims = textureDimensions(outFull);
    if (gid.x >= full_dims.x || gid.y >= full_dims.y) {
        return;
    }

    let half_uv = half_uv_from_full(gid.xy);
    if (uSsgi.use_edge_aware == 0u) {
        let color = textureSampleLevel(tHalf, sLinear, half_uv, 0.0).rgb;
        textureStore(outFull, gid.xy, vec4<f32>(color, 1.0));
        return;
    }

    let center_depth = textureLoad(tDepthFull, gid.xy, 0).r;
    let center_normal = decode_normal(textureLoad(tNormalFull, gid.xy, 0));
    let half_dims = vec2<f32>(textureDimensions(tHalf));

    var accum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    let sigma_d = max(uSsgi.upsample_depth_sigma, 1e-4);
    let sigma_n = max(uSsgi.upsample_normal_sigma, 1e-4);

    for (var j: i32 = -1; j <= 1; j = j + 1) {
        for (var i: i32 = -1; i <= 1; i = i + 1) {
            let offset = vec2<f32>(f32(i), f32(j));
            let sample_uv = clamp(half_uv + offset / half_dims, vec2<f32>(0.0), vec2<f32>(1.0));
            let half_rgb = textureSampleLevel(tHalf, sLinear, sample_uv, 0.0).rgb;

            let sample_xy_i = vec2<i32>(clamp(vec2<i32>(gid.xy) + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(full_dims) - vec2<i32>(1)));
            let sample_xy = vec2<u32>(u32(sample_xy_i.x), u32(sample_xy_i.y));
            let sample_depth = textureLoad(tDepthFull, sample_xy, 0).r;
            let sample_normal = decode_normal(textureLoad(tNormalFull, sample_xy, 0));

            let w_spatial = exp(-dot(offset, offset) * 0.5);
            let w_depth = exp(-abs(sample_depth - center_depth) / sigma_d);
            let w_normal = exp(-(1.0 - dot(sample_normal, center_normal)) / sigma_n);
            let weight = w_spatial * w_depth * w_normal;
            accum += half_rgb * weight;
            weight_sum += weight;
        }
    }

    let color = accum / max(weight_sum, 1e-4);
    textureStore(outFull, gid.xy, vec4<f32>(color, 1.0));
}

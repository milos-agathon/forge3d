// src/shaders/filters/edge_aware_upsample.wgsl
// P5.2: Edge-aware upsample from half to full resolution

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

@group(0) @binding(0) var tHalf: texture_2d<f32>;
@group(0) @binding(1) var outFull: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var sLinear: sampler;
@group(0) @binding(3) var tDepthFull: texture_2d<f32>;
@group(0) @binding(4) var tNormalFull: texture_2d<f32>;
@group(0) @binding(5) var<uniform> uSsgi: SsgiSettings;

fn unpack_normal(packed: vec4<f32>) -> vec3<f32> {
    return normalize(packed.xyz * 2.0 - 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs_edge_aware_upsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let xy = gid.xy;
    let dims = textureDimensions(outFull);
    if (xy.x >= dims.x || xy.y >= dims.y) { return; }

    let uv = (vec2<f32>(xy) + vec2<f32>(0.5)) / vec2<f32>(dims);

    let center_d = textureLoad(tDepthFull, xy, 0).r;
    let center_n = unpack_normal(textureLoad(tNormalFull, xy, 0));

    var sum = vec3<f32>(0.0);
    var wsum = 0.0;

    // 3x3 kernel around uv sampling half-res
    let sigma_d = max(uSsgi.upsample_depth_sigma, 1e-4);
    let n_exp = max(uSsgi.upsample_normal_exp, 0.0);

    for (var j: i32 = -1; j <= 1; j = j + 1) {
        for (var i: i32 = -1; i <= 1; i = i + 1) {
            let offs = vec2<f32>(f32(i), f32(j)) / vec2<f32>(dims);
            let suv = clamp(uv + offs, vec2<f32>(0.0), vec2<f32>(1.0));
            let half_dims = textureDimensions(tHalf);
            let half_px = vec2<u32>(suv * vec2<f32>(half_dims));
            let half_rgb = textureLoad(tHalf, half_px, 0).rgb;

            let sp = vec2<u32>(clamp(vec2<f32>(xy) + vec2<f32>(f32(i), f32(j)), vec2<f32>(0.0), vec2<f32>(dims) - vec2<f32>(1.0)));
            let sd = textureLoad(tDepthFull, sp, 0).r;
            let sn = unpack_normal(textureLoad(tNormalFull, sp, 0));

            let w_spatial = exp(-dot(offs, offs) * 4.0);
            let w_depth = exp(-abs(sd - center_d) / sigma_d);
            let w_normal = pow(max(dot(sn, center_n), 0.0), n_exp);
            let w = w_spatial * w_depth * w_normal;
            sum += half_rgb * w;
            wsum += w;
        }
    }

    let rgb = sum / max(wsum, 1e-5);
    textureStore(outFull, xy, vec4<f32>(rgb, 1.0));
}

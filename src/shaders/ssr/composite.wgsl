// src/shaders/ssr/composite.wgsl
// Add SSR contribution into the main color buffer prior to tonemapping

struct CompositeParams {
    boost: f32,
    exposure: f32,
    gamma: f32,
    weight_floor: f32,
    tone_white: f32,
    tone_bias: f32,
    reinhard_k: f32,
    _pad0: f32,
}

@group(0) @binding(0) var base_color: texture_2d<f32>;
@group(0) @binding(1) var ssr_final: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: CompositeParams;
@group(0) @binding(3) var composite_out: texture_storage_2d<rgba8unorm, write>;

fn tone_map(color: vec3<f32>, exposure: f32, white: f32, k: f32) -> vec3<f32> {
    let scaled = color * exposure;
    let numerator = scaled * (k * scaled + vec3<f32>(1.0));
    let denom = scaled + vec3<f32>(white);
    return numerator / denom;
}

@compute @workgroup_size(8, 8, 1)
fn cs_ssr_composite(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(base_color);
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    let base = textureLoad(base_color, pixel, 0).rgb;
    let spec_sample = textureLoad(ssr_final, pixel, 0);
    let spec_rgb = spec_sample.rgb;
    let weight = max(spec_sample.a, params.weight_floor);
    let boosted = spec_rgb * (params.boost * weight);

    let combined = base + boosted;
    let tone = tone_map(combined, params.exposure, params.tone_white, params.reinhard_k);
    let gamma = max(params.gamma, 0.001);
    let corrected = pow(tone, vec3<f32>(1.0 / gamma));
    textureStore(composite_out, pixel, vec4<f32>(clamp(corrected, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}

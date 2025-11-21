// src/shaders/ssr/composite.wgsl
// Add SSR contribution into the main color buffer prior to tonemapping.
// base_color   : pre-tonemap lit buffer (direct + IBL diffuse/spec) sampled from viewer's
//                `lit_output_view`.
// ssr_final    : Rgba16Float specular buffer after fallback/temporal resolve
//                (rgb = SSR+env spec, a encodes reflection weight/miss info).
// composite_out: Rgba8Unorm LDR view used by the P5 SSR harness for visualization.

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
    let raw_alpha = spec_sample.a;
    // Pixels with alpha <= 0.0 are treated as env-only misses and should not
    // add extra specular on top of the base lighting. For hits, we use the
    // stored alpha with a small floor as before.
    let is_miss = raw_alpha <= 0.0;
    if (is_miss) {
        textureStore(
            composite_out,
            pixel,
            vec4<f32>(clamp(base, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0),
        );
        return;
    }
    let weight = max(raw_alpha, params.weight_floor);
    let boosted = spec_rgb * (params.boost * weight);

    let combined = base + boosted;
    let tone = tone_map(combined, params.exposure, params.tone_white, params.reinhard_k);
    let gamma = max(params.gamma, 0.001);
    let corrected = pow(tone, vec3<f32>(1.0 / gamma));
    textureStore(
        composite_out,
        pixel,
        vec4<f32>(clamp(corrected, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0),
    );
}

// shaders/ssao.wgsl
// SSAO shader draft for Workstream B implementation.
// Provides ambient occlusion estimation and blur pass over the G-buffer.
// RELEVANT FILES:src/core/postfx.rs,python/forge3d/postfx.py,tests/test_b3_ssao.py

struct SsaoSettings {
    radius: f32;
    intensity: f32;
    bias: f32;
    _pad0: f32;
    inv_resolution: vec2<f32>;
    _pad1: vec2<f32>;
};

@group(0) @binding(0) var normal_depth_tex: texture_2d<f32>;
@group(0) @binding(1) var normal_depth_sampler: sampler;
@group(0) @binding(2) var<storage, write> ssao_output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> settings: SsaoSettings;

const SSAO_KERNEL: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(0.8660254, 0.5),
    vec2<f32>(0.8660254, -0.5),
    vec2<f32>(0.0, -1.0),
    vec2<f32>(-0.8660254, -0.5),
    vec2<f32>(-0.8660254, 0.5),
    vec2<f32>(0.5, 0.0),
    vec2<f32>(-0.5, 0.0)
);

@group(1) @binding(0) var ssao_input: texture_2d<f32>;
@group(1) @binding(1) var ssao_sampler: sampler;
@group(1) @binding(2) var<storage, write> ssao_blur_output: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(3) var<uniform> blur_settings: SsaoSettings;
@group(1) @binding(4) var normal_depth_tex_blur: texture_2d<f32>;
@group(1) @binding(5) var normal_depth_sampler_blur: sampler;

@group(2) @binding(0) var color_storage: texture_storage_2d<rgba8unorm, read_write>;
@group(2) @binding(1) var ssao_blurred_tex: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn cs_ssao(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(normal_depth_tex, 0);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let resolution = vec2<f32>(dims);
    let uv = (vec2<f32>(global_id.xy) + vec2<f32>(0.5, 0.5)) / resolution;
    let center_sample = textureSample(normal_depth_tex, normal_depth_sampler, uv);
    let center_normal = normalize(center_sample.xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0));
    let center_depth = center_sample.w;

    var occlusion = 0.0;
    let radius_uv = settings.radius * settings.inv_resolution;
    for (var i = 0u; i < SSAO_KERNEL.length(); i = i + 1u) {
        let sample_offset = SSAO_KERNEL[i] * radius_uv;
        let sample_uv = clamp(uv + sample_offset, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
        let sample_data = textureSample(normal_depth_tex, normal_depth_sampler, sample_uv);
        let sample_depth = sample_data.w;
        let sample_normal = normalize(sample_data.xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0));

        let depth_difference = sample_depth - center_depth;
        let occlude = step(settings.bias, depth_difference);
        let range_falloff = 1.0 - clamp(abs(depth_difference) / (settings.radius + 1e-4), 0.0, 1.0);
        let normal_weight = clamp(dot(center_normal, sample_normal), 0.0, 1.0);
        occlusion = occlusion + occlude * range_falloff * normal_weight;
    }

    let sample_count = f32(SSAO_KERNEL.length());
    let raw_ao = 1.0 - clamp((occlusion / sample_count) * settings.intensity, 0.0, 1.0);
    textureStore(ssao_output, vec2<i32>(global_id.xy), vec4<f32>(raw_ao, raw_ao, raw_ao, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_ssao_blur(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(ssao_input, 0);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let resolution = vec2<f32>(dims);
    let base_coord = vec2<i32>(global_id.xy);
    let uv_center = (vec2<f32>(global_id.xy) + vec2<f32>(0.5, 0.5)) / resolution;
    let center_sample = textureSample(normal_depth_tex_blur, normal_depth_sampler_blur, uv_center);
    let center_depth = center_sample.w;
    let center_normal = normalize(center_sample.xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0));

    var accum = 0.0;
    var weight_sum = 0.0;
    let offsets = array<vec2<i32>, 9>(
        vec2<i32>(-1, -1), vec2<i32>(0, -1), vec2<i32>(1, -1),
        vec2<i32>(-1, 0),  vec2<i32>(0, 0),  vec2<i32>(1, 0),
        vec2<i32>(-1, 1),  vec2<i32>(0, 1),  vec2<i32>(1, 1)
    );
    let spatial_weights = array<f32, 9>(1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0);

    for (var i = 0u; i < offsets.length(); i = i + 1u) {
        let offset = offsets[i];
        let sample_coord = clamp(base_coord + offset, vec2<i32>(0, 0), vec2<i32>(i32(dims.x - 1u), i32(dims.y - 1u)));
        let sample_uv = (vec2<f32>(sample_coord) + vec2<f32>(0.5, 0.5)) / resolution;
        let sample_data = textureSample(normal_depth_tex_blur, normal_depth_sampler_blur, sample_uv);
        let sample_depth = sample_data.w;
        let sample_normal = normalize(sample_data.xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0));
        let normal_weight = clamp(dot(center_normal, sample_normal), 0.0, 1.0);
        let depth_delta = abs(sample_depth - center_depth);
        let range_weight = exp(-depth_delta * (25.0 + blur_settings.bias * 200.0));
        let weight = spatial_weights[i] * range_weight * max(normal_weight, 0.1);
        let sample = textureLoad(ssao_input, sample_coord, 0).r;
        accum = accum + sample * weight;
        weight_sum = weight_sum + weight;
    }

    let blurred = accum / max(weight_sum, 1e-5);
    textureStore(ssao_blur_output, vec2<i32>(global_id.xy), vec4<f32>(blurred, blurred, blurred, 1.0));
}
@compute @workgroup_size(8, 8, 1)
fn cs_ssao_composite(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(color_storage);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let coord = vec2<i32>(global_id.xy);
    let color = textureLoad(color_storage, coord);
    let ao = textureLoad(ssao_blurred_tex, coord, 0).r;
    let occlusion = clamp(ao, 0.0, 1.0);
    let shaded = vec3<f32>(color.rgb) * occlusion;
    textureStore(color_storage, coord, vec4<f32>(shaded, color.a));
}

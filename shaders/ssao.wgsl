// shaders/ssao.wgsl
// SSAO shader draft for Workstream B implementation.
// Provides ambient occlusion estimation and blur pass over the G-buffer.
// RELEVANT FILES:src/core/postfx.rs,python/forge3d/postfx.py,tests/test_b3_ssao.py

struct SsaoSettings {
    radius: f32,
    intensity: f32,
    bias: f32,
    _pad0: f32,
    inv_resolution: vec2<f32>,
    _pad1: vec2<f32>,
};

@group(0) @binding(0) var normal_depth_tex: texture_2d<f32>;
@group(0) @binding(1) var ssao_output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> settings: SsaoSettings;

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

// TEMPORARILY DISABLED: Group 1 bindings for blur function
// @group(1) @binding(0) var ssao_input: texture_2d<f32>;
// @group(1) @binding(1) var ssao_blur_output: texture_storage_2d<rgba8unorm, write>;
// @group(1) @binding(2) var<uniform> blur_settings: SsaoSettings;
// @group(1) @binding(3) var normal_depth_tex_blur: texture_2d<f32>;

@group(2) @binding(0) var color_input: texture_2d<f32>;
@group(2) @binding(1) var color_storage: texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(2) var ssao_blurred_tex: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn cs_ssao(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(normal_depth_tex, 0);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let resolution = vec2<f32>(dims);
    let center_coord = vec2<i32>(global_id.xy);
    let center_sample = textureLoad(normal_depth_tex, center_coord, 0);
    let center_normal = normalize(center_sample.xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0));
    let center_depth = center_sample.w;

    var occlusion = 0.0;
    let radius_uv = settings.radius * settings.inv_resolution;

    // Unrolled loop for 8 kernel samples
    {
        let sample_offset = SSAO_KERNEL[0] * radius_uv;
        let sample_coord = clamp(center_coord + vec2<i32>(sample_offset * resolution), vec2<i32>(0, 0), vec2<i32>(i32(dims.x - 1u), i32(dims.y - 1u)));
        let sample_data = textureLoad(normal_depth_tex, sample_coord, 0);
        let sample_depth = sample_data.w;
        let sample_normal = normalize(sample_data.xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0));
        let depth_difference = sample_depth - center_depth;
        let occlude = step(settings.bias, depth_difference);
        let range_falloff = 1.0 - clamp(abs(depth_difference) / (settings.radius + 1e-4), 0.0, 1.0);
        let normal_weight = clamp(dot(center_normal, sample_normal), 0.0, 1.0);
        occlusion = occlusion + occlude * range_falloff * normal_weight;
    }
    {
        let sample_offset = SSAO_KERNEL[1] * radius_uv;
        let sample_coord = clamp(center_coord + vec2<i32>(sample_offset * resolution), vec2<i32>(0, 0), vec2<i32>(i32(dims.x - 1u), i32(dims.y - 1u)));
        let sample_data = textureLoad(normal_depth_tex, sample_coord, 0);
        let sample_depth = sample_data.w;
        let sample_normal = normalize(sample_data.xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0));
        let depth_difference = sample_depth - center_depth;
        let occlude = step(settings.bias, depth_difference);
        let range_falloff = 1.0 - clamp(abs(depth_difference) / (settings.radius + 1e-4), 0.0, 1.0);
        let normal_weight = clamp(dot(center_normal, sample_normal), 0.0, 1.0);
        occlusion = occlusion + occlude * range_falloff * normal_weight;
    }
    {
        let sample_offset = SSAO_KERNEL[2] * radius_uv;
        let sample_coord = clamp(center_coord + vec2<i32>(sample_offset * resolution), vec2<i32>(0, 0), vec2<i32>(i32(dims.x - 1u), i32(dims.y - 1u)));
        let sample_data = textureLoad(normal_depth_tex, sample_coord, 0);
        let sample_depth = sample_data.w;
        let sample_normal = normalize(sample_data.xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0));
        let depth_difference = sample_depth - center_depth;
        let occlude = step(settings.bias, depth_difference);
        let range_falloff = 1.0 - clamp(abs(depth_difference) / (settings.radius + 1e-4), 0.0, 1.0);
        let normal_weight = clamp(dot(center_normal, sample_normal), 0.0, 1.0);
        occlusion = occlusion + occlude * range_falloff * normal_weight;
    }
    {
        let sample_offset = SSAO_KERNEL[3] * radius_uv;
        let sample_coord = clamp(center_coord + vec2<i32>(sample_offset * resolution), vec2<i32>(0, 0), vec2<i32>(i32(dims.x - 1u), i32(dims.y - 1u)));
        let sample_data = textureLoad(normal_depth_tex, sample_coord, 0);
        let sample_depth = sample_data.w;
        let sample_normal = normalize(sample_data.xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0));
        let depth_difference = sample_depth - center_depth;
        let occlude = step(settings.bias, depth_difference);
        let range_falloff = 1.0 - clamp(abs(depth_difference) / (settings.radius + 1e-4), 0.0, 1.0);
        let normal_weight = clamp(dot(center_normal, sample_normal), 0.0, 1.0);
        occlusion = occlusion + occlude * range_falloff * normal_weight;
    }
    {
        let sample_offset = SSAO_KERNEL[4] * radius_uv;
        let sample_coord = clamp(center_coord + vec2<i32>(sample_offset * resolution), vec2<i32>(0, 0), vec2<i32>(i32(dims.x - 1u), i32(dims.y - 1u)));
        let sample_data = textureLoad(normal_depth_tex, sample_coord, 0);
        let sample_depth = sample_data.w;
        let sample_normal = normalize(sample_data.xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0));
        let depth_difference = sample_depth - center_depth;
        let occlude = step(settings.bias, depth_difference);
        let range_falloff = 1.0 - clamp(abs(depth_difference) / (settings.radius + 1e-4), 0.0, 1.0);
        let normal_weight = clamp(dot(center_normal, sample_normal), 0.0, 1.0);
        occlusion = occlusion + occlude * range_falloff * normal_weight;
    }
    {
        let sample_offset = SSAO_KERNEL[5] * radius_uv;
        let sample_coord = clamp(center_coord + vec2<i32>(sample_offset * resolution), vec2<i32>(0, 0), vec2<i32>(i32(dims.x - 1u), i32(dims.y - 1u)));
        let sample_data = textureLoad(normal_depth_tex, sample_coord, 0);
        let sample_depth = sample_data.w;
        let sample_normal = normalize(sample_data.xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0));
        let depth_difference = sample_depth - center_depth;
        let occlude = step(settings.bias, depth_difference);
        let range_falloff = 1.0 - clamp(abs(depth_difference) / (settings.radius + 1e-4), 0.0, 1.0);
        let normal_weight = clamp(dot(center_normal, sample_normal), 0.0, 1.0);
        occlusion = occlusion + occlude * range_falloff * normal_weight;
    }
    {
        let sample_offset = SSAO_KERNEL[6] * radius_uv;
        let sample_coord = clamp(center_coord + vec2<i32>(sample_offset * resolution), vec2<i32>(0, 0), vec2<i32>(i32(dims.x - 1u), i32(dims.y - 1u)));
        let sample_data = textureLoad(normal_depth_tex, sample_coord, 0);
        let sample_depth = sample_data.w;
        let sample_normal = normalize(sample_data.xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0));
        let depth_difference = sample_depth - center_depth;
        let occlude = step(settings.bias, depth_difference);
        let range_falloff = 1.0 - clamp(abs(depth_difference) / (settings.radius + 1e-4), 0.0, 1.0);
        let normal_weight = clamp(dot(center_normal, sample_normal), 0.0, 1.0);
        occlusion = occlusion + occlude * range_falloff * normal_weight;
    }
    {
        let sample_offset = SSAO_KERNEL[7] * radius_uv;
        let sample_coord = clamp(center_coord + vec2<i32>(sample_offset * resolution), vec2<i32>(0, 0), vec2<i32>(i32(dims.x - 1u), i32(dims.y - 1u)));
        let sample_data = textureLoad(normal_depth_tex, sample_coord, 0);
        let sample_depth = sample_data.w;
        let sample_normal = normalize(sample_data.xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0));
        let depth_difference = sample_depth - center_depth;
        let occlude = step(settings.bias, depth_difference);
        let range_falloff = 1.0 - clamp(abs(depth_difference) / (settings.radius + 1e-4), 0.0, 1.0);
        let normal_weight = clamp(dot(center_normal, sample_normal), 0.0, 1.0);
        occlusion = occlusion + occlude * range_falloff * normal_weight;
    }

    let sample_count = f32(8u);
    let raw_ao = 1.0 - clamp((occlusion / sample_count) * settings.intensity, 0.0, 1.0);
    textureStore(ssao_output, vec2<i32>(global_id.xy), vec4<f32>(raw_ao, raw_ao, raw_ao, 1.0));
}

// TEMPORARILY DISABLED: Blur function has group binding issues
// @compute @workgroup_size(8, 8, 1)
// fn cs_ssao_blur(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Function body temporarily disabled due to group binding issues
    // let dims = textureDimensions(ssao_input, 0);
    // if (global_id.x >= dims.x || global_id.y >= dims.y) {
    //     return;
    // }
// }
@compute @workgroup_size(8, 8, 1)
fn cs_ssao_composite(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(color_storage);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let coord = vec2<i32>(global_id.xy);
    let color = textureLoad(color_input, coord, 0);
    let ao = textureLoad(ssao_blurred_tex, coord, 0).r;
    let occlusion = clamp(ao, 0.0, 1.0);
    let shaded = vec3<f32>(color.rgb) * occlusion;
    textureStore(color_storage, coord, vec4<f32>(shaded, color.a));
}

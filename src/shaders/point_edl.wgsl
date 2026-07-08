//! Eye-Dome Lighting post-process for point-cloud/vector OIT overlays.

struct EdlUniforms {
    texel_size: vec2<f32>,
    strength: f32,
    radius_px: f32,
}

@group(0) @binding(0)
var color_tex: texture_2d<f32>;

@group(0) @binding(1)
var depth_tex: texture_depth_2d;

@group(0) @binding(2)
var color_sampler: sampler;

@group(0) @binding(3)
var<uniform> edl: EdlUniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vertex_index & 2u) * 2.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x, -y) * 0.5 + 0.5;
    return out;
}

fn clamped_depth(coords: vec2<i32>, dims: vec2<i32>) -> f32 {
    let p = clamp(coords, vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    return textureLoad(depth_tex, p, 0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(color_tex, color_sampler, in.uv);
    if (color.a <= 0.001) {
        return color;
    }

    let dims = vec2<i32>(textureDimensions(depth_tex));
    let center_px = clamp(vec2<i32>(in.clip_position.xy), vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    let center_depth = clamped_depth(center_px, dims);
    if (center_depth >= 0.999999) {
        return color;
    }

    let radius = max(1, i32(round(max(edl.radius_px, 1.0))));
    var response = 0.0;
    let n0 = clamped_depth(center_px + vec2<i32>(radius, 0), dims);
    let n1 = clamped_depth(center_px + vec2<i32>(-radius, 0), dims);
    let n2 = clamped_depth(center_px + vec2<i32>(0, radius), dims);
    let n3 = clamped_depth(center_px + vec2<i32>(0, -radius), dims);
    let n4 = clamped_depth(center_px + vec2<i32>(radius, radius), dims);
    let n5 = clamped_depth(center_px + vec2<i32>(-radius, radius), dims);
    let n6 = clamped_depth(center_px + vec2<i32>(radius, -radius), dims);
    let n7 = clamped_depth(center_px + vec2<i32>(-radius, -radius), dims);
    response = response + max(0.0, n0 - center_depth);
    response = response + max(0.0, n1 - center_depth);
    response = response + max(0.0, n2 - center_depth);
    response = response + max(0.0, n3 - center_depth);
    response = response + max(0.0, n4 - center_depth);
    response = response + max(0.0, n5 - center_depth);
    response = response + max(0.0, n6 - center_depth);
    response = response + max(0.0, n7 - center_depth);

    let shade = exp(-response * max(edl.strength, 0.0) * 64.0);
    return vec4<f32>(color.rgb * shade, color.a);
}

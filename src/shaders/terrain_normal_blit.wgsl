// src/shaders/terrain_normal_blit.wgsl
// Fullscreen blit shader for terrain normal AOV resolves.
// RELEVANT FILES: src/shaders/terrain_blit.wgsl, src/terrain/renderer/aov.rs

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_id : u32) -> VertexOutput {
    var out : VertexOutput;

    let uv_x = f32((vertex_id << 1u) & 2u);
    let uv_y = f32(vertex_id & 2u);
    let uv = vec2<f32>(uv_x, uv_y);

    out.clip_position = vec4<f32>(uv * vec2<f32>(2.0, 2.0) - vec2<f32>(1.0, 1.0), 0.0, 1.0);
    out.uv = uv * 0.5;
    return out;
}

@group(0) @binding(0)
var source_tex : texture_2d<f32>;

@group(0) @binding(1)
var source_sampler : sampler;

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
    let sampled = textureSample(source_tex, source_sampler, input.uv);
    let len_sq = dot(sampled.xyz, sampled.xyz);
    if (len_sq <= 1e-8) {
        return sampled;
    }
    return vec4<f32>(normalize(sampled.xyz), sampled.w);
}

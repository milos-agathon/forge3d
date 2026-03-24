struct OfflineLuminanceUniforms {
    width: u32,
    height: u32,
    sample_count: u32,
    _pad: u32,
}

@group(0) @binding(0) var accumulation: texture_2d<f32>;
@group(0) @binding(1) var luminance_out: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> uniforms: OfflineLuminanceUniforms;

fn luminance(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_width = (uniforms.width + 3u) / 4u;
    let out_height = (uniforms.height + 3u) / 4u;
    if (gid.x >= out_width || gid.y >= out_height) {
        return;
    }

    let base = vec2<i32>(gid.xy) * 4;
    var sum: f32 = 0.0;
    var count: u32 = 0u;
    for (var dy: i32 = 0; dy < 4; dy = dy + 1) {
        for (var dx: i32 = 0; dx < 4; dx = dx + 1) {
            let src = base + vec2<i32>(dx, dy);
            if (src.x < i32(uniforms.width) && src.y < i32(uniforms.height)) {
                let avg = textureLoad(accumulation, src, 0) / f32(max(uniforms.sample_count, 1u));
                sum = sum + luminance(avg.rgb);
                count = count + 1u;
            }
        }
    }

    let mean = sum / f32(max(count, 1u));
    textureStore(luminance_out, vec2<i32>(gid.xy), vec4<f32>(mean, 0.0, 0.0, 1.0));
}

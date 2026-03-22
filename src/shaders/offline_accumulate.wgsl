// TV12: Additive ping-pong accumulation for offline terrain rendering.
//
// Reads current sample + previous accumulation, writes sum to next accumulation.
// Division by sample count happens at resolve time, not here.
// Ping-pong between two textures avoids the ReadWrite storage limitation
// that the previous accumulation_blend.wgsl could not work around.

@group(0) @binding(0) var current_sample: texture_2d<f32>;
@group(0) @binding(1) var prev_accumulation: texture_2d<f32>;
@group(0) @binding(2) var next_accumulation: texture_storage_2d<rgba32float, write>;

struct AccumParams {
    width: u32,
    height: u32,
    sample_index: u32,
    _pad: u32,
}
@group(0) @binding(3) var<uniform> params: AccumParams;

@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let current = textureLoad(current_sample, coords, 0);

    var result: vec4<f32>;
    if (params.sample_index == 0u) {
        // First sample: just store it (no previous accumulation)
        result = current;
    } else {
        // Subsequent samples: add to running sum
        let prev = textureLoad(prev_accumulation, coords, 0);
        result = prev + current;
    }
    textureStore(next_accumulation, coords, result);
}

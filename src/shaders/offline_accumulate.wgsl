struct OfflineAccumulateUniforms {
    sample_index: u32,
    width: u32,
    height: u32,
    _pad: u32,
}

@group(0) @binding(0) var current_sample: texture_2d<f32>;
@group(0) @binding(1) var prev_accumulation: texture_2d<f32>;
@group(0) @binding(2) var next_accumulation: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> uniforms: OfflineAccumulateUniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    let coords = vec2<i32>(gid.xy);
    let sample_value = textureLoad(current_sample, coords, 0);
    let prev_value = textureLoad(prev_accumulation, coords, 0);
    textureStore(next_accumulation, coords, prev_value + sample_value);
}

struct OfflineTonemapUniforms {
    width: u32,
    height: u32,
    operator_index: u32,
    _pad0: u32,
    white_point: f32,
    gamma: f32,
    _pad1: vec2<f32>,
}

@group(0) @binding(0) var hdr_input: texture_2d<f32>;
@group(0) @binding(1) var ldr_output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> uniforms: OfflineTonemapUniforms;

fn apply_operator(color: vec3<f32>) -> vec3<f32> {
    return tonemap_apply_operator(color, uniforms.operator_index, uniforms.white_point);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    let coords = vec2<i32>(gid.xy);
    let hdr = textureLoad(hdr_input, coords, 0);
    let mapped = apply_operator(hdr.rgb);
    let encoded = linear_to_srgb(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)));
    textureStore(ldr_output, coords, vec4<f32>(encoded, clamp(hdr.a, 0.0, 1.0)));
}

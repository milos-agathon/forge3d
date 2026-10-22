@group(0) @binding(0) var base: texture_2d<f32>;
@group(0) @binding(1) var opaque: texture_2d<f32>;
@group(0) @binding(2) var lookup: texture_2d<u32>;
@group(0) @binding(3) var output_image: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= textureDimensions(base))) { return; }
    let below = vec4<u32>(round(textureLoad(base, gid.xy, 0) * 255.0));
    let above = vec4<u32>(round(textureLoad(opaque, gid.xy, 0) * 255.0));
    var result: vec4<f32>;
    for (var channel = 0u; channel < 4u; channel = channel + 1u) {
        result[channel] = f32(textureLoad(lookup, vec2<u32>(above[channel], below[channel]), 0).r) / 255.0;
    }
    textureStore(output_image, gid.xy, result);
}

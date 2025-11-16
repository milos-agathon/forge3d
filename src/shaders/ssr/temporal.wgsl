// src/shaders/ssr/temporal.wgsl
// Simple exponential moving average for SSR output

@group(0) @binding(0) var current_ssr: texture_2d<f32>;
@group(0) @binding(1) var history_ssr: texture_2d<f32>;
@group(0) @binding(2) var filtered_ssr: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(current_ssr);
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    let cur = textureLoad(current_ssr, pixel, 0);
    let hist = textureLoad(history_ssr, pixel, 0);
    let blended = mix(cur, hist, 0.2);
    textureStore(filtered_ssr, pixel, blended);
}

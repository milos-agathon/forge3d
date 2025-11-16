// src/shaders/ssr/composite.wgsl
// Add SSR contribution into the main color buffer prior to tonemapping

@group(0) @binding(0) var base_color: texture_2d<f32>;
@group(0) @binding(1) var ssr_final: texture_2d<f32>;
@group(0) @binding(2) var composite_out: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_ssr_composite(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(base_color);
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    let base = textureLoad(base_color, pixel, 0).rgb;
    let spec = textureLoad(ssr_final, pixel, 0).rgb;
    let summed = clamp(base + spec, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(composite_out, pixel, vec4<f32>(summed, 1.0));
}

// GTAO kernel entry point (uses helpers from common.wgsl)

@compute @workgroup_size(8, 8, 1)
fn cs_gtao(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(depth_texture);
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(dims);
    let depth = textureLoad(depth_texture, pixel, 0).r;

    if (depth <= 1e-6) {
        textureStore(output_ao, pixel, vec4<f32>(1.0));
        return;
    }

    let normal_packed = textureLoad(normal_texture, pixel, 0);
    let view_pos = reconstruct_view_pos_linear(uv, depth);
    let view_normal = unpack_normal(normal_packed);

    let ao = compute_gtao(pixel, uv, view_pos, view_normal);
    textureStore(output_ao, pixel, vec4<f32>(ao));
}

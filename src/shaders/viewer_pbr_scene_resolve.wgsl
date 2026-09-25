// viewer_pbr_scene_resolve.wgsl
// SSAA tent-filter resolve for the viewer PBR scene stage.
// Mirrors the CPU filter the adjudication raster adapter applied:
//   w_k = 1 - |((k+0.5)/S - 0.5)| / 0.5
//   out = sum_sy sum_sx src * w_y * w_x / (sum w)^2, alpha = 1
// RELEVANT FILES: src/viewer/pbr_scene/mod.rs, src/core/tonemap.rs

struct ResolveParams {
    ssaa: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: ResolveParams;

fn tent_weight(k: u32, s: u32) -> f32 {
    let sk = (f32(k) + 0.5) / f32(s);
    return 1.0 - abs(sk - 0.5) / 0.5;
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let s = params.ssaa;
    var weight_sum = 0.0;
    for (var k = 0u; k < s; k = k + 1u) {
        weight_sum += tent_weight(k, s);
    }
    let normalization = weight_sum * weight_sum;
    var acc = vec4<f32>(0.0);
    for (var sy = 0u; sy < s; sy = sy + 1u) {
        let wy = tent_weight(sy, s);
        for (var sx = 0u; sx < s; sx = sx + 1u) {
            let wx = tent_weight(sx, s);
            let p = textureLoad(
                src_tex,
                vec2<i32>(i32(gid.x * s + sx), i32(gid.y * s + sy)),
                0
            );
            // Keep the CPU loop's exact per-term order: p * wy * wx / norm.
            acc += p * wy * wx / normalization;
        }
    }
    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(acc.rgb, 1.0));
}

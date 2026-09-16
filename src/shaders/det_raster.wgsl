// src/shaders/det_raster.wgsl
// TERRA-DETERMINATA raster canary.
//
// A minimal real RASTER pass (fullscreen triangle -> 64x64 rgba32float) whose
// fragment math runs through the pinned det_* helpers. The same WGSL runs
// natively (forge3d.determinism_probe raster leg) and in browser WebGPU
// (tools/determinism_browser/det_probe.js), and the readback bytes are hashed
// as `raster_sha256` — an executed cross-implementation raster-identity leg.
// Inputs derive only from the fragment position; no uniforms, no textures.
//
// The target is rgba32float, NOT rgba8unorm: an 8-bit unorm write quantizes
// every channel to 1/255, which hides single-ULP f32 differences entirely.
// Raw f32 readback is compared bit-for-bit.
//
// Every mul->add / add->add / mul->mul edge is barriered per the IR lint
// (src/verify/determinism_lint.rs): raw products feeding adds are fma
// contraction candidates and bare add chains are reassociation candidates.

struct RasterOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_id: u32) -> RasterOut {
    det_seed(f32(vertex_id));
    var out: RasterOut;
    let uv_x = f32((vertex_id << 1u) & 2u);
    let uv_y = f32(vertex_id & 2u);
    let uv = vec2<f32>(uv_x, uv_y);
    out.pos = vec4<f32>(det_fma2(uv, vec2<f32>(2.0, 2.0), vec2<f32>(-1.0, -1.0)), 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(inp: RasterOut) -> @location(0) vec4<f32> {
    det_seed(inp.pos.x);
    // uv in [0, 2]; bring into a spread of operand classes.
    let a = inp.uv.x - 1.0;
    let b = det_barrier(inp.uv.y - 0.375);
    let v3 = vec3<f32>(a, b, det_fma(a, 0.5, det_barrier(b * -0.25)));
    let r = det_fma(a, b, det_dot3(v3, vec3<f32>(0.5, 0.25, 0.75)));
    let g = det_fma(
        det_sqrt(abs(a) + 0.125),
        det_rcp(b + 2.0),
        det_barrier(det_length3(v3) * 0.1),
    );
    // Caller products are barriered before entering det_sin/det_cos: each
    // helper multiplies its argument again internally, so a bare product arg
    // would contract into the helper's own mul chain after inlining.
    let bl = det_fma(
        det_sin(det_barrier(a * 2.1)),
        0.5,
        det_fma(
            det_cos(det_barrier(b * 1.7)),
            0.5,
            det_barrier(det_pow(abs(b) + 0.25, 1.3) * 0.05),
        ),
    );
    let w = det_fma(
        det_atan2(a, b + 1.5),
        det_rcp(3.14159265359),
        det_smoothstep(-1.0, 1.0, a),
    );
    return vec4<f32>(r, g, bl, w);
}

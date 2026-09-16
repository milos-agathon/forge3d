// src/shaders/det_probe.wgsl
// TERRA-DETERMINATA arithmetic canary.
//
// A standalone compute shader that exercises the pinned det_* helper battery
// (plus the few raw ops that remain in the canonical path by design: adds,
// multiplies, floor/fract/trunc) on lane-varying inputs derived ONLY from the
// lane index — no uniforms, no textures, no atomics, no wall clock. The host
// hashes the raw output-buffer bytes (64 x f32) as `probe_sha256`; any backend
// that lowers a det_* helper differently shows up as a hash mismatch, and the
// per-slot values localize which operation diverged.
//
// The SAME WGSL source runs natively (via the forge3d.determinism_probe
// pyfunction) and in browser WebGPU (tools/determinism_browser/det_probe.js),
// so the probe hash is a cross-implementation arithmetic-identity leg that is
// far cheaper to run on a browser than a full terrain frame.

@group(0) @binding(0)
var<storage, read_write> probe_out: array<vec4<f32>, 16>;

// Lane inputs: a, b, c span different magnitudes/signs so det_rcp, det_sqrt,
// det_log2, det_atan2, det_exp2 each see a spread of operand classes.
// The mul-adds go through det_fma: a bare i*k + c is a contraction candidate
// (FXC emits a fused mad where naga->SPIR-V keeps separate FMul/FAdd), which
// would diverge the inputs before the helper battery even runs.
fn probe_inputs(lane: u32) -> vec3<f32> {
    let i = f32(lane);
    // Barrier every det_fma result: its tail is an unbarriered add, so a
    // caller-side +/- extends the add tree and a driver JIT may reassociate
    // (measured 2026-09-14: Dawn/tint folded `a + 0.004913 - 0.125` into
    // `a + (0.004913 - 0.125)`, diverging browser-vs-native by 1 ULP).
    let a = det_barrier(det_barrier(det_fma(i, 0.1234517, -2.0)) + 0.004913);
    let b = det_barrier(det_fma(i, -0.077777, 1.6180339887));
    let c = det_barrier(det_fma(i, 0.03125, -0.25));
    return vec3<f32>(a, b, c);
}

@compute @workgroup_size(4, 4, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    det_seed(f32(gid.x));
    let lane = gid.x + gid.y * 4u;
    if (lane >= 16u) {
        return;
    }
    let inp = probe_inputs(lane);
    let a = inp.x;
    let b = inp.y;
    let c = inp.z;
    let v3 = vec3<f32>(a, b, c);
    // Raw-mul elements are barriered at construction: a bare product stored in
    // a vector/matrix and consumed later by an add (w3 + 4.0) or by a helper's
    // internal mul chain is still a contraction/reassociation surface.
    let w3 = vec3<f32>(det_barrier(b * 0.5), det_barrier(c + 0.75), det_barrier(a - 0.125));
    let n3 = det_normalize3(v3);
    let m4 = mat4x4<f32>(
        vec4<f32>(1.0, det_barrier(a * 0.03125), 0.0, 0.0),
        vec4<f32>(det_barrier(b * 0.03125), 1.0, det_barrier(c * 0.03125), 0.0),
        vec4<f32>(0.0, det_barrier(c * 0.0625), 1.0, 0.0),
        vec4<f32>(a, b, c, 1.0),
    );

    // Group 0: reductions, barriers, and fused products.
    let g0a = det_barrier(det_fma(a, b, c));
    let g0b = det_barrier(det_dot3(v3, w3));
    let g0c = det_barrier(det_dot4(m4[0], vec4<f32>(a, b, c, 0.5)));
    let g0d = det_barrier(det_barrier(a * b) + det_barrier3(v3 * w3).x);
    // Group 1: divisions, roots, norms. Partial sums are barriered: a bare
    // three-term add is an add tree the driver can reassociate into a
    // differently-rounded order.
    let g1a = det_div(a, b + 3.0);
    let g1b = det_rcp(b + 3.0);
    let g1c = det_sqrt(abs(a) + 0.5);
    let g1d = det_barrier(det_barrier(det_inverse_sqrt(abs(c) + 0.125) + det_length3(v3)) + det_distance3(v3, w3));
    // Group 2: transcendentals. Raw-mul args are barriered where the callee
    // multiplies again internally (det_exp/det_sin/det_cos), so the product
    // cannot be reassociated into the helper's own mul chain.
    let g2a = det_barrier(det_exp2(c) + det_barrier(det_log2(abs(a) + 0.75)));
    let g2b = det_barrier(det_pow(abs(b) + 0.25, 1.7) + det_exp(det_barrier(c * 0.5)));
    let g2c = det_barrier(det_sin(det_barrier(a * 3.1)) + det_cos(det_barrier(b * 2.3)));
    let g2d = det_barrier(det_barrier(det_barrier(det_atan2(a, b) + det_acos(clamp(c, -1.0, 1.0))) + det_asin(clamp(a * 0.3, -1.0, 1.0))) + det_atan01(abs(c)));
    // Group 3: vector/structural helpers.
    let g3a = det_barrier(det_barrier(det_mix(a, b, abs(c))) + det_smoothstep(-1.0, 2.0, a));
    let g3b = det_barrier(det_barrier(det_cross3(v3, w3).y) + det_barrier(det_reflect3(n3, w3).z));
    let g3c = det_barrier(det_mat4_mul_vec4(m4, vec4<f32>(a, b, c, 1.0)).w);
    // Each helper result is barriered at the extraction site: a helper value
    // consumed only through a swizzle inline can be scalarized and re-fused
    // into the surrounding adds by the driver JIT (measured 2026-07-16: the
    // inline spelling diverged dx12-vs-vulkan; barriered operands make the
    // adds see opaque bitcast results, not contractable products).
    let g3d = det_barrier(det_barrier(n3.x + det_barrier(det_div3(v3, vec3<f32>(w3 + vec3<f32>(4.0))).z)) + det_barrier(det_rcp3(abs(w3) + vec3<f32>(2.0)).x));

    probe_out[lane] = vec4<f32>(
        det_barrier(det_barrier(g0a + g0b) + g0c) + g0d,
        det_barrier(det_barrier(g1a + g1b) + g1c) + g1d,
        det_barrier(det_barrier(g2a + g2b) + g2c) + g2d,
        det_barrier(det_barrier(g3a + g3b) + g3c) + g3d,
    );
}

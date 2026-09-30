// src/shaders/includes/determinism.wgsl
// TERRA-DETERMINATA: pinned-order float helpers for cross-vendor bit-exact rendering.
// Loaded by Rust shader assembly (include_str!); WGSL has no preprocessor, so every
// module that calls a det_* helper must have this file concatenated exactly once
// ahead of it (see terrain pipeline_cache.rs, terrain offline.rs, core/tonemap.rs,
// hdr_offscreen/pipeline.rs, pbr/tone_mapping.rs, pbr/rendering.rs, pbr/shadow.rs).
//
// Every helper removes a degree of freedom the driver would otherwise have:
//  - contraction:      a * b + c may or may not become one hardware FMA per driver.
//    det_fma* splits the multiply and the add into separate statements so naga
//    emits them as distinct operations. This trades the one-ULP-better fused
//    result for cross-vendor identity.
//  - reduction-order:  dot(), normalize(), and matrix*vector products are sums
//    whose association order is unspecified. det_dot*/det_normalize*/det_mat4_*
//    spell out a fixed left-to-right reduction tree.
//  - transcendental:   pow/exp/log lowering differs per driver in ULP. det_pow/
//    det_exp/det_log2 centralize the choice: pow/exp are spelled as explicit
//    exp2/log2 compositions (measured a NO-OP on dx12-FXC and vulkan-NVIDIA,
//    which already lower pow/exp exactly this way — kept to pin vendors not
//    yet measured; see the Transcendental pins section).
//  - intrinsic formula: mix() has no single mandated lowering (HLSL lerp vs
//    SPIR-V FMix differ in mad/formula choice) and cross() is contraction-
//    prone inside the intrinsic. det_mix/det_mix3/det_cross3 spell both out
//    (measured a NO-OP on dx12-FXC/vulkan-NVIDIA — kept as a pin for vendors
//    not yet measured).
//  - per-API precision contracts: f32 divide, sqrt and inverseSqrt are
//    correctly rounded under D3D but only ~2.5 ULP under Vulkan, so ONE GPU
//    legitimately runs DIFFERENT refinement sequences per API. THIS was the
//    measured dx12/vulkan divergence on the canonical scene (52/262144
//    pixels one 8-bit LSB apart, root-caused 2026-07-10 to the shading-normal
//    Sobel chain). det_rcp/det_div/det_sqrt/det_inverse_sqrt replace the
//    native ops with bit-trick seeds + pinned Newton-Raphson; routing the
//    normal-chain divide/sqrt sites through them made dx12 == vulkan
//    byte-exact (the pinned hash moved with each later instrumentation
//    change; the current values are the committed per-scene goldens
//    under tests/goldens/determinism/).
//
// Residual-source enforcement policy:
//  - [S] fixed-function texture sampling remains in deterministic-path
//    modules. The naga-IR lint reports every ImageSample expression as a
//    hardware_sample advisory because converting an arbitrary sampling
//    contract to textureLoad is not safely automatable; the source rewriter
//    leaves those sites for manual work. Deterministic mode pins sampler
//    filtering to nearest through src/core/gpu.rs::deterministic_filter_mode,
//    removing vendor filter-weight arithmetic, but coordinate-to-texel
//    selection remains hardware-defined. Runtime canaries and the cross-vendor
//    hash matrix gate the resulting bytes. This is not an all-sampling-replaced
//    claim.
//  - [C] downstream fast-math/FMA: three layers. Source: every det_*
//    helper barriers its products through the runtime-opaque det_zu OR
//    (see det_barrier) and the IR lint rejects any unbarriered
//    mul->add / add->add / mul->mul edge plus raw divide/sqrt/dot/mix/
//    pow/exp/trig/mat*vec outside the det_* layer, so there is no
//    contractible edge left for a driver to fuse — this is what makes
//    Metal's forced fast-math harmless (wgpu-hal 0.19.5 sets only the
//    language version and preserve_invariance; with no unbarriered
//    contraction edge there is nothing for fast-math to change).
//    Backend: under FORGE3D_DETERMINISTIC a DX12-only instance sets the
//    wgpu DEBUG flag so wgpu-hal compiles HLSL with FXC
//    D3DCOMPILE_SKIP_OPTIMIZATION (-Od); measured cost ~490 s for the
//    terrain module on this machine vs ~1.8 s on Vulkan, so the
//    deterministic CI legs get generous timeouts. Detection: the
//    det_probe arithmetic canary (src/shaders/det_probe.wgsl) runs at
//    deterministic context start and its SHA-256 is compared against the
//    committed golden — a compiler that re-fuses a barriered mul-add or a
//    backend whose native exp2/log2/sqrt deviates from the pinned
//    software results changes the hash and the context refuses to render.
//    The same canary also runs in browser WebGPU
//    (tools/determinism_browser/) so naga->FXC/SPIR-V/MSL and Dawn/tint
//    are measured on one text.
//  - Seed discipline: det_zu initializes to 0u, so an UNSEEDED stage is
//    merely unpinned (the OR folds away) rather than wrong. The IR lint
//    makes a missing det_seed a hard failure: every entry point whose
//    call graph reaches det_zu must call det_seed first.
//
// Residual sources that remain hardware-defined and are covered only by
// the cross-vendor hash matrix (kept loud in review):
//  - Vertex varying interpolation and the rasterizer's clip->ndc divide are
//    fixed-function per vendor; no source-level pin exists.
//  - dpdx/dpdy (pinned to the Coarse variants in the terrain fs) differ in
//    value across vendors whenever the varyings they difference differ.
//
// ---------------------------------------------------------------------------
// Nondeterminism-site inventory (step 1 of the TERRA-DETERMINATA task).
// Risk classes: [C] contraction, [R] reduction-order, [T] transcendental ULP,
// [F] intrinsic-formula freedom (mix -> lerp/FMix, cross -> per-component
//     mad choices; no single mandated lowering across HLSL/SPIR-V/MSL),
// [P] per-API precision contract (f32 divide/sqrt/inverseSqrt: correctly
//     rounded under D3D, ~2.5 ULP under Vulkan — one GPU, two refinement
//     sequences; THE measured dx12/vulkan divergence class on this machine),
// [D] derivative choice (dpdx/dpdy coarse-vs-fine is implementation-defined;
//     pinned to the Coarse variants throughout the terrain fs).
//
// src/shaders/includes/tonemap_common.wgsl
//  - tonemap_filmic_terrain (curve + white_curve polynomials)      [C] highest priority: last shared terrain math before write-out
//  - tonemap_reinhard / _extended / _aces / _uncharted2 polynomials [C]
//  - tonemap_exposure: exp(-max(color,0))                           [T]
//  - gamma_correct: pow(color, 1/gamma)                             [T]
//  - linear_to_srgb: pow(clamped, 1/2.4)                            [T]
//
// src/shaders/lighting_ibl.wgsl
//  - eval_ibl: dot(n, v)                                            [R]
//  - eval_ibl: reflect(-v, n) (contains a dot reduction)            [R]
//  - fresnel_schlick_roughness: pow(1-cos, 5)                       [T]
//  - fresnel/split-sum mul-add chains                               [C]
//  - mip_level = r*r*9.0                                            [C]
//
// (csm.wgsl was deleted upstream; the shadow sites now live inside
// terrain_pbr_pom.wgsl as the sample_shadow_*_terrain family and in
// shadow_moments.wgsl:)
//  - calculate_depth_bias / select_cascade: normalize/dot reductions,
//    float boundary compares (ULP flip at edges)                    [R][C]
//  - world_to_light_space: mat4 x mat4 x vec4 chain                 [R]
//  - sample_shadow_pcf_terrain: += accumulation with data-dependent
//    tap set; textureSampleCompare hardware PCF replaced by
//    textureLoad + manual det_* compares                            [R][S]
//  - VSM/EVSM/MSM moments: exp(+-c*depth) warps                     [T]
//  - cascade transition / debug paths: mix()                        [C]
//
// src/shaders/terrain_pbr_pom.wgsl (fs hot path; ~92 sites total)
//  - normal reconstruction/blend: normalize() at 1578,1660,1682,
//    1694,1976-1978,2027,2038,2142,2146,2178-2232,2738,2783,3044,
//    3576,3666,4280                                                 [R]
//  - lighting dots: dot(n,l)/dot(n,v)/dot(n,h)/dot(v,h) at 801,806,
//    878,1083,2318-2323,2384-2392,2443-2453,2533,3278,3865-3905,4010 [R]
//  - luminance dots 2575,2696; plane distance 2728; normal-variance
//    dots 3222,3587,3672                                            [R]
//  - fresnel/specular pow at 806,879,1368,1704,2340,2364,2412,2473,
//    3267,3917                                                      [T]
//  - EVSM warp exp at 983-991; fog/absorption exp at 2669,2675,2690,
//    2891,3000                                                      [T]
//  - POM ray march / detail blends: sequential += in fixed loops    [R] order-safe as written (single accumulator, fixed trip order)
//  - sqrt at 3245,3593 (Toksvig): sqrt itself is order-safe; its
//    mul-add argument is pinned at the call site                    [C]
//  - mix(): 41 fs-path sites (19 scalar, 22 vec3) ROUTED to
//    det_mix/det_mix3 (gap-closure; measured a no-op on the
//    dx12/vulkan pair — kept as a vendor pin). Comment-only
//    mentions remain at 128, 3896.                                  [F]
//  - cross(): 3 live sites (geometry normal from ddx/ddy; tangent
//    frame construction) ROUTED to det_cross3                       [F][C]
//  - matrix-vector products: VS clip chain (proj*view*pos), fs
//    view/light-space transforms, TBN*vec — ROUTED through
//    det_mat4_mul_vec4/det_mat3_mul_vec3 (measured no-op on this
//    pair; kept as a vendor pin)                                    [R][C]
//  - dpdx/dpdy: 18 sites ROUTED to dpdxCoarse/dpdyCoarse            [D]
//  - divide/sqrt in the shading-normal chain (Sobel gradient
//    divisions, get_height_geom_t, texel_uv, blended-normal
//    length): ROUTED through det_div/det_sqrt/det_rcp; together
//    with det_inverse_sqrt in det_normalize*, this ZEROED the
//    52-pixel dx12/vulkan diff (2026-07-10). Divisions elsewhere
//    in the fs (fog, water, tonemap curves) remain native: they
//    were measured NON-divergent on this pair once the normal
//    chain was pinned; revisit per-vendor if a new pair diverges.   [P]
//
// src/shaders/ibl_equirect.wgsl / ibl_prefilter.wgsl / ibl_brdf.wgsl
// (IBL PRECOMPUTE — outside the original four-shader scope, pulled in by
// gap-closure because its output textures feed the deterministic hash):
//  - atan2/acos (equirect projection), sin/cos (hemisphere/GGX
//    importance sampling) ROUTED to det_atan2/det_acos/det_sin/
//    det_cos (measured no-op here: rgba16float targets absorb
//    trig ULP; kept as a vendor pin)                                [T]
//  - accumulation loops and normalize/cross/dot reductions remain
//    native: measured non-divergent through the f16 quantization
//    of every precompute target on this pair                        [C][R]
// ---------------------------------------------------------------------------

// --- Contraction pins -------------------------------------------------------

// Runtime-opaque 0u used by det_barrier. Per-invocation private storage;
// every entry point that (transitively) calls a det_* helper must run
// det_seed() first — the IR lint (src/verify/determinism_lint.rs) rejects a
// seeded-discipline violation. The initializer is 0u, NOT garbage: an
// unseeded stage computes the CORRECT values but is merely unpinned (the
// OR is an exact identity then and a driver may fold it), which keeps
// callers that forgot the seed wrong-but-safe and lets the lint — not a
// NaN flood — report the defect. det_seed writes u32(fract(s)), which is
// 0 for every finite s but cannot be constant-folded: folding requires
// range-modeling fract, which neither FXC nor the driver JITs implement.
var<private> det_zu: u32 = 0u;

fn det_seed(s: f32) -> u32 {
    // u32(fract(s)) = 0 for every finite s but cannot be constant-folded:
    // folding s*0.0 is legal only under fast-math (drivers DO fold it —
    // measured 2026-07-16, which made det_zu a provable 0 and the OR
    // transparent), while folding u32(fract(s)) requires range-modeling
    // fract, which neither FXC nor the NVIDIA JITs implement. The stored
    // index is returned so the proof contract can observe it.
    det_zu = u32(fract(s));
    return det_zu;
}

// Optimization barrier: bitcast<f32>(bitcast<u32>(x) | det_zu). The bitwise
// OR is exact bit-identity for every f32 pattern (NaN, inf, denormals,
// -0.0) because det_zu is 0 at runtime but unprovable at compile time, so
// it cannot fold. Crucially it crosses the integer domain: the FP
// optimizer cannot pull an operand mul out through bitcast+or, and the
// returned value is a bitcast result — not a mul — so no mul+add edge
// exists that a driver could contract into an fma, under any
// reassociation.
//
// This idiom replaced four FP-domain identity barriers, all measured
// defeated 2026-07-16 on RTX 3070 (det-selftest bisection; the NVIDIA
// backend JIT optimizes BOTH the DX12 DXBC->SASS and Vulkan SPIR-V->SASS
// paths):
//  - bitcast<f32>(bitcast<u32>(x)): folded as a no-op pair, then contracted.
//  - bitcast<f32>(reverseBits(reverseBits(bitcast<u32>(x)))): literal bfrev
//    pairs in DXBC, still folded by the DXBC->SASS JIT.
//  - fma(x, 1.0, -0.0): folded — det_barrier(a*b)+c was measured identical
//    to spelled fma(a,b,c) on all 16 probe lanes.
//  - x * det_one (opaque 1.0 in var<private>): the JIT reassociates the
//    multiply chain — (a*b)*det_one + c -> fma(a, b*det_one, c) — which
//    fuses the inner product with single rounding: det_dot3(v,v) still
//    diverged +-1 ULP on 7 of 16 lanes.
// FP-domain identities can always be folded or reassociated into a
// contractable mul+add; the integer OR cannot participate in FP
// contraction at all.
fn det_barrier(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | det_zu);
}

fn det_barrier2(v: vec2<f32>) -> vec2<f32> {
    return bitcast<vec2<f32>>(bitcast<vec2<u32>>(v) | vec2<u32>(det_zu));
}

fn det_barrier3(v: vec3<f32>) -> vec3<f32> {
    return bitcast<vec3<f32>>(bitcast<vec3<u32>>(v) | vec3<u32>(det_zu));
}

fn det_barrier4(v: vec4<f32>) -> vec4<f32> {
    return bitcast<vec4<f32>>(bitcast<vec4<u32>>(v) | vec4<u32>(det_zu));
}

// a * b + c with the multiply barriered so no compiler can contract it into a
// hardware FMA. One ULP worse than fused, but the same one ULP everywhere.
// c is barriered at entry: a caller's raw product passed as the addend would
// otherwise fuse into the add (fma(p, q, ab) after inlining).
fn det_fma(a: f32, b: f32, c: f32) -> f32 {
    let p = det_barrier(a) * det_barrier(b);
    return det_barrier(p) + det_barrier(c);
}

fn det_fma2(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
    let p = det_barrier2(a) * det_barrier2(b);
    return det_barrier2(p) + det_barrier2(c);
}

fn det_fma3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let p = det_barrier3(a) * det_barrier3(b);
    return det_barrier3(p) + det_barrier3(c);
}

fn det_fma4(a: vec4<f32>, b: vec4<f32>, c: vec4<f32>) -> vec4<f32> {
    let p = det_barrier4(a) * det_barrier4(b);
    return det_barrier4(p) + det_barrier4(c);
}

// mix(a, b, t) restated as a + (b - a) * t with pinned intermediate steps.
// a and b are barriered at entry because both feed a subtract/add.
fn det_mix(a: f32, b: f32, t: f32) -> f32 {
    let ab = det_barrier(a);
    let d = det_barrier(b) - ab;
    let s = d * det_barrier(t);
    return ab + det_barrier(s);
}

fn det_mix3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    let ab = det_barrier3(a);
    let d = det_barrier3(b) - ab;
    let s = d * det_barrier(t);
    return ab + det_barrier3(s);
}

// --- Reduction-order pins ---------------------------------------------------

// Explicit left-to-right reduction trees: ((x)+(y))+(z). No driver may
// reassociate a sum that is spelled out as sequential binary adds.
fn det_dot2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = det_barrier2(a);
    let bb = det_barrier2(b);
    let px = det_barrier(ab.x * bb.x);
    let py = det_barrier(ab.y * bb.y);
    return px + py;
}

// Pinned left-assoc product chains: a bare a*b*c is a multiply tree the
// driver can reassociate (a*(b*c) rounds differently than (a*b)*c); each
// step barriered forces the spelled order.
fn det_mul3(a: f32, b: f32, c: f32) -> f32 {
    return det_barrier(det_barrier(det_barrier(a) * det_barrier(b)) * det_barrier(c));
}

fn det_mul4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    return det_barrier(det_barrier(det_barrier(det_barrier(a) * det_barrier(b)) * det_barrier(c)) * det_barrier(d));
}

fn det_mul5(a: f32, b: f32, c: f32, d: f32, e: f32) -> f32 {
    return det_barrier(det_barrier(det_barrier(det_barrier(det_barrier(det_barrier(a) * det_barrier(b)) * det_barrier(c)) * det_barrier(d)) * det_barrier(e)));
}

fn det_mul3_2(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
    return det_barrier2(det_barrier2(det_barrier2(a) * det_barrier2(b)) * det_barrier2(c));
}

fn det_mul3_3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    return det_barrier3(det_barrier3(det_barrier3(a) * det_barrier3(b)) * det_barrier3(c));
}

fn det_mul4_3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return det_barrier3(det_barrier3(det_barrier3(det_barrier3(a) * det_barrier3(b)) * det_barrier3(c)) * det_barrier3(d));
}

fn det_dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let ab = det_barrier3(a);
    let bb = det_barrier3(b);
    let px = det_barrier(ab.x * bb.x);
    let py = det_barrier(ab.y * bb.y);
    let pz = det_barrier(ab.z * bb.z);
    // barrier the partial sum: a bare `s01 + pz` is an add tree the driver
    // can reassociate to px + (py + pz), which rounds differently.
    return det_barrier(px + py) + pz;
}

fn det_dot4(a: vec4<f32>, b: vec4<f32>) -> f32 {
    let ab = det_barrier4(a);
    let bb = det_barrier4(b);
    let px = det_barrier(ab.x * bb.x);
    let py = det_barrier(ab.y * bb.y);
    let pz = det_barrier(ab.z * bb.z);
    let pw = det_barrier(ab.w * bb.w);
    return det_barrier(det_barrier(px + py) + pz) + pw;
}

// Deterministic 1/sqrt(x). Native inverseSqrt is NOT a correctly-rounded op:
// D3D mandates tighter precision than Vulkan (which allows ~2 ULP), so the
// SAME GPU legitimately runs DIFFERENT refinement sequences under FXC-DXBC
// vs SPIR-V — measured 2026-07-10 as residual f32-ULP divergence in the
// shading-normal chain (visible as f16-boundary flips per terrain quad).
// This version is bit-exact everywhere by construction: an integer bit-trick
// seed plus three pinned Newton-Raphson steps built only from barriered
// mul/add (correctly rounded on every backend).
fn det_inverse_sqrt(x: f32) -> f32 {
    let xc = max(x, 1.17549435e-38); // clamp off zero/denormals; callers pass squared lengths
    var y = bitcast<f32>(0x5f3759dfu - (bitcast<u32>(xc) >> 1u));
    let half_x = 0.5 * xc;
    y = y * (1.5 - det_barrier(half_x * det_barrier(y * y)));
    y = y * (1.5 - det_barrier(half_x * det_barrier(y * y)));
    y = y * (1.5 - det_barrier(half_x * det_barrier(y * y)));
    return det_barrier(y);
}

// Deterministic reciprocal / division / sqrt. Same rationale as
// det_inverse_sqrt: D3D requires correctly-rounded f32 divide and sqrt while
// Vulkan permits ~2.5 ULP, so one GPU runs different refinement sequences per
// API. Bit-trick seed + pinned Newton-Raphson = identical bits everywhere.
fn det_rcp(x: f32) -> f32 {
    let ax = abs(x);
    var y = bitcast<f32>(0x7EF311C3u - bitcast<u32>(ax));
    y = y * (2.0 - det_barrier(ax * y));
    y = y * (2.0 - det_barrier(ax * y));
    y = y * (2.0 - det_barrier(ax * y));
    return select(y, -y, x < 0.0);
}

// Callers routinely sum det_* results; a helper whose LAST op is a bare
// multiply returns a product the caller's `+` could contract into a mad once
// inlined. Every tail product below is therefore barriered so no caller-side
// contraction can reach into a helper.
fn det_div(a: f32, b: f32) -> f32 {
    return det_barrier(det_barrier(a) * det_rcp(b));
}

fn det_sqrt(x: f32) -> f32 {
    // barrier x on both edges: bare `x * isqrt(x)` is the canonical native
    // sqrt lowering and a driver that pattern-matches it substitutes its
    // own per-implementation sqrt.
    let xb = det_barrier(x);
    let r = det_barrier(xb * det_inverse_sqrt(xb));
    return select(r, 0.0, x <= 0.0);
}

// normalize() without the driver-specific lowering: the length reduction goes
// through det_dot*, the scale through det_inverse_sqrt (native inverseSqrt is
// per-API precision — see det_inverse_sqrt).
fn det_normalize2(v: vec2<f32>) -> vec2<f32> {
    let vb = det_barrier2(v);
    let inv_len = det_inverse_sqrt(det_barrier(det_dot2(vb, vb)));
    return det_barrier2(vb * inv_len);
}

fn det_normalize3(v: vec3<f32>) -> vec3<f32> {
    // barrier the dot->isqrt edge: driver JITs pattern-match
    // `v * isqrt(dot(v,v))` and can substitute a native normalize whose
    // internal reduction then feeds back into a standalone det_dot3 call
    // via CSE (measured 2026-07-16: det_dot3(v,v) diverged ONLY when
    // det_normalize3 was present in the same module).
    let vb = det_barrier3(v);
    let inv_len = det_inverse_sqrt(det_barrier(det_dot3(vb, vb)));
    return det_barrier3(vb * inv_len);
}

// length(v) = sqrt(dot(v, v)) through the pinned dot + sqrt chain; native
// length() has the same reduction-order and per-API-precision freedom as
// normalize().
fn det_length2(v: vec2<f32>) -> f32 {
    return det_sqrt(det_barrier(det_dot2(v, v)));
}

fn det_length3(v: vec3<f32>) -> f32 {
    return det_sqrt(det_barrier(det_dot3(v, v)));
}

fn det_length4(v: vec4<f32>) -> f32 {
    return det_sqrt(det_barrier(det_dot4(v, v)));
}

// distance(a, b) = length(a - b) through the pinned chain. The operands are
// barriered because `a - b` is a subtract — caller products would fuse into it.
fn det_distance2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return det_length2(det_barrier2(a) - det_barrier2(b));
}

fn det_distance3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return det_length3(det_barrier3(a) - det_barrier3(b));
}

// smoothstep(lo, hi, x) = t*t*(3 - 2t) with t = clamp((x - lo)/(hi - lo)).
// The divide goes through det_div (per-API precision class) and the
// polynomial is barriered so no step can contract. lo/hi/x are barriered at
// entry because all three feed subtracts (x - lo, hi - lo).
fn det_smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    let lob = det_barrier(lo);
    // det_barrier(hi - lo) keeps the subtract's operand places visible to the
    // prover: the difference_ge:hi:lo invariant bounds the divisor away from 0.
    let t = clamp(det_div(det_barrier(x) - lob, det_barrier(hi - lo)), 0.0, 1.0);
    let s = det_barrier(t * t);
    let u = 3.0 - det_barrier(2.0 * t);
    return det_barrier(s * u);
}

// Per-component det_rcp / det_div for vector operands; the canonical path
// divides vec2/vec3 by scalars and vectors (perspective divides, PCF sums,
// tonemap curve ratios) and each lane must see the same pinned sequence.
fn det_rcp2(v: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(det_rcp(v.x), det_rcp(v.y));
}

fn det_rcp3(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(det_rcp(v.x), det_rcp(v.y), det_rcp(v.z));
}

fn det_rcp4(v: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(det_rcp(v.x), det_rcp(v.y), det_rcp(v.z), det_rcp(v.w));
}

fn det_div2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return det_barrier2(det_barrier2(a) * det_rcp2(b));
}

fn det_div3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return det_barrier3(det_barrier3(a) * det_rcp3(b));
}

fn det_div4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return det_barrier4(det_barrier4(a) * det_rcp4(b));
}

// reflect(i, n) = i - 2*dot(n, i)*n with the dot and both products pinned.
// i is barriered because it feeds the final subtract.
fn det_reflect3(i: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let ib = det_barrier3(i);
    let nb = det_barrier3(n);
    let d = det_dot3(nb, ib);
    let s = 2.0 * d;
    let offset = nb * s;
    return ib - det_barrier3(offset);
}

// cross(a, b) with each component spelled as two barriered products and a
// pinned subtraction, so neither term can be contracted into an FMA.
fn det_cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let ab = det_barrier3(a);
    let bb = det_barrier3(b);
    let px = det_barrier(ab.y * bb.z);
    let qx = det_barrier(ab.z * bb.y);
    let py = det_barrier(ab.z * bb.x);
    let qy = det_barrier(ab.x * bb.z);
    let pz = det_barrier(ab.x * bb.y);
    let qz = det_barrier(ab.y * bb.x);
    return vec3<f32>(px - qx, py - qy, pz - qz);
}

// Column-major mat3 * vec3 as a fixed left-to-right sum of scaled columns.
// Same rationale as det_mat4_mul_vec4: OpMatrixTimesVector / HLSL mul() have
// per-compiler mad/fma freedom, measured as the dx12-vs-vulkan divergence
// class on the canonical scene (TBN transforms and the VS clip chain).
fn det_mat3_mul_vec3(m: mat3x3<f32>, v: vec3<f32>) -> vec3<f32> {
    let vb = det_barrier3(v);
    let c0 = det_barrier3(det_barrier3(m[0]) * vb.x);
    let c1 = det_barrier3(det_barrier3(m[1]) * vb.y);
    let c2 = det_barrier3(det_barrier3(m[2]) * vb.z);
    return det_barrier3(c0 + c1) + c2;
}

// Column-major mat4 * vec4 as a fixed left-to-right sum of scaled columns.
fn det_mat4_mul_vec4(m: mat4x4<f32>, v: vec4<f32>) -> vec4<f32> {
    let vb = det_barrier4(v);
    let c0 = det_barrier4(det_barrier4(m[0]) * vb.x);
    let c1 = det_barrier4(det_barrier4(m[1]) * vb.y);
    let c2 = det_barrier4(det_barrier4(m[2]) * vb.z);
    let c3 = det_barrier4(det_barrier4(m[3]) * vb.w);
    return det_barrier4(det_barrier4(c0 + c1) + c2) + c3;
}

// --- Transcendental pins ----------------------------------------------------
// REPINNED 2026-07-09 (gap-closure): pow/exp are spelled as explicit exp2/log2
// compositions so no compiler is free to choose its own pow/exp fixup
// sequence. FINALIZED 2026-07-15: exp2 and log2 themselves are now SOFTWARE
// implementations below, because the native ops are per-vendor/per-API in
// ULP (the same freedom class as sin/cos) and the browser WebGPU leg
// (Dawn/SwiftShader vs hardware ex2/lg2) cannot be expected to match them.
// Both are built only from correctly-rounded mul/add plus exact ops (floor,
// select, comparisons, integer bit ops) with every mul+add barriered —
// bit-identical on every backend by construction. Accuracy: det_exp2 max rel
// error ~1.2e-7 (~2 ulp) on [-10, 10]; det_log2 max abs error ~1.9e-6 on
// [1e-6, 65536]. DETERMINISM, not last-ULP accuracy, is the contract.
//
// Domain guard: native pow is undefined for x < 0 and pow(0, y>0) = 0. The
// max() keeps log2 off the undefined x <= 0 domain; the select() then forces
// an exact 0 for x <= 0 (all in-scope call sites use non-negative bases with
// positive exponents: fresnel (1-cos)^5, gamma/sRGB powers, spec lobes).
// pow(0, 0) would return 0 here instead of native's customary 1 — no in-scope
// call site can hit that (exponents are positive constants or clamped).
//
// det_exp2(x): x = k + r with k = round(x), r in [-0.5, 0.5]; 2^r via a
// degree-9 Horner in pinned det_fma steps, then exact exponent-bit scaling
// by 2^k. k is clamped to the normal-exponent range [-126, 127]: |x| > ~128
// is outside every in-scope caller (det_pow sees y*log2(x) for x in
// [1e-38, 64k] and |y| <= 8 -> [-160, 160], still deterministic because the
// clamp is identical everywhere; accuracy is only contracted within the
// documented input ranges).
fn det_exp2(x: f32) -> f32 {
    // x is barriered at entry: a caller's raw product would otherwise contract
    // into the `+ 0.5` or `- k` below.
    let xb = det_barrier(x);
    let k = floor(xb + 0.5);
    let r = xb - k;
    var p = det_fma(r, 0.00000010312740612007, 0.00000132697009347531);
    p = det_fma(r, p, 0.00001525273380405950);
    p = det_fma(r, p, 0.00015403530393381606);
    p = det_fma(r, p, 0.00133335581463984430);
    p = det_fma(r, p, 0.00961812910762847700);
    p = det_fma(r, p, 0.05550410866482158000);
    p = det_fma(r, p, 0.24022650695910070000);
    p = det_fma(r, p, 0.69314718055994530000);
    p = det_fma(r, p, 1.0);
    let ki = clamp(i32(k), -126, 127);
    let scale = bitcast<f32>(u32(ki + 127) << 23u);
    return det_barrier(p * scale);
}

// det_log2(x) for normal x > 0: decompose x = 2^e * m with m in [1, 2) via
// the exponent/mantissa bit fields, then log2(m) = (2/ln2) * s * P(s^2)
// where s = (m-1)/(m+1) (pinned det_div) and P is the atanh series folded
// to degree 7 in s^2 (s in [0, 1/3]). Subnormal x (e = -127) is outside the
// documented input range.
fn det_log2(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let e = f32(i32((bits >> 23u) & 0xFFu) - 127);
    let m = bitcast<f32>((bits & 0x7FFFFFu) | 0x3F800000u);
    let s = det_div(m - 1.0, m + 1.0);
    let u = det_barrier(s * s);
    var p = det_fma(u, 0.06666666666666667, 0.07692307692307693);
    p = det_fma(u, p, 0.09090909090909091);
    p = det_fma(u, p, 0.11111111111111111);
    p = det_fma(u, p, 0.14285714285714285);
    p = det_fma(u, p, 0.20000000000000000);
    p = det_fma(u, p, 0.33333333333333333);
    p = det_fma(u, p, 1.0);
    return e + det_barrier(det_barrier(2.8853900817779268 * s) * p);
}

fn det_pow(x: f32, y: f32) -> f32 {
    let lg = det_log2(max(x, 1.17549435e-38));
    // barrier the product: det_exp2's first ops are `x + 0.5` / `x - k`, and
    // an unbarriered arg mul would contract into either after inlining.
    let r = det_exp2(det_barrier(det_barrier(y) * lg));
    return select(r, 0.0, x <= 0.0);
}

fn det_pow3(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(det_pow(x.x, y.x), det_pow(x.y, y.y), det_pow(x.z, y.z));
}

// exp(x) = exp2(x * log2(e)); the constant is log2(e) rounded to f64 then f32.
// The product is barriered so it cannot contract into det_exp2's `x + 0.5` /
// `x - k` after inlining.
fn det_exp(x: f32) -> f32 {
    return det_exp2(det_barrier(det_barrier(x) * 1.4426950408889634));
}

fn det_exp3(x: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(det_exp(x.x), det_exp(x.y), det_exp(x.z));
}

// --- Trigonometric pins ------------------------------------------------------
// ADDED 2026-07-09 (gap-closure): native sin/cos/atan2/acos lowering is a
// per-compiler choice, and the IBL precompute chain (equirect projection +
// irradiance convolution) leans on all four. On the measured dx12-FXC /
// vulkan-NVIDIA pair this pin was a NO-OP — the precompute writes rgba16float
// targets, whose quantization absorbs trig-level ULP differences, and the
// real 52-pixel divergence was the per-API divide/sqrt class (see header).
// The pins are KEPT because a vendor whose trig differs by more than the f16
// quantum would silently poison every downstream IBL sample. These are
// SOFTWARE implementations composed only of correctly-rounded ops (+, -, *)
// plus exact ops (abs, floor, select, comparisons), with every mul+add step
// barriered — bit-identical on every backend by construction. Accuracy is
// minimax-polynomial grade (~1e-6 absolute, plus the inherent error of
// non-exact range reduction near large multiples of pi/2); acceptable because
// callers are offline precompute integrators, and DETERMINISM, not last-ULP
// accuracy, is the contract here. Do not substitute native intrinsics back
// without re-running the cross-backend hash measurement.

// sin(x) for |x| within a few periods (IBL callers pass phi in [0, 2*pi)).
fn det_sin(x: f32) -> f32 {
    // Quadrant reduction: k = round(x / (pi/2)), r = x - k*(pi/2).
    // The mul and sub are correctly rounded, so r is identical everywhere.
    // x is barriered so a caller's product arg cannot reassociate into either.
    let xb = det_barrier(x);
    let k = floor(det_barrier(xb * 0.6366197723675814) + 0.5);
    let r = xb - det_barrier(k * 1.5707963267948966);
    let q = i32(k) & 3;
    let r2 = det_barrier(r * r);
    // Pinned Horner: sin(r) ~= r + r^3 c3 + r^5 c5 + r^7 c7 on [-pi/4, pi/4]
    var ps = det_fma(r2, -0.00019840874, 0.0083333310);
    ps = det_fma(r2, ps, -0.16666667);
    ps = det_fma(r2, ps, 1.0);
    let s = r * ps;
    // Pinned Horner: cos(r) ~= 1 + r^2 c2 + r^4 c4 + r^6 c6 on [-pi/4, pi/4]
    var pc = det_fma(r2, -0.0013888378, 0.041666638);
    pc = det_fma(r2, pc, -0.5);
    pc = det_fma(r2, pc, 1.0);
    let c = pc;
    let use_cos = (q & 1) == 1;
    let negate = (q & 2) == 2;
    let v = select(s, c, use_cos);
    return select(v, -v, negate);
}

fn det_cos(x: f32) -> f32 {
    // barrier(x): a caller's unbarriered product passed as x would otherwise
    // contract into the `+ pi/2` here (fma(b, k, pi/2) after inlining).
    return det_sin(det_barrier(x) + 1.5707963267948966);
}

// atan(a) for a in [0, 1] — minimax polynomial in a^2, pinned Horner order.
fn det_atan01(a: f32) -> f32 {
    let ab = det_barrier(a);
    let s = det_barrier(ab * ab);
    var p = det_fma(s, -0.0117212, 0.05265332);
    p = det_fma(s, p, -0.11643287);
    p = det_fma(s, p, 0.19354346);
    p = det_fma(s, p, -0.33262347);
    p = det_fma(s, p, 0.99997726);
    return det_barrier(ab * p);
}

fn det_atan2(y: f32, x: f32) -> f32 {
    let ax = abs(x);
    let ay = abs(y);
    let hi = max(ax, ay);
    if (hi == 0.0) {
        return 0.0; // atan2(0, 0): native is undefined; pin an exact 0
    }
    let lo = min(ax, ay);
    // Pinned division: native / is the per-API precision class (D3D requires
    // correctly-rounded, Vulkan permits ~2.5 ULP), so a raw divide here would
    // reintroduce the exact divergence this file exists to remove.
    let a = det_div(lo, hi);
    var p = det_atan01(a);
    p = select(p, 1.5707963267948966 - p, ay > ax);
    p = select(p, 3.141592653589793 - p, x < 0.0);
    return select(p, -p, y < 0.0);
}

// acos(x) via the atan2 identity; the sqrt goes through det_sqrt (per-API
// precision class) so this composes only pinned/exact steps.
fn det_acos(x: f32) -> f32 {
    let xc = clamp(x, -1.0, 1.0);
    let x2 = det_barrier(xc * xc);
    let s = det_sqrt(max(1.0 - x2, 0.0));
    return det_atan2(s, xc);
}

// asin(x) = atan2(x, sqrt(1 - x^2)) through the pinned chain.
fn det_asin(x: f32) -> f32 {
    let xc = clamp(x, -1.0, 1.0);
    let x2 = det_barrier(xc * xc);
    let s = det_sqrt(max(1.0 - x2, 0.0));
    return det_atan2(xc, s);
}

// --- Vector-width pin surface --------------------------------------------
// Per-lane compositions of the scalar pins: every lane runs the identical
// pinned sequence, so a vecN form is N independent copies of the scalar
// proof. These complete the det_* surface the rewriter and the proof
// contract expect (one entry per name in
// shaders/contracts/determinism.toml).
fn det_sqrt2(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(det_sqrt(v.x), det_sqrt(v.y)); }
fn det_sqrt3(v: vec3<f32>) -> vec3<f32> { return vec3<f32>(det_sqrt(v.x), det_sqrt(v.y), det_sqrt(v.z)); }
fn det_sqrt4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(det_sqrt(v.x), det_sqrt(v.y), det_sqrt(v.z), det_sqrt(v.w)); }
fn det_inverse_sqrt2(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(det_inverse_sqrt(v.x), det_inverse_sqrt(v.y)); }
fn det_inverse_sqrt3(v: vec3<f32>) -> vec3<f32> { return vec3<f32>(det_inverse_sqrt(v.x), det_inverse_sqrt(v.y), det_inverse_sqrt(v.z)); }
fn det_inverse_sqrt4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(det_inverse_sqrt(v.x), det_inverse_sqrt(v.y), det_inverse_sqrt(v.z), det_inverse_sqrt(v.w)); }

fn det_distance4(a: vec4<f32>, b: vec4<f32>) -> f32 {
    return det_length4(det_barrier4(a) - det_barrier4(b));
}
fn det_smoothstep2(lo: f32, hi: f32, x: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(det_smoothstep(lo, hi, x.x), det_smoothstep(lo, hi, x.y));
}
fn det_smoothstep3(lo: f32, hi: f32, x: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(det_smoothstep(lo, hi, x.x), det_smoothstep(lo, hi, x.y), det_smoothstep(lo, hi, x.z));
}
fn det_smoothstep4(lo: f32, hi: f32, x: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(det_smoothstep(lo, hi, x.x), det_smoothstep(lo, hi, x.y), det_smoothstep(lo, hi, x.z), det_smoothstep(lo, hi, x.w));
}

fn det_normalize4(v: vec4<f32>) -> vec4<f32> {
    let vb = det_barrier4(v);
    let inv_len = det_inverse_sqrt(det_barrier(det_dot4(vb, vb)));
    return det_barrier4(vb * inv_len);
}

fn det_reflect4(i: vec4<f32>, n: vec4<f32>) -> vec4<f32> {
    let ib = det_barrier4(i);
    let nb = det_barrier4(n);
    let d = det_dot4(nb, ib);
    let s = 2.0 * d;
    let offset = nb * s;
    return ib - det_barrier4(offset);
}

// Column-major mat2 * vec2 as a fixed left-to-right sum of scaled columns.
fn det_mat2_mul_vec2(m: mat2x2<f32>, v: vec2<f32>) -> vec2<f32> {
    let vb = det_barrier2(v);
    let c0 = det_barrier2(det_barrier2(m[0]) * vb.x);
    let c1 = det_barrier2(det_barrier2(m[1]) * vb.y);
    return c0 + c1;
}
fn det_vec2_mul_mat2(v: vec2<f32>, m: mat2x2<f32>) -> vec2<f32> {
    return vec2<f32>(det_dot2(v, m[0]), det_dot2(v, m[1]));
}
fn det_vec3_mul_mat3(v: vec3<f32>, m: mat3x3<f32>) -> vec3<f32> {
    return vec3<f32>(det_dot3(v, m[0]), det_dot3(v, m[1]), det_dot3(v, m[2]));
}
fn det_vec4_mul_mat4(v: vec4<f32>, m: mat4x4<f32>) -> vec4<f32> {
    return vec4<f32>(det_dot4(v, m[0]), det_dot4(v, m[1]), det_dot4(v, m[2]), det_dot4(v, m[3]));
}
fn det_mat2_mul_mat2(a: mat2x2<f32>, b: mat2x2<f32>) -> mat2x2<f32> {
    return mat2x2<f32>(det_mat2_mul_vec2(a, b[0]), det_mat2_mul_vec2(a, b[1]));
}
fn det_mat3_mul_mat3(a: mat3x3<f32>, b: mat3x3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(det_mat3_mul_vec3(a, b[0]), det_mat3_mul_vec3(a, b[1]), det_mat3_mul_vec3(a, b[2]));
}
fn det_mat4_mul_mat4(a: mat4x4<f32>, b: mat4x4<f32>) -> mat4x4<f32> {
    return mat4x4<f32>(det_mat4_mul_vec4(a, b[0]), det_mat4_mul_vec4(a, b[1]), det_mat4_mul_vec4(a, b[2]), det_mat4_mul_vec4(a, b[3]));
}

fn det_exp_2(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(det_exp(v.x), det_exp(v.y)); }
fn det_exp4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(det_exp(v.x), det_exp(v.y), det_exp(v.z), det_exp(v.w)); }
fn det_exp2_2(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(det_exp2(v.x), det_exp2(v.y)); }
fn det_exp2_3(v: vec3<f32>) -> vec3<f32> { return vec3<f32>(det_exp2(v.x), det_exp2(v.y), det_exp2(v.z)); }
fn det_exp2_4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(det_exp2(v.x), det_exp2(v.y), det_exp2(v.z), det_exp2(v.w)); }

// ln(x) = log2(x) * ln(2); the product is barriered so it cannot fuse.
fn det_log(x: f32) -> f32 {
    return det_barrier(det_log2(x) * 0.6931471805599453);
}
fn det_log_2(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(det_log(v.x), det_log(v.y)); }
fn det_log3(v: vec3<f32>) -> vec3<f32> { return vec3<f32>(det_log(v.x), det_log(v.y), det_log(v.z)); }
fn det_log4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(det_log(v.x), det_log(v.y), det_log(v.z), det_log(v.w)); }
fn det_log2_2(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(det_log2(v.x), det_log2(v.y)); }
fn det_log2_3(v: vec3<f32>) -> vec3<f32> { return vec3<f32>(det_log2(v.x), det_log2(v.y), det_log2(v.z)); }
fn det_log2_4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(det_log2(v.x), det_log2(v.y), det_log2(v.z), det_log2(v.w)); }

fn det_pow2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(det_pow(a.x, b.x), det_pow(a.y, b.y)); }
fn det_pow4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> { return vec4<f32>(det_pow(a.x, b.x), det_pow(a.y, b.y), det_pow(a.z, b.z), det_pow(a.w, b.w)); }

fn det_sin2(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(det_sin(v.x), det_sin(v.y)); }
fn det_sin3(v: vec3<f32>) -> vec3<f32> { return vec3<f32>(det_sin(v.x), det_sin(v.y), det_sin(v.z)); }
fn det_sin4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(det_sin(v.x), det_sin(v.y), det_sin(v.z), det_sin(v.w)); }
fn det_cos2(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(det_cos(v.x), det_cos(v.y)); }
fn det_cos3(v: vec3<f32>) -> vec3<f32> { return vec3<f32>(det_cos(v.x), det_cos(v.y), det_cos(v.z)); }
fn det_cos4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(det_cos(v.x), det_cos(v.y), det_cos(v.z), det_cos(v.w)); }

// tan = sin/cos through the pinned divide; callers stay inside |x| <= 1.4
// where cos is provably away from zero.
fn det_tan(x: f32) -> f32 {
    return det_div(det_sin(x), det_cos(x));
}
fn det_tan2(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(det_tan(v.x), det_tan(v.y)); }
fn det_tan3(v: vec3<f32>) -> vec3<f32> { return vec3<f32>(det_tan(v.x), det_tan(v.y), det_tan(v.z)); }
fn det_tan4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(det_tan(v.x), det_tan(v.y), det_tan(v.z), det_tan(v.w)); }

fn det_asin2(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(det_asin(v.x), det_asin(v.y)); }
fn det_asin3(v: vec3<f32>) -> vec3<f32> { return vec3<f32>(det_asin(v.x), det_asin(v.y), det_asin(v.z)); }
fn det_asin4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(det_asin(v.x), det_asin(v.y), det_asin(v.z), det_asin(v.w)); }
fn det_acos2(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(det_acos(v.x), det_acos(v.y)); }
fn det_acos3(v: vec3<f32>) -> vec3<f32> { return vec3<f32>(det_acos(v.x), det_acos(v.y), det_acos(v.z)); }
fn det_acos4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(det_acos(v.x), det_acos(v.y), det_acos(v.z), det_acos(v.w)); }

fn det_atan(x: f32) -> f32 {
    return det_atan2(x, 1.0);
}
fn det_atan_2(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(det_atan(v.x), det_atan(v.y)); }
fn det_atan3(v: vec3<f32>) -> vec3<f32> { return vec3<f32>(det_atan(v.x), det_atan(v.y), det_atan(v.z)); }
fn det_atan4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(det_atan(v.x), det_atan(v.y), det_atan(v.z), det_atan(v.w)); }
fn det_atan2_2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(det_atan2(a.x, b.x), det_atan2(a.y, b.y)); }
fn det_atan2_3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> { return vec3<f32>(det_atan2(a.x, b.x), det_atan2(a.y, b.y), det_atan2(a.z, b.z)); }
fn det_atan2_4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> { return vec4<f32>(det_atan2(a.x, b.x), det_atan2(a.y, b.y), det_atan2(a.z, b.z), det_atan2(a.w, b.w)); }

fn det_mix2(a: vec2<f32>, b: vec2<f32>, t: f32) -> vec2<f32> {
    return vec2<f32>(det_mix(a.x, b.x, t), det_mix(a.y, b.y, t));
}
fn det_mix4(a: vec4<f32>, b: vec4<f32>, t: f32) -> vec4<f32> {
    return vec4<f32>(det_mix(a.x, b.x, t), det_mix(a.y, b.y, t), det_mix(a.z, b.z, t), det_mix(a.w, b.w, t));
}
fn det_mix2v(a: vec2<f32>, b: vec2<f32>, t: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(det_mix(a.x, b.x, t.x), det_mix(a.y, b.y, t.y));
}
fn det_mix3v(a: vec3<f32>, b: vec3<f32>, t: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(det_mix(a.x, b.x, t.x), det_mix(a.y, b.y, t.y), det_mix(a.z, b.z, t.z));
}
fn det_mix4v(a: vec4<f32>, b: vec4<f32>, t: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(det_mix(a.x, b.x, t.x), det_mix(a.y, b.y, t.y), det_mix(a.z, b.z, t.z), det_mix(a.w, b.w, t.w));
}

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
//    det_exp/det_log2 centralize the choice: they currently forward to the native
//    intrinsic (measurement has not yet shown native pow divergence); if a vendor
//    diverges, flipping ONE body here (e.g. pow(x,y) -> exp2(y*log2(x))) repins
//    every call site at once.
//
// Residual sources this file cannot pin (documented, kept loud in review):
//  - Fixed-function texture filtering precision (bilinear/trilinear weights) is
//    vendor hardware. In-scope shaders sample with explicit LOD so mip selection
//    is not driver-chosen; filter arithmetic itself remains hardware-defined.
//  - Downstream compilers (DXC/FXC/Metal) may still re-fuse across statements at
//    their default optimization level; wgpu 0.19 exposes no fast-math toggle to
//    forbid it. FORGE3D_DETERMINISTIC (src/core/gpu.rs) pins the backend so a
//    given hash is at least stable per backend+driver, and the CI matrix diffs
//    the survivors.
//
// ---------------------------------------------------------------------------
// Nondeterminism-site inventory (step 1 of the TERRA-DETERMINATA task).
// Risk classes: [C] contraction, [R] reduction-order, [T] transcendental ULP.
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
// src/shaders/csm.wgsl
//  - calculate_depth_bias: normalize(light_dir), dot(n, l),
//    sqrt argument n_dot_l*n_dot_l                                  [R][C]
//  - world_to_light_space: mat4 x mat4 x vec4 chain                 [R]
//  - select_cascade: float boundary compare (ULP flip at edges)     [R]
//  - sample_shadow_pcf: += accumulation with data-dependent tap set [R]
//  - sample_shadow_poisson: += accumulation (order already fixed)   [R]
//  - sample_shadow_evsm: exp(+-c*depth) warps                       [T]
//  - smooth_cascade_transition / debug paths: mix()                 [C]
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
// ---------------------------------------------------------------------------

// --- Contraction pins -------------------------------------------------------

// a * b + c with the multiply forced into its own statement so the compiler
// cannot contract it into a hardware FMA. One ULP worse than fused, but the
// same one ULP everywhere.
fn det_fma(a: f32, b: f32, c: f32) -> f32 {
    let p = a * b;
    return p + c;
}

fn det_fma3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let p = a * b;
    return p + c;
}

// mix(a, b, t) restated as a + (b - a) * t with pinned intermediate steps.
fn det_mix(a: f32, b: f32, t: f32) -> f32 {
    let d = b - a;
    let s = d * t;
    return a + s;
}

fn det_mix3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    let d = b - a;
    let s = d * t;
    return a + s;
}

// --- Reduction-order pins ---------------------------------------------------

// Explicit left-to-right reduction trees: ((x)+(y))+(z). No driver may
// reassociate a sum that is spelled out as sequential binary adds.
fn det_dot2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    let px = a.x * b.x;
    let py = a.y * b.y;
    return px + py;
}

fn det_dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let px = a.x * b.x;
    let py = a.y * b.y;
    let pz = a.z * b.z;
    let s01 = px + py;
    return s01 + pz;
}

fn det_dot4(a: vec4<f32>, b: vec4<f32>) -> f32 {
    let px = a.x * b.x;
    let py = a.y * b.y;
    let pz = a.z * b.z;
    let pw = a.w * b.w;
    let s01 = px + py;
    let s012 = s01 + pz;
    return s012 + pw;
}

// normalize() without the driver-specific lowering: the length reduction goes
// through det_dot*, the scale through inverseSqrt (correctly-rounded class op).
fn det_normalize2(v: vec2<f32>) -> vec2<f32> {
    let inv_len = inverseSqrt(det_dot2(v, v));
    return v * inv_len;
}

fn det_normalize3(v: vec3<f32>) -> vec3<f32> {
    let inv_len = inverseSqrt(det_dot3(v, v));
    return v * inv_len;
}

// reflect(i, n) = i - 2*dot(n, i)*n with the dot and both products pinned.
fn det_reflect3(i: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let d = det_dot3(n, i);
    let s = 2.0 * d;
    let offset = n * s;
    return i - offset;
}

// Column-major mat4 * vec4 as a fixed left-to-right sum of scaled columns.
fn det_mat4_mul_vec4(m: mat4x4<f32>, v: vec4<f32>) -> vec4<f32> {
    let c0 = m[0] * v.x;
    let c1 = m[1] * v.y;
    let c2 = m[2] * v.z;
    let c3 = m[3] * v.w;
    let s01 = c0 + c1;
    let s012 = s01 + c2;
    return s012 + c3;
}

// --- Transcendental pins ----------------------------------------------------
// Native intrinsics for now (see header): the point is that the CHOICE lives
// here, so repinning to exp2/log2 forms is a one-line edit per helper.

fn det_pow(x: f32, y: f32) -> f32 {
    return pow(x, y);
}

fn det_pow3(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
    return pow(x, y);
}

fn det_exp(x: f32) -> f32 {
    return exp(x);
}

fn det_exp3(x: vec3<f32>) -> vec3<f32> {
    return exp(x);
}

fn det_exp2(x: f32) -> f32 {
    return exp2(x);
}

fn det_log2(x: f32) -> f32 {
    return log2(x);
}

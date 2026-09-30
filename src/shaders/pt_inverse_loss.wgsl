// src/shaders/pt_inverse_loss.wgsl
// DIFFERENTIA: shared inverse declarations + per-eval state clear + the
// reservoir weight snapshot + the Reinhard-space loss adjoint.
//
// This file is appended to the FORWARD kernel module (see
// shader_sources.rs::inverse_kernel). The inverse solver re-dispatches the
// verbatim `main_terrain` primal per accumulation frame — there is NO
// separate inverse primal, so geometry, jittering, candidate generation,
// reservoir shading, atmosphere and tonemap can never drift from the forward
// path. Uniforms, LightingUniforms, Ray, intersect_hybrid,
// intersect_shadow_ray, the terrain_* helpers, RestirReservoir and the
// forward group-2/3 bindings are all already in scope.
//
//   main_inverse_clear   zeroes the per-eval accumulation state once before
//                        frame 0: accum_hdr, the merged (prev) reservoirs,
//                        the welford record, the scalar-gradient slots and
//                        the per-texel albedo gradient table. 1-D dispatch
//                        over max(pixels, 4*texels, 8).
//   main_inv_replay_clear
//                        zeroes only the merged (prev) reservoir state; the
//                        checkpointed adjoint replay re-runs the frame chain
//                        after the loss pass and must reproduce the same
//                        reservoir trajectory the forward sweep saw.
//   main_inv_wsnap       runs immediately after `main_terrain` in each
//                        frame's dispatch chain (before the temporal merge
//                        rewrites the prev buffer) and records the reservoir
//                        weight/spectral channel that frame's shading used
//                        into inv_whist[frame_index % slots]. 1-D over pixels.
//   main_inverse_loss    per-pixel loss in tonemapped sRGB display space:
//                        the current linear mean runs through the same
//                        Reinhard operator the forward applies to out_tex and
//                        then the sRGB OETF; the observed u8 beauty (already
//                        the forward's Reinhard-space quantization) is lifted
//                        through the same OETF so both sides live in display
//                        sRGB. Writes the adjoint dL/d(linear radiance)
//                        carrying srgb' . reinhard', pre-folded with
//                        1/N_pixels so every downstream sum IS the mean-loss
//                        gradient.
// RELEVANT FILES: src/path_tracing/inverse/{params,grads,mod}.rs

struct InvParams {
    dims: vec4<u32>,   // albedo texel w, h, texel count, frames
    ctrl: vec4<u32>,   // flags (bit0 spatial, bit1 edge, bit2 score),
                       // whist slot count K = min(tile_size, frames), unused x2
    sunv: vec4<f32>,   // unit-intensity sun color rgb, sun intensity
    fl: vec4<f32>,     // struct_w, reserved legacy clamp, 1/pixel_count, 1/(frames*spp)
    rsv0: vec4<f32>,   // x = 1/replicate_count, y = replicate ordinal (0/1)
    rsv1: vec4<f32>,
}
// ctrl.x flags
const INV_FLAG_SPATIAL: u32 = 1u;
const INV_FLAG_EDGE: u32 = 2u;
const INV_FLAG_SCORE: u32 = 4u;
// Loss reads the replicate-mean image region instead of this eval's accum.
// Its expectation is E[loss((sum_r cur_r)/R)] for the finite replicate count;
// a finite mean inside a nonlinear loss is not loss(E[cur]).
const INV_FLAG_MEAN_LOSS: u32 = 8u;
// Edge provenance preflight: run the certified-edge entry point against the
// real replay history, capture its camera/light inputs, and return before any
// receiver-dependent event work or gradient accumulation. The host certifies
// the subsequent full event batch only when it reports this exact snapshot.
const INV_FLAG_EDGE_PROBE: u32 = 16u;

// gid.y folds 1-D dispatches wider than the 65535-workgroup x limit: the
// host dispatches x = min(wg, 65535), y = ceil(wg/65535), so the flat index
// is gid.y * (65535*256) + gid.x.
const INV_DISPATCH_X: u32 = 16776960u; // 65535 * 256

@group(0) @binding(2) var<uniform> inv_params: InvParams;

// Consolidated inverse state — two storage buffers carry every array so the
// shared group-2 layout stays within the storage-buffer budget alongside the
// forward bindings (0..7, 10, 16):
//   inv_v4  [0, px)              per-pixel adjoint dL/d(linear radiance)
//           [px, px + K*px)      per-(slot, pixel) reservoir weight/channel
//                              snapshot written by main_inv_wsnap; K =
//                              min(tile_size, frames) bounds the retained
//                              per-frame reservoir history — when frames > K
//                              the host replays the frame chain per
//                              K-frame tile instead of caching all F slots
//           [px + K*px, px + K*px + px)
//                              replicate-mean linear image accumulated by
//                              main_inv_accum_mean and consumed by the loss
//                              pass under INV_FLAG_MEAN_LOSS
//   inv_u32 [0, 8)               scalar gradient slots (f32 CAS)
//           [8, 8 + px)          per-pixel loss (bitcast f32, atomicStore)
//           [8+px, 8+px+4*T)     per-texel albedo gradient (f32 CAS, rgb+spare)
@group(2) @binding(11) var<storage, read_write> inv_v4: array<vec4<f32>>;
@group(2) @binding(12) var<storage, read_write> inv_u32: array<atomic<u32>>;

@group(3) @binding(8) var inv_target_tex: texture_2d<f32>;
// Sampled views of the Depth/Normal AOVs the primal writes on frame 0 — read
// only by main_inverse_edge, which binds its own group 3 so the AOV textures
// never appear as both storage and sampled in one dispatch.
@group(3) @binding(9) var inv_aov_normal: texture_2d<f32>;
@group(3) @binding(10) var inv_aov_depth: texture_2d<f32>;

// Scalar gradient slots inside inv_u32 (order must match params.rs/mod.rs).
const INV_GRAD_SUN_X: u32 = 0u;
const INV_GRAD_SUN_Y: u32 = 1u;
const INV_GRAD_SUN_Z: u32 = 2u;
const INV_GRAD_INTENSITY: u32 = 3u;
const INV_GRAD_TURBIDITY: u32 = 4u;
const INV_GRAD_EDGE_MASS: u32 = 5u;  // sum |boundary jump| (diagnostic)
const INV_GRAD_NONFINITE: u32 = 6u;  // dropped non-finite adds (diagnostic)
const INV_GRAD_SPARE7: u32 = 7u;
// Region bases inside inv_u32.
const INV_U32_SCALARS: u32 = 0u;
fn inv_u32_loss_base() -> u32 { return 8u; }
fn inv_u32_alb_base() -> u32 {
    return 8u + uniforms.width * uniforms.height;
}
// inv_v4 regions: adjoint at [pix], w-history at [px + slot*px + pix].
// The slot is the frame's position inside its checkpoint tile (frame % K),
// NOT the global frame index — the history buffer holds at most K frames.
fn inv_whist_index(slot: u32, pix: u32) -> u32 {
    return uniforms.width * uniforms.height * (slot + 1u) + pix;
}
// Slots of per-frame reservoir history retained in inv_v4.
fn inv_whist_slots() -> u32 {
    return max(inv_params.ctrl.y, 1u);
}
// Replicate-mean image region, after the adjoint and K whist slots.
fn inv_mean_index(pix: u32) -> u32 {
    return uniforms.width * uniforms.height * (inv_whist_slots() + 1u) + pix;
}

// ---------------------------------------------------------------------------
// Parameter-space helpers
// ---------------------------------------------------------------------------

// Stream seed identical to the forward kernel: avalanche-hashed
// (seed, pixel, frame) — the reverse pass replays the identical stream so
// the primal needs no recorded per-sample state.
fn inv_seed(gpix: u32, frame: u32) -> u32 {
    let st = terrain_seed_hash(uniforms.seed_hi ^ uniforms.seed_lo
        ^ terrain_seed_hash(gpix) ^ terrain_seed_hash(frame + 1u));
    return select(st, 0x6d2b79f5u, st == 0u);
}

// Camera ray for a pixel + jitter (same convention as main_terrain).
fn inv_cam_ray(gx: u32, gy: u32, jx: f32, jy: f32) -> Ray {
    let half_h = tan(0.5 * uniforms.cam_fov_y);
    let half_w = uniforms.cam_aspect * half_h;
    let ndc_x = ((f32(gx) + 0.5 + jx) / f32(uniforms.width)) * 2.0 - 1.0;
    let ndc_y = (1.0 - (f32(gy) + 0.5 + jy) / f32(uniforms.height)) * 2.0 - 1.0;
    var rd = normalize(vec3<f32>(ndc_x * half_w, ndc_y * half_h, -1.0));
    rd = normalize(rd.x * uniforms.cam_right + rd.y * uniforms.cam_up
        + rd.z * (-uniforms.cam_forward));
    return Ray(uniforms.cam_origin, 1e-3, rd, 1e30);
}

// Unjittered center-ray direction for a pixel (AOV depth -> world position).
fn inv_center_dir(gx: u32, gy: u32) -> vec3<f32> {
    let half_h = tan(0.5 * uniforms.cam_fov_y);
    let half_w = uniforms.cam_aspect * half_h;
    let ndc_x = ((f32(gx) + 0.5) / f32(uniforms.width)) * 2.0 - 1.0;
    let ndc_y = (1.0 - (f32(gy) + 0.5) / f32(uniforms.height)) * 2.0 - 1.0;
    var rd = normalize(vec3<f32>(ndc_x * half_w, ndc_y * half_h, -1.0));
    rd = normalize(rd.x * uniforms.cam_right + rd.y * uniforms.cam_up
        + rd.z * (-uniforms.cam_forward));
    return rd;
}

// sRGB OETF and its derivative (IEC 61966-2-1 piecewise curve). The loss
// metric lives in display-encoded sRGB space, so the adjoint must carry the
// OETF Jacobian between the Reinhard output and the encoded observation.
fn inv_srgb(c: f32) -> f32 {
    return select(1.055 * pow(max(c, 1e-8), 1.0 / 2.4) - 0.055,
        12.92 * c, c <= 0.0031308);
}
fn inv_srgb_d(c: f32) -> f32 {
    return select(1.055 * (1.0 / 2.4) * pow(max(c, 1e-8), 1.0 / 2.4 - 1.0),
        12.92, c <= 0.0031308);
}

// ---------------------------------------------------------------------------
// Deterministic f32 accumulation through atomic<u32> CAS
// ---------------------------------------------------------------------------
// The scalar and per-texel albedo gradients are plain f32 sums; the previous
// fixed-point i32 scheme wrapped silently at photo resolutions and dropped
// every contribution |v| >= 8. A CAS loop on the bit pattern gives exact
// f32 accumulation and is order-independent because f32 addition is
// commutative — though NOT associative, so the result still depends on add
// order; the bitwise result is deterministic on a fixed GPU only per launch
// order, which is all the tests require (cross-checks use finite
// differences, not bitwise equality).
fn inv_is_finite(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}

// Index-based CAS add — Naga rejects storage-space pointers as function
// arguments, so the helpers take a slot index into the global inv_u32.
fn inv_cas_add(idx: u32, v: f32) {
    var old = atomicLoad(&inv_u32[idx]);
    loop {
        let res = atomicCompareExchangeWeak(&inv_u32[idx], old,
            bitcast<u32>(bitcast<f32>(old) + v));
        if (res.exchanged) { break; }
        old = res.old_value;
    }
}

fn inv_add_scalar(idx: u32, v: f32) {
    if (!inv_is_finite(v)) {
        inv_cas_add(INV_U32_SCALARS + INV_GRAD_NONFINITE, 1.0);
        return;
    }
    inv_cas_add(INV_U32_SCALARS + idx, v);
}

// Per-texel albedo gradient: four f32 slots per texel (rgb + spare) after the
// loss region.
fn inv_add_albedo(texel: u32, g: vec3<f32>) {
    if (texel == 0xffffffffu) { return; }
    if (!(inv_is_finite(g.x) && inv_is_finite(g.y) && inv_is_finite(g.z))) {
        inv_cas_add(INV_U32_SCALARS + INV_GRAD_NONFINITE, 1.0);
        return;
    }
    let base = inv_u32_alb_base() + texel * 4u;
    inv_cas_add(base, g.x);
    inv_cas_add(base + 1u, g.y);
    inv_cas_add(base + 2u, g.z);
}

// ---------------------------------------------------------------------------
// Per-eval state clear
// ---------------------------------------------------------------------------
// Pixel state only — accum_hdr, merged reservoirs, welford. Runs once per
// replicate (each replicate is an independent ReSTIR chain); the gradient
// region is cleared separately so it accumulates across the replicate set.
@compute @workgroup_size(256)
fn main_inverse_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = uniforms.width * uniforms.height;
    let i = gid.y * INV_DISPATCH_X + gid.x;
    if (i < px) {
        accum_hdr[i] = vec4<f32>(0.0);
        terrain_reservoirs_prev[i] = RestirReservoir(
            RestirLightSample(vec3<f32>(0.0), 0u, vec3<f32>(0.0), 0.0, 0u,
                vec3<f32>(0.0)),
            0.0, 0u, 0.0, 0.0);
        terrain_welford[i] = TerrainStatistics(0.0, 0.0, 0u, 0u,
            vec4<u32>(0u), vec4<u32>(0u));
    }
}

// Gradient-state clear — scalars + per-pixel loss + per-texel albedo region.
// Runs ONCE per multi-replicate eval, before the forward sweep of the first
// replicate; per-replicate clears must not touch this region.
@compute @workgroup_size(256)
fn main_inv_clear_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * INV_DISPATCH_X + gid.x;
    let n_u32 = inv_u32_alb_base() + inv_params.dims.z * 4u;
    if (i < n_u32) {
        atomicStore(&inv_u32[i], 0u);
    }
}

// Reservoir-only clear for the checkpointed adjoint replay: the seed stream
// is a pure function of (seed, pixel, frame), so re-running the frame chain
// from a cleared prev-buffer reproduces the forward sweep's reservoir
// trajectory exactly. accum/welford are not gradient inputs.
@compute @workgroup_size(256)
fn main_inv_replay_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = uniforms.width * uniforms.height;
    let i = gid.y * INV_DISPATCH_X + gid.x;
    if (i < px) {
        terrain_reservoirs_prev[i] = RestirReservoir(
            RestirLightSample(vec3<f32>(0.0), 0u, vec3<f32>(0.0), 0.0, 0u,
                vec3<f32>(0.0)),
            0.0, 0u, 0.0, 0.0);
    }
}

// ---------------------------------------------------------------------------
// Reservoir weight snapshot — the shading state each frame actually used
// ---------------------------------------------------------------------------
// Runs after `main_terrain` (which M-clamps prev in place before shading)
// and before the temporal merge overwrites the buffer for the next frame.
// Records W_used and the selected spectral residual channel so the reverse
// pass differentiates the exact importance-sampled contribution.
@compute @workgroup_size(256)
fn main_inv_wsnap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = uniforms.width * uniforms.height;
    let pix = gid.y * INV_DISPATCH_X + gid.x;
    if (pix >= px) { return; }
    let r = terrain_reservoirs_prev[pix];
    let valid = terrain_spectral_restir_enabled() && uniforms.frame_index > 0u
        && r.m > 0u && r.weight > 0.0 && r.target_pdf > 0.0
        && r.sample.light_type == 1u && r.sample.light_index < 3u;
    let w_used = select(1.0, clamp(r.weight, 0.0, TERRAIN_RESTIR_W_CAP), valid);
    let channel = select(3u, r.sample.light_index, valid);
    // Checkpointed slot: only K frame histories are live at once.
    let slot = uniforms.frame_index % inv_whist_slots();
    inv_v4[inv_whist_index(slot, pix)] = vec4<f32>(w_used, f32(channel), 0.0, 0.0);
}

// ---------------------------------------------------------------------------
// Replicate-mean image accumulation
// ---------------------------------------------------------------------------
// Runs once per replicate after that replicate's forward sweep. Folds the
// replicate's mean linear radiance into the shared mean image with weight
// 1/R; the first replicate overwrites (rsv0.y < 0.5) so no prior clear of
// the mean region is needed.
@compute @workgroup_size(256)
fn main_inv_accum_mean(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = uniforms.width * uniforms.height;
    let pix = gid.y * INV_DISPATCH_X + gid.x;
    if (pix >= px) { return; }
    let acc = accum_hdr[pix];
    let lin = acc.rgb / max(acc.a, 1e-8);
    let mi = inv_mean_index(pix);
    let contrib = vec4<f32>(lin * inv_params.rsv0.x, 0.0);
    let prev = select(inv_v4[mi], vec4<f32>(0.0), inv_params.rsv0.y < 0.5);
    inv_v4[mi] = prev + contrib;
}

// ---------------------------------------------------------------------------
// Loss + adjoint in tonemapped sRGB display space
// ---------------------------------------------------------------------------
// loss_p = mean(diff^2) + struct_w * lum(diff)^2
// where cur = srgb(reinhard(lin * exposure)) and tgt = srgb(target_u8): the
// observed beauty quantizes the forward's Reinhard output (the path writes
// no OETF), so both sides are lifted through the sRGB OETF and the metric
// lives in display-encoded sRGB as the spec requires.
// adj_lin = dL/dlin = dL/dcur . srgb'(reinhard(lin)) . reinhard'(lin),
// folded with 1/pixel_count.
struct InvPixelLoss {
    value: f32,
    adjoint: vec3<f32>,
}

// Pure loss evaluation shared by the image loss pass and boundary replay.
// target_raw is the observed quantized Reinhard output, before the OETF.
// Both outputs are per pixel; callers apply the image normalization.
fn inv_pixel_loss(lin: vec3<f32>, target_raw: vec3<f32>, exposure: f32,
    struct_weight: f32) -> InvPixelLoss {
    let tm = reinhard_tonemap(lin, exposure);
    let cur = vec3<f32>(inv_srgb(tm.r), inv_srgb(tm.g), inv_srgb(tm.b));
    let tgt = vec3<f32>(inv_srgb(target_raw.r), inv_srgb(target_raw.g),
        inv_srgb(target_raw.b));
    let diff = cur - tgt;
    let lumw = vec3<f32>(0.2126, 0.7152, 0.0722);
    let ld = dot(lumw, diff);
    let value = dot(diff, diff) / 3.0 + struct_weight * ld * ld;
    var adj = (2.0 / 3.0) * diff + 2.0 * struct_weight * ld * lumw;
    adj *= vec3<f32>(inv_srgb_d(tm.r), inv_srgb_d(tm.g), inv_srgb_d(tm.b));
    let denom = vec3<f32>(1.0) + exposure * lin;
    let dtm = exposure / (denom * denom);
    return InvPixelLoss(value, adj * dtm);
}

@compute @workgroup_size(8, 8, 1)
fn main_inverse_loss(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) { return; }
    let gpix = gid.y * uniforms.width + gid.x;

    let acc = accum_hdr[gpix];
    var lin = acc.rgb / max(acc.a, 1e-8);
    if ((inv_params.ctrl.x & INV_FLAG_MEAN_LOSS) != 0u) {
        lin = inv_v4[inv_mean_index(gpix)].rgb;
    }
    let tgt_raw = textureLoad(inv_target_tex, vec2<i32>(i32(gid.x), i32(gid.y)), 0).rgb;
    let loss = inv_pixel_loss(lin, tgt_raw, uniforms.cam_exposure, inv_params.fl.x);
    atomicStore(&inv_u32[inv_u32_loss_base() + gpix],
        bitcast<u32>(loss.value));
    inv_v4[gpix] = vec4<f32>(loss.adjoint * inv_params.fl.z, 0.0);
}

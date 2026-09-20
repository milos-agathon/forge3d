// src/shaders/pt_edge_sample.wgsl
// DIFFERENTIA: replay complete nonlinear loss jumps at host-certified terrain
// visibility events. The host proves event support, first-hit ordering,
// receiver density and signed boundary velocity for the restricted contract.
// RELEVANT FILES: src/path_tracing/inverse/geometry_cert.rs,
// src/path_tracing/inverse/mod.rs, src/shaders/pt_inverse_loss.wgsl

struct InvEdgeEvent {
    ids: vec4<u32>, // pixel, frame, camera sample, certification marker (1)
    receiver: vec4<f32>, // xyz: receiver, w: certified receiver cell x
    normal: vec4<f32>, // xyz: normal, w: certified receiver cell z
    grad_weight: vec4<f32>,
}
@group(2) @binding(13) var<storage, read> inv_edge_events: array<InvEdgeEvent>;

// Actual shader inputs/results, consumed by the conservative host validator.
// Thirty-three sixteen-byte fields give the host an exact 528-byte storage
// stride.  The source rows bind the normal-offset IBL ray and both event
// radiances to the live UBOs/textures/random stream before the host accepts a
// terrain visibility branch or nonlinear-loss contribution.
struct InvEdgeAudit {
    camera_ray: vec4<f32>,
    camera_hit: vec4<f32>,
    ibl_ray: vec4<f32>,
    ibl_origin: vec4<f32>,
    lit_linear: vec4<f32>,
    shadow_linear: vec4<f32>,
    loss_jump: vec4<f32>,
    gradient: vec4<f32>,
    radiance_base: vec4<f32>, // without_original.rgb, inv_rfs
    radiance_sun: vec4<f32>, // unscaled direct sun.rgb, reserved
    radiance_ibl: vec4<f32>, // unscaled IBL.rgb, visibility (1 = visible)
    target_raw: vec4<f32>, // actual textureLoad target.rgb, reserved
    mean_linear: vec4<f32>, // exact inv_mean_index load.rgb, reserved
    original_linear: vec4<f32>, // exact selected primal replay.rgb, reserved
    camera_origin: vec4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    camera_forward: vec4<f32>,
    camera_projection: vec4<f32>,
    camera_dimensions: vec4<u32>,
    sun_provenance: vec4<f32>,
    reservoir_provenance: vec4<f32>,
    source_seed: vec4<u32>,
    source_terrain: vec4<u32>,
    source_light_direction: vec4<f32>,
    source_lighting: vec4<f32>,
    source_environment: vec4<f32>,
    environment_lookup: vec4<u32>,
    surface_albedo: vec4<f32>,
    surface_weights: vec4<f32>,
    surface_texels: vec4<u32>,
    environment_effective: vec4<f32>,
    sampled_spectrum: vec4<f32>,
}
@group(2) @binding(14) var<storage, read_write> inv_edge_audits: array<InvEdgeAudit>;
// Compact edge-only x/y/z sums, separate from smooth-path accumulation.
@group(2) @binding(15) var<storage, read_write> inv_edge_sums: array<atomic<u32>>;

fn inv_edge_add_scalar(index: u32, value: f32) {
    var old = atomicLoad(&inv_edge_sums[index]);
    loop {
        let next = bitcast<f32>(old) + value;
        if (!inv_is_finite(next)) {
            atomicOr(&inv_u32[INV_GRAD_SPARE7], 32u);
            return;
        }
        let result = atomicCompareExchangeWeak(&inv_edge_sums[index], old, bitcast<u32>(next));
        if (result.exchanged) { return; }
        old = result.old_value;
    }
}

fn inv_edge_finite3(v: vec3<f32>) -> bool {
    return inv_is_finite(v.x) && inv_is_finite(v.y) && inv_is_finite(v.z);
}

fn inv_edge_receiver_cell_valid(event: InvEdgeEvent) -> bool {
    let cell = vec2<f32>(event.receiver.w, event.normal.w);
    let spacing = terrain.origin_spacing.zw;
    if (!inv_is_finite(cell.x) || !inv_is_finite(cell.y)
        || any(cell < vec2<f32>(0.0)) || any(floor(cell) != cell)
        || any(cell >= vec2<f32>(terrain.dims.zw))
        || !inv_is_finite(spacing.x) || !inv_is_finite(spacing.y)
        || any(spacing <= vec2<f32>(0.0))) {
        return false;
    }
    // Recompute with the actual shader arithmetic: a host projection can
    // round onto a crease even when the ideal receiver is inside its cell.
    let grid = (event.receiver.xz - terrain.origin_spacing.xy) / spacing;
    return inv_is_finite(grid.x) && inv_is_finite(grid.y)
        && all(grid > cell) && all(grid < cell + vec2<f32>(1.0));
}

fn inv_edge_sun_occluded(p: vec3<f32>, n: vec3<f32>, d: vec3<f32>) -> bool {
    return intersect_shadow_ray(Ray(p + n * 1e-3, 1e-3, d, 1e30), 1e30);
}

fn inv_edge_ibl_occluded(p: vec3<f32>, n: vec3<f32>, d: vec3<f32>) -> bool {
    return intersect_ibl_occlusion_ray(Ray(p + n * 1e-3, 1e-3, d, 1e30), 1e30);
}

fn inv_edge_ibl(p: vec3<f32>, n: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
    let d = terrain_cosine_dir(n, u1, u2);
    let visibility = select(1.0, 0.0, inv_edge_ibl_occluded(p, n, d));
    return terrain_albedo_at(p) * terrain_env_effective(d) * visibility;
}

// Keep the certified edge path's phase within WGSL's documented sin/cos
// domain [-pi, pi].  This is the same cosine distribution as the forward
// sampler: values in (pi, 2*pi) differ only by one full turn.  It is local
// to the independent boundary IBL draw, so it does not alter primal replay.
fn inv_edge_cosine_dir(n: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
    let sign = select(1.0, -1.0, n.z < 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bt = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
    let r = sqrt(u1);
    let raw_phi = 2.0 * TERRAIN_PI * u2;
    let phi = select(raw_phi, raw_phi - 2.0 * TERRAIN_PI, raw_phi > TERRAIN_PI);
    let local = vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
    return normalize(local.x * t + local.y * bt + local.z * n);
}

@compute @workgroup_size(256)
fn main_inverse_edge_certified(@builtin(global_invocation_id) gid: vec3<u32>) {
    if ((inv_params.ctrl.x & INV_FLAG_EDGE) == 0u) { return; }
    let index = gid.y * INV_DISPATCH_X + gid.x;
    if (index >= inv_params.ctrl.z) { return; }
    if (index >= arrayLength(&inv_edge_events)) {
        atomicOr(&inv_u32[INV_GRAD_SPARE7], 1u);
        return;
    }
    if (index >= arrayLength(&inv_edge_audits) || arrayLength(&inv_edge_sums) < 3u) {
        atomicOr(&inv_u32[INV_GRAD_SPARE7], 4u);
        return;
    }
    let event = inv_edge_events[index];
    let pix = event.ids.x;
    let frame = event.ids.y;
    let sample_index = event.ids.z;
    let spp = max(terrain.extra.x, 1u);
    let pixels = uniforms.width * uniforms.height;
    if (event.ids.w != 1u || pix >= pixels || frame >= inv_params.dims.w
        || sample_index >= spp || earth_curvature.enabled != 0u
        || !terrain_spectral_restir_enabled()) {
        atomicOr(&inv_u32[INV_GRAD_SPARE7], 1u);
        return;
    }
    // The host dispatches every record for each replay tile; consume a record
    // only while its exact reservoir weight/channel snapshot is resident.
    let tile_end = uniforms.frame_index;
    let tile_base = tile_end - tile_end % inv_whist_slots();
    if (frame < tile_base || frame > tile_end) { return; }
    let wh = inv_v4[inv_whist_index(frame % inv_whist_slots(), pix)];
    let half_h = tan(0.5 * uniforms.cam_fov_y);
    let half_w = uniforms.cam_aspect * half_h;
    if (!inv_is_finite(wh.x) || !inv_is_finite(wh.y)
        || wh.x < 0.0 || wh.y < 0.0 || wh.y > 3.0 || floor(wh.y) != wh.y
        || !inv_edge_finite3(uniforms.cam_origin)
        || !inv_edge_finite3(uniforms.cam_right)
        || !inv_edge_finite3(uniforms.cam_up)
        || !inv_edge_finite3(uniforms.cam_forward)
        || !inv_edge_finite3(lighting.light_dir)
        || !(dot(lighting.light_dir, lighting.light_dir) > 0.0)
        || !inv_is_finite(uniforms.cam_fov_y) || !inv_is_finite(uniforms.cam_aspect)
        || !inv_is_finite(uniforms.cam_exposure)
        || !inv_is_finite(half_h) || !inv_is_finite(half_w)
        || !(half_h > 0.0) || !(half_w > 0.0)
        || uniforms.width == 0u || uniforms.height == 0u
        || !inv_is_finite(inv_params.rsv0.x) || !inv_is_finite(inv_params.fl.w)
        || !(inv_params.rsv0.x > 0.0) || !(inv_params.fl.w > 0.0)) {
        atomicOr(&inv_u32[INV_GRAD_SPARE7], 1u);
        return;
    }
    var omega = normalize(lighting.light_dir);
    // Match the forward pass's extra normalization for reused reservoirs.
    if (wh.y < 3.0) { omega = normalize(omega); }
    if (!inv_edge_finite3(omega) || !(dot(omega, omega) > 0.0)) {
        atomicOr(&inv_u32[INV_GRAD_SPARE7], 1u);
        return;
    }
    inv_edge_audits[index].camera_origin = vec4<f32>(uniforms.cam_origin, uniforms.cam_fov_y);
    inv_edge_audits[index].camera_right = vec4<f32>(uniforms.cam_right, uniforms.cam_aspect);
    inv_edge_audits[index].camera_up = vec4<f32>(uniforms.cam_up, uniforms.cam_exposure);
    inv_edge_audits[index].camera_forward = vec4<f32>(uniforms.cam_forward, 0.0);
    inv_edge_audits[index].camera_projection = vec4<f32>(0.0, 0.0, half_h, half_w);
    inv_edge_audits[index].camera_dimensions = vec4<u32>(uniforms.width, uniforms.height,
        pix % uniforms.width, pix / uniforms.width);
    // Capture all source operands before the probe early return.  The final
    // event must match this f32 token bit-for-bit, and the host separately
    // checks it against the values uploaded for the selected replica.
    var env_st = terrain_seed_hash(inv_seed(pix, frame)
        ^ terrain_seed_hash(sample_index) ^ 0xa511e9b3u);
    env_st = select(env_st, 0x6d2b79f5u, env_st == 0u);
    let env_u1 = xorshift32(&env_st);
    let env_u2 = xorshift32(&env_st);
    inv_edge_audits[index].sun_provenance = vec4<f32>(omega, select(0.0, 1.0, wh.y < 3.0));
    inv_edge_audits[index].reservoir_provenance = vec4<f32>(wh.xy, env_u1, env_u2);
    inv_edge_audits[index].source_seed = vec4<u32>(
        uniforms.seed_hi, uniforms.seed_lo, frame, sample_index);
    inv_edge_audits[index].source_terrain = vec4<u32>(
        terrain.dims.x, terrain.dims.y, terrain.mips.y, terrain.mips.z);
    inv_edge_audits[index].source_light_direction = vec4<f32>(lighting.light_dir, 0.0);
    inv_edge_audits[index].source_lighting = vec4<f32>(lighting.light_color, terrain.albedo_pad.w);
    inv_edge_audits[index].source_environment = vec4<f32>(
        terrain.h_params.w, f32(terrain.mips.w), 0.0, 0.0);
    if ((inv_params.ctrl.x & INV_FLAG_EDGE_PROBE) != 0u) { return; }
    if (!inv_edge_finite3(event.receiver.xyz) || !inv_edge_finite3(event.normal.xyz)
        || !inv_edge_finite3(event.grad_weight.xyz) || !inv_edge_receiver_cell_valid(event)
        || !(dot(event.normal.xyz, event.normal.xyz) > 0.0)) {
        atomicOr(&inv_u32[INV_GRAD_SPARE7], 1u);
        return;
    }
    let camera_delta = event.receiver.xyz - uniforms.cam_origin;
    if (!inv_edge_finite3(camera_delta) || !(dot(camera_delta, camera_delta) > 0.0)) {
        atomicOr(&inv_u32[INV_GRAD_SPARE7], 8u);
        return;
    }
    // Invert the actual (not assumed orthonormal) UBO basis. The determinant
    // cancels in the perspective ratios, but a singular basis is unresolved.
    let basis_cross = cross(uniforms.cam_right, uniforms.cam_up);
    let determinant = dot(uniforms.cam_forward, basis_cross);
    let projected_z = dot(camera_delta, basis_cross);
    let projected_x = dot(camera_delta, cross(uniforms.cam_up, uniforms.cam_forward));
    let projected_y = dot(camera_delta, cross(uniforms.cam_forward, uniforms.cam_right));
    let ndc_x = (projected_x / projected_z) / half_w;
    let ndc_y = (projected_y / projected_z) / half_h;
    let jx = ((ndc_x + 1.0) * 0.5) * f32(uniforms.width)
        - (f32(pix % uniforms.width) + 0.5);
    let jy = ((1.0 - ndc_y) * 0.5) * f32(uniforms.height)
        - (f32(pix / uniforms.width) + 0.5);
    if (!inv_is_finite(determinant) || determinant == 0.0
        || !inv_is_finite(projected_z) || !(projected_z / determinant > 0.0)
        || !inv_is_finite(jx) || !inv_is_finite(jy)
        || !(jx > -0.5 && jx < 0.5 && jy > -0.5 && jy < 0.5)) {
        atomicOr(&inv_u32[INV_GRAD_SPARE7], 8u);
        return;
    }
    inv_edge_audits[index].camera_projection = vec4<f32>(jx, jy, half_h, half_w);
    let camera_ray = inv_cam_ray(pix % uniforms.width, pix / uniforms.width, jx, jy);
    let camera_direction = camera_ray.direction;
    let camera_hit = terrain_intersect(camera_ray);
    inv_edge_audits[index].camera_ray = vec4<f32>(camera_direction, f32(camera_hit.hit));
    inv_edge_audits[index].camera_hit = vec4<f32>(camera_hit.point, camera_hit.t);
    let camera_grid = (camera_hit.point.xz - terrain.origin_spacing.xy) / terrain.origin_spacing.zw;
    let receiver_cell = vec2<f32>(event.receiver.w, event.normal.w);
    if (camera_hit.hit == 0u || !inv_edge_finite3(camera_direction)
        || !inv_edge_finite3(camera_hit.point) || !inv_is_finite(camera_hit.t)
        || !inv_is_finite(camera_grid.x) || !inv_is_finite(camera_grid.y)
        || any(camera_grid <= receiver_cell) || any(camera_grid >= receiver_cell + vec2<f32>(1.0))) {
        atomicOr(&inv_u32[INV_GRAD_SPARE7], 8u);
        return;
    }
    let spectrum = lighting.light_color * terrain_sun_transmit(omega.y);
    var sampled_spectrum = spectrum;
    if (wh.y < 3.0) {
        let baseline = min(spectrum.x, min(spectrum.y, spectrum.z));
        let residual = max(spectrum - vec3<f32>(baseline), vec3<f32>(0.0));
        sampled_spectrum = vec3<f32>(baseline)
            + terrain_spectral_channel_mask(u32(wh.y)) * residual * wh.x;
    }

    // Replay the original sample exactly, including the conditional draws
    // consumed by every earlier hit. Misses consume no environment draws.
    var st = inv_seed(pix, frame);
    var original = vec3<f32>(0.0);
    for (var s = 0u; s <= sample_index; s += 1u) {
        let jx = terrain_tent_offset(xorshift32(&st)) * 0.5;
        let jy = terrain_tent_offset(xorshift32(&st)) * 0.5;
        let ray = inv_cam_ray(pix % uniforms.width, pix / uniforms.width, jx, jy);
        let hit = terrain_intersect(ray);
        if (hit.hit == 0u) {
            if (s == sample_index) { original = terrain_env_effective(ray.direction); }
            continue;
        }
        let u1 = xorshift32(&st);
        let u2 = xorshift32(&st);
        if (s != sample_index) { continue; }
        var visibility = 1.0;
        let nd = max(dot(hit.normal, omega), 0.0);
        if (nd > 0.0 && lighting.shadows_enabled != 0u
            && inv_edge_sun_occluded(hit.point, hit.normal, omega)) {
            visibility = 0.0;
        }
        let sun = terrain_albedo_at(hit.point) * sampled_spectrum * nd * visibility;
        original = sun + inv_edge_ibl(hit.point, hit.normal, u1, u2);
    }

    // This is an independent coupling for the ideal independent-draw
    // objective, not a deterministic counterfactual of the xorshift tape.
    // The certified event must remain a terrain camera hit on both shadow
    // sides. Companion samples and spectral history retain their original
    // independent draws; boundary IBL uses a separate keyed random stream.
    let p = event.receiver.xyz;
    let n = event.normal.xyz;
    let ibl_direction = inv_edge_cosine_dir(n, env_u1, env_u2);
    let ibl_origin = p + n * 1e-3;
    let ibl_occluded = intersect_ibl_occlusion_ray(Ray(ibl_origin, 1e-3, ibl_direction, 1e30), 1e30);
    let surface = terrain_albedo_taps_at(p);
    let environment = terrain_env_effective(ibl_direction);
    let env_d = normalize(ibl_direction);
    let env_ew = terrain.mips.z;
    let env_eh = terrain.mips.w;
    var env_lookup = vec4<u32>(0u, 0u, env_ew, env_eh);
    if (env_ew != 0u && env_eh != 0u) {
        let env_u = (atan2(env_d.z, env_d.x) / (2.0 * TERRAIN_PI)) + 0.5;
        let env_v = acos(clamp(env_d.y, -1.0, 1.0)) / TERRAIN_PI;
        env_lookup.x = min(u32(env_u * f32(env_ew)), env_ew - 1u);
        env_lookup.y = min(u32(env_v * f32(env_eh)), env_eh - 1u);
    }
    let nd = max(dot(n, omega), 0.0);
    let ibl = surface.alb * environment
        * select(1.0, 0.0, ibl_occluded);
    inv_edge_audits[index].ibl_ray = vec4<f32>(ibl_direction, select(0.0, 1.0, ibl_occluded));
    inv_edge_audits[index].ibl_origin = vec4<f32>(ibl_origin, 0.0);
    inv_edge_audits[index].surface_albedo = vec4<f32>(surface.alb, nd);
    inv_edge_audits[index].surface_weights = surface.weights;
    inv_edge_audits[index].surface_texels = surface.texels;
    inv_edge_audits[index].environment_lookup = env_lookup;
    inv_edge_audits[index].environment_effective = vec4<f32>(environment, 0.0);
    inv_edge_audits[index].sampled_spectrum = vec4<f32>(sampled_spectrum, 0.0);
    let sun = surface.alb * sampled_spectrum * nd;
    let inv_rfs = inv_params.rsv0.x * inv_params.fl.w;
    let mean_linear = inv_v4[inv_mean_index(pix)].rgb;
    let without_original = mean_linear - original * inv_rfs;
    let observed_rgb = textureLoad(inv_target_tex,
        vec2<i32>(i32(pix % uniforms.width), i32(pix / uniforms.width)), 0).rgb;
    inv_edge_audits[index].target_raw = vec4<f32>(observed_rgb, 0.0);
    inv_edge_audits[index].mean_linear = vec4<f32>(mean_linear, 0.0);
    inv_edge_audits[index].original_linear = vec4<f32>(original, 0.0);
    inv_edge_audits[index].radiance_base = vec4<f32>(without_original, inv_rfs);
    inv_edge_audits[index].radiance_sun = vec4<f32>(sun, 0.0);
    inv_edge_audits[index].radiance_ibl = vec4<f32>(ibl, select(1.0, 0.0, ibl_occluded));
    let lit_linear = without_original + (sun + ibl) * inv_rfs;
    let shadow_linear = without_original + ibl * inv_rfs;
    let lit = inv_pixel_loss(lit_linear,
        observed_rgb, uniforms.cam_exposure, inv_params.fl.x).value;
    let shadow = inv_pixel_loss(shadow_linear,
        observed_rgb, uniforms.cam_exposure, inv_params.fl.x).value;
    // Host divides the accumulated scalar gradients by R after all replay
    // replicates. Cancel that averaging for the complete nonlinear jump.
    let jump = (lit - shadow) * inv_params.fl.z / inv_params.rsv0.x;
    let gradient = jump * event.grad_weight.xyz;
    inv_edge_audits[index].lit_linear = vec4<f32>(lit_linear, 0.0);
    inv_edge_audits[index].shadow_linear = vec4<f32>(shadow_linear, 0.0);
    inv_edge_audits[index].loss_jump = vec4<f32>(lit, shadow, jump, 0.0);
    inv_edge_audits[index].gradient = vec4<f32>(gradient, 0.0);
    if (!inv_edge_finite3(ibl_direction) || !inv_edge_finite3(ibl_origin)
        || !inv_edge_finite3(mean_linear) || !inv_edge_finite3(original)
        || !inv_edge_finite3(lit_linear) || !inv_edge_finite3(shadow_linear)
        || !inv_is_finite(lit) || !inv_is_finite(shadow)) {
        atomicOr(&inv_u32[INV_GRAD_SPARE7], 16u);
        return;
    }
    if (!inv_is_finite(jump) || !inv_edge_finite3(gradient)) {
        atomicOr(&inv_u32[INV_GRAD_SPARE7], 2u);
        return;
    }
    inv_add_scalar(INV_GRAD_EDGE_MASS, abs(jump));
    inv_edge_add_scalar(0u, gradient.x);
    inv_edge_add_scalar(1u, gradient.y);
    inv_edge_add_scalar(2u, gradient.z);
}

// gid.y folds 1-D dispatches wider than the 65535-workgroup x limit:
// INV_DISPATCH_X (declared in pt_inverse_loss.wgsl, same module) scales it.

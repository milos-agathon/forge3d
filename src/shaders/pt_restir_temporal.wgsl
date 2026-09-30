// src/shaders/pt_restir_temporal.wgsl
// ReSTIR DI: temporal reuse — canonical reservoir merge of the previous
// frame's merged reservoir with the current frame's fresh candidates.
// Each input contributes its reuse weight w_j = w_sum_j / target_pdf_j (the
// receiver re-evaluation term is the identity here: both reservoirs describe
// the same pixel, so the selected sample's re-evaluated pdf is its stored
// target pdf by construction). The selected sample is drawn stochastically
// with probability proportional to its weight — a real reservoir update,
// not an argmax heuristic.

struct Uniforms {
    width: u32,
    height: u32,
    frame_index: u32,
    spp: u32,
    cam_origin: vec3<f32>,
    cam_fov_y: f32,
    cam_right: vec3<f32>,
    cam_aspect: f32,
    cam_up: vec3<f32>,
    cam_exposure: f32,
    cam_forward: vec3<f32>,
    seed_hi: u32,
    seed_lo: u32,
    camera_model: u32,
    full_width: u32,
    full_height: u32,
    pixel_offset_x: u32,
    pixel_offset_y: u32,
    ortho_half_height: f32,
    camera_flags: u32,
    sensor_rect: vec4<f32>,
}

struct LightSample {
    position: vec3<f32>,
    light_index: u32,
    direction: vec3<f32>,
    intensity: f32,
    light_type: u32,
    params: vec3<f32>,
}

struct Reservoir {
    sample: LightSample,
    w_sum: f32,
    m: u32,
    weight: f32,
    target_pdf: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// Temporal bind group (Group 2): prev, curr, out
@group(2) @binding(0) var<storage, read> prev_reservoirs: array<Reservoir>;
@group(2) @binding(1) var<storage, read> curr_reservoirs: array<Reservoir>;
@group(2) @binding(2) var<storage, read_write> out_reservoirs: array<Reservoir>;

fn xorshift32(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x ^= (x << 13u);
    x ^= (x >> 17u);
    x ^= (x << 5u);
    *state = x;
    return f32(x) / 4294967296.0;
}

// gid.y folds 1-D dispatches wider than the 65535-workgroup x limit: the
// host dispatches x = min(wg, 65535), y = ceil(wg/65535).
const DISPATCH_X_WG: u32 = 16776960u; // 65535 * 256

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.y * DISPATCH_X_WG + gid.x;
    let pixel_count = uniforms.width * uniforms.height;
    if (idx >= pixel_count) { return; }

    let local_x = idx % uniforms.width;
    let local_y = idx / uniforms.width;
    let global_idx = (local_y + uniforms.pixel_offset_y) * uniforms.full_width
        + local_x + uniforms.pixel_offset_x;
    var seed = (uniforms.seed_hi ^ uniforms.frame_index) + global_idx * 7411u + 7u;

    let rp = prev_reservoirs[idx];
    let rc = curr_reservoirs[idx];
    var ro: Reservoir;

    let prev_valid = (rp.m > 0u) && (rp.weight > 0.0) && (rp.target_pdf > 0.0);
    let curr_valid = (rc.m > 0u) && (rc.weight > 0.0) && (rc.target_pdf > 0.0);

    if (!prev_valid && !curr_valid) {
        ro = rc;
        out_reservoirs[idx] = ro;
        return;
    }
    if (!prev_valid) {
        ro = rc;
        out_reservoirs[idx] = ro;
        return;
    }
    if (!curr_valid) {
        ro = rp;
        out_reservoirs[idx] = ro;
        return;
    }

    // Stochastic merge over the reuse weights: the chosen sample lands with
    // probability w_j / (w_p + w_c). The canonical contribution
    // p̂_r(y_j)·W_j·M_j telescopes to w_sum_j for a same-pixel merge because
    // the receiver re-evaluation equals the stored target pdf exactly.
    let w_p = rp.w_sum;
    let w_c = rc.w_sum;
    let w_sum = w_p + w_c;
    let u = xorshift32(&seed);
    let pick_prev = (u * w_sum) < w_p;

    if (pick_prev) {
        ro.sample = rp.sample;
    } else {
        ro.sample = rc.sample;
    }
    ro.target_pdf = select(rc.target_pdf, rp.target_pdf, pick_prev);
    ro.m = rp.m + rc.m;
    ro.w_sum = w_sum;
    if (ro.m > 0u && ro.target_pdf > 0.0) {
        ro.weight = ro.w_sum / (f32(ro.m) * ro.target_pdf);
    } else {
        ro.weight = 0.0;
    }

    out_reservoirs[idx] = ro;
}

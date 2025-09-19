// src/shaders/pt_restir_init.wgsl
// ReSTIR DI: Initial reservoir population (MVP stub)
// Writes a minimal reservoir per pixel by sampling a light via alias table.

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
    _pad: u32,
}

// Scene lights (Group 1) and per-pixel G-buffer (written at primary hit in shading)
const PI: f32 = 3.14159265358979323846;
struct AreaLight {
    position: vec3<f32>,
    radius: f32,
    normal: vec3<f32>,
    intensity: f32,
    color: vec3<f32>,
    importance: f32,
}

struct DirectionalLight {
    direction: vec3<f32>,
    intensity: f32,
    color: vec3<f32>,
    importance: f32,
}

@group(1) @binding(4) var<storage, read> area_lights: array<AreaLight>;
@group(1) @binding(5) var<storage, read> directional_lights: array<DirectionalLight>;
@group(1) @binding(10) var<storage, read_write> restir_gbuffer: array<vec4<f32>>;      // normal.xyz, roughness
@group(1) @binding(11) var<storage, read_write> restir_gbuffer_pos: array<vec4<f32>>;  // world pos.xyz, 1

struct LightSample {
    position: vec3<f32>,
    light_index: u32,
    direction: vec3<f32>,
    intensity: f32,
    light_type: u32,
    params: vec3<f32>,
}

struct AliasEntry {
    prob: f32,
    alias_idx: u32,
}

struct Reservoir {
    sample: LightSample,
    w_sum: f32,
    m: u32,
    weight: f32,
    target_pdf: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// Group 1 (scene) is present in pipeline layout but not used here in MVP
@group(2) @binding(0) var<storage, read_write> reservoirs: array<Reservoir>;
@group(2) @binding(1) var<storage, read> light_samples: array<LightSample>;
@group(2) @binding(2) var<storage, read> alias_entries: array<AliasEntry>;
@group(2) @binding(3) var<storage, read> light_probs: array<f32>;

fn xorshift32(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x ^= (x << 13u);
    x ^= (x >> 17u);
    x ^= (x << 5u);
    *state = x;
    return f32(x) / 4294967296.0;
}

fn alias_sample(u: f32) -> u32 {
    let n = arrayLength(&alias_entries);
    if (n == 0u) { return 0u; }
    let s = u * f32(n);
    let bin = u32(clamp(floor(s), 0.0, f32(n - 1u)));
    let frac = s - floor(s);
    let e = alias_entries[bin];
    return select(e.alias_idx, bin, frac < e.prob);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let pixel_count = uniforms.width * uniforms.height;
    if (idx >= pixel_count) { return; }

    var seed = (uniforms.seed_hi ^ uniforms.frame_index) + idx * 9781u + 1u;
    let n_alias = arrayLength(&alias_entries);
    let n_lights = arrayLength(&light_samples);

    var r: Reservoir;
    if (n_alias > 0u && n_lights > 0u) {
        // Reservoir sampling over K candidates using alias table
        let K: u32 = 4u;
        r.w_sum = 0.0;
        r.m = 0u;
        r.weight = 0.0;
        r.target_pdf = 0.0;
        // Initialize sample to zero
        r.sample.position = vec3<f32>(0.0, 0.0, 0.0);
        r.sample.light_index = 0u;
        r.sample.direction = vec3<f32>(0.0, 1.0, 0.0);
        r.sample.intensity = 0.0;
        r.sample.light_type = 0u;
        r.sample.params = vec3<f32>(0.0, 0.0, 0.0);
        // Fetch per-pixel shading context from last frame's G-buffer
        let P = restir_gbuffer_pos[idx].xyz;
        let N = normalize(restir_gbuffer[idx].xyz);

        for (var i: u32 = 0u; i < K; i = i + 1u) {
            let u = xorshift32(&seed);
            let li = alias_sample(u);
            let lidx = min(li, n_lights - 1u);
            let ls = light_samples[lidx];
            let p_sel = select(1.0 / f32(n_lights), light_probs[lidx], arrayLength(&light_probs) >= n_lights);

            // Evaluate exact per-pixel target pdf
            var p_curr: f32 = 0.0;
            if (ls.light_type == 1u) {
                // Directional: delta distribution -> selection probability only
                let wi = normalize(ls.direction);
                let cosTheta = max(dot(N, wi), 0.0);
                if (cosTheta > 0.0) {
                    p_curr = p_sel;
                }
            } else if (ls.light_type == 2u) {
                // Area disc: selection probability times area->solid-angle conversion
                let L = ls.position;
                let dirv = L - P;
                let d = length(dirv);
                if (d > 1e-6) {
                    let wi = dirv / d;
                    let cosTheta = max(dot(N, wi), 0.0);
                    if (cosTheta > 0.0) {
                        let aidx = ls.light_index;
                        let nL = normalize(area_lights[min(aidx, arrayLength(&area_lights) - 1u)].normal);
                        let R = max(area_lights[min(aidx, arrayLength(&area_lights) - 1u)].radius, 1e-6);
                        let cos_on_light = max(dot(nL, -wi), 0.0);
                        if (cos_on_light > 0.0) {
                            let area = PI * R * R;
                            let p_area = 1.0 / area;
                            let p_solid = p_area * (d * d) / max(cos_on_light, 1e-6);
                            p_curr = p_sel * p_solid;
                        }
                    }
                }
            } else {
                // Unknown light type: fall back to selection-only
                p_curr = p_sel;
            }

            // Candidate weight is the target pdf
            let w = p_curr;
            r.w_sum = r.w_sum + w;
            r.m = r.m + 1u;
            // Reservoir update: accept with probability w / w_sum
            let accept = xorshift32(&seed) * r.w_sum <= w;
            if (accept) {
                r.sample = ls;
                r.target_pdf = p_curr;
            }
        }
        // Finalize weight
        if (r.w_sum > 0.0 && r.target_pdf > 0.0 && r.m > 0u) {
            r.weight = r.w_sum / (f32(r.m) * r.target_pdf);
        } else {
            r.weight = 0.0;
        }
    } else {
        // Zero/identity sample
        r.sample.position = vec3<f32>(0.0, 0.0, 0.0);
        r.sample.light_index = 0u;
        r.sample.direction = vec3<f32>(0.0, 1.0, 0.0);
        r.sample.intensity = 0.0;
        r.sample.light_type = 0u;
        r.sample.params = vec3<f32>(0.0, 0.0, 0.0);
        r.w_sum = 0.0;
        r.m = 0u;
        r.weight = 0.0;
        r.target_pdf = 0.0;
    }

    reservoirs[idx] = r;
}

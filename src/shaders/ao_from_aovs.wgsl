// src/shaders/ao_from_aovs.wgsl
// Screen-space Ambient Occlusion from AOV buffers (depth, normal)
// Minimal cosine-hemisphere sampling around normal for offline validation.

struct AOUniforms {
    width: u32,
    height: u32,
    samples: u32,
    intensity: f32,
    bias: f32,
    seed: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read> aov_depth_buf: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> aov_normal_buf: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> ao_out_buf: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> ao_params: AOUniforms;

fn rng_hash(x: u32) -> u32 { return x ^ 0x9E3779B9u ^ (x << 6u) ^ (x >> 2u); }

fn rand01(seed_in: ptr<function, u32>) -> f32 {
    var s = *seed_in;
    s ^= (s << 13u);
    s ^= (s >> 17u);
    s ^= (s << 5u);
    *seed_in = s;
    return f32(s) / 4294967296.0;
}

fn cosine_hemisphere(u1: f32, u2: f32) -> vec3<f32> {
    let r = sqrt(u1);
    let phi = 6.283185307179586 * u2;
    let x = r * cos(phi);
    let y = r * sin(phi);
    let z = sqrt(max(0.0, 1.0 - u1));
    return vec3<f32>(x, y, z);
}

// Build ONB from normal; columns are (t, b, n)
fn make_basis(n: vec3<f32>) -> mat3x3<f32> {
    let sign = select(1.0, -1.0, n.z < 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bb = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
    return mat3x3<f32>(t, bb, n);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = ao_params.width;
    let height = ao_params.height;
    if (gid.x >= width || gid.y >= height) { return; }
    let pix = gid.y * width + gid.x;

    let n = normalize(aov_normal_buf[pix].xyz);
    // Basic guard: if no normal (background), write 1.0 (fully unoccluded)
    if (all(n == vec3<f32>(0.0))) {
        ao_out_buf[pix] = vec4<f32>(1.0, 0.0, 0.0, 1.0);
        return;
    }

    let basis = make_basis(n);
    var seed = rng_hash(pix ^ ao_params.seed);

    var sum_occ = 0.0;
    let s_count = max(1u, ao_params.samples);
    for (var i: u32 = 0u; i < s_count; i = i + 1u) {
        let u1 = rand01(&seed);
        let u2 = rand01(&seed);
        let d_local = cosine_hemisphere(u1, u2);
        let d_world = basis * d_local;
        // Cosine AO proxy: occlusion proportional to n·d
        let occ = max(0.0, dot(n, d_world));
        sum_occ = sum_occ + occ;
    }
    var ao = sum_occ / f32(s_count);
    // Convert to occlusion factor (1 - AO) and apply intensity
    ao = max(0.0, 1.0 - ao * ao_params.intensity);

    ao_out_buf[pix] = vec4<f32>(ao, 0.0, 0.0, 1.0);
}

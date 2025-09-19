// src/shaders/pt_raygen.wgsl
// Wavefront Path Tracer: Ray Generation Stage
// This file exists to generate primary rays and push them into the ray queue for wavefront processing.
// RELEVANT FILES:src/path_tracing/wavefront/mod.rs,src/shaders/pt_intersect.wgsl,src/shaders/pt_kernel.wgsl

// Bind Group 0: Uniforms (width, height, frame_index, camera params, exposure, seed_hi/lo)
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

// Optional RestirSettings in scene group for QMC mode/control (A16)
struct RestirSettings {
    debug_aov_mode: u32,
    qmc_mode: u32,                 // 0=halton/vdc, 1=sobol
    adaptive_threshold_u32: u32,
    _pad: u32,
};
@group(1) @binding(12) var<uniform> restir_settings: RestirSettings;

// Bind Group 1: Scene (readonly storage: materials, textures/handles, accel/BVH)
struct Sphere {
    center: vec3<f32>,
    radius: f32,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ior: f32,
    emissive: vec3<f32>,
    ax: f32, // anisotropic alpha_x
    ay: f32, // anisotropic alpha_y
}

// Bind Group 2: Queues (read/write storage buffers with atomic counters)
struct Ray {
    o: vec3<f32>,           // origin
    tmin: f32,              // minimum ray parameter  
    d: vec3<f32>,           // direction
    tmax: f32,              // maximum ray parameter
    throughput: vec3<f32>,  // path throughput
    pdf: f32,               // path pdf
    pixel: u32,             // pixel index
    depth: u32,             // bounce depth
    rng_hi: u32,            // RNG state high
    rng_lo: u32,            // RNG state low
}

struct QueueHeader {
    in_count: atomic<u32>,  // number of items pushed
    out_count: atomic<u32>, // number of items popped
    capacity: u32,          // maximum capacity
    _pad: u32,
}

// Bind Group 3: Accum/Output (HDR accum buffer or storage texture)
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> scene_spheres: array<Sphere>;
@group(2) @binding(0) var<storage, read_write> ray_queue_header: QueueHeader;
@group(2) @binding(1) var<storage, read_write> ray_queue: array<Ray>;
@group(3) @binding(0) var<storage, read_write> accum_hdr: array<vec4<f32>>;

// XorShift32 RNG for consistency with mega-kernel
fn xorshift32(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x ^= (x << 13u);
    x ^= (x >> 17u);
    x ^= (x << 5u);
    *state = x;
    return f32(x) / 4294967296.0;
}

// Tent filter for anti-aliasing
fn tent_filter(u: f32) -> f32 {
    let t = 2.0 * u - 1.0;
    return select(1.0 + t, 1.0 - t, t < 0.0);
}

// -----------------------------------------------------------------------------
// QMC helpers (A16): Sobol/Owen + Halton/VDC fallback with CP rotation
// -----------------------------------------------------------------------------
fn radical_inverse_vdc(n_in: u32) -> f32 {
    var n = n_in;
    // Van der Corput base 2
    n = (n << 16u) | (n >> 16u);
    n = ((n & 0x55555555u) << 1u) | ((n & 0xAAAAAAAAu) >> 1u);
    n = ((n & 0x33333333u) << 2u) | ((n & 0xCCCCCCCCu) >> 2u);
    n = ((n & 0x0F0F0F0Fu) << 4u) | ((n & 0xF0F0F0F0u) >> 4u);
    n = ((n & 0x00FF00FFu) << 8u) | ((n & 0xFF00FF00u) >> 8u);
    return f32(n) * 2.3283064365386963e-10; // 1/2^32
}

fn halton_base3(i: u32) -> f32 {
    var f: f32 = 1.0;
    var r: f32 = 0.0;
    var n: u32 = i;
    let b: f32 = 3.0;
    loop {
        if (n == 0u) { break; }
        f = f / b;
        let digit: f32 = f32(n % 3u);
        r = r + digit * f;
        n = n / 3u;
    }
    return r;
}
// Sobol 2D with simple direction numbers; Owen-style scramble via XOR hash.
fn sobol_dir_v(d: u32, i: u32) -> u32 {
    // First 32 direction numbers for dim 0/1 (precomputed for primitive polynomials)
    // dim 0: v_j = 1 << (31-j)
    if (d == 0u) {
        return 0x80000000u >> i;
    }
    // dim 1: use standard set for direction numbers (m = [1,3,5,15,17,51,...]) approximated by shifts
    // Fallback to simple pattern to avoid large tables
    let base: u32 = 0x80000000u >> i;
    // XOR a rotated version for dim 1
    let rot = (base >> 1) ^ (base >> 3);
    return base ^ rot;
}

fn sobol2(i: u32) -> vec2<f32> {
    var xbits: u32 = 0u;
    var ybits: u32 = 0u;
    var idx = i;
    var j: u32 = 0u;
    loop {
        if (j >= 32u) { break; }
        if ((idx & 1u) == 1u) {
            xbits ^= sobol_dir_v(0u, j);
            ybits ^= sobol_dir_v(1u, j);
        }
        idx >>= 1u;
        j = j + 1u;
    }
    let inv = 1.0 / 4294967296.0; // 1/2^32
    return vec2<f32>(f32(xbits) * inv, f32(ybits) * inv);
}

fn cp_rotate(u: f32, r: f32) -> f32 {
    // Cranley-Patterson rotation in [0,1)
    let x = u + r;
    return x - floor(x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_idx = gid.x;
    let total_pixels = uniforms.width * uniforms.height;
    
    if pixel_idx >= total_pixels {
        return;
    }
    
    let px = pixel_idx % uniforms.width;
    let py = pixel_idx / uniforms.width;
    
    // Generate multiple samples per pixel for SPP > 1 (A16: Sobol/Owen + CP rotation)
    // If adaptive_threshold_u32 != 0, clamp spp to that threshold for this pass.
    let spp_target = select(uniforms.spp, min(uniforms.spp, restir_settings.adaptive_threshold_u32), restir_settings.adaptive_threshold_u32 != 0u);
    for (var sample: u32 = 0u; sample < spp_target; sample = sample + 1u) {
        // Sample index with frame offset to vary sequences over time
        let sidx = sample + uniforms.frame_index * max(1u, uniforms.spp);
        // Select QMC sequence
        var u1: f32;
        var u2: f32;
        if (restir_settings.qmc_mode != 0u) {
            let s2 = sobol2(sidx);
            u1 = s2.x; u2 = s2.y;
        } else {
            u1 = radical_inverse_vdc(sidx);
            u2 = halton_base3(sidx);
        }
        // Per-pixel rotation using hashed seed for blue-noise-like decorrelation
        var rr_state = uniforms.seed_lo ^ (px * 9781u) ^ (py * 6271u) ^ (uniforms.seed_hi * 13007u);
        let r1 = xorshift32(&rr_state);
        let r2 = xorshift32(&rr_state);
        // Apply Cranley-Patterson rotation and tent-filter to concentrate around pixel center
        let jx = tent_filter(cp_rotate(u1, r1)) * 0.5;
        let jy = tent_filter(cp_rotate(u2, r2)) * 0.5;
        
        // Generate camera ray
        let ndc_x = ((f32(px) + 0.5 + jx) / f32(uniforms.width)) * 2.0 - 1.0;
        let ndc_y = (1.0 - (f32(py) + 0.5 + jy) / f32(uniforms.height)) * 2.0 - 1.0;
        let half_h = tan(0.5 * uniforms.cam_fov_y);
        let half_w = uniforms.cam_aspect * half_h;
        
        var rd = normalize(vec3<f32>(ndc_x * half_w, ndc_y * half_h, -1.0));
        rd = normalize(rd.x * uniforms.cam_right + rd.y * uniforms.cam_up + rd.z * (-uniforms.cam_forward));
        
        // Create primary ray
        var rng_state = uniforms.seed_hi ^ (pixel_idx * 9781u) ^ (uniforms.frame_index * 6271u);
        var primary_ray: Ray;
        primary_ray.o = uniforms.cam_origin;
        primary_ray.tmin = 1e-4;
        primary_ray.d = rd;
        primary_ray.tmax = 1e30;
        primary_ray.throughput = vec3<f32>(1.0, 1.0, 1.0);
        primary_ray.pdf = 1.0;
        primary_ray.pixel = pixel_idx;
        primary_ray.depth = 0u;
        primary_ray.rng_hi = rng_state;
        primary_ray.rng_lo = uniforms.seed_lo ^ sample;
        
        // Push ray to queue atomically
        let queue_idx = atomicAdd(&ray_queue_header.in_count, 1u);
        if queue_idx < ray_queue_header.capacity {
            ray_queue[queue_idx] = primary_ray;
        }
    }
}
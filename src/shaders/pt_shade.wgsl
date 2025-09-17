// src/shaders/pt_shade.wgsl
// Placeholder WGSL for path tracing shading stage.
// Exists to reserve file location and naming for future compute pipeline wiring.
// RELEVANT FILES:src/shaders/pbr_textured.wgsl,src/shaders/pt_kernel.wgsl

// Wavefront Path Tracer: Shading Stage (Lambertian MVP)
// Pops hit records, evaluates simple diffuse BRDF, samples next-bounce direction,
// and pushes ScatterRay into the scatter queue. Accumulation of background is
// handled in pt_scatter.wgsl's miss processing.

// -----------------------------------------------------------------------------
// Bindings and shared structures (keep in sync with other PT shaders)
// -----------------------------------------------------------------------------
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

// Bind Group 1: Scene (readonly storage: materials, textures/handles, accel/BVH)
struct Sphere {
    center: vec3<f32>,
    radius: f32,
    albedo: vec3<f32>,
    _pad0: f32,
}

// Bind Group 2: Queues (read/write storage buffers with atomic counters)
struct Hit {
    p: vec3<f32>,           // hit position
    t: f32,                 // ray parameter
    n: vec3<f32>,           // surface normal
    mat: u32,               // material index
    throughput: vec3<f32>,  // inherited throughput
    pdf: f32,               // inherited pdf
    pixel: u32,             // pixel index
    depth: u32,             // bounce depth
    rng_hi: u32,            // RNG state high
    rng_lo: u32,            // RNG state low
}

struct ScatterRay {
    o: vec3<f32>,           // origin
    tmin: f32,              // minimum ray parameter
    d: vec3<f32>,           // direction
    tmax: f32,              // maximum ray parameter
    throughput: vec3<f32>,  // updated throughput
    pdf: f32,               // updated pdf
    pixel: u32,             // pixel index
    depth: u32,             // bounce depth + 1
    rng_hi: u32,            // updated RNG state high
    rng_lo: u32,            // updated RNG state low
}

struct QueueHeader {
    in_count: u32,          // number of items pushed
    out_count: u32,         // number of items popped
    capacity: u32,          // maximum capacity
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> scene_spheres: array<Sphere>;
@group(2) @binding(0) var<storage, read_write> hit_queue_header: QueueHeader;
@group(2) @binding(1) var<storage, read_write> hit_queue: array<Hit>;
@group(2) @binding(2) var<storage, read_write> scatter_queue_header: QueueHeader;
@group(2) @binding(3) var<storage, read_write> scatter_queue: array<ScatterRay>;
@group(3) @binding(0) var<storage, read_write> accum_hdr: array<vec4<f32>>;

// -----------------------------------------------------------------------------
// Utilities: RNG and sampling
// -----------------------------------------------------------------------------
let PI: f32 = 3.14159265358979323846;

// XorShift32 RNG for consistency with other stages
fn xorshift32(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x ^= (x << 13u);
    x ^= (x >> 17u);
    x ^= (x << 5u);
    *state = x;
    return f32(x) / 4294967296.0;
}

// Orthonormal basis from normal; returns matrix whose columns are (t, b, n)
fn make_tangent_basis(n: vec3<f32>) -> mat3x3<f32> {
    let sign = select(1.0, -1.0, n.z < 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bvec = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
    // Columns are tangent, bitangent, normal
    return mat3x3<f32>(t, bvec, n);
}

// Cosine-weighted hemisphere sample in local space (z up)
fn sample_cosine_hemisphere(u1: f32, u2: f32) -> vec3<f32> {
    let r = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let x = r * cos(phi);
    let y = r * sin(phi);
    let z = sqrt(max(0.0, 1.0 - u1));
    return vec3<f32>(x, y, z);
}

// -----------------------------------------------------------------------------
// Main shading kernel
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Persistent threads: pop hits until queue is empty
    loop {
        let hit_idx = atomicAdd(&hit_queue_header.out_count, 1u);
        if hit_idx >= hit_queue_header.in_count { break; }
        if hit_idx >= hit_queue_header.capacity { break; }

        let h = hit_queue[hit_idx];

        // Fetch material albedo from scene
        let sphere_count = arrayLength(&scene_spheres);
        let mat_idx = select(0u, h.mat, h.mat < sphere_count);
        let albedo = scene_spheres[mat_idx].albedo;

        // Prepare RNG from hit state
        var rng_state = h.rng_hi ^ (h.pixel * 9781u) ^ (uniforms.frame_index * 6271u);

        // Cosine-weighted hemisphere sample around normal
        let u1 = xorshift32(&rng_state);
        let u2 = xorshift32(&rng_state);
        let local_dir = sample_cosine_hemisphere(u1, u2);
        let basis = make_tangent_basis(normalize(h.n));
        let wi = normalize(basis * local_dir); // world-space direction

        let cos_theta = max(0.0, dot(normalize(h.n), wi));
        let pdf = cos_theta / PI + 1e-8; // avoid div-by-zero

        // Lambertian BRDF = albedo / PI; throughput update: * (f * cos / pdf)
        let brdf = albedo / PI;
        let new_throughput = h.throughput * brdf * (cos_theta / pdf);

        // Russian roulette after a few bounces (optional MVP)
        var continue_path = true;
        if (h.depth >= 4u) {
            let max_c = max(new_throughput.x, max(new_throughput.y, new_throughput.z));
            let q = clamp(1.0 - max_c, 0.0, 0.95);
            let u = xorshift32(&rng_state);
            if (u < q) {
                continue_path = false;
            } else {
                // Balance throughput for RR
                let rr_scale = 1.0 / (1.0 - q);
                // Multiply component-wise
                // Note: scalar multiply auto-applies to vec3
                // so we can reuse new_throughput scaled later.
            }
        }

        if continue_path && (h.depth + 1u) < 16u {
            // Create scatter ray
            var s: ScatterRay;
            s.o = h.p + normalize(h.n) * 1e-3;
            s.tmin = 1e-3;
            s.d = wi;
            s.tmax = 1e30;
            s.throughput = new_throughput;
            s.pdf = pdf;
            s.pixel = h.pixel;
            s.depth = h.depth + 1u;
            s.rng_hi = rng_state;
            s.rng_lo = h.rng_lo ^ uniforms.seed_lo;

            let qidx = atomicAdd(&scatter_queue_header.in_count, 1u);
            if qidx < scatter_queue_header.capacity {
                scatter_queue[qidx] = s;
            }
        } else {
            // Path terminated at surface: no direct accumulation here for MVP.
            // Background accumulation is handled by miss processing in pt_scatter.
        }
    }
}


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
    metallic: f32,
    roughness: f32,
    ior: f32,
    emissive: vec3<f32>,
    _pad0: f32,
}

// Bind Group 2: Queues (read/write storage buffers with atomic counters)
struct Hit {
    p: vec3<f32>,           // hit position
    t: f32,                 // ray parameter
    n: vec3<f32>,           // surface normal
    wo: vec3<f32>,          // outgoing (to camera) direction
    _pad_wo: f32,           // alignment
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

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

// Fresnel-Schlick (vector F0)
fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - saturate(cos_theta), 5.0);
}

// GGX/Trowbridge-Reitz normal distribution (isotropic)
fn ggx_D(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ndh2 = n_dot_h * n_dot_h;
    let denom = PI * pow(ndh2 * (a2 - 1.0) + 1.0, 2.0);
    return a2 / max(denom, 1e-6);
}

// Smith's masking-shadowing (height-correlated approx)
fn smith_G1(n_dot_v: f32, alpha: f32) -> f32 {
    // k from UE4: (a+1)^2 / 8
    let k = pow(alpha + 1.0, 2.0) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn smith_G(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32 {
    return smith_G1(n_dot_l, alpha) * smith_G1(n_dot_v, alpha);
}

// Isotropic GGX half-vector sampling around +Z
fn sample_ggx_isotropic(u1: f32, u2: f32, alpha: f32) -> vec3<f32> {
    let a2 = alpha * alpha;
    let cos_theta_h = sqrt((1.0 - u1) / (1.0 + (a2 - 1.0) * u1));
    let sin_theta_h = sqrt(max(0.0, 1.0 - cos_theta_h * cos_theta_h));
    let phi = 2.0 * PI * u2;
    let x = sin_theta_h * cos(phi);
    let y = sin_theta_h * sin(phi);
    let z = cos_theta_h;
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

        // Fetch material parameters from scene
        let sphere_count = arrayLength(&scene_spheres);
        let mat_idx = select(0u, h.mat, h.mat < sphere_count);
        let albedo = scene_spheres[mat_idx].albedo;
        let metallic = scene_spheres[mat_idx].metallic;
        let roughness = scene_spheres[mat_idx].roughness;
        let ior = scene_spheres[mat_idx].ior;
        let emissive = scene_spheres[mat_idx].emissive;

        // Direct emissive accumulation (energy conserving via throughput)
        if (emissive.x > 0.0 || emissive.y > 0.0 || emissive.z > 0.0) {
            let add = h.throughput * emissive;
            let pix = h.pixel;
            accum_hdr[pix] = accum_hdr[pix] + vec4<f32>(add, 0.0);
        }

        // Prepare RNG from hit state
        var rng_state = h.rng_hi ^ (h.pixel * 9781u) ^ (uniforms.frame_index * 6271u);

        let n = normalize(h.n);
        let wo = normalize(h.wo); // to camera
        let n_dot_v = max(dot(n, wo), 0.0);

        // Dispatch to BSDF
        var wi: vec3<f32>;
        var pdf: f32;
        var new_throughput: vec3<f32>;
        let basis = make_tangent_basis(n);

        // Parameters
        let a = max(0.02, roughness * roughness);
        let F0 = mix(vec3<f32>(0.04), albedo, saturate(metallic));

        if (metallic > 0.5) {
            // GGX metal reflection
            let u1 = xorshift32(&rng_state);
            let u2 = xorshift32(&rng_state);
            let h_local = sample_ggx_isotropic(u1, u2, a);
            let h_world = normalize(basis * h_local);
            wi = normalize(reflect(-wo, h_world));
            let n_dot_l = max(dot(n, wi), 0.0);
            let n_dot_h = max(dot(n, h_world), 0.0);
            let v_dot_h = max(dot(wo, h_world), 0.0);
            if (n_dot_l > 0.0 && n_dot_v > 0.0) {
                let D = ggx_D(n_dot_h, a);
                let G = smith_G(n_dot_l, n_dot_v, a);
                let F = fresnel_schlick(v_dot_h, F0);
                let spec = (D * G) / max(4.0 * n_dot_l * n_dot_v, 1e-6) * F;
                pdf = (D * n_dot_h) / max(4.0 * v_dot_h, 1e-6);
                new_throughput = h.throughput * spec * (n_dot_l / max(pdf, 1e-6));
            } else {
                // Invalid sample
                continue;
            }
        } else if (ior > 1.01) {
            // Dielectric (perfect specular): reflect or refract using Schlick
            let cosi = saturate(dot(n, wo));
            let n1 = 1.0;
            let n2 = ior;
            let F0s = pow((n2 - n1) / (n2 + n1), 2.0);
            let F = F0s + (1.0 - F0s) * pow(1.0 - cosi, 5.0);
            let u = xorshift32(&rng_state);
            if (u < F) {
                // Reflect
                wi = normalize(reflect(-wo, n));
            } else {
                // Refract (assume exiting/entering based on sign)
                let entering = dot(n, wo) > 0.0;
                let eta = select(n2 / n1, n1 / n2, entering);
                let N = select(-n, n, entering);
                // WGSL refract(I, N, eta)
                wi = normalize(refract(-wo, N, eta));
                // If TIR produced zero-length/invalid, fallback to reflection
                let len2 = dot(wi, wi);
                if (len2 < 1e-12) {
                    wi = normalize(reflect(-wo, n));
                }
            }
            // Delta lobe: pdf=1, color via albedo as tint
            pdf = 1.0;
            new_throughput = h.throughput * max(albedo, vec3<f32>(0.0));
        } else {
            // Lambertian
            let u1 = xorshift32(&rng_state);
            let u2 = xorshift32(&rng_state);
            let local_dir = sample_cosine_hemisphere(u1, u2);
            wi = normalize(basis * local_dir);
            let cos_theta = max(0.0, dot(n, wi));
            pdf = cos_theta / PI + 1e-8;
            let brdf = albedo / PI;
            new_throughput = h.throughput * brdf * (cos_theta / pdf);
        }

        // Russian roulette after a few bounces (optional MVP)
        var continue_path = true;
        var rr_scale = 1.0;
        if (h.depth >= 4u) {
            let max_c = max(new_throughput.x, max(new_throughput.y, new_throughput.z));
            let q = clamp(1.0 - max_c, 0.0, 0.95);
            let u = xorshift32(&rng_state);
            if (u < q) {
                continue_path = false;
            } else {
                // Balance throughput for RR
                rr_scale = 1.0 / (1.0 - q);
            }
        }

        if continue_path && (h.depth + 1u) < 16u {
            // Create scatter ray
            var s: ScatterRay;
            s.o = h.p + normalize(h.n) * 1e-3;
            s.tmin = 1e-3;
            s.d = wi;
            s.tmax = 1e30;
            s.throughput = new_throughput * rr_scale;
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


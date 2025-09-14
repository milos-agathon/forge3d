// src/shaders/pt_shade.wgsl
// Wavefront Path Tracer: Shading Stage
// This file exists to evaluate BSDF and direct lighting, writing scatter rays or marking termination.
// RELEVANT FILES:src/path_tracing/wavefront/mod.rs,src/shaders/pt_intersect.wgsl,src/shaders/pt_scatter.wgsl

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
    _pad0: f32 
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

// Bind Group 3: Accum/Output (HDR accum buffer or storage texture)
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> scene_spheres: array<Sphere>;
@group(2) @binding(0) var<storage, read_write> hit_queue_header: QueueHeader;
@group(2) @binding(1) var<storage, read_write> hit_queue: array<Hit>;
@group(2) @binding(2) var<storage, read_write> scatter_queue_header: QueueHeader;
@group(2) @binding(3) var<storage, read_write> scatter_queue: array<ScatterRay>;
@group(3) @binding(0) var<storage, read_write> accum_hdr: array<vec4<f32>>;

// Constants
let MAX_DEPTH: u32 = 8u;
let PI: f32 = 3.14159265359;

// XorShift32 RNG
fn xorshift32(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x ^= (x << 13u);
    x ^= (x >> 17u);
    x ^= (x << 5u);
    *state = x;
    return f32(x) / 4294967296.0;
}

// Cosine-weighted hemisphere sampling
fn sample_cosine_hemisphere(u1: f32, u2: f32) -> vec3<f32> {
    let cos_theta = sqrt(1.0 - u1);
    let sin_theta = sqrt(u1);
    let phi = 2.0 * PI * u2;
    return vec3<f32>(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );
}

// Create orthonormal basis from normal
fn create_basis(n: vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return mat3x3<f32>(tangent, bitangent, n);
}

// Fresnel Schlick approximation
fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    let ct = clamp(cos_theta, 0.0, 1.0);
    return F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - ct, 5.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Persistent threads loop: keep pulling hits until queue is empty
    loop {
        let hit_idx = atomicAdd(&hit_queue_header.out_count, 1u);
        if hit_idx >= hit_queue_header.in_count {
            break; // No more hits to process
        }
        
        if hit_idx >= hit_queue_header.capacity {
            break; // Safety check
        }
        
        let hit = hit_queue[hit_idx];
        var rng_state = hit.rng_hi;
        
        // Get material properties
        let sphere = scene_spheres[hit.mat];
        let albedo = sphere.albedo;
        
        // Direct lighting (simple point light at (2, 2, 2))
        let light_pos = vec3<f32>(2.0, 2.0, 2.0);
        let light_color = vec3<f32>(4.0, 4.0, 4.0); // Compensate for distance falloff
        let light_dir = normalize(light_pos - hit.p);
        let light_distance = length(light_pos - hit.p);
        let ndotl = max(0.0, dot(hit.n, light_dir));
        
        // Simple Lambert BRDF for direct lighting
        let direct_contrib = (albedo / PI) * light_color * ndotl / (light_distance * light_distance);
        
        // Accumulate direct lighting contribution
        let contrib = hit.throughput * direct_contrib;
        let pixel_idx = hit.pixel;
        accum_hdr[pixel_idx] = accum_hdr[pixel_idx] + vec4<f32>(contrib, 0.0);
        
        // Russian roulette termination (after a few bounces)
        var continue_probability = 1.0;
        if hit.depth >= 3u {
            let max_component = max(hit.throughput.x, max(hit.throughput.y, hit.throughput.z));
            continue_probability = min(0.95, max_component);
            
            if xorshift32(&rng_state) > continue_probability {
                continue; // Terminate this path
            }
        }
        
        // Check maximum depth
        if hit.depth >= MAX_DEPTH {
            continue; // Terminate this path
        }
        
        // Sample next bounce direction (cosine-weighted hemisphere)
        let u1 = xorshift32(&rng_state);
        let u2 = xorshift32(&rng_state);
        let local_dir = sample_cosine_hemisphere(u1, u2);
        
        // Transform to world space
        let basis = create_basis(hit.n);
        let world_dir = basis * local_dir;
        
        // Compute BRDF and PDF
        let cos_theta = dot(world_dir, hit.n);
        if cos_theta <= 0.0 {
            continue; // Invalid direction
        }
        
        let brdf = albedo / PI; // Lambert BRDF
        let pdf = cos_theta / PI; // Cosine-weighted hemisphere PDF
        
        // Update throughput
        let new_throughput = hit.throughput * brdf * cos_theta / (pdf * continue_probability);
        
        // Create scatter ray
        var scatter_ray: ScatterRay;
        scatter_ray.o = hit.p + hit.n * 1e-4; // Offset to avoid self-intersection
        scatter_ray.tmin = 1e-4;
        scatter_ray.d = world_dir;
        scatter_ray.tmax = 1e30;
        scatter_ray.throughput = new_throughput;
        scatter_ray.pdf = hit.pdf * pdf;
        scatter_ray.pixel = hit.pixel;
        scatter_ray.depth = hit.depth + 1u;
        scatter_ray.rng_hi = rng_state;
        scatter_ray.rng_lo = hit.rng_lo;
        
        // Push to scatter queue
        let scatter_queue_idx = atomicAdd(&scatter_queue_header.in_count, 1u);
        if scatter_queue_idx < scatter_queue_header.capacity {
            scatter_queue[scatter_queue_idx] = scatter_ray;
        }
    }
}
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

// Bind Group 1: Scene (readonly storage: materials, textures/handles, accel/BVH)
struct Sphere { 
    center: vec3<f32>, 
    radius: f32, 
    albedo: vec3<f32>, 
    _pad0: f32 
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
    in_count: u32,          // number of items pushed
    out_count: u32,         // number of items popped
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_idx = gid.x;
    let total_pixels = uniforms.width * uniforms.height;
    
    if pixel_idx >= total_pixels {
        return;
    }
    
    let px = pixel_idx % uniforms.width;
    let py = pixel_idx / uniforms.width;
    
    // Generate multiple samples per pixel for SPP > 1
    for (var sample: u32 = 0u; sample < uniforms.spp; sample = sample + 1u) {
        // Initialize per-pixel, per-sample RNG state
        var rng_state = uniforms.seed_hi ^ px ^ (py * 1664525u) ^ (sample * 1013904223u) ^ uniforms.frame_index;
        
        // Anti-aliasing jitter
        let jx = tent_filter(xorshift32(&rng_state)) * 0.5;
        let jy = tent_filter(xorshift32(&rng_state)) * 0.5;
        
        // Generate camera ray
        let ndc_x = ((f32(px) + 0.5 + jx) / f32(uniforms.width)) * 2.0 - 1.0;
        let ndc_y = (1.0 - (f32(py) + 0.5 + jy) / f32(uniforms.height)) * 2.0 - 1.0;
        let half_h = tan(0.5 * uniforms.cam_fov_y);
        let half_w = uniforms.cam_aspect * half_h;
        
        var rd = normalize(vec3<f32>(ndc_x * half_w, ndc_y * half_h, -1.0));
        rd = normalize(rd.x * uniforms.cam_right + rd.y * uniforms.cam_up + rd.z * (-uniforms.cam_forward));
        
        // Create primary ray
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
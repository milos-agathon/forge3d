// src/shaders/pt_intersect.wgsl
// Wavefront Path Tracer: Intersection Stage
// This file exists to intersect rays with scene acceleration structures and write hit information to the hit queue.
// RELEVANT FILES:src/path_tracing/wavefront/mod.rs,src/shaders/pt_raygen.wgsl,src/shaders/pt_shade.wgsl

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

struct QueueHeader {
    in_count: u32,          // number of items pushed
    out_count: u32,         // number of items popped
    capacity: u32,          // maximum capacity
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> scene_spheres: array<Sphere>;
@group(2) @binding(0) var<storage, read_write> ray_queue_header: QueueHeader;
@group(2) @binding(1) var<storage, read_write> ray_queue: array<Ray>;
@group(2) @binding(2) var<storage, read_write> hit_queue_header: QueueHeader;
@group(2) @binding(3) var<storage, read_write> hit_queue: array<Hit>;
@group(2) @binding(4) var<storage, read_write> miss_queue_header: QueueHeader;
@group(2) @binding(5) var<storage, read_write> miss_queue: array<Ray>;

// Ray-sphere intersection
fn ray_sphere(ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
    let oc = ro - c;
    let b = dot(oc, rd);
    let cterm = dot(oc, oc) - r * r;
    let disc = b * b - cterm;
    if (disc <= 0.0) { return 1e30; }
    let s = sqrt(disc);
    let t0 = -b - s;
    let t1 = -b + s;
    if (t0 > 1e-3) { return t0; }
    if (t1 > 1e-3) { return t1; }
    return 1e30;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Persistent threads loop: keep pulling rays until queue is empty
    loop {
        let ray_idx = atomicAdd(&ray_queue_header.out_count, 1u);
        if ray_idx >= ray_queue_header.in_count {
            break; // No more rays to process
        }
        
        if ray_idx >= ray_queue_header.capacity {
            break; // Safety check
        }
        
        let ray = ray_queue[ray_idx];
        
        // Test intersection with scene
        var t_best = 1e30;
        var hit_normal = vec3<f32>(0.0, 1.0, 0.0);
        var hit_albedo = vec3<f32>(0.5);
        var material_idx = 0u;
        
        // Intersect with spheres
        let sphere_count = arrayLength(&scene_spheres);
        for (var i: u32 = 0u; i < sphere_count; i = i + 1u) {
            let sphere = scene_spheres[i];
            let t = ray_sphere(ray.o, ray.d, sphere.center, sphere.radius);
            
            if t >= ray.tmin && t < min(t_best, ray.tmax) {
                t_best = t;
                let hit_pos = ray.o + ray.d * t;
                hit_normal = normalize(hit_pos - sphere.center);
                hit_albedo = sphere.albedo;
                material_idx = i;
            }
        }
        
        if t_best < 1e20 {
            // Hit: create hit record
            var hit: Hit;
            hit.p = ray.o + ray.d * t_best;
            hit.t = t_best;
            hit.n = hit_normal;
            hit.mat = material_idx;
            hit.throughput = ray.throughput;
            hit.pdf = ray.pdf;
            hit.pixel = ray.pixel;
            hit.depth = ray.depth;
            hit.rng_hi = ray.rng_hi;
            hit.rng_lo = ray.rng_lo;
            
            // Push to hit queue
            let hit_queue_idx = atomicAdd(&hit_queue_header.in_count, 1u);
            if hit_queue_idx < hit_queue_header.capacity {
                hit_queue[hit_queue_idx] = hit;
            }
        } else {
            // Miss: push to miss queue for background evaluation
            let miss_queue_idx = atomicAdd(&miss_queue_header.in_count, 1u);
            if miss_queue_idx < miss_queue_header.capacity {
                miss_queue[miss_queue_idx] = ray;
            }
        }
    }
}
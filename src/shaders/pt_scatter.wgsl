// src/shaders/pt_scatter.wgsl
// Wavefront Path Tracer: Scatter Stage
// This file exists to spawn next-bounce rays from shade outputs and push them back to the ray queue.
// RELEVANT FILES:src/path_tracing/wavefront/mod.rs,src/shaders/pt_shade.wgsl,src/shaders/pt_intersect.wgsl

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
@group(2) @binding(0) var<storage, read_write> scatter_queue_header: QueueHeader;
@group(2) @binding(1) var<storage, read_write> scatter_queue: array<ScatterRay>;
@group(2) @binding(2) var<storage, read_write> ray_queue_header: QueueHeader;
@group(2) @binding(3) var<storage, read_write> ray_queue: array<Ray>;
@group(2) @binding(4) var<storage, read_write> miss_queue_header: QueueHeader;
@group(2) @binding(5) var<storage, read_write> miss_queue: array<Ray>;
@group(3) @binding(0) var<storage, read_write> accum_hdr: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Persistent threads loop: keep pulling scatter rays until queue is empty
    loop {
        let scatter_idx = atomicAdd(&scatter_queue_header.out_count, 1u);
        if scatter_idx >= scatter_queue_header.in_count {
            break; // No more scatter rays to process
        }
        
        if scatter_idx >= scatter_queue_header.capacity {
            break; // Safety check
        }
        
        let scatter_ray = scatter_queue[scatter_idx];
        
        // Convert scatter ray back to regular ray format
        var next_ray: Ray;
        next_ray.o = scatter_ray.o;
        next_ray.tmin = scatter_ray.tmin;
        next_ray.d = scatter_ray.d;
        next_ray.tmax = scatter_ray.tmax;
        next_ray.throughput = scatter_ray.throughput;
        next_ray.pdf = scatter_ray.pdf;
        next_ray.pixel = scatter_ray.pixel;
        next_ray.depth = scatter_ray.depth;
        next_ray.rng_hi = scatter_ray.rng_hi;
        next_ray.rng_lo = scatter_ray.rng_lo;
        
        // Push ray back to ray queue for next iteration
        let ray_queue_idx = atomicAdd(&ray_queue_header.in_count, 1u);
        if ray_queue_idx < ray_queue_header.capacity {
            ray_queue[ray_queue_idx] = next_ray;
        }
    }
    
    // Also handle miss rays (background contribution)
    loop {
        let miss_idx = atomicAdd(&miss_queue_header.out_count, 1u);
        if miss_idx >= miss_queue_header.in_count {
            break; // No more miss rays to process
        }
        
        if miss_idx >= miss_queue_header.capacity {
            break; // Safety check
        }
        
        let miss_ray = miss_queue[miss_idx];
        
        // Compute background color (simple sky gradient)
        let sky_t = 0.5 * (miss_ray.d.y + 1.0);
        let sky_color = mix(vec3<f32>(0.6, 0.7, 0.9), vec3<f32>(0.1, 0.2, 0.5), sky_t);
        
        // Apply throughput and accumulate to pixel
        let contrib = miss_ray.throughput * sky_color;
        let pixel_idx = miss_ray.pixel;
        accum_hdr[pixel_idx] = accum_hdr[pixel_idx] + vec4<f32>(contrib, 0.0);
    }
}
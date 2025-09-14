// src/shaders/pt_compact.wgsl
// Wavefront Path Tracer: Stream Compaction Stage
// This file exists to compact queues by removing terminated/invalid rays to maintain efficiency.
// RELEVANT FILES:src/path_tracing/wavefront/mod.rs,src/shaders/pt_scatter.wgsl

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

struct QueueHeader {
    in_count: u32,          // number of items pushed
    out_count: u32,         // number of items popped
    capacity: u32,          // maximum capacity
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(2) @binding(0) var<storage, read_write> ray_queue_header: QueueHeader;
@group(2) @binding(1) var<storage, read_write> ray_queue: array<Ray>;
@group(2) @binding(2) var<storage, read_write> ray_queue_compacted: array<Ray>;
@group(2) @binding(3) var<storage, read_write> ray_flags: array<u32>;
@group(2) @binding(4) var<storage, read_write> prefix_sums: array<u32>;

// Workgroup size must be power of 2 for prefix sum
@compute @workgroup_size(256)
fn compact_rays(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_id = gid.x;
    let active_rays = ray_queue_header.in_count - ray_queue_header.out_count;
    
    // Phase 1: Flag valid rays
    if thread_id < active_rays {
        let ray_idx = ray_queue_header.out_count + thread_id;
        if ray_idx < ray_queue_header.capacity {
            let ray = ray_queue[ray_idx];
            
            // Ray is valid if it has non-zero throughput and reasonable depth
            let max_throughput = max(ray.throughput.x, max(ray.throughput.y, ray.throughput.z));
            let is_valid = max_throughput > 1e-6 && ray.depth < 16u;
            ray_flags[thread_id] = select(0u, 1u, is_valid);
        } else {
            ray_flags[thread_id] = 0u;
        }
    } else if thread_id < ray_queue_header.capacity {
        ray_flags[thread_id] = 0u;
    }
    
    workgroupBarrier();
    
    // Phase 2: Parallel prefix sum (Hillis-Steele scan)
    // Initialize prefix sums
    if thread_id < active_rays {
        prefix_sums[thread_id] = ray_flags[thread_id];
    } else if thread_id < ray_queue_header.capacity {
        prefix_sums[thread_id] = 0u;
    }
    
    workgroupBarrier();
    
    // Up-sweep (reduction phase)
    var step = 1u;
    let max_threads = min(256u, active_rays);
    
    while step < max_threads {
        if thread_id < max_threads && thread_id >= step {
            if (thread_id & (step - 1u)) == (step - 1u) {
                let partner = thread_id - step;
                if partner < max_threads {
                    prefix_sums[thread_id] += prefix_sums[partner];
                }
            }
        }
        workgroupBarrier();
        step <<= 1u;
    }
    
    // Down-sweep (distribution phase)
    step = max_threads >> 1u;
    while step > 0u {
        if thread_id < max_threads && (thread_id & (step - 1u)) == (step - 1u) {
            let partner = thread_id + step;
            if partner < max_threads && partner < active_rays {
                prefix_sums[partner] += prefix_sums[thread_id];
            }
        }
        workgroupBarrier();
        step >>= 1u;
    }
    
    workgroupBarrier();
    
    // Phase 3: Scatter valid rays to compacted array
    if thread_id < active_rays && ray_flags[thread_id] == 1u {
        let ray_idx = ray_queue_header.out_count + thread_id;
        if ray_idx < ray_queue_header.capacity {
            let compact_idx = prefix_sums[thread_id] - 1u; // Convert to 0-based index
            if compact_idx < ray_queue_header.capacity {
                ray_queue_compacted[compact_idx] = ray_queue[ray_idx];
            }
        }
    }
    
    workgroupBarrier();
    
    // Update queue header with compacted count
    if thread_id == 0u {
        let total_valid = select(0u, prefix_sums[active_rays - 1u], active_rays > 0u);
        ray_queue_header.out_count = 0u;
        ray_queue_header.in_count = total_valid;
        
        // Copy compacted rays back to main queue
        for (var i = 0u; i < total_valid; i = i + 1u) {
            if i < ray_queue_header.capacity {
                ray_queue[i] = ray_queue_compacted[i];
            }
        }
    }
}

// Alternative simpler compaction for smaller queues
@compute @workgroup_size(256)
fn compact_rays_simple(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_id = gid.x;
    
    if thread_id == 0u {
        let active_rays = ray_queue_header.in_count - ray_queue_header.out_count;
        var write_idx = 0u;
        
        // Serial compaction (acceptable for smaller queues)
        for (var read_idx = ray_queue_header.out_count; read_idx < ray_queue_header.in_count; read_idx = read_idx + 1u) {
            if read_idx < ray_queue_header.capacity {
                let ray = ray_queue[read_idx];
                
                // Ray is valid if it has non-zero throughput and reasonable depth
                let max_throughput = max(ray.throughput.x, max(ray.throughput.y, ray.throughput.z));
                let is_valid = max_throughput > 1e-6 && ray.depth < 16u;
                
                if is_valid && write_idx < ray_queue_header.capacity {
                    ray_queue[write_idx] = ray;
                    write_idx = write_idx + 1u;
                }
            }
        }
        
        // Update queue header
        ray_queue_header.out_count = 0u;
        ray_queue_header.in_count = write_idx;
    }
}
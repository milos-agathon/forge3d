// Optimized radix sort with single bind group layout (8-bit digits, 4 passes)
// Uses 256 bins per pass, ping-pong buffers, and efficient shared memory

struct Uniforms {
    prim_count: u32,
    pass_shift: u32,  // 0, 8, 16, 24 for each pass
    num_workgroups: u32,
    _pad: u32,
}

// Single bind group with all resources
@group(0) @binding(0) var<storage, read> src_keys: array<u32>;
@group(0) @binding(1) var<storage, read> src_vals: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst_keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst_vals: array<u32>;
@group(0) @binding(4) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: Uniforms;

// Shared memory for local histogram (256 bins)
var<workgroup> local_hist: array<atomic<u32>, 256>;

// Clear histogram buffer
@compute @workgroup_size(256)
fn clear_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_bins = params.num_workgroups * 256u;
    if idx < total_bins {
        atomicStore(&histogram[idx], 0u);
    }
}

// Build histogram: each workgroup processes a chunk and accumulates to global histogram
@compute @workgroup_size(256)
fn build_histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Clear local histogram
    atomicStore(&local_hist[lid.x], 0u);
    workgroupBarrier();
    
    // Each thread processes multiple keys (coalesced reads)
    let items_per_thread = 4u;
    let total_threads = params.num_workgroups * 256u;
    let thread_id = gid.x;
    
    for (var i = 0u; i < items_per_thread; i = i + 1u) {
        let idx = thread_id + i * total_threads;
        if idx < params.prim_count {
            let key = src_keys[idx];
            let digit = (key >> params.pass_shift) & 0xffu;
            atomicAdd(&local_hist[digit], 1u);
        }
    }
    
    workgroupBarrier();
    
    // Write local histogram to global (one thread per bin)
    let bin = lid.x;
    let count = atomicLoad(&local_hist[bin]);
    if count > 0u {
        let global_idx = wid.x * 256u + bin;
        atomicAdd(&histogram[global_idx], count);
    }
}

// Exclusive prefix scan across all histogram bins (single workgroup)
// Uses Blelloch scan algorithm
var<workgroup> scan_temp: array<u32, 512>;

@compute @workgroup_size(256)
fn scan_histogram(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let total_bins = params.num_workgroups * 256u;
    
    // Load two elements per thread
    let idx1 = tid * 2u;
    let idx2 = tid * 2u + 1u;
    
    if idx1 < total_bins {
        scan_temp[idx1] = atomicLoad(&histogram[idx1]);
    } else {
        scan_temp[idx1] = 0u;
    }
    
    if idx2 < total_bins {
        scan_temp[idx2] = atomicLoad(&histogram[idx2]);
    } else {
        scan_temp[idx2] = 0u;
    }
    workgroupBarrier();
    
    // Up-sweep (reduce phase)
    var offset = 1u;
    for (var d = total_bins >> 1u; d > 0u; d = d >> 1u) {
        if tid < d {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            if bi < total_bins {
                scan_temp[bi] = scan_temp[bi] + scan_temp[ai];
            }
        }
        offset = offset << 1u;
        workgroupBarrier();
    }
    
    // Clear last element
    if tid == 0u {
        scan_temp[total_bins - 1u] = 0u;
    }
    workgroupBarrier();
    
    // Down-sweep phase
    for (var d = 1u; d < total_bins; d = d << 1u) {
        offset = offset >> 1u;
        if tid < d {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            if bi < total_bins {
                let temp = scan_temp[ai];
                scan_temp[ai] = scan_temp[bi];
                scan_temp[bi] = scan_temp[bi] + temp;
            }
        }
        workgroupBarrier();
    }
    
    // Write back (exclusive scan result)
    if idx1 < total_bins {
        atomicStore(&histogram[idx1], scan_temp[idx1]);
    }
    if idx2 < total_bins {
        atomicStore(&histogram[idx2], scan_temp[idx2]);
    }
}

// Scatter elements to sorted positions
@compute @workgroup_size(256)
fn scatter_keys(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let items_per_thread = 4u;
    let total_threads = params.num_workgroups * 256u;
    let thread_id = gid.x;
    
    for (var i = 0u; i < items_per_thread; i = i + 1u) {
        let idx = thread_id + i * total_threads;
        if idx < params.prim_count {
            let key = src_keys[idx];
            let val = src_vals[idx];
            let digit = (key >> params.pass_shift) & 0xffu;
            
            // Get base offset from scanned histogram and atomically increment
            let hist_idx = wid.x * 256u + digit;
            let pos = atomicAdd(&histogram[hist_idx], 1u);
            
            if pos < params.prim_count {
                dst_keys[pos] = key;
                dst_vals[pos] = val;
            }
        }
    }
}

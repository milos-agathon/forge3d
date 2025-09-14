// src/shaders/radix_sort_pairs.wgsl
// WGSL compute kernel for key-value radix sort of Morton code pairs with 4-bit digit passes.
// This file exists to implement GPU radix sort for LBVH construction: sorting (Morton codes, primitive indices) pairs for spatial ordering.
// RELEVANT FILES:src/accel/lbvh_gpu.rs,src/shaders/lbvh_morton.wgsl,src/shaders/lbvh_link.wgsl

struct Uniforms {
    prim_count: u32,
    pass_shift: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> input_keys: array<u32>;
@group(1) @binding(1) var<storage, read> input_values: array<u32>;
@group(2) @binding(0) var<storage, read_write> output_keys: array<u32>;
@group(2) @binding(1) var<storage, read_write> output_values: array<u32>;
@group(3) @binding(0) var<storage, read_write> histogram: array<u32>;
@group(3) @binding(1) var<storage, read_write> prefix_sums: array<u32>;

var<workgroup> local_histogram: array<u32, 16>;
var<workgroup> local_data: array<vec2<u32>, 256>; // key, value pairs

@compute @workgroup_size(256)
fn count_pass(@builtin(global_invocation_id) gid: vec3<u32>, 
              @builtin(local_invocation_index) lid: u32,
              @builtin(workgroup_id) wgid: vec3<u32>) {
    
    // Initialize local histogram
    if lid < 16u {
        local_histogram[lid] = 0u;
    }
    workgroupBarrier();
    
    let global_idx = gid.x;
    let digit_mask = 0xfu;
    
    // Count digits in this workgroup
    if global_idx < uniforms.prim_count {
        let key = input_keys[global_idx];
        let digit = (key >> uniforms.pass_shift) & digit_mask;
        atomicAdd(&local_histogram[digit], 1u);
    }
    
    workgroupBarrier();
    
    // Write workgroup histogram to global memory
    if lid < 16u {
        histogram[wgid.x * 16u + lid] = local_histogram[lid];
    }
}

@compute @workgroup_size(256)
fn scan_pass(@builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(local_invocation_index) lid: u32) {
    
    let global_idx = gid.x;
    
    // Simple parallel prefix sum for 16 buckets per workgroup
    if global_idx < arrayLength(&histogram) {
        var sum = 0u;
        for (var i = 0u; i <= (global_idx % 16u); i = i + 1u) {
            sum += histogram[(global_idx / 16u) * 16u + i];
        }
        prefix_sums[global_idx] = sum;
    }
}

@compute @workgroup_size(256)
fn scatter_pass(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(local_invocation_index) lid: u32,
                @builtin(workgroup_id) wgid: vec3<u32>) {
    
    // Load data into local memory
    let global_idx = gid.x;
    if global_idx < uniforms.prim_count {
        local_data[lid] = vec2<u32>(input_keys[global_idx], input_values[global_idx]);
    } else {
        local_data[lid] = vec2<u32>(0xffffffffu, 0u); // Invalid key for padding
    }
    
    workgroupBarrier();
    
    // Initialize local histogram for scattering
    if lid < 16u {
        local_histogram[lid] = 0u;
    }
    workgroupBarrier();
    
    let digit_mask = 0xfu;
    var local_pos: array<u32, 16>;
    
    // Calculate local positions
    if global_idx < uniforms.prim_count {
        let key = local_data[lid].x;
        let digit = (key >> uniforms.pass_shift) & digit_mask;
        let local_offset = atomicAdd(&local_histogram[digit], 1u);
        
        // Get global offset from prefix sums
        let global_offset = prefix_sums[wgid.x * 16u + digit];
        local_pos[lid] = global_offset + local_offset;
    }
    
    workgroupBarrier();
    
    // Scatter to output
    if global_idx < uniforms.prim_count {
        let output_pos = local_pos[lid];
        if output_pos < uniforms.prim_count {
            output_keys[output_pos] = local_data[lid].x;
            output_values[output_pos] = local_data[lid].y;
        }
    }
}
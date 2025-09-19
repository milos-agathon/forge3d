// src/shaders/lbvh_link.wgsl
// WGSL compute kernel for building linear BVH topology from sorted Morton codes using Karras-style algorithm.
// This file exists to implement parallel BVH node linking: determining parent-child relationships and computing AABBs from sorted primitives.
// RELEVANT FILES:src/accel/lbvh_gpu.rs,src/accel/types.rs,src/shaders/radix_sort_pairs.wgsl,src/shaders/bvh_refit.wgsl

struct Uniforms {
    prim_count: u32,
    node_count: u32,
    _pad0: u32,
    _pad1: u32,
}

struct Aabb {
    min_val: vec3<f32>,
    _pad0: f32,
    max_val: vec3<f32>,
    _pad1: f32,
}

struct BvhNode {
    aabb: Aabb,
    kind: u32,     // 0 = internal, 1 = leaf
    left_idx: u32, // for internal: left child index; for leaf: first primitive index
    right_idx: u32, // for internal: right child index; for leaf: primitive count
    parent_idx: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> sorted_codes: array<u32>;
@group(1) @binding(1) var<storage, read> sorted_indices: array<u32>;
@group(1) @binding(2) var<storage, read> primitive_aabbs: array<Aabb>;
@group(2) @binding(0) var<storage, read_write> nodes: array<BvhNode>;

fn delta(i: i32, j: i32) -> i32 {
    if j < 0 || j >= i32(uniforms.prim_count) {
        return -1;
    }
    if i == j {
        return 32; // All bits match for identical indices
    }
    
    let code_i = sorted_codes[i];
    let code_j = sorted_codes[j];
    
    if code_i == code_j {
        // Same Morton code, use primitive indices to break ties
        let idx_i = sorted_indices[i];
        let idx_j = sorted_indices[j];
        return i32(32u + countLeadingZeros(idx_i ^ idx_j));
    }
    
    return i32(countLeadingZeros(code_i ^ code_j));
}

fn sign(x: i32) -> i32 {
    if x > 0 { return 1; }
    if x < 0 { return -1; }
    return 0;
}

fn determine_range(i: i32) -> vec2<i32> {
    // Determine direction
    let d = sign(delta(i, i + 1) - delta(i, i - 1));
    
    // Find minimum number of bits differing with neighbor
    let delta_min = delta(i, i - d);
    
    // Binary search to find the other end
    var l_max = 2;
    while delta(i, i + l_max * d) > delta_min {
        l_max *= 2;
    }
    
    var l = 0;
    for (var t = l_max / 2; t >= 1; t /= 2) {
        if delta(i, i + (l + t) * d) > delta_min {
            l += t;
        }
    }
    
    let j = i + l * d;
    return vec2<i32>(min(i, j), max(i, j));
}

fn find_split(first: i32, last: i32) -> i32 {
    let first_code = sorted_codes[first];
    let last_code = sorted_codes[last];
    
    if first_code == last_code {
        // Same Morton code, split by primitive index
        return (first + last) >> 1;
    }
    
    let common_prefix: u32 = countLeadingZeros(first_code ^ last_code);
    var split = first;
    let step = last - first;
    
    var current_step = step;
    while current_step > 1 {
        current_step = (current_step + 1) >> 1;
        let new_split = split + current_step;
        
        if new_split < last {
            let split_code = sorted_codes[new_split];
            let split_prefix: u32 = countLeadingZeros(first_code ^ split_code);
            if split_prefix > common_prefix {
                split = new_split;
            }
        }
    }
    
    return split;
}

@compute @workgroup_size(64)
fn link_nodes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    
    if i >= i32(uniforms.prim_count - 1u) {
        return;
    }
    
    // Internal node index (leaf nodes come after internal nodes)
    let internal_idx = u32(i);
    
    // Determine range of keys covered by this internal node
    let range = determine_range(i);
    let first = range.x;
    let last = range.y;
    
    // Find split position
    let split = find_split(first, last);
    
    // Determine child node indices
    var left_child: u32;
    var right_child: u32;
    
    if split == first {
        // Left child is a leaf
        left_child = uniforms.prim_count - 1u + u32(split);
    } else {
        // Left child is an internal node
        left_child = u32(split);
    }
    
    if split + 1 == last {
        // Right child is a leaf
        right_child = uniforms.prim_count - 1u + u32(split + 1);
    } else {
        // Right child is an internal node
        right_child = u32(split + 1);
    }
    
    // Initialize internal node
    nodes[internal_idx].kind = 0u; // Internal node
    nodes[internal_idx].left_idx = left_child;
    nodes[internal_idx].right_idx = right_child;
    nodes[internal_idx].parent_idx = 0xffffffffu; // Will be set by parent
    
    // Set parent pointers for children
    nodes[left_child].parent_idx = internal_idx;
    nodes[right_child].parent_idx = internal_idx;
}

@compute @workgroup_size(64)
fn init_leaves(@builtin(global_invocation_id) gid: vec3<u32>) {
    let leaf_idx = gid.x;
    
    if leaf_idx >= uniforms.prim_count {
        return;
    }
    
    let node_idx = uniforms.prim_count - 1u + leaf_idx;
    let prim_idx = sorted_indices[leaf_idx];
    
    // Initialize leaf node
    nodes[node_idx].aabb = primitive_aabbs[prim_idx];
    nodes[node_idx].kind = 1u; // Leaf node
    nodes[node_idx].left_idx = prim_idx; // First primitive index
    nodes[node_idx].right_idx = 1u; // Primitive count
    nodes[node_idx].parent_idx = 0xffffffffu; // Will be set by parent
}
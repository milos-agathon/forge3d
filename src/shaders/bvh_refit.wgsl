// src/shaders/bvh_refit.wgsl
// WGSL compute kernel for bottom-up BVH AABB refit without topology changes.
// This file exists to implement fast BVH updating: propagating updated primitive AABBs up the tree for dynamic scenes.
// RELEVANT FILES:src/accel/lbvh_gpu.rs,src/accel/types.rs,src/shaders/lbvh_link.wgsl

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
@group(1) @binding(0) var<storage, read> primitive_aabbs: array<Aabb>;
@group(1) @binding(1) var<storage, read> sorted_indices: array<u32>;
@group(2) @binding(0) var<storage, read_write> nodes: array<BvhNode>;
@group(2) @binding(1) var<storage, read_write> node_flags: array<u32>; // Atomic flags for processing

fn aabb_union(a: Aabb, b: Aabb) -> Aabb {
    var result: Aabb;
    result.min_val = min(a.min_val, b.min_val);
    result.max_val = max(a.max_val, b.max_val);
    return result;
}

fn aabb_empty() -> Aabb {
    var result: Aabb;
    result.min_val = vec3<f32>(1e30);
    result.max_val = vec3<f32>(-1e30);
    return result;
}

@compute @workgroup_size(64)
fn refit_leaves(@builtin(global_invocation_id) gid: vec3<u32>) {
    let leaf_idx = gid.x;
    
    if leaf_idx >= uniforms.prim_count {
        return;
    }
    
    let node_idx = uniforms.prim_count - 1u + leaf_idx;
    let prim_idx = sorted_indices[leaf_idx];
    
    // Update leaf AABB with new primitive AABB
    nodes[node_idx].aabb = primitive_aabbs[prim_idx];
    
    // Signal that this leaf has been processed
    node_flags[node_idx] = 1u;
}

@compute @workgroup_size(64)
fn refit_internal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let internal_idx = gid.x;
    
    if internal_idx >= uniforms.prim_count - 1u {
        return;
    }
    
    let left_idx = nodes[internal_idx].left_idx;
    let right_idx = nodes[internal_idx].right_idx;
    
    // Wait for children to be processed
    // This is a simplified approach; in practice, you'd use a more sophisticated
    // synchronization scheme or multiple passes
    var left_ready = false;
    var right_ready = false;
    
    // Check if children are ready (this is a simplified check)
    for (var attempts = 0u; attempts < 1000u; attempts = attempts + 1u) {
        if node_flags[left_idx] != 0u {
            left_ready = true;
        }
        if node_flags[right_idx] != 0u {
            right_ready = true;
        }
        
        if left_ready && right_ready {
            break;
        }
        
        // Simple busy wait (not ideal, but works for small trees)
        if attempts % 10u == 0u {
            storageBarrier();
        }
    }
    
    if left_ready && right_ready {
        // Compute union of child AABBs
        let left_aabb = nodes[left_idx].aabb;
        let right_aabb = nodes[right_idx].aabb;
        let union_aabb = aabb_union(left_aabb, right_aabb);
        
        // Update this node's AABB
        nodes[internal_idx].aabb = union_aabb;
        
        // Signal that this node has been processed
        node_flags[internal_idx] = 1u;
    }
}

@compute @workgroup_size(1)
fn refit_iterative(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Alternative approach: single-threaded bottom-up traversal
    // More reliable than the parallel version above
    
    if gid.x != 0u {
        return;
    }
    
    // Clear all flags
    for (var i = 0u; i < uniforms.node_count; i = i + 1u) {
        node_flags[i] = 0u;
    }
    
    // Update leaf nodes first
    for (var leaf_idx = 0u; leaf_idx < uniforms.prim_count; leaf_idx = leaf_idx + 1u) {
        let node_idx = uniforms.prim_count - 1u + leaf_idx;
        let prim_idx = sorted_indices[leaf_idx];
        nodes[node_idx].aabb = primitive_aabbs[prim_idx];
        node_flags[node_idx] = 1u;
    }
    
    // Process internal nodes from leaves to root
    var processed_count = 0u;
    let max_iterations = uniforms.prim_count - 1u;
    
    while processed_count < max_iterations {
        var progress_made = false;
        
        for (var internal_idx = 0u; internal_idx < uniforms.prim_count - 1u; internal_idx = internal_idx + 1u) {
            if node_flags[internal_idx] != 0u {
                continue; // Already processed
            }
            
            let left_idx = nodes[internal_idx].left_idx;
            let right_idx = nodes[internal_idx].right_idx;
            
            // Check if both children are ready
            if node_flags[left_idx] != 0u && node_flags[right_idx] != 0u {
                // Compute union of child AABBs
                let left_aabb = nodes[left_idx].aabb;
                let right_aabb = nodes[right_idx].aabb;
                let union_aabb = aabb_union(left_aabb, right_aabb);
                
                // Update this node's AABB
                nodes[internal_idx].aabb = union_aabb;
                node_flags[internal_idx] = 1u;
                
                processed_count = processed_count + 1u;
                progress_made = true;
            }
        }
        
        if !progress_made {
            break; // Prevent infinite loop
        }
    }
}
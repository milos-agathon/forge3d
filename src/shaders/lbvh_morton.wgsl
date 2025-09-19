// src/shaders/lbvh_morton.wgsl
// WGSL compute kernel for LBVH Morton code generation from primitive centroids.
// This file exists to implement Morton code computation for LBVH construction: normalizing world coordinates to [0,1]³ and computing 30-bit Morton codes.
// RELEVANT FILES:src/accel/lbvh_gpu.rs,src/accel/types.rs,src/shaders/lbvh_link.wgsl

struct Uniforms {
    prim_count: u32,
    frame_index: u32,
    _pad0: u32,
    _pad1: u32,
    world_min: vec3<f32>,
    _pad2: f32,
    world_extent: vec3<f32>,
    _pad3: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> centroids: array<vec3<f32>>;
@group(1) @binding(1) var<storage, read> prim_indices: array<u32>;
@group(2) @binding(0) var<storage, read_write> morton_codes: array<u32>;
@group(2) @binding(1) var<storage, read_write> sorted_indices: array<u32>;

fn expand_bits(v: u32) -> u32 {
    var x: u32 = v & 0x000003ffu;                  // mask to 10 bits
    x = (x | (x << 16u)) & 0x030000ffu; // x = ---- ---- ---- ---- ---- --98 ---- ---- 7654 3210
    x = (x | (x << 8u))  & 0x0300f00fu; // x = ---- ---- 98-- ---- ---- 7654 ---- ---- 3210
    x = (x | (x << 4u))  & 0x030c30c3u; // x = ---- --98 ---- 76-- ---- 54-- ---- 32-- ---- 10
    x = (x | (x << 2u))  & 0x09249249u; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
}

fn morton3d(x: u32, y: u32, z: u32) -> u32 {
    let xx = expand_bits(x);
    let yy = expand_bits(y);
    let zz = expand_bits(z);
    return xx | (yy << 1u) | (zz << 2u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= uniforms.prim_count {
        return;
    }
    
    // Get centroid for this primitive
    let centroid = centroids[index];
    
    // Normalize to [0, 1]³ using world AABB
    let normalized = (centroid - uniforms.world_min) / max(uniforms.world_extent, vec3<f32>(1e-6));
    let clamped = clamp(normalized, vec3<f32>(0.0), vec3<f32>(1.0));
    
    // Convert to Morton grid coordinates (1024³ grid)
    let grid_coords = vec3<u32>(
        min(u32(clamped.x * 1023.0), 1023u),
        min(u32(clamped.y * 1023.0), 1023u),
        min(u32(clamped.z * 1023.0), 1023u)
    );
    
    // Compute Morton code
    let morton_code = morton3d(grid_coords.x, grid_coords.y, grid_coords.z);
    
    // Store Morton code and initialize sorted indices
    morton_codes[index] = morton_code;
    sorted_indices[index] = prim_indices[index];
}
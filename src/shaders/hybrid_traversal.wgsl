// src/shaders/hybrid_traversal.wgsl
// Hybrid traversal combining mesh BVH traversal with legacy hooks for SDF.

// GPU hybrid traversal currently focuses on mesh geometry; SDF support is CPU-only.

// Hybrid traversal configuration (GPU path currently supports mesh traversal only)

// Hybrid scene data structures
struct HybridUniforms {
    sdf_primitive_count: u32,
    sdf_node_count: u32,
    mesh_vertex_count: u32,
    mesh_index_count: u32,
    mesh_bvh_node_count: u32,
    mesh_bvh_root_index: u32,
    traversal_mode: u32, // 0 = hybrid, 1 = SDF only, 2 = mesh only
    _pad: u32,
}

struct HybridHitResult {
    t: f32,
    point: vec3f,
    normal: vec3f,
    color: vec3f,       // Interpolated vertex color
    material_id: u32,
    hit_type: u32, // 0 = mesh, retained for compatibility
    hit: u32, // 0 = false, 1 = true
    _pad: u32,
}

struct Ray {
    origin: vec3f,
    tmin: f32,
    direction: vec3f,
    tmax: f32,
}

// BVH structures (match accel::types::BvhNode layout)
// Rust layout:
//   struct Aabb { min: [f32;3], _pad0: f32, max: [f32;3], _pad1: f32 }
//   struct BvhNode { aabb: Aabb, kind: u32, left_idx: u32, right_idx: u32, parent_idx: u32 }
struct Aabb {
    min: vec3f,
    _pad0: f32,
    max: vec3f,
    _pad1: f32,
}

struct BvhNode {
    aabb: Aabb,
    kind: u32,       // 0 = internal, 1 = leaf
    left_idx: u32,   // internal: left child; leaf: first triangle index
    right_idx: u32,  // internal: right child; leaf: triangle count
    parent_idx: u32,
}

struct MeshVertex {
    position: vec3f,
    _pad: f32,
    color: vec3f,
    _pad2: f32,
}

// Bind groups for hybrid traversal
// NOTE: Consolidated into group(1) to stay within max_bind_groups=4
@group(1) @binding(1) var<uniform> hybrid_uniforms: HybridUniforms;
@group(1) @binding(2) var<storage, read> mesh_vertices: array<MeshVertex>;
@group(1) @binding(3) var<storage, read> mesh_indices: array<u32>;
@group(1) @binding(4) var<storage, read> mesh_bvh_nodes: array<BvhNode>;

// Ray-AABB intersection for BVH traversal
fn ray_aabb_intersect(ray: Ray, aabb_min: vec3f, aabb_max: vec3f) -> bool {
    var tmin = ray.tmin;
    var tmax = ray.tmax;

    for (var i = 0u; i < 3u; i = i + 1u) {
        let inv_dir = 1.0 / ray.direction[i];
        var t0 = (aabb_min[i] - ray.origin[i]) * inv_dir;
        var t1 = (aabb_max[i] - ray.origin[i]) * inv_dir;

        if (inv_dir < 0.0) {
            let temp = t0;
            t0 = t1;
            t1 = temp;
        }

        tmin = max(tmin, t0);
        tmax = min(tmax, t1);

        if (tmin > tmax) {
            return false;
        }
    }

    return true;
}

// Ray-triangle intersection with barycentric color interpolation
fn ray_triangle_intersect(
    ray: Ray,
    v0: vec3f,
    v1: vec3f,
    v2: vec3f,
    c0: vec3f,
    c1: vec3f,
    c2: vec3f
) -> HybridHitResult {
    var result: HybridHitResult;
    result.hit = 0u;
    result.t = ray.tmax;
    result.hit_type = 0u; // mesh
    result.color = vec3f(0.7, 0.7, 0.8); // default

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(ray.direction, edge2);
    let a = dot(edge1, h);

    let epsilon = 1e-7;
    if (abs(a) < epsilon) {
        return result;
    }

    let f = 1.0 / a;
    let s = ray.origin - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return result;
    }

    let q = cross(s, edge1);
    let v = f * dot(ray.direction, q);
    if (v < 0.0 || u + v > 1.0) {
        return result;
    }

    let t = f * dot(edge2, q);
    if (t > ray.tmin && t < ray.tmax) {
        // Compute geometric normal from triangle winding
        var normal = normalize(cross(edge1, edge2));
        
        // Flip normal if it faces away from camera (needed for terrain heightmaps)
        if (dot(normal, -ray.direction) < 0.0) {
            normal = -normal;
        }
        
        // Barycentric interpolation: w = 1-u-v, u, v
        let w = 1.0 - u - v;
        result.hit = 1u;
        result.t = t;
        result.point = ray.origin + ray.direction * t;
        result.normal = normal;
        result.color = w * c0 + u * c1 + v * c2;  // Interpolate colors
        result.material_id = 0u;
        result.hit_type = 0u; // mesh
    }

    return result;
}

// BVH traversal for mesh intersection
const MAX_BVH_STACK_SIZE: u32 = 128u;

fn intersect_mesh(ray: Ray) -> HybridHitResult {
    var result: HybridHitResult;
    result.hit = 0u;
    result.t = ray.tmax;
    result.hit_type = 0u; // mesh

    let node_count = hybrid_uniforms.mesh_bvh_node_count;
    let index_count = hybrid_uniforms.mesh_index_count;
    let vertex_count = hybrid_uniforms.mesh_vertex_count;

    // Fast path: use BVH traversal when we have data
    if (node_count > 0u && vertex_count > 0u) {
        // Stack-based BVH traversal (iterative DFS)
        var stack: array<u32, MAX_BVH_STACK_SIZE>;
        var stack_size = 1u;
        let root = hybrid_uniforms.mesh_bvh_root_index;
        stack[0] = root; // Push root node
        var stack_overflow = false;

        while (stack_size > 0u) {
            // Pop node from stack
            stack_size -= 1u;
            let node_idx = stack[stack_size];

            if (node_idx >= node_count) {
                continue;
            }

            let node = mesh_bvh_nodes[node_idx];

            // Test ray against AABB - skip if no intersection
            if (!ray_aabb_intersect(ray, node.aabb.min, node.aabb.max)) {
                continue;
            }

            // Check if leaf node (kind bit 0 == 1)
            let is_leaf = (node.kind & 1u) != 0u;

            if (is_leaf) {
                // Leaf: test all triangles
                let tri_start = node.left_idx;  // first triangle index
                let tri_count = node.right_idx; // number of triangles

                for (var i = 0u; i < tri_count; i += 1u) {
                    let tri_idx = (tri_start + i) * 3u;
                    if (tri_idx + 2u >= index_count) { break; }

                    let i0 = mesh_indices[tri_idx];
                    let i1 = mesh_indices[tri_idx + 1u];
                    let i2 = mesh_indices[tri_idx + 2u];

                    if (i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count) {
                        continue;
                    }

                    let v0 = mesh_vertices[i0].position;
                    let v1 = mesh_vertices[i1].position;
                    let v2 = mesh_vertices[i2].position;
                    let c0 = mesh_vertices[i0].color;
                    let c1 = mesh_vertices[i1].color;
                    let c2 = mesh_vertices[i2].color;

                    let tri_hit = ray_triangle_intersect(ray, v0, v1, v2, c0, c1, c2);
                    if (tri_hit.hit != 0u && tri_hit.t < result.t) {
                        result = tri_hit;
                    }
                }
            } else {
                // Internal node: push children onto stack (both should be valid)
                let left_idx = node.left_idx;
                let right_idx = node.right_idx;

                // Push right child first (so left is processed first - DFS left-to-right)
                if (right_idx < node_count) {
                    if (stack_size < MAX_BVH_STACK_SIZE) {
                        stack[stack_size] = right_idx;
                        stack_size += 1u;
                    } else {
                        stack_overflow = true;
                    }
                }
                if (left_idx < node_count) {
                    if (stack_size < MAX_BVH_STACK_SIZE) {
                        stack[stack_size] = left_idx;
                        stack_size += 1u;
                    } else {
                        stack_overflow = true;
                    }
                }
            }
        }

        if (stack_overflow) {
            // Force brute-force fallback for safety
            result.hit = 0u;
            result.t = ray.tmax;
        }
    }

    // Brute-force fallback if BVH traversal was unavailable or missed
    if (result.hit == 0u && index_count >= 3u && vertex_count > 0u) {
        for (var tri = 0u; tri + 2u < index_count; tri = tri + 3u) {
            let i0 = mesh_indices[tri];
            let i1 = mesh_indices[tri + 1u];
            let i2 = mesh_indices[tri + 2u];

            if (i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count) {
                continue;
            }

            let v0 = mesh_vertices[i0].position;
            let v1 = mesh_vertices[i1].position;
            let v2 = mesh_vertices[i2].position;
            let c0 = mesh_vertices[i0].color;
            let c1 = mesh_vertices[i1].color;
            let c2 = mesh_vertices[i2].color;

            let tri_hit = ray_triangle_intersect(ray, v0, v1, v2, c0, c1, c2);
            if (tri_hit.hit != 0u && tri_hit.t < result.t) {
                result = tri_hit;
            }
        }
    }

    return result;
}

// Main hybrid intersection function
fn intersect_hybrid(ray: Ray) -> HybridHitResult {
    var best_hit: HybridHitResult;
    best_hit.hit = 0u;
    best_hit.t = ray.tmax;

    // Test mesh geometry if enabled
    if (hybrid_uniforms.traversal_mode == 0u || hybrid_uniforms.traversal_mode == 2u) {
        let mesh_hit = intersect_mesh(ray);
        if (mesh_hit.hit != 0u && mesh_hit.t < best_hit.t) {
            best_hit = mesh_hit;
        }
    }

    return best_hit;
}

// Performance-optimized early termination
fn intersect_hybrid_optimized(ray: Ray, early_exit_distance: f32) -> HybridHitResult {
    var best_hit: HybridHitResult;
    best_hit.hit = 0u;
    best_hit.t = ray.tmax;

    if (hybrid_uniforms.traversal_mode == 0u || hybrid_uniforms.traversal_mode == 2u) {
        let mesh_hit = intersect_mesh(ray);
        if (mesh_hit.hit != 0u && mesh_hit.t < early_exit_distance) {
            return mesh_hit;
        }
        if (mesh_hit.hit != 0u && mesh_hit.t < best_hit.t) {
            best_hit = mesh_hit;
        }
    }

    return best_hit;
}

// Utility function to get surface properties at hit point
fn get_surface_properties(hit: HybridHitResult) -> vec3f {
    // Return interpolated color from hit result
    return hit.color;
}

// Shadow ray testing for both SDF and mesh geometry
// Culls backface hits to prevent shadow spikes from triangle backsides
fn intersect_shadow_ray(ray: Ray, max_distance: f32) -> bool {
    let hit = intersect_hybrid_optimized(ray, 0.05);  // Larger early exit for shadows
    if (hit.hit == 0u || hit.t >= max_distance) {
        return false;
    }
    // Backface culling: only count hits where normal faces toward ray origin
    // If dot(normal, -ray_dir) > 0, we hit a front face (valid shadow)
    let is_frontface = dot(hit.normal, -ray.direction) > 0.0;
    return is_frontface;
}

// Test occlusion for soft shadows (SDF can provide smoother shadows)
fn soft_shadow_factor(ray: Ray, max_distance: f32, softness: f32) -> f32 {
    return select(1.0, 0.0, intersect_shadow_ray(ray, max_distance));
}
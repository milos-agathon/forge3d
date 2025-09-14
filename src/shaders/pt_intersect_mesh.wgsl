// src/shaders/pt_intersect_mesh.wgsl
// BVH traversal and watertight triangle intersection for GPU path tracing (Task A3).
// This file implements iterative BVH traversal with Möller-Trumbore intersection testing.
// RELEVANT FILES:src/accel/cpu_bvh.rs,src/path_tracing/mesh.rs,src/shaders/pt_kernel.wgsl

// Expected bind group layout:
// Bind Group 0: Uniforms (width, height, frame_index, camera params, seed_hi/lo)
// Bind Group 1: Scene (vertices[], indices[], bvh_nodes[])
// Bind Group 2: Output/Accum (HDR accum or storage texture)

// GPU-compatible BVH node layout (matches src/accel/cpu_bvh.rs)
struct BvhNode {
    aabb_min: vec3<f32>,   // AABB minimum bounds
    left: u32,             // Internal: left child index; Leaf: first triangle index
    aabb_max: vec3<f32>,   // AABB maximum bounds
    right: u32,            // Internal: right child index; Leaf: triangle count
    flags: u32,            // Bit 0: leaf flag (1 = leaf, 0 = internal)
    _pad: u32,             // Padding for 16-byte alignment
}

// Vertex data (vec3 or vec4 with w unused for alignment)
struct Vertex {
    position: vec3<f32>,
    _pad: f32,
}

// Ray structure for intersection testing
struct Ray {
    origin: vec3<f32>,
    tmin: f32,
    direction: vec3<f32>,
    tmax: f32,
}

// Hit result from ray-triangle intersection
struct HitResult {
    t: f32,                // Ray parameter at hit point
    triangle_idx: u32,     // Index of hit triangle
    barycentric: vec2<f32>, // Barycentric coordinates (u, v; w = 1-u-v)
    normal: vec3<f32>,     // Surface normal at hit point
    hit: bool,             // Whether intersection occurred
}

// BVH traversal stack entry (for iterative traversal)
struct StackEntry {
    node_idx: u32,
    _pad: u32, // Alignment
}

// Maximum stack depth for BVH traversal (conservative estimate)
const MAX_STACK_DEPTH: u32 = 64u;

/// Test ray against axis-aligned bounding box
fn ray_aabb_intersect(ray: Ray, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    var tmin = ray.tmin;
    var tmax = ray.tmax;

    for (var i = 0u; i < 3u; i = i + 1u) {
        let inv_dir = 1.0 / ray.direction[i];
        var t0 = (aabb_min[i] - ray.origin[i]) * inv_dir;
        var t1 = (aabb_max[i] - ray.origin[i]) * inv_dir;

        // Ensure t0 <= t1
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

/// Watertight Möller-Trumbore ray-triangle intersection with robust epsilon handling
fn ray_triangle_intersect(
    ray: Ray,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = ray.tmax;

    // Compute triangle edges
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;

    // Cross product: ray.direction × edge2
    let h = cross(ray.direction, edge2);

    // Dot product: edge1 · h (determinant)
    let a = dot(edge1, h);

    // Ray is parallel to triangle (within epsilon)
    let epsilon = 1e-7;
    if (abs(a) < epsilon) {
        return result;
    }

    let f = 1.0 / a;
    let s = ray.origin - v0;

    // Compute u parameter and test bounds
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return result;
    }

    // Cross product: s × edge1
    let q = cross(s, edge1);

    // Compute v parameter and test bounds
    let v = f * dot(ray.direction, q);
    if (v < 0.0 || u + v > 1.0) {
        return result;
    }

    // Compute t parameter
    let t = f * dot(edge2, q);

    // Test ray parameter bounds
    if (t > ray.tmin && t < ray.tmax) {
        // Valid intersection - compute surface normal
        let normal = normalize(cross(edge1, edge2));

        result.hit = true;
        result.t = t;
        result.barycentric = vec2<f32>(u, v);
        result.normal = normal;
    }

    return result;
}

/// Iterative BVH traversal with stack-based node processing
/// Returns closest intersection along ray
fn bvh_intersect(
    ray: Ray,
    bvh_nodes: ptr<storage, array<BvhNode>, read>,
    vertices: ptr<storage, array<Vertex>, read>,
    indices: ptr<storage, array<u32>, read>
) -> HitResult {
    var closest_hit: HitResult;
    closest_hit.hit = false;
    closest_hit.t = ray.tmax;

    // Early exit if no nodes
    let node_count = arrayLength(bvh_nodes);
    if (node_count == 0u) {
        return closest_hit;
    }

    // Traversal stack (local memory per thread)
    var stack: array<u32, MAX_STACK_DEPTH>;
    var stack_ptr = 0u;

    // Start with root node
    stack[0] = 0u;
    stack_ptr = 1u;

    var current_ray = ray;

    // Iterative traversal loop
    while (stack_ptr > 0u) {
        // Pop node from stack
        stack_ptr = stack_ptr - 1u;
        let node_idx = stack[stack_ptr];

        // Bounds check
        if (node_idx >= node_count) {
            continue;
        }

        let node = (*bvh_nodes)[node_idx];

        // Test ray against node AABB
        if (!ray_aabb_intersect(current_ray, node.aabb_min, node.aabb_max)) {
            continue;
        }

        // Check if leaf node (bit 0 of flags)
        if ((node.flags & 1u) != 0u) {
            // Leaf node - test triangles
            let first_tri = node.left;
            let tri_count = node.right;

            for (var i = 0u; i < tri_count; i = i + 1u) {
                let tri_idx = first_tri + i;

                // Bounds check for triangle indices
                let indices_length = arrayLength(indices);
                if (tri_idx * 3u + 2u >= indices_length) {
                    continue;
                }

                // Get triangle vertex indices
                let i0 = (*indices)[tri_idx * 3u];
                let i1 = (*indices)[tri_idx * 3u + 1u];
                let i2 = (*indices)[tri_idx * 3u + 2u];

                // Bounds check for vertices
                let vertex_count = arrayLength(vertices);
                if (max(max(i0, i1), i2) >= vertex_count) {
                    continue;
                }

                // Get triangle vertices
                let v0 = (*vertices)[i0].position;
                let v1 = (*vertices)[i1].position;
                let v2 = (*vertices)[i2].position;

                // Test ray-triangle intersection
                var hit = ray_triangle_intersect(current_ray, v0, v1, v2);

                if (hit.hit && hit.t < closest_hit.t) {
                    closest_hit = hit;
                    closest_hit.triangle_idx = tri_idx;
                    // Update ray tmax for early termination
                    current_ray.tmax = hit.t;
                }
            }
        } else {
            // Internal node - add children to stack
            let left_idx = node.left;
            let right_idx = node.right;

            // Add children in order that may improve traversal efficiency
            // For better performance, we could order based on ray direction,
            // but for MVP we use simple order

            // Add right child first (will be processed after left)
            if (right_idx < node_count && stack_ptr < MAX_STACK_DEPTH) {
                stack[stack_ptr] = right_idx;
                stack_ptr = stack_ptr + 1u;
            }

            // Add left child second (will be processed first)
            if (left_idx < node_count && stack_ptr < MAX_STACK_DEPTH) {
                stack[stack_ptr] = left_idx;
                stack_ptr = stack_ptr + 1u;
            }
        }
    }

    return closest_hit;
}

/// Test ray against scene bounds for early rejection
fn ray_scene_bounds_test(
    ray: Ray,
    scene_aabb_min: vec3<f32>,
    scene_aabb_max: vec3<f32>
) -> bool {
    return ray_aabb_intersect(ray, scene_aabb_min, scene_aabb_max);
}

/// Optimized BVH intersection for shadow rays (any-hit, not closest-hit)
fn bvh_intersect_any(
    ray: Ray,
    bvh_nodes: ptr<storage, array<BvhNode>, read>,
    vertices: ptr<storage, array<Vertex>, read>,
    indices: ptr<storage, array<u32>, read>
) -> bool {
    let node_count = arrayLength(bvh_nodes);
    if (node_count == 0u) {
        return false;
    }

    var stack: array<u32, MAX_STACK_DEPTH>;
    var stack_ptr = 0u;

    stack[0] = 0u;
    stack_ptr = 1u;

    while (stack_ptr > 0u) {
        stack_ptr = stack_ptr - 1u;
        let node_idx = stack[stack_ptr];

        if (node_idx >= node_count) {
            continue;
        }

        let node = (*bvh_nodes)[node_idx];

        if (!ray_aabb_intersect(ray, node.aabb_min, node.aabb_max)) {
            continue;
        }

        if ((node.flags & 1u) != 0u) {
            // Leaf node
            let first_tri = node.left;
            let tri_count = node.right;

            for (var i = 0u; i < tri_count; i = i + 1u) {
                let tri_idx = first_tri + i;

                let indices_length = arrayLength(indices);
                if (tri_idx * 3u + 2u >= indices_length) {
                    continue;
                }

                let i0 = (*indices)[tri_idx * 3u];
                let i1 = (*indices)[tri_idx * 3u + 1u];
                let i2 = (*indices)[tri_idx * 3u + 2u];

                let vertex_count = arrayLength(vertices);
                if (max(max(i0, i1), i2) >= vertex_count) {
                    continue;
                }

                let v0 = (*vertices)[i0].position;
                let v1 = (*vertices)[i1].position;
                let v2 = (*vertices)[i2].position;

                let hit = ray_triangle_intersect(ray, v0, v1, v2);

                if (hit.hit) {
                    return true; // Any hit found - early exit
                }
            }
        } else {
            // Internal node
            let left_idx = node.left;
            let right_idx = node.right;

            if (right_idx < node_count && stack_ptr < MAX_STACK_DEPTH) {
                stack[stack_ptr] = right_idx;
                stack_ptr = stack_ptr + 1u;
            }

            if (left_idx < node_count && stack_ptr < MAX_STACK_DEPTH) {
                stack[stack_ptr] = left_idx;
                stack_ptr = stack_ptr + 1u;
            }
        }
    }

    return false;
}
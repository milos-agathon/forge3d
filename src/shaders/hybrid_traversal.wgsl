// src/shaders/hybrid_traversal.wgsl
// Hybrid traversal combining SDF raymarching with BVH mesh traversal
// Integrates with existing path tracer infrastructure

#include "sdf_primitives.wgsl"
#include "sdf_operations.wgsl"

// Hybrid traversal configuration
const MAX_RAYMARCH_STEPS: u32 = 128u;
const MIN_RAYMARCH_DISTANCE: f32 = 0.001;
const MAX_RAYMARCH_DISTANCE: f32 = 100.0;
const RAYMARCH_EPSILON: f32 = 0.001;

// Hybrid scene data structures
struct HybridUniforms {
    sdf_primitive_count: u32,
    sdf_node_count: u32,
    mesh_vertex_count: u32,
    mesh_index_count: u32,
    mesh_bvh_node_count: u32,
    traversal_mode: u32, // 0 = hybrid, 1 = SDF only, 2 = mesh only
    _pad: vec2u,
}

struct HybridHitResult {
    t: f32,
    point: vec3f,
    normal: vec3f,
    material_id: u32,
    hit_type: u32, // 0 = mesh, 1 = SDF
    hit: u32, // 0 = false, 1 = true
    _pad: vec2u,
}

struct Ray {
    origin: vec3f,
    tmin: f32,
    direction: vec3f,
    tmax: f32,
}

// BVH structures (matching existing pt_kernel.wgsl)
struct BvhNode {
    aabb_min: vec3f,
    left: u32,
    aabb_max: vec3f,
    right: u32,
    flags: u32,
    _pad: u32,
}

struct MeshVertex {
    position: vec3f,
    _pad: f32,
}

// Bind groups for hybrid traversal
@group(5) @binding(0) var<uniform> hybrid_uniforms: HybridUniforms;
@group(5) @binding(1) var<storage, read> sdf_primitives: array<SdfPrimitive>;
@group(5) @binding(2) var<storage, read> sdf_nodes: array<CsgNode>;
@group(5) @binding(3) var<storage, read> mesh_vertices: array<MeshVertex>;
@group(5) @binding(4) var<storage, read> mesh_indices: array<u32>;
@group(5) @binding(5) var<storage, read> mesh_bvh_nodes: array<BvhNode>;

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

// Ray-triangle intersection
fn ray_triangle_intersect(
    ray: Ray,
    v0: vec3f,
    v1: vec3f,
    v2: vec3f
) -> HybridHitResult {
    var result: HybridHitResult;
    result.hit = 0u;
    result.t = ray.tmax;
    result.hit_type = 0u; // mesh

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
        let normal = normalize(cross(edge1, edge2));
        result.hit = 1u;
        result.t = t;
        result.point = ray.origin + ray.direction * t;
        result.normal = normal;
        result.material_id = 0u; // Default mesh material
        result.hit_type = 0u; // mesh
    }

    return result;
}

// BVH traversal for mesh intersection
const MAX_BVH_STACK_SIZE: u32 = 32u;

fn intersect_mesh(ray: Ray) -> HybridHitResult {
    var result: HybridHitResult;
    result.hit = 0u;
    result.t = ray.tmax;
    result.hit_type = 0u; // mesh

    if (hybrid_uniforms.mesh_bvh_node_count == 0u) {
        return result;
    }

    // Stack-based BVH traversal
    var stack: array<u32, MAX_BVH_STACK_SIZE>;
    var stack_ptr = 0u;
    stack[0] = 0u; // Start with root node

    while (stack_ptr != 4294967295u) { // while stack_ptr >= 0 (u32 underflow check)
        let node_idx = stack[stack_ptr];
        stack_ptr = stack_ptr - 1u;

        if (node_idx >= hybrid_uniforms.mesh_bvh_node_count) {
            continue;
        }

        let node = mesh_bvh_nodes[node_idx];

        // Test ray against AABB
        if (!ray_aabb_intersect(ray, node.aabb_min, node.aabb_max)) {
            continue;
        }

        // Check if leaf node
        if ((node.flags & 1u) != 0u) {
            // Leaf node: test triangles
            let start_idx = node.left;
            let end_idx = node.right;

            for (var tri_idx = start_idx; tri_idx < end_idx && tri_idx + 2u < hybrid_uniforms.mesh_index_count; tri_idx = tri_idx + 3u) {
                let i0 = mesh_indices[tri_idx];
                let i1 = mesh_indices[tri_idx + 1u];
                let i2 = mesh_indices[tri_idx + 2u];

                if (i0 < hybrid_uniforms.mesh_vertex_count &&
                    i1 < hybrid_uniforms.mesh_vertex_count &&
                    i2 < hybrid_uniforms.mesh_vertex_count) {

                    let v0 = mesh_vertices[i0].position;
                    let v1 = mesh_vertices[i1].position;
                    let v2 = mesh_vertices[i2].position;

                    let tri_hit = ray_triangle_intersect(ray, v0, v1, v2);
                    if (tri_hit.hit != 0u && tri_hit.t < result.t) {
                        result = tri_hit;
                    }
                }
            }
        } else {
            // Internal node: add children to stack
            if (stack_ptr + 1u < MAX_BVH_STACK_SIZE - 1u) {
                stack_ptr = stack_ptr + 1u;
                stack[stack_ptr] = node.left;
                stack_ptr = stack_ptr + 1u;
                stack[stack_ptr] = node.right;
            }
        }
    }

    return result;
}

// Evaluate SDF scene at a point
fn evaluate_sdf_scene(point: vec3f) -> CsgResult {
    if (hybrid_uniforms.sdf_node_count == 0u) {
        return CsgResult(1e20, 0u);
    }

    // For simplicity, evaluate the last (root) node
    let root_idx = hybrid_uniforms.sdf_node_count - 1u;

    // Create local copies for the iterative evaluator
    var local_nodes: array<CsgNode, 64>;
    var local_primitives: array<SdfPrimitive, 64>;

    let max_nodes = min(hybrid_uniforms.sdf_node_count, 64u);
    let max_primitives = min(hybrid_uniforms.sdf_primitive_count, 64u);

    for (var i = 0u; i < max_nodes; i = i + 1u) {
        local_nodes[i] = sdf_nodes[i];
    }

    for (var i = 0u; i < max_primitives; i = i + 1u) {
        local_primitives[i] = sdf_primitives[i];
    }

    return evaluate_csg_tree_iterative(
        point,
        root_idx,
        &local_nodes,
        &local_primitives,
        max_nodes,
        max_primitives
    );
}

// Calculate SDF normal using finite differences
fn calculate_sdf_normal(point: vec3f) -> vec3f {
    let eps = RAYMARCH_EPSILON;
    let normal = vec3f(
        evaluate_sdf_scene(point + vec3f(eps, 0.0, 0.0)).distance -
        evaluate_sdf_scene(point - vec3f(eps, 0.0, 0.0)).distance,

        evaluate_sdf_scene(point + vec3f(0.0, eps, 0.0)).distance -
        evaluate_sdf_scene(point - vec3f(0.0, eps, 0.0)).distance,

        evaluate_sdf_scene(point + vec3f(0.0, 0.0, eps)).distance -
        evaluate_sdf_scene(point - vec3f(0.0, 0.0, eps)).distance
    );

    return normalize(normal);
}

// SDF raymarching
fn raymarch_sdf(ray: Ray) -> HybridHitResult {
    var result: HybridHitResult;
    result.hit = 0u;
    result.t = ray.tmax;
    result.hit_type = 1u; // SDF

    if (hybrid_uniforms.sdf_primitive_count == 0u) {
        return result;
    }

    var t = ray.tmin;
    var steps = 0u;

    while (steps < MAX_RAYMARCH_STEPS && t < min(ray.tmax, MAX_RAYMARCH_DISTANCE)) {
        let point = ray.origin + ray.direction * t;
        let sdf_result = evaluate_sdf_scene(point);

        if (sdf_result.distance < MIN_RAYMARCH_DISTANCE) {
            // Hit!
            result.hit = 1u;
            result.t = t;
            result.point = point;
            result.normal = calculate_sdf_normal(point);
            result.material_id = sdf_result.material_id;
            result.hit_type = 1u; // SDF
            break;
        }

        // March forward
        t += max(abs(sdf_result.distance), MIN_RAYMARCH_DISTANCE * 0.1);
        steps = steps + 1u;
    }

    return result;
}

// Main hybrid intersection function
fn intersect_hybrid(ray: Ray) -> HybridHitResult {
    var best_hit: HybridHitResult;
    best_hit.hit = 0u;
    best_hit.t = ray.tmax;

    // Test SDF geometry if enabled
    if (hybrid_uniforms.traversal_mode == 0u || hybrid_uniforms.traversal_mode == 1u) {
        let sdf_hit = raymarch_sdf(ray);
        if (sdf_hit.hit != 0u && sdf_hit.t < best_hit.t) {
            best_hit = sdf_hit;
        }
    }

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

    // For performance, test meshes first since they're typically faster
    if (hybrid_uniforms.traversal_mode == 0u || hybrid_uniforms.traversal_mode == 2u) {
        let mesh_hit = intersect_mesh(ray);
        if (mesh_hit.hit != 0u && mesh_hit.t < early_exit_distance) {
            // Early exit if mesh hit is very close
            return mesh_hit;
        }
        if (mesh_hit.hit != 0u && mesh_hit.t < best_hit.t) {
            best_hit = mesh_hit;
        }
    }

    // Test SDF geometry with potentially reduced max distance
    if (hybrid_uniforms.traversal_mode == 0u || hybrid_uniforms.traversal_mode == 1u) {
        let sdf_ray = Ray(ray.origin, ray.tmin, ray.direction, min(ray.tmax, best_hit.t));
        let sdf_hit = raymarch_sdf(sdf_ray);
        if (sdf_hit.hit != 0u && sdf_hit.t < best_hit.t) {
            best_hit = sdf_hit;
        }
    }

    return best_hit;
}

// Utility function to get surface properties at hit point
fn get_surface_properties(hit: HybridHitResult) -> vec3f {
    // Return albedo based on hit type and material
    if (hit.hit_type == 1u) {
        // SDF hit - use material-based coloring
        switch (hit.material_id) {
            case 1u: { return vec3f(0.8, 0.2, 0.2); } // Red
            case 2u: { return vec3f(0.2, 0.8, 0.2); } // Green
            case 3u: { return vec3f(0.2, 0.2, 0.8); } // Blue
            default: { return vec3f(0.9, 0.6, 0.3); } // Orange
        }
    } else {
        // Mesh hit - default mesh color
        return vec3f(0.7, 0.7, 0.8);
    }
}

// Shadow ray testing for both SDF and mesh geometry
fn intersect_shadow_ray(ray: Ray, max_distance: f32) -> bool {
    // Use optimized traversal with early exit
    let hit = intersect_hybrid_optimized(ray, 0.01);
    return hit.hit != 0u && hit.t < max_distance;
}

// Test occlusion for soft shadows (SDF can provide smoother shadows)
fn soft_shadow_factor(ray: Ray, max_distance: f32, softness: f32) -> f32 {
    if (hybrid_uniforms.sdf_primitive_count == 0u) {
        // No SDFs, use hard shadows
        return select(1.0, 0.0, intersect_shadow_ray(ray, max_distance));
    }

    var t = ray.tmin;
    var shadow_factor = 1.0;
    let k = softness;

    // March through SDF field to compute soft shadows
    while (t < max_distance) {
        let point = ray.origin + ray.direction * t;
        let sdf_result = evaluate_sdf_scene(point);

        if (sdf_result.distance < 0.001) {
            return 0.0; // In shadow
        }

        // Compute soft shadow contribution
        shadow_factor = min(shadow_factor, k * sdf_result.distance / t);
        t += max(sdf_result.distance, 0.001);
    }

    // Also check mesh occlusion
    if (intersect_shadow_ray(ray, max_distance)) {
        return 0.0;
    }

    return clamp(shadow_factor, 0.0, 1.0);
}
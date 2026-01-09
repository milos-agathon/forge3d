// src/shaders/bvh_ray_query.wgsl
// GPU BVH Ray Query Shader
// Traverses the LBVH structure to find ray-triangle intersections

struct BvhNode {
    aabb_min: vec3<f32>,
    left: u32,
    aabb_max: vec3<f32>,
    right: u32,
    flags: u32, // bit 0: leaf
    pad: u32,
};

struct Triangle {
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
};

struct Ray {
    origin: vec3<f32>,
    t_min: f32,
    direction: vec3<f32>,
    t_max: f32,
};

struct Hit {
    t: f32,
    u: f32,
    v: f32,
    prim_idx: u32,
    instance_idx: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<BvhNode>;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read> indices: array<u32>; // If indexed
@group(0) @binding(3) var<storage, read> rays: array<Ray>;
@group(0) @binding(4) var<storage, read_write> hits: array<Hit>;

// Möller-Trumbore intersection
fn intersect_triangle(ray: Ray, tri: Triangle) -> Hit {
    var hit: Hit;
    hit.t = ray.t_max;
    hit.prim_idx = 0xFFFFFFFFu;

    let e1 = tri.v1 - tri.v0;
    let e2 = tri.v2 - tri.v0;
    let h = cross(ray.direction, e2);
    let a = dot(e1, h);

    if (abs(a) < 1e-7) { return hit; }

    let f = 1.0 / a;
    let s = ray.origin - tri.v0;
    let u = f * dot(s, h);

    if (u < 0.0 || u > 1.0) { return hit; }

    let q = cross(s, e1);
    let v = f * dot(ray.direction, q);

    if (v < 0.0 || u + v > 1.0) { return hit; }

    let t = f * dot(e2, q);

    if (t > ray.t_min && t < ray.t_max) {
        hit.t = t;
        hit.u = u;
        hit.v = v;
        hit.prim_idx = 1u; // Valid hit
    }
    return hit;
}

fn intersect_aabb(ray: Ray, min_b: vec3<f32>, max_b: vec3<f32>) -> bool {
    let inv_d = 1.0 / ray.direction;
    let t0 = (min_b - ray.origin) * inv_d;
    let t1 = (max_b - ray.origin) * inv_d;

    let tmin = min(t0, t1);
    let tmax = max(t0, t1);

    let t_enter = max(max(tmin.x, tmin.y), tmin.z);
    let t_exit = min(min(tmax.x, tmax.y), tmax.z);

    return t_enter <= t_exit && t_exit >= ray.t_min && t_enter <= ray.t_max;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_idx = global_id.x;
    if (ray_idx >= arrayLength(&rays)) { return; }

    var ray = rays[ray_idx];
    var closest_hit: Hit;
    closest_hit.t = ray.t_max;
    closest_hit.prim_idx = 0xFFFFFFFFu;

    // Stack for traversal (fixed size)
    var stack: array<u32, 32>;
    var stack_ptr = 0u;
    stack[stack_ptr] = 0u; // Root node
    stack_ptr++;

    while (stack_ptr > 0u) {
        stack_ptr--;
        let node_idx = stack[stack_ptr];
        let node = nodes[node_idx];

        if (!intersect_aabb(ray, node.aabb_min, node.aabb_max)) {
            continue;
        }

        if ((node.flags & 1u) != 0u) {
            // Leaf
            let count = node.right;
            let first = node.left;
            for (var i = 0u; i < count; i++) {
                let tri_idx = first + i;
                let tri = triangles[tri_idx];
                let hit = intersect_triangle(ray, tri);
                if (hit.prim_idx != 0xFFFFFFFFu && hit.t < closest_hit.t) {
                    closest_hit = hit;
                    closest_hit.prim_idx = tri_idx;
                    ray.t_max = hit.t; // Prune search
                }
            }
        } else {
            // Internal - push children
            // Order by distance to split plane or simple heuristic could improve perf
            if (stack_ptr + 2u <= 32u) {
                stack[stack_ptr] = node.left;
                stack_ptr++;
                stack[stack_ptr] = node.right;
                stack_ptr++;
            }
        }
    }

    hits[ray_idx] = closest_hit;
}

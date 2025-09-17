// -----------------------------------------------------------------------------
// Mesh BVH traversal helpers (adapted from pt_intersect_mesh.wgsl)
// -----------------------------------------------------------------------------
struct HitResult {
    t: f32,
    triangle_idx: u32,
    barycentric: vec2<f32>,
    normal: vec3<f32>,
    hit: bool,
}

fn ray_aabb_intersect(ray: Ray, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    var tmin = ray.tmin;
    var tmax = ray.tmax;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let inv_dir = 1.0 / ray.d[i];
        var t0 = (aabb_min[i] - ray.o[i]) * inv_dir;
        var t1 = (aabb_max[i] - ray.o[i]) * inv_dir;
        if (inv_dir < 0.0) { let tmp = t0; t0 = t1; t1 = tmp; }
        tmin = max(tmin, t0);
        tmax = min(tmax, t1);
        if (tmin > tmax) { return false; }
    }
    return true;
}

fn ray_triangle_intersect(ray: Ray, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = ray.tmax;
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = cross(ray.d, e2);
    let a = dot(e1, h);
    let eps = 1e-7;
    if (abs(a) < eps) { return result; }
    let f = 1.0 / a;
    let s = ray.o - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) { return result; }
    let q = cross(s, e1);
    let v = f * dot(ray.d, q);
    if (v < 0.0 || u + v > 1.0) { return result; }
    let t = f * dot(e2, q);
    if (t > ray.tmin && t < ray.tmax) {
        result.hit = true;
        result.t = t;
        result.triangle_idx = 0u; // set by caller
        result.barycentric = vec2<f32>(u, v);
        result.normal = normalize(cross(e1, e2));
    }
    return result;
}

fn bvh_intersect_mesh(ray: Ray) -> HitResult {
    var closest: HitResult;
    closest.hit = false;
    closest.t = ray.tmax;

    let node_count = arrayLength(&mesh_bvh_nodes);
    if (node_count == 0u) { return closest; }

    // Traversal stack
    var stack: array<u32, 64u>;
    var sp = 0u;
    stack[sp] = 0u; sp = sp + 1u;
    var current_ray = ray;

    while (sp > 0u) {
        sp = sp - 1u;
        let node_idx = stack[sp];
        if (node_idx >= node_count) { continue; }
        let node = mesh_bvh_nodes[node_idx];
        if (!ray_aabb_intersect(current_ray, node.aabb_min, node.aabb_max)) { continue; }
        if ((node.flags & 1u) != 0u) {
            // Leaf: triangles
            let first_tri = node.left;
            let tri_count = node.right;
            let idx_len = arrayLength(&mesh_indices);
            for (var i = 0u; i < tri_count; i = i + 1u) {
                let tri_idx = first_tri + i;
                if (tri_idx * 3u + 2u >= idx_len) { continue; }
                let i0 = mesh_indices[tri_idx * 3u + 0u];
                let i1 = mesh_indices[tri_idx * 3u + 1u];
                let i2 = mesh_indices[tri_idx * 3u + 2u];
                let vcount = arrayLength(&mesh_vertices);
                if (max(max(i0, i1), i2) >= vcount) { continue; }
                let v0 = mesh_vertices[i0].position;
                let v1 = mesh_vertices[i1].position;
                let v2 = mesh_vertices[i2].position;
                var hit = ray_triangle_intersect(current_ray, v0, v1, v2);
                if (hit.hit && hit.t < closest.t) {
                    hit.triangle_idx = tri_idx;
                    closest = hit;
                    current_ray.tmax = hit.t; // tighten
                }
            }
        } else {
            // Internal: push children
            let l = node.left;
            let r = node.right;
            if (r < node_count && sp < 64u) { stack[sp] = r; sp = sp + 1u; }
            if (l < node_count && sp < 64u) { stack[sp] = l; sp = sp + 1u; }
        }
    }
    return closest;
}
// src/shaders/pt_intersect.wgsl
// Wavefront Path Tracer: Intersection Stage
// This file exists to intersect rays with scene acceleration structures and write hit information to the hit queue.
// RELEVANT FILES:src/path_tracing/wavefront/mod.rs,src/shaders/pt_raygen.wgsl,src/shaders/pt_shade.wgsl

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

// Bind Group 1: Scene (readonly storage: materials, mesh data, BVH)
struct Sphere {
    center: vec3<f32>,
    radius: f32,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ior: f32,
    emissive: vec3<f32>,
    _pad0: f32,
}

struct Vertex {
    position: vec3<f32>,
    _pad: f32,
}

struct BvhNode {
    aabb_min: vec3<f32>,
    left: u32,
    aabb_max: vec3<f32>,
    right: u32,
    flags: u32,
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

struct Hit {
    p: vec3<f32>,           // hit position
    t: f32,                 // ray parameter
    n: vec3<f32>,           // surface normal
    wo: vec3<f32>,          // outgoing (to camera) direction
    _pad_wo: f32,           // alignment
    mat: u32,               // material index
    throughput: vec3<f32>,  // inherited throughput
    pdf: f32,               // inherited pdf
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
@group(1) @binding(0) var<storage, read> scene_spheres: array<Sphere>;
@group(1) @binding(1) var<storage, read> mesh_vertices: array<Vertex>;
@group(1) @binding(2) var<storage, read> mesh_indices: array<u32>;
@group(1) @binding(3) var<storage, read> mesh_bvh_nodes: array<BvhNode>;
@group(2) @binding(0) var<storage, read_write> ray_queue_header: QueueHeader;
@group(2) @binding(1) var<storage, read_write> ray_queue: array<Ray>;
@group(2) @binding(2) var<storage, read_write> hit_queue_header: QueueHeader;
@group(2) @binding(3) var<storage, read_write> hit_queue: array<Hit>;
@group(2) @binding(4) var<storage, read_write> miss_queue_header: QueueHeader;
@group(2) @binding(5) var<storage, read_write> miss_queue: array<Ray>;

// Ray-sphere intersection
fn ray_sphere(ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
    let oc = ro - c;
    let b = dot(oc, rd);
    let cterm = dot(oc, oc) - r * r;
    let disc = b * b - cterm;
    if (disc <= 0.0) { return 1e30; }
    let s = sqrt(disc);
    let t0 = -b - s;
    let t1 = -b + s;
    if (t0 > 1e-3) { return t0; }
    if (t1 > 1e-3) { return t1; }
    return 1e30;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Persistent threads loop: keep pulling rays until queue is empty
    loop {
        let ray_idx = atomicAdd(&ray_queue_header.out_count, 1u);
        if ray_idx >= ray_queue_header.in_count {
            break; // No more rays to process
        }
        
        if ray_idx >= ray_queue_header.capacity {
            break; // Safety check
        }
        
        let ray = ray_queue[ray_idx];
        
        // Test intersection with spheres (analytic)
        var t_best = 1e30;
        var hit_normal = vec3<f32>(0.0, 1.0, 0.0);
        var material_idx = 0u;
        let sphere_count = arrayLength(&scene_spheres);
        for (var i: u32 = 0u; i < sphere_count; i = i + 1u) {
            let s = scene_spheres[i];
            let t = ray_sphere(ray.o, ray.d, s.center, s.radius);
            if t >= ray.tmin && t < min(t_best, ray.tmax) {
                t_best = t;
                let hp = ray.o + ray.d * t;
                hit_normal = normalize(hp - s.center);
                material_idx = i;
            }
        }

        // Test intersection with mesh BVH
        let mesh_hit = bvh_intersect_mesh(ray);
        if (mesh_hit.hit && mesh_hit.t < t_best) {
            t_best = mesh_hit.t;
            hit_normal = mesh_hit.normal;
            material_idx = 0u; // TODO: per-triangle material indices (future)
        }
        
        if t_best < 1e20 {
            // Hit: create hit record
            var hit: Hit;
            hit.p = ray.o + ray.d * t_best;
            hit.t = t_best;
            hit.n = hit_normal;
            hit.wo = normalize(-ray.d);
            hit._pad_wo = 0.0;
            hit.mat = material_idx;
            hit.throughput = ray.throughput;
            hit.pdf = ray.pdf;
            hit.pixel = ray.pixel;
            hit.depth = ray.depth;
            hit.rng_hi = ray.rng_hi;
            hit.rng_lo = ray.rng_lo;
            
            // Push to hit queue
            let hit_queue_idx = atomicAdd(&hit_queue_header.in_count, 1u);
            if hit_queue_idx < hit_queue_header.capacity {
                hit_queue[hit_queue_idx] = hit;
            }
        } else {
            // Miss: push to miss queue for background evaluation
            let miss_queue_idx = atomicAdd(&miss_queue_header.in_count, 1u);
            if miss_queue_idx < miss_queue_header.capacity {
                miss_queue[miss_queue_idx] = ray;
            }
        }
    }
}
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

// Hair width multiplier (host can approximate by scaling CPU-provided radii)
const HAIR_RADIUS_SCALE: f32 = 1.0;
// Build a simple tangent from a normal
fn tangent_from_normal(n: vec3<f32>) -> vec3<f32> {
    let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
    return normalize(cross(a, n));
}

// Ray vs finite cylinder aligned with (p0->p1) with constant radius r
fn ray_cylinder_segment(ray: Ray, p0: vec3<f32>, p1: vec3<f32>, r: f32) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = ray.tmax;
    let axis = p1 - p0;
    let L = length(axis);
    if (L < 1e-6 || r <= 0.0) { return result; }
    let n = axis / L;
    let w0 = ray.o - p0;
    let d_par = dot(ray.d, n);
    let d_perp = ray.d - d_par * n;
    let w_perp = w0 - dot(w0, n) * n;
    let A = dot(d_perp, d_perp);
    let B = 2.0 * dot(d_perp, w_perp);
    let C = dot(w_perp, w_perp) - r * r;
    if (A < 1e-12) { return result; }
    let disc = B * B - 4.0 * A * C;
    if (disc < 0.0) { return result; }
    let sdisc = sqrt(max(disc, 0.0));
    var t0 = (-B - sdisc) / (2.0 * A);
    var t1 = (-B + sdisc) / (2.0 * A);
    var thit = 1e30;
    if (t0 > ray.tmin && t0 < ray.tmax) { thit = t0; }
    if (t1 > ray.tmin && t1 < thit) { thit = t1; }
    if (thit >= 1e20) { return result; }
    // Check axial bounds
    let s = dot(w0 + thit * ray.d, n);
    if (s < 0.0 || s > L) { return result; }
    result.hit = true;
    result.t = thit;
    // Cylinder normal is perpendicular component
    let n_world = normalize(w_perp + thit * d_perp);
    result.normal = n_world;
    result.barycentric = vec2<f32>(0.0);
    result.triangle_idx = 0u;
    return result;
}

// Hair curve segment (world-space)
struct HairSegment {
    p0: vec3<f32>,
    r0: f32,
    p1: vec3<f32>,
    r1: f32,
    material_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Instance data for TLAS-style instancing (A22)
struct Instance {
    // Object-to-world and world-to-object transforms
    // Use full matrices for clarity; could be packed as 3x4 for efficiency
    object_to_world: mat4x4<f32>,
    world_to_object: mat4x4<f32>,
    // Extended fields
    blas_index: u32,     // currently unused (single BLAS MVP)
    material_id: u32,    // per-instance material selection (indexes scene_spheres)
    _pad: vec2<u32>,     // 16-byte alignment
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

// Watertight ray/triangle intersection (Kensler/Walther style)
fn ray_triangle_intersect(ray: Ray, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = ray.tmax;

    // Translate vertices relative to ray origin
    let A = v0 - ray.o;
    let B = v1 - ray.o;
    let C = v2 - ray.o;

    // Permute ray direction to align kz with the largest magnitude component
    let ad = abs(ray.d);
    var kz: u32 = 2u;
    var kx: u32 = 0u;
    var ky: u32 = 1u;
    if (ad.x > ad.y && ad.x > ad.z) { kz = 0u; kx = 1u; ky = 2u; }
    else if (ad.y > ad.z) { kz = 1u; kx = 2u; ky = 0u; }
    // Shear constants
    let Sz = 1.0 / ray.d[kz];
    let Sx = ray.d[kx] * Sz;
    let Sy = ray.d[ky] * Sz;

    // Apply shear to vertices so ray direction becomes (0,0,1)
    let ax = A[kx] - Sx * A[kz];
    let ay = A[ky] - Sy * A[kz];
    let bx = B[kx] - Sx * B[kz];
    let by = B[ky] - Sy * B[kz];
    let cx = C[kx] - Sx * C[kz];
    let cy = C[ky] - Sy * C[kz];
    let az = A[kz] * Sz;
    let bz = B[kz] * Sz;
    let cz = C[kz] * Sz;

    // 2D edge functions
    let U = (bx * cy) - (by * cx);
    let V = (cx * ay) - (cy * ax);
    let W = (ax * by) - (ay * bx);

    // Reject if not on the same side of all edges
    if ((U < 0.0 || V < 0.0 || W < 0.0) && (U > 0.0 || V > 0.0 || W > 0.0)) {
        return result;
    }

    // Compute determinant and t scaled
    let det = U + V + W;
    if (det == 0.0) { return result; }

    // Compute hit distance in the same units as det (note: az,bz,cz already scaled)
    let T = U * az + V * bz + W * cz;
    let t = T / det;

    if (t > ray.tmin && t < ray.tmax) {
        result.hit = true;
        result.t = t;
        // Barycentrics
        let inv_det = 1.0 / det;
        let u = U * inv_det;
        let v = V * inv_det;
        result.barycentric = vec2<f32>(u, v);
        // Geometric normal
        let e1 = v1 - v0;
        let e2 = v2 - v0;
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
    ax: f32, // anisotropic alpha_x
    ay: f32, // anisotropic alpha_y
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
// BVH node layout:
//  - 'flags & 1u' indicates a leaf node; for leaves, 'left' is the first triangle index
//    and 'right' is the triangle count. For internal nodes, 'left' and 'right' are child indices.
//  - AABBs are in world/object space depending on traversal context; desc-based traversal applies offsets.

// BLAS descriptor for atlas selection (one entry per BLAS)
struct BlasDesc {
    node_offset: u32,
    node_count: u32,
    tri_offset: u32,
    tri_count: u32,
    vtx_offset: u32,
    vtx_count: u32,
    _pad0: u32,
    _pad1: u32,
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
    tangent: vec3<f32>,     // strand or surface tangent
    flags: u32,             // bit0 = is_hair
}

struct QueueHeader {
    in_count: atomic<u32>,  // number of items pushed
    out_count: atomic<u32>, // number of items popped
    capacity: u32,          // maximum capacity
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> scene_spheres: array<Sphere>;
@group(1) @binding(1) var<storage, read> mesh_vertices: array<Vertex>;
@group(1) @binding(2) var<storage, read> mesh_indices: array<u32>;
@group(1) @binding(3) var<storage, read> mesh_bvh_nodes: array<BvhNode>;
@group(1) @binding(14) var<storage, read> instances: array<Instance>;
@group(1) @binding(15) var<storage, read> blas_descs: array<BlasDesc>;
@group(1) @binding(20) var<storage, read> hair_segments: array<HairSegment>;
@group(2) @binding(0) var<storage, read_write> ray_queue_header: QueueHeader;
@group(2) @binding(1) var<storage, read_write> ray_queue: array<Ray>;
@group(2) @binding(2) var<storage, read_write> hit_queue_header: QueueHeader;
@group(2) @binding(3) var<storage, read_write> hit_queue: array<Hit>;
@group(2) @binding(4) var<storage, read_write> miss_queue_header: QueueHeader;
@group(2) @binding(5) var<storage, read_write> miss_queue: array<Ray>;

// --- Transform helpers for instancing ---
fn transform_point(m: mat4x4<f32>, p: vec3<f32>) -> vec3<f32> {
    return (m * vec4<f32>(p, 1.0)).xyz;
}

fn transform_vector(m: mat4x4<f32>, v: vec3<f32>) -> vec3<f32> {
    return (m * vec4<f32>(v, 0.0)).xyz;
}

fn transform_ray_to_object(ray: Ray, world_to_object: mat4x4<f32>) -> Ray {
    var r = ray;
    r.o = transform_point(world_to_object, ray.o);
    r.d = normalize(transform_vector(world_to_object, ray.d));
    // tmin/tmax and payload fields preserved
    return r;
}

fn transform_normal_to_world(n_obj: vec3<f32>, world_to_object: mat4x4<f32>) -> vec3<f32> {
    // Use transpose(world_to_object) which equals inverse(object_to_world) transposed
    let n4 = vec4<f32>(n_obj, 0.0);
    let n_world = (transpose(world_to_object) * n4).xyz;
    return normalize(n_world);
}

fn bvh_intersect_mesh_desc(ray: Ray, desc_idx: u32) -> HitResult {
    var closest: HitResult;
    closest.hit = false;
    closest.t = ray.tmax;

    let desc_count = arrayLength(&blas_descs);
    if (desc_idx >= desc_count) { return closest; }
    let desc = blas_descs[desc_idx];

    if (desc.node_count == 0u) { return closest; }

    var stack: array<u32, 64u>;
    var sp = 0u;
    stack[sp] = desc.node_offset; sp = sp + 1u;
    var current_ray = ray;

    while (sp > 0u) {
        sp = sp - 1u;
        let node_idx = stack[sp];
        if (node_idx < desc.node_offset || node_idx >= desc.node_offset + desc.node_count) { continue; }
        let node = mesh_bvh_nodes[node_idx];
        if (!ray_aabb_intersect(current_ray, node.aabb_min, node.aabb_max)) { continue; }
        if ((node.flags & 1u) != 0u) {
            // Leaf: triangles in [desc.tri_offset ..)
            let first_tri = desc.tri_offset + node.left;
            let tri_count = node.right;
            for (var i = 0u; i < tri_count; i = i + 1u) {
                let tri_idx = first_tri + i;
                if (tri_idx >= desc.tri_offset + desc.tri_count) { continue; }
                let base = tri_idx * 3u;
                let i0 = mesh_indices[base + 0u] + desc.vtx_offset;
                let i1 = mesh_indices[base + 1u] + desc.vtx_offset;
                let i2 = mesh_indices[base + 2u] + desc.vtx_offset;
                let vcount = arrayLength(&mesh_vertices);
                if (max(max(i0, i1), i2) >= vcount) { continue; }
                let v0 = mesh_vertices[i0].position;
                let v1 = mesh_vertices[i1].position;
                let v2 = mesh_vertices[i2].position;
                var hit = ray_triangle_intersect(current_ray, v0, v1, v2);
                if (hit.hit && hit.t < closest.t) {
                    hit.triangle_idx = tri_idx;
                    closest = hit;
                    current_ray.tmax = hit.t;
                }
            }
        } else {
            // Internal: push children (with node offset)
            let l = node.left + desc.node_offset;
            let r = node.right + desc.node_offset;
            if (sp < 64u) { stack[sp] = r; sp = sp + 1u; }
            if (sp < 64u) { stack[sp] = l; sp = sp + 1u; }
        }
    }
    return closest;
}
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

        // Track hair hit state
        var is_hair: bool = false;
        var hair_tangent: vec3<f32> = vec3<f32>(0.0);

        // If no instances are present, test non-instanced mesh BVH
        let inst_count = arrayLength(&instances);
        if (inst_count == 0u) {
            let mesh_hit = bvh_intersect_mesh(ray);
            if (mesh_hit.hit && mesh_hit.t < t_best) {
                t_best = mesh_hit.t;
                hit_normal = mesh_hit.normal;
                material_idx = 0u; // TODO: per-triangle material indices (future)
            }
        }

        // Test instanced meshes (per-instance BLAS selection)
        if (inst_count > 0u) {
            for (var ii: u32 = 0u; ii < inst_count; ii = ii + 1u) {
                let inst = instances[ii];
                let r_obj = transform_ray_to_object(ray, inst.world_to_object);
                let h_obj = bvh_intersect_mesh_desc(r_obj, inst.blas_index);
                if (h_obj.hit && h_obj.t < t_best) {
                    t_best = h_obj.t;
                    hit_normal = transform_normal_to_world(h_obj.normal, inst.world_to_object);
                    // Per-instance material selection (clamped to available materials)
                    let mat_count = arrayLength(&scene_spheres);
                    if (mat_count > 0u) {
                        let mid = min(inst.material_id, mat_count - 1u);
                        material_idx = mid;
                    } else {
                        material_idx = 0u;
                    }
                }
            }
        }
        
        // Test hair segments (world-space cylinders)
        let hcount = arrayLength(&hair_segments);
        if (hcount > 0u) {
            for (var hi: u32 = 0u; hi < hcount; hi = hi + 1u) {
                let seg = hair_segments[hi];
                let axis = seg.p1 - seg.p0;
                let r = max(0.0, 0.5 * (seg.r0 + seg.r1) * HAIR_RADIUS_SCALE);
                var hseg = ray_cylinder_segment(ray, seg.p0, seg.p1, r);
                if (hseg.hit && hseg.t < t_best) {
                    t_best = hseg.t;
                    hit_normal = hseg.normal;
                    // Clamp material id
                    let mat_count = arrayLength(&scene_spheres);
                    if (mat_count > 0u) {
                        material_idx = min(seg.material_id, mat_count - 1u);
                    } else {
                        material_idx = 0u;
                    }
                    is_hair = true;
                    hair_tangent = normalize(axis);
                }
            }
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
            if (is_hair) {
                hit.tangent = hair_tangent;
                hit.flags = 1u;
            } else {
                hit.tangent = tangent_from_normal(hit_normal);
                hit.flags = 0u;
            }
            
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
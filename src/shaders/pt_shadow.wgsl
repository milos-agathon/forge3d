// --- Transform helpers ---
fn transform_pos(m: mat4x4<f32>, p: vec3<f32>) -> vec3<f32> { return (m * vec4<f32>(p, 1.0)).xyz; }

fn mesh_any_hit_desc(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32, desc_idx: u32) -> bool {
    let desc_count = arrayLength(&blas_descs);
    if (desc_idx >= desc_count) { return false; }
    let desc = blas_descs[desc_idx];
    if (desc.node_count == 0u) { return false; }
    var stack: array<u32, 64u>;
    var sp = 0u; stack[sp] = desc.node_offset; sp = sp + 1u;
    while (sp > 0u) {
        sp = sp - 1u;
        let idx = stack[sp];
        if (idx < desc.node_offset || idx >= desc.node_offset + desc.node_count) { continue; }
        let node = mesh_bvh_nodes[idx];
        if (!ray_aabb_intersect(ro, rd, tmin, tmax, node.aabb_min, node.aabb_max)) { continue; }
        if ((node.flags & 1u) != 0u) {
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
                // Moller-Trumbore any-hit
                let e1 = v1 - v0;
                let e2 = v2 - v0;
                let h = cross(rd, e2);
                let a = dot(e1, h);
                let eps = 1e-7;
                if (abs(a) < eps) { continue; }
                let f = 1.0 / a;
                let s = ro - v0;
                let u = f * dot(s, h);
                if (u < 0.0 || u > 1.0) { continue; }
                let q = cross(s, e1);
                let v = f * dot(rd, q);
                if (v < 0.0 || u + v > 1.0) { continue; }
                let t = f * dot(e2, q);
                if (t > tmin && t < tmax) { return true; }
            }
        } else {
            let l = node.left + desc.node_offset;
            let r = node.right + desc.node_offset;
            if (r < desc.node_offset + desc.node_count && sp < 64u) { stack[sp] = r; sp = sp + 1u; }
            if (l < desc.node_offset + desc.node_count && sp < 64u) { stack[sp] = l; sp = sp + 1u; }
        }
    }
    return false;
}
fn transform_dir(m: mat4x4<f32>, v: vec3<f32>) -> vec3<f32> { return (m * vec4<f32>(v, 0.0)).xyz; }
// src/shaders/pt_shadow.wgsl
// Wavefront Path Tracer: Shadow (visibility) Stage
// Pops shadow rays, tests visibility against spheres and mesh BVH, and accumulates contribution if visible.
// RELEVANT FILES: src/path_tracing/wavefront/pipeline.rs, src/shaders/pt_intersect.wgsl, src/shaders/pt_shade.wgsl

// Bind Group 0: Uniforms
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

// Instance data (matches pt_intersect.wgsl)
struct Instance {
    object_to_world: mat4x4<f32>,
    world_to_object: mat4x4<f32>,
    blas_index: u32,
    material_id: u32,
    _pad: vec2<u32>,
}

// Bind Group 1: Scene (spheres + mesh)
struct Sphere {
    center: vec3<f32>,
    radius: f32,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ior: f32,
    emissive: vec3<f32>,
    ax: f32,
    ay: f32,
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

// Bind Group 2: Queues (shadow)
struct QueueHeader {
    in_count: atomic<u32>,
    out_count: atomic<u32>,
    capacity: u32,
    _pad: u32,
}

struct ShadowRay {
    o: vec3<f32>,
    tmin: f32,
    d: vec3<f32>,
    tmax: f32,
    contrib: vec3<f32>,
    _pad0: f32,
    pixel: u32,
    _pad1: vec3<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> scene_spheres: array<Sphere>;
@group(1) @binding(1) var<storage, read> mesh_vertices: array<Vertex>;
@group(1) @binding(2) var<storage, read> mesh_indices: array<u32>;
@group(1) @binding(3) var<storage, read> mesh_bvh_nodes: array<BvhNode>;
@group(1) @binding(14) var<storage, read> instances: array<Instance>;
// BLAS descriptor table (matches pt_intersect)
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
@group(1) @binding(15) var<storage, read> blas_descs: array<BlasDesc>;
@group(2) @binding(0) var<storage, read_write> shadow_queue_header: QueueHeader;
@group(2) @binding(1) var<storage, read_write> shadow_queue: array<ShadowRay>;
@group(3) @binding(0) var<storage, read_write> accum_hdr: array<vec4<f32>>;

// Ray-sphere intersection
fn ray_sphere(ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, r: f32, tmin: f32, tmax: f32) -> bool {
    let oc = ro - c;
    let b = dot(oc, rd);
    let cterm = dot(oc, oc) - r * r;
    let disc = b * b - cterm;
    if (disc <= 0.0) { return false; }
    let s = sqrt(disc);
    let t0 = -b - s;
    let t1 = -b + s;
    let t = select(t1, t0, t0 > tmin && t0 < tmax);
    // If t0 not in range, test t1
    let hit0 = t0 > tmin && t0 < tmax;
    let hit1 = t1 > tmin && t1 < tmax;
    return hit0 || hit1;
}

// Ray-AABB test
fn ray_aabb_intersect(ro: vec3<f32>, rd: vec3<f32>, tmin_in: f32, tmax_in: f32, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    var tmin = tmin_in;
    var tmax = tmax_in;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let inv_dir = 1.0 / rd[i];
        var t0 = (aabb_min[i] - ro[i]) * inv_dir;
        var t1 = (aabb_max[i] - ro[i]) * inv_dir;
        if (inv_dir < 0.0) { let tmp = t0; t0 = t1; t1 = tmp; }
        tmin = max(tmin, t0);
        tmax = min(tmax, t1);
        if (tmin > tmax) { return false; }
    }
    return true;
}

// Any-hit mesh BVH traversal
fn mesh_any_hit(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32) -> bool {
    let node_count = arrayLength(&mesh_bvh_nodes);
    if (node_count == 0u) { return false; }
    var stack: array<u32, 64u>;
    var sp = 0u; stack[sp] = 0u; sp = sp + 1u;
    while (sp > 0u) {
        sp = sp - 1u;
        let idx = stack[sp];
        if (idx >= node_count) { continue; }
        let node = mesh_bvh_nodes[idx];
        if (!ray_aabb_intersect(ro, rd, tmin, tmax, node.aabb_min, node.aabb_max)) { continue; }
        if ((node.flags & 1u) != 0u) {
            // Leaf
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
                // Moller-Trumbore any-hit
                let e1 = v1 - v0;
                let e2 = v2 - v0;
                let h = cross(rd, e2);
                let a = dot(e1, h);
                let eps = 1e-7;
                if (abs(a) < eps) { continue; }
                let f = 1.0 / a;
                let s = ro - v0;
                let u = f * dot(s, h);
                if (u < 0.0 || u > 1.0) { continue; }
                let q = cross(s, e1);
                let v = f * dot(rd, q);
                if (v < 0.0 || u + v > 1.0) { continue; }
                let t = f * dot(e2, q);
                if (t > tmin && t < tmax) { return true; }
            }
        } else {
            // Internal
            let l = node.left;
            let r = node.right;
            if (r < node_count && sp < 64u) { stack[sp] = r; sp = sp + 1u; }
            if (l < node_count && sp < 64u) { stack[sp] = l; sp = sp + 1u; }
        }
    }
    return false;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Persistent threads loop: consume shadow rays
    loop {
        let idx = atomicAdd(&shadow_queue_header.out_count, 1u);
        if (idx >= shadow_queue_header.in_count) { break; }
        if (idx >= shadow_queue_header.capacity) { break; }
        let sr = shadow_queue[idx];
        let ro = sr.o;
        let rd = sr.d;
        let tmin = sr.tmin;
        let tmax = sr.tmax;

        // Test spheres
        var occluded = false;
        let sphere_count = arrayLength(&scene_spheres);
        for (var i: u32 = 0u; i < sphere_count; i = i + 1u) {
            let s = scene_spheres[i];
            if (ray_sphere(ro, rd, s.center, s.radius, tmin, tmax)) {
                occluded = true; break;
            }
        }

        // If no instances are present, test non-instanced mesh BVH
        let inst_count = arrayLength(&instances);
        if (!occluded && inst_count == 0u) {
            if (mesh_any_hit(ro, rd, tmin, tmax)) { occluded = true; }
        }

        // Test instanced meshes: transform ray into each instance's object space, using per-instance BLAS
        if (!occluded && inst_count > 0u) {
            let count = arrayLength(&instances);
            for (var ii: u32 = 0u; ii < count; ii = ii + 1u) {
                let inst = instances[ii];
                let ro_obj = transform_pos(inst.world_to_object, ro);
                let rd_obj = normalize(transform_dir(inst.world_to_object, rd));
                if (mesh_any_hit_desc(ro_obj, rd_obj, tmin, tmax, inst.blas_index)) { occluded = true; break; }
            }
        }

        if (!occluded) {
            // Accumulate contribution
            let p = sr.pixel;
            accum_hdr[p] = accum_hdr[p] + vec4<f32>(sr.contrib, 0.0);
        }
    }
}

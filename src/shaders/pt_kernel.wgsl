// src/shaders/pt_kernel.wgsl
// WGSL compute kernel for A3: GPU path tracer with sphere and triangle mesh intersection support.
// This file implements the GPU path tracer with BVH traversal for triangle meshes and sphere primitives.
// RELEVANT FILES:src/path_tracing/mod.rs,src/path_tracing/compute.rs,src/path_tracing/mesh.rs,python/forge3d/path_tracing.py,tests/test_path_tracing_gpu.py

struct Uniforms {
  width: u32,
  height: u32,
  frame_index: u32,
  _pad: u32,
  cam_origin: vec3<f32>,
  cam_fov_y: f32,
  cam_right: vec3<f32>,
  cam_aspect: f32,
  cam_up: vec3<f32>,
  cam_exposure: f32,
  cam_forward: vec3<f32>,
  seed_hi: u32,
  seed_lo: u32,
};

struct Sphere { center: vec3<f32>, radius: f32, albedo: vec3<f32>, _pad0: f32 };

// Mesh structures from pt_intersect_mesh.wgsl
struct BvhNode {
    aabb_min: vec3<f32>,
    left: u32,
    aabb_max: vec3<f32>,
    right: u32,
    flags: u32,
    _pad: u32,
}

struct Vertex {
    position: vec3<f32>,
    _pad: f32,
}

struct Ray {
    origin: vec3<f32>,
    tmin: f32,
    direction: vec3<f32>,
    tmax: f32,
}

struct HitResult {
    t: f32,
    triangle_idx: u32,
    barycentric: vec2<f32>,
    normal: vec3<f32>,
    hit: bool,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> scene_spheres: array<Sphere>;
@group(1) @binding(1) var<storage, read> mesh_vertices: array<Vertex>;
@group(1) @binding(2) var<storage, read> mesh_indices: array<u32>;
@group(1) @binding(3) var<storage, read> mesh_bvh_nodes: array<BvhNode>;
@group(2) @binding(0) var<storage, read_write> accum_hdr: array<vec4<f32>>;
@group(3) @binding(0) var out_tex: texture_storage_2d<rgba16float, write>;

fn xorshift32(state: ptr<function, u32>) -> f32 {
  var x = *state;
  x ^= (x << 13u);
  x ^= (x >> 17u);
  x ^= (x << 5u);
  *state = x;
  return f32(x) / 4294967296.0;
}

fn tent_filter(u: f32) -> f32 {
  let t = 2.0 * u - 1.0;
  return select(1.0 + t, 1.0 - t, t < 0.0);
}

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

// Mesh intersection functions from pt_intersect_mesh.wgsl

fn ray_aabb_intersect(ray: Ray, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
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

fn ray_triangle_intersect(
    ray: Ray,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = ray.tmax;

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
        result.hit = true;
        result.t = t;
        result.barycentric = vec2<f32>(u, v);
        result.normal = normal;
    }

    return result;
}

const MAX_STACK_DEPTH: u32 = 64u;

fn bvh_intersect(ray: Ray) -> HitResult {
    var closest_hit: HitResult;
    closest_hit.hit = false;
    closest_hit.t = ray.tmax;

    let node_count = arrayLength(&mesh_bvh_nodes);
    if (node_count == 0u) {
        return closest_hit;
    }

    var stack: array<u32, MAX_STACK_DEPTH>;
    var stack_ptr = 0u;
    stack[0] = 0u;
    stack_ptr = 1u;

    var current_ray = ray;

    while (stack_ptr > 0u) {
        stack_ptr = stack_ptr - 1u;
        let node_idx = stack[stack_ptr];

        if (node_idx >= node_count) {
            continue;
        }

        let node = mesh_bvh_nodes[node_idx];

        if (!ray_aabb_intersect(current_ray, node.aabb_min, node.aabb_max)) {
            continue;
        }

        if ((node.flags & 1u) != 0u) {
            // Leaf node
            let first_tri = node.left;
            let tri_count = node.right;

            for (var i = 0u; i < tri_count; i = i + 1u) {
                let tri_idx = first_tri + i;
                let indices_length = arrayLength(&mesh_indices);
                if (tri_idx * 3u + 2u >= indices_length) {
                    continue;
                }

                let i0 = mesh_indices[tri_idx * 3u];
                let i1 = mesh_indices[tri_idx * 3u + 1u];
                let i2 = mesh_indices[tri_idx * 3u + 2u];

                let vertex_count = arrayLength(&mesh_vertices);
                if (max(max(i0, i1), i2) >= vertex_count) {
                    continue;
                }

                let v0 = mesh_vertices[i0].position;
                let v1 = mesh_vertices[i1].position;
                let v2 = mesh_vertices[i2].position;

                var hit = ray_triangle_intersect(current_ray, v0, v1, v2);

                if (hit.hit && hit.t < closest_hit.t) {
                    closest_hit = hit;
                    closest_hit.triangle_idx = tri_idx;
                    current_ray.tmax = hit.t;
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

    return closest_hit;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = uniforms.width;
  let H = uniforms.height;
  if (gid.x >= W || gid.y >= H) { return; }
  let px = f32(gid.x);
  let py = f32(gid.y);

  // Seed per-pixel RNG from seed_hi/lo + pixel index + frame_index
  var st: u32 = uniforms.seed_hi ^ (gid.x * 1664525u) ^ (gid.y * 1013904223u) ^ uniforms.frame_index;
  let jx = tent_filter(xorshift32(&st)) * 0.5;
  let jy = tent_filter(xorshift32(&st)) * 0.5;

  let ndc_x = ((px + 0.5 + jx) / f32(W)) * 2.0 - 1.0;
  let ndc_y = (1.0 - (py + 0.5 + jy) / f32(H)) * 2.0 - 1.0;
  let half_h = tan(0.5 * uniforms.cam_fov_y);
  let half_w = uniforms.cam_aspect * half_h;
  var rd = normalize(vec3<f32>(ndc_x * half_w, ndc_y * half_h, -1.0));
  // re-orient with camera basis (right, up, forward)
  rd = normalize(rd.x * uniforms.cam_right + rd.y * uniforms.cam_up + rd.z * (-uniforms.cam_forward));
  let ro = uniforms.cam_origin;

  // Create ray for intersection testing
  let ray = Ray(ro, 1e-3, rd, 1e30);

  // Test against spheres
  var t_best = 1e30;
  var hit_color = vec3<f32>(0.5); // Default mesh albedo
  var hit_n = vec3<f32>(0.0, 0.0, 1.0);
  var hit_material_type = 0u; // 0 = mesh, 1 = sphere

  let sphere_count = arrayLength(&scene_spheres);
  for (var i: u32 = 0u; i < sphere_count; i = i + 1u) {
    let s = scene_spheres[i];
    let t = ray_sphere(ro, rd, s.center, s.radius);
    if (t < t_best) {
      t_best = t;
      let p = ro + rd * t;
      hit_n = normalize(p - s.center);
      hit_color = s.albedo;
      hit_material_type = 1u;
    }
  }

  // Test against mesh (BVH traversal)
  let mesh_hit = bvh_intersect(ray);
  if (mesh_hit.hit && mesh_hit.t < t_best) {
    t_best = mesh_hit.t;
    hit_n = mesh_hit.normal;
    hit_color = vec3<f32>(0.7, 0.7, 0.8); // Default mesh color
    hit_material_type = 0u;
  }

  var rgb = vec3<f32>(0.0);
  if (t_best < 1e20) {
    let light_dir = normalize(vec3<f32>(0.5, 0.8, 0.2));
    let ndotl = max(0.0, dot(hit_n, light_dir));
    rgb = hit_color * ndotl;
  } else {
    // simple sky
    let tsky = 0.5 * (rd.y + 1.0);
    rgb = mix(vec3<f32>(0.6, 0.7, 0.9), vec3<f32>(0.1, 0.2, 0.5), tsky);
  }

  let idx = gid.y * W + gid.x;
  // HDR accumulate
  let prev = accum_hdr[idx];
  let acc = vec4<f32>(prev.xyz + rgb, 1.0);
  accum_hdr[idx] = acc;
  // Tonemap (Reinhard) with exposure and write
  let color = (acc.xyz) / (vec3<f32>(1.0) + acc.xyz);
  let exposed = color * uniforms.cam_exposure;
  textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(exposed, 1.0));
}

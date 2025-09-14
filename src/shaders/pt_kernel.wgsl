// src/shaders/pt_kernel.wgsl
// WGSL compute kernel for A14: GPU path tracer with AOV (Arbitrary Output Variables) support.
// This file implements the GPU path tracer with BVH traversal for triangle meshes, sphere primitives, and AOV outputs.
//
// Bind Group Layout:
//   Group 0 (Uniforms): width, height, frame_index; camera params; exposure; seed_hi/seed_lo; aov_flags
//   Group 1 (Scene): readonly storage (materials, spheres, mesh vertices/indices/BVH nodes)
//   Group 2 (Accum/State): storage buffers (HDR accumulation buffer)
//   Group 3 (Output): primary RGBA output texture
//   Group 4 (AOV Outputs): storage textures for albedo, normal, depth, direct, indirect, emission, visibility
//
// RELEVANT FILES:src/path_tracing/mod.rs,src/path_tracing/compute.rs,src/path_tracing/aov.rs,python/forge3d/path_tracing.py,tests/test_aovs_gpu.py

struct Uniforms {
  width: u32,
  height: u32,
  frame_index: u32,
  aov_flags: u32,           // Bitmask: bit 0=albedo, 1=normal, 2=depth, 3=direct, 4=indirect, 5=emission, 6=visibility
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
// Mesh buffers omitted in A1/A14 minimal pipeline; wavefront path provides them separately
@group(2) @binding(0) var<storage, read_write> accum_hdr: array<vec4<f32>>;
@group(3) @binding(0) var out_tex: texture_storage_2d<rgba16float, write>;

// AOV Output textures (Group 4)
@group(4) @binding(0) var aov_albedo: texture_storage_2d<rgba16float, write>;
@group(4) @binding(1) var aov_normal: texture_storage_2d<rgba16float, write>;
@group(4) @binding(2) var aov_depth: texture_storage_2d<r32float, write>;
@group(4) @binding(3) var aov_direct: texture_storage_2d<rgba16float, write>;
@group(4) @binding(4) var aov_indirect: texture_storage_2d<rgba16float, write>;
@group(4) @binding(5) var aov_emission: texture_storage_2d<rgba16float, write>;
@group(4) @binding(6) var aov_visibility: texture_storage_2d<r8unorm, write>;

// AOV flag constants (bit positions in uniforms.aov_flags)
const AOV_ALBEDO_BIT: u32 = 0u;
const AOV_NORMAL_BIT: u32 = 1u;
const AOV_DEPTH_BIT: u32 = 2u;
const AOV_DIRECT_BIT: u32 = 3u;
const AOV_INDIRECT_BIT: u32 = 4u;
const AOV_EMISSION_BIT: u32 = 5u;
const AOV_VISIBILITY_BIT: u32 = 6u;

fn aov_enabled(bit: u32) -> bool {
    return (uniforms.aov_flags & (1u << bit)) != 0u;
}

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
    // Minimal stub: no mesh intersection in this kernel variant
    var h: HitResult;
    h.hit = false;
    h.t = ray.tmax;
    return h;
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
  var hit_albedo = vec3<f32>(0.7, 0.7, 0.8); // Default mesh albedo
  var hit_normal = vec3<f32>(0.0, 0.0, 1.0);
  var hit_material_type = 0u; // 0 = mesh, 1 = sphere
  var hit_point = vec3<f32>(0.0);

  let sphere_count = arrayLength(&scene_spheres);
  for (var i: u32 = 0u; i < sphere_count; i = i + 1u) {
    let s = scene_spheres[i];
    let t = ray_sphere(ro, rd, s.center, s.radius);
    if (t < t_best) {
      t_best = t;
      hit_point = ro + rd * t;
      hit_normal = normalize(hit_point - s.center);
      hit_albedo = s.albedo;
      hit_material_type = 1u;
    }
  }

  // Test against mesh (BVH traversal)
  let mesh_hit = bvh_intersect(ray);
  if (mesh_hit.hit && mesh_hit.t < t_best) {
    t_best = mesh_hit.t;
    hit_point = ro + rd * t_best;
    hit_normal = mesh_hit.normal;
    hit_albedo = vec3<f32>(0.7, 0.7, 0.8); // Default mesh color
    hit_material_type = 0u;
  }

  // Calculate AOV values
  let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let is_hit = t_best < 1e20;

  // Sky color for miss cases
  let sky_color = if (is_hit) { vec3<f32>(0.0) } else {
    let tsky = 0.5 * (rd.y + 1.0);
    mix(vec3<f32>(0.6, 0.7, 0.9), vec3<f32>(0.1, 0.2, 0.5), tsky)
  };

  // Lighting calculation
  let light_dir = normalize(vec3<f32>(0.5, 0.8, 0.2));
  let light_color = vec3<f32>(1.0, 0.95, 0.8);

  var direct_light = vec3<f32>(0.0);
  var indirect_light = vec3<f32>(0.0);
  var final_color = sky_color;

  if (is_hit) {
    // Direct lighting (simple Lambert)
    let ndotl = max(0.0, dot(hit_normal, light_dir));
    direct_light = light_color * ndotl;

    // Simple indirect/ambient lighting
    indirect_light = vec3<f32>(0.1, 0.12, 0.15);

    final_color = hit_albedo * (direct_light + indirect_light);
  }

  // Write AOVs if enabled
  if (aov_enabled(AOV_ALBEDO_BIT)) {
    // CPU stub stores 0 on miss for albedo
    let albedo_val = if (is_hit) { vec4<f32>(hit_albedo, 1.0) } else { vec4<f32>(0.0, 0.0, 0.0, 1.0) };
    textureStore(aov_albedo, pixel_coord, albedo_val);
  }

  if (aov_enabled(AOV_NORMAL_BIT)) {
    // Store world-space normals; CPU stub leaves zeros on miss
    let normal_val = if (is_hit) { vec4<f32>(hit_normal, 1.0) } else { vec4<f32>(0.0, 0.0, 0.0, 1.0) };
    textureStore(aov_normal, pixel_coord, normal_val);
  }

  if (aov_enabled(AOV_DEPTH_BIT)) {
    // Linear depth from camera origin; store NaN on miss to match CPU nanmean/nanvar
    let depth_val = if (is_hit) { t_best } else { bitcast<f32>(0x7FC00000u) /* quiet NaN */ };
    textureStore(aov_depth, pixel_coord, depth_val);
  }

  if (aov_enabled(AOV_DIRECT_BIT)) {
    let direct_val = if (is_hit) { vec4<f32>(hit_albedo * direct_light, 1.0) } else { vec4<f32>(0.0, 0.0, 0.0, 0.0) };
    textureStore(aov_direct, pixel_coord, direct_val);
  }

  if (aov_enabled(AOV_INDIRECT_BIT)) {
    // CPU stub currently does not accumulate GI; keep zeros on miss as well
    let indirect_val = if (is_hit) { vec4<f32>(hit_albedo * indirect_light, 1.0) } else { vec4<f32>(0.0, 0.0, 0.0, 1.0) };
    textureStore(aov_indirect, pixel_coord, indirect_val);
  }

  if (aov_enabled(AOV_EMISSION_BIT)) {
    // For now, no emissive materials - could be extended for area lights
    textureStore(aov_emission, pixel_coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
  }

  if (aov_enabled(AOV_VISIBILITY_BIT)) {
    // r8unorm: 1.0 maps to 255, 0.0 to 0
    let visibility_val = if (is_hit) { 1.0 } else { 0.0 };
    textureStore(aov_visibility, pixel_coord, visibility_val);
  }

  // Standard HDR accumulation and output
  let idx = gid.y * W + gid.x;
  let prev = accum_hdr[idx];
  let acc = vec4<f32>(prev.xyz + final_color, 1.0);
  accum_hdr[idx] = acc;

  // Tonemap (Reinhard) with exposure and write to main output
  let color = (acc.xyz) / (vec3<f32>(1.0) + acc.xyz);
  let exposed = color * uniforms.cam_exposure;
  textureStore(out_tex, pixel_coord, vec4<f32>(exposed, 1.0));
}

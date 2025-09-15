// src/shaders/hybrid_kernel.wgsl
// Main compute kernel for hybrid path tracing combining SDF raymarching with BVH traversal
// Extends pt_kernel.wgsl functionality with SDF support

#include "hybrid_traversal.wgsl"

// Base uniforms (Group 0)
struct Uniforms {
  width: u32,
  height: u32,
  frame_index: u32,
  aov_flags: u32,
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

// Sphere primitive for legacy support
struct Sphere {
    center: vec3<f32>,
    radius: f32,
    albedo: vec3<f32>,
    _pad0: f32
}

// Bind groups
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> scene_spheres: array<Sphere>;
@group(1) @binding(1) var<storage, read> legacy_mesh_vertices: array<u32>; // Dummy
@group(1) @binding(2) var<storage, read> legacy_mesh_indices: array<u32>;  // Dummy
@group(1) @binding(3) var<storage, read> legacy_mesh_bvh: array<u32>;      // Dummy
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

// AOV flag constants
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

// Random number generation
fn xorshift32(state: ptr<function, u32>) -> f32 {
  var x = *state;
  x ^= (x << 13u);
  x ^= (x >> 17u);
  x ^= (x << 5u);
  *state = x;
  return f32(x) / 4294967296.0;
}

// Tent filter for antialiasing
fn tent_filter(u: f32) -> f32 {
  let t = 2.0 * u - 1.0;
  return select(1.0 + t, 1.0 - t, t < 0.0);
}

// Ray-sphere intersection (for legacy sphere support)
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

// Simple tonemap function
fn reinhard_tonemap(color: vec3<f32>, exposure: f32) -> vec3<f32> {
    let exposed = color * exposure;
    return exposed / (vec3<f32>(1.0) + exposed);
}

// Main compute shader
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = uniforms.width;
  let H = uniforms.height;
  if (gid.x >= W || gid.y >= H) { return; }
  let px = f32(gid.x);
  let py = f32(gid.y);

  // Seed per-pixel RNG
  var st: u32 = uniforms.seed_hi ^ (gid.x * 1664525u) ^ (gid.y * 1013904223u) ^ uniforms.frame_index;
  let jx = tent_filter(xorshift32(&st)) * 0.5;
  let jy = tent_filter(xorshift32(&st)) * 0.5;

  // Generate camera ray
  let ndc_x = ((px + 0.5 + jx) / f32(W)) * 2.0 - 1.0;
  let ndc_y = (1.0 - (py + 0.5 + jy) / f32(H)) * 2.0 - 1.0;
  let half_h = tan(0.5 * uniforms.cam_fov_y);
  let half_w = uniforms.cam_aspect * half_h;
  var rd = normalize(vec3<f32>(ndc_x * half_w, ndc_y * half_h, -1.0));
  rd = normalize(rd.x * uniforms.cam_right + rd.y * uniforms.cam_up + rd.z * (-uniforms.cam_forward));
  let ro = uniforms.cam_origin;

  // Create ray
  let camera_ray = Ray(ro, 1e-3, rd, 1e30);

  // Initialize results
  var t_best = 1e30;
  var hit_albedo = vec3<f32>(0.7, 0.7, 0.8);
  var hit_normal = vec3<f32>(0.0, 0.0, 1.0);
  var hit_material_type = 0u; // 0 = mesh, 1 = sphere, 2 = SDF
  var hit_point = vec3<f32>(0.0);

  // Test legacy spheres first
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

  // Test hybrid scene (SDF + mesh)
  let hybrid_hit = intersect_hybrid(camera_ray);
  if (hybrid_hit.hit != 0u && hybrid_hit.t < t_best) {
    t_best = hybrid_hit.t;
    hit_point = hybrid_hit.point;
    hit_normal = hybrid_hit.normal;
    hit_albedo = get_surface_properties(hybrid_hit);
    hit_material_type = hybrid_hit.hit_type + 2u; // 2 = mesh, 3 = SDF
  }

  // Calculate final color
  let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let is_hit = t_best < 1e20;

  // Sky color for miss cases
  let sky_color = if (is_hit) { vec3<f32>(0.0) } else {
    let tsky = 0.5 * (rd.y + 1.0);
    mix(vec3<f32>(0.6, 0.7, 0.9), vec3<f32>(0.1, 0.2, 0.5), tsky)
  };

  // Lighting
  let light_dir = normalize(vec3<f32>(0.5, 0.8, 0.2));
  let light_color = vec3<f32>(1.0, 0.95, 0.8);

  var direct_light = vec3<f32>(0.0);
  var indirect_light = vec3<f32>(0.0);
  var final_color = sky_color;

  if (is_hit) {
    // Direct lighting with shadow testing
    let ndotl = max(0.0, dot(hit_normal, light_dir));
    let shadow_ray = Ray(hit_point + hit_normal * 0.001, 0.001, light_dir, 1000.0);

    // Use soft shadows if SDF geometry is present
    let shadow_factor = soft_shadow_factor(shadow_ray, 1000.0, 4.0);
    direct_light = light_color * ndotl * shadow_factor;

    // Indirect/ambient lighting
    indirect_light = vec3<f32>(0.1, 0.12, 0.15);

    final_color = hit_albedo * (direct_light + indirect_light);
  }

  // Apply tonemapping
  final_color = reinhard_tonemap(final_color, uniforms.cam_exposure);

  // Write AOVs if enabled
  if (aov_enabled(AOV_ALBEDO_BIT)) {
    let albedo_val = if (is_hit) { vec4<f32>(hit_albedo, 1.0) } else { vec4<f32>(0.0, 0.0, 0.0, 1.0) };
    textureStore(aov_albedo, pixel_coord, albedo_val);
  }

  if (aov_enabled(AOV_NORMAL_BIT)) {
    let normal_val = if (is_hit) { vec4<f32>(hit_normal, 1.0) } else { vec4<f32>(0.0, 0.0, 0.0, 1.0) };
    textureStore(aov_normal, pixel_coord, normal_val);
  }

  if (aov_enabled(AOV_DEPTH_BIT)) {
    let depth_val = if (is_hit) { t_best } else { 0.0 / 0.0 }; // NaN for miss
    textureStore(aov_depth, pixel_coord, vec4<f32>(depth_val, 0.0, 0.0, 0.0));
  }

  if (aov_enabled(AOV_DIRECT_BIT)) {
    let direct_val = if (is_hit) { vec4<f32>(direct_light, 1.0) } else { vec4<f32>(0.0, 0.0, 0.0, 1.0) };
    textureStore(aov_direct, pixel_coord, direct_val);
  }

  if (aov_enabled(AOV_INDIRECT_BIT)) {
    let indirect_val = if (is_hit) { vec4<f32>(indirect_light, 1.0) } else { vec4<f32>(0.0, 0.0, 0.0, 1.0) };
    textureStore(aov_indirect, pixel_coord, indirect_val);
  }

  if (aov_enabled(AOV_EMISSION_BIT)) {
    // No emission in this simple implementation
    textureStore(aov_emission, pixel_coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
  }

  if (aov_enabled(AOV_VISIBILITY_BIT)) {
    let visibility_val = if (is_hit) { 1.0 } else { 0.0 };
    textureStore(aov_visibility, pixel_coord, vec4<f32>(visibility_val, 0.0, 0.0, 0.0));
  }

  // Write final output
  textureStore(out_tex, pixel_coord, vec4<f32>(final_color, 1.0));
}
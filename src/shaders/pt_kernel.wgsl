// src/shaders/pt_kernel.wgsl
// WGSL compute kernel for A1: minimal GPU path tracer with jittered primary rays and sphere hits.
// This file exists to implement the GPU MVP: RNG (xorshift64*), ray-sphere, HDR accumulation, and tonemap to storage texture.
// RELEVANT FILES:src/path_tracing/mod.rs,src/path_tracing/compute.rs,python/forge3d/path_tracing.py,tests/test_path_tracing_gpu.py

struct Uniforms {
  width: u32;
  height: u32;
  frame_index: u32;
  _pad: u32;
  cam_origin: vec3<f32>;
  cam_fov_y: f32;
  cam_right: vec3<f32>;
  cam_aspect: f32;
  cam_up: vec3<f32>;
  cam_exposure: f32;
  cam_forward: vec3<f32>;
  seed_hi: u32;
  seed_lo: u32;
};

struct Sphere { center: vec3<f32>; radius: f32; albedo: vec3<f32>; _pad0: f32; };

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> scene_spheres: array<Sphere>;
@group(2) @binding(0) var<storage, read_write> accum_hdr: array<vec4<f32>>;
@group(3) @binding(0) var out_tex: texture_storage_2d<rgba16float, write>;

fn xorshift64star(state: ptr<function, vec2<u32>>) -> f32 {
  // emulate 64-bit xorshift* using two u32 lanes: hi, lo
  var s = *state; // hi, lo
  // combine to 64
  var x_hi = s.x;
  var x_lo = s.y;
  // x ^= x >> 12
  var t_hi: u32 = x_hi >> 12u;
  var t_lo: u32 = (x_lo >> 12u) | (x_hi << (32u - 12u));
  x_hi = x_hi ^ t_hi;
  x_lo = x_lo ^ t_lo;
  // x ^= x << 25
  t_hi = (x_hi << 25u) | (x_lo >> (32u - 25u));
  t_lo = x_lo << 25u;
  x_hi = x_hi ^ t_hi;
  x_lo = x_lo ^ t_lo;
  // x ^= x >> 27
  t_hi = x_hi >> 27u;
  t_lo = (x_lo >> 27u) | (x_hi << (32u - 27u));
  x_hi = x_hi ^ t_hi;
  x_lo = x_lo ^ t_lo;
  // * 0x2545F4914F6CDD1D
  let C_hi: u32 = 0x2545F491u;
  let C_lo: u32 = 0x4F6CDD1Du;
  let m0 = u64((x_lo)) * u64((C_lo));
  let m1 = u64((x_lo)) * u64((C_hi));
  let m2 = u64((x_hi)) * u64((C_lo));
  // accumulate 128-bit product collapsed to 64 bits (hi:lo)
  let lo = m0 & 0xFFFF_FFFFu64;
  let carry = (m0 >> 32u) + (m1 & 0xFFFF_FFFFu64) + (m2 & 0xFFFF_FFFFu64);
  let hi = (m1 >> 32u) + (m2 >> 32u) + (carry >> 32u);
  x_lo = u32(lo & 0xFFFF_FFFFu64);
  x_hi = u32(hi & 0xFFFF_FFFFu64);
  *state = vec2<u32>(x_hi, x_lo);
  let mant = (f32(x_hi) / 4294967296.0) + (f32(x_lo) / 18446744073709551616.0);
  return fract(mant);
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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = uniforms.width;
  let H = uniforms.height;
  if (gid.x >= W || gid.y >= H) { return; }
  let px = f32(gid.x);
  let py = f32(gid.y);

  // Seed per-pixel RNG from seed_hi/lo + pixel index + frame_index
  var st = vec2<u32>(uniforms.seed_hi ^ gid.x ^ (uniforms.frame_index * 1664525u),
                     uniforms.seed_lo ^ gid.y ^ (uniforms.frame_index * 1013904223u));
  let jx = tent_filter(xorshift64star(&st)) * 0.5;
  let jy = tent_filter(xorshift64star(&st)) * 0.5;

  let ndc_x = ((px + 0.5 + jx) / f32(W)) * 2.0 - 1.0;
  let ndc_y = (1.0 - (py + 0.5 + jy) / f32(H)) * 2.0 - 1.0;
  let half_h = tan(0.5 * uniforms.cam_fov_y);
  let half_w = uniforms.cam_aspect * half_h;
  var rd = normalize(vec3<f32>(ndc_x * half_w, ndc_y * half_h, -1.0));
  // re-orient with camera basis (right, up, forward)
  rd = normalize(rd.x * uniforms.cam_right + rd.y * uniforms.cam_up + rd.z * (-uniforms.cam_forward));
  let ro = uniforms.cam_origin;

  // trace MVP spheres
  var t_best = 1e30;
  var hit_color = vec3<f32>(0.0);
  var hit_n = vec3<f32>(0.0, 0.0, 1.0);
  // Iterate small list; in MVP it's fine to linear scan
  let sphere_count = arrayLength(&scene_spheres);
  for (var i: u32 = 0u; i < sphere_count; i = i + 1u) {
    let s = scene_spheres[i];
    let t = ray_sphere(ro, rd, s.center, s.radius);
    if (t < t_best) {
      t_best = t;
      let p = ro + rd * t;
      hit_n = normalize(p - s.center);
      hit_color = s.albedo;
    }
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

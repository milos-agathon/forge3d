// src/shaders/pt_kernel.wgsl
// Path tracing compute kernel with minimal ground plane + sky and AOVs.
// Exists to provide a deterministic GPU path for tests and a simple PBR-like plane with reflections.
// RELEVANT FILES:src/path_tracing/compute.rs,src/path_tracing/aov.rs,python/forge3d/path_tracing.py

// Bind Group 0 (Uniforms): width, height, frame_index, aov_flags; camera params; exposure; seed_hi/seed_lo
// Bind Group 1 (Scene): readonly storage (materials, spheres/triangles, etc.)
// Bind Group 2 (Accum/State): storage buffers (if using accumulation)
// Bind Group 3 (Out): storage texture RGBA16F for tonemapped output
// Bind Group 4 (AOV outputs): storage textures per AOV (enabled by aov_flags)

struct Uniforms {
  width: u32,
  height: u32,
  frame_index: u32,
  aov_flags: u32,
  cam_origin: vec3<f32>, cam_fov_y: f32,
  cam_right: vec3<f32>, cam_aspect: f32,
  cam_up: vec3<f32>, cam_exposure: f32,
  cam_forward: vec3<f32>, seed_hi: u32,
  seed_lo: u32,
  _pad0: vec3<u32>,
};

@group(0) @binding(0) var<uniform> ubo: Uniforms;

// Output texture (tonemapped RGBA)
@group(3) @binding(0) var out_rgba: texture_storage_2d<rgba16float, write>;

// AOV outputs (each is optional, enabled via bits in ubo.aov_flags)
// Bit layout (LSB..): 0=albedo,1=normal,2=depth,3=direct,4=indirect,5=emission,6=visibility
@group(4) @binding(0) var aov_albedo: texture_storage_2d<rgba16float, write>;
@group(4) @binding(1) var aov_normal: texture_storage_2d<rgba16float, write>;
@group(4) @binding(2) var aov_depth: texture_storage_2d<r32float, write>;
@group(4) @binding(3) var aov_direct: texture_storage_2d<rgba16float, write>;
@group(4) @binding(4) var aov_indirect: texture_storage_2d<rgba16float, write>;
@group(4) @binding(5) var aov_emission: texture_storage_2d<rgba16float, write>;
@group(4) @binding(6) var aov_visibility: texture_storage_2d<r8unorm, write>;

fn aov_enabled(bit: u32) -> bool {
  return (ubo.aov_flags & (1u << bit)) != 0u;
}

fn tonemap_reinhard(c: vec3<f32>, exposure: f32) -> vec3<f32> {
  let x = c * max(exposure, 0.0001);
  return x / (x + vec3<f32>(1.0));
}

fn env_color(dir: vec3<f32>) -> vec3<f32> {
  // Simple gradient sky: up = blue, horizon = white, below = darker ground tint.
  let t = clamp(0.5 * (dir.y + 1.0), 0.0, 1.0);
  let sky = mix(vec3<f32>(0.9, 0.95, 1.0), vec3<f32>(0.2, 0.4, 0.8), t);
  let ground = vec3<f32>(0.08, 0.08, 0.08);
  return mix(ground, sky, t);
}

fn generate_primary_ray(px: u32, py: u32) -> vec3<f32> {
  let sx = (f32(px) + 0.5) / max(1.0, f32(ubo.width));
  let sy = (f32(py) + 0.5) / max(1.0, f32(ubo.height));
  let ndc_x = 2.0 * sx - 1.0;
  let ndc_y = 1.0 - 2.0 * sy;
  let tan_half = tan(0.5 * ubo.cam_fov_y);
  let cam_dir = normalize(
    ubo.cam_forward +
    ndc_x * ubo.cam_aspect * tan_half * ubo.cam_right +
    ndc_y * tan_half * ubo.cam_up);
  return cam_dir;
}

struct Hit {
  hit: bool,
  t: f32,
  n: vec3<f32>,
  p: vec3<f32>,
}

fn intersect_ground_plane(ro: vec3<f32>, rd: vec3<f32>) -> Hit {
  // Plane y=0 with normal (0,1,0). Only consider hits in front of the ray and above plane.
  let n = vec3<f32>(0.0, 1.0, 0.0);
  if (rd.y >= -1e-5) {
    return Hit(false, 1e30, n, ro);
  }
  let t = -ro.y / rd.y;
  if (t > 0.0) {
    let p = ro + t * rd;
    return Hit(true, t, n, p);
  }
  return Hit(false, 1e30, n, ro);
}

fn shade_ground(ro: vec3<f32>, rd: vec3<f32>, h: Hit) -> struct {
  color: vec3<f32>,
  albedo: vec3<f32>,
  direct: vec3<f32>,
  indirect: vec3<f32>,
} {
  // Minimal PBR-ish shading with a single directional light and env reflection.
  let base_color = vec3<f32>(0.6, 0.6, 0.6);
  let metallic = 0.0;
  let roughness = 0.2; // keep fairly glossy for reflections
  let n = h.n;
  let p = h.p;
  let v = normalize(-rd);
  let l = normalize(vec3<f32>(0.4, 1.0, 0.2));
  let radiance = vec3<f32>(1.0, 0.95, 0.90) * 2.5;
  let hvec = normalize(l + v);
  let n_dot_l = max(dot(n, l), 0.0);
  let n_dot_v = max(dot(n, v), 0.0);
  let n_dot_h = max(dot(n, hvec), 0.0);
  let v_dot_h = max(dot(v, hvec), 0.0);
  let pi = 3.14159265359;
  // GGX helpers
  let a = roughness * roughness;
  let a2 = a * a;
  let D = a2 / max(pi * pow(n_dot_h * n_dot_h * (a2 - 1.0) + 1.0, 2.0), 1e-6);
  let k = pow(roughness + 1.0, 2.0) / 8.0;
  let Gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
  let Gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
  let G = Gv * Gl;
  let F0 = mix(vec3<f32>(0.04), base_color, metallic);
  let F = F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - v_dot_h, 5.0);
  let spec = (D * G) / max(4.0 * n_dot_v * n_dot_l, 1e-6) * F;
  let kS = F;
  let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);
  let diffuse = kD * base_color / pi;
  let direct = (diffuse + spec) * radiance * n_dot_l;
  // Simple env reflection for indirect
  let r = reflect(-v, n);
  let env = env_color(r);
  // Approximate BRDF integration by scaling env with Fresnel and roughness factor
  let F_ibl = F0 + (max(vec3<f32>(1.0 - roughness), F0) - F0) * pow(1.0 - n_dot_v, 5.0);
  let indirect = env * (F_ibl * 0.5 + 0.5 * kD * base_color);
  // Combine and apply a trivial ground tint based on distance for visual depth
  let dist = length(p - ro);
  let fog = clamp(dist / 50.0, 0.0, 1.0);
  let color = mix(direct + indirect, env_color(vec3<f32>(0.0, 1.0, 0.0)), fog);
  return .{ color, base_color, direct, indirect };
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ubo.width || gid.y >= ubo.height) { return; }
  let xy = vec2<i32>(i32(gid.x), i32(gid.y));

  let ro = ubo.cam_origin;
  let rd = generate_primary_ray(gid.x, gid.y);
  let hit = intersect_ground_plane(ro, rd);

  var color = vec3<f32>(0.0);
  var albedo = vec3<f32>(0.0);
  var direct = vec3<f32>(0.0);
  var indirect = vec3<f32>(0.0);
  var depth = 1.0;
  var vis: f32 = 0.0;

  if (hit.hit) {
    let s = shade_ground(ro, rd, hit);
    color = s.color;
    albedo = s.albedo;
    direct = s.direct;
    indirect = s.indirect;
    depth = hit.t;
    vis = 1.0;
  } else {
    color = env_color(rd);
    albedo = vec3<f32>(0.0);
    direct = vec3<f32>(0.0);
    indirect = color;
    depth = 1.0;
    vis = 0.0;
  }

  // Tonemap and output RGBA16F
  let tm = tonemap_reinhard(color, ubo.cam_exposure);
  textureStore(out_rgba, xy, vec4<f16>(vec4<f32>(tm, 1.0)));

  // AOV writes (optional)
  if (aov_enabled(0u)) { textureStore(aov_albedo, xy, vec4<f16>(vec4<f32>(albedo, 1.0))); }
  if (aov_enabled(1u)) {
    let n = normalize(select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), hit.hit));
    textureStore(aov_normal, xy, vec4<f16>(vec4<f32>(n, 1.0)));
  }
  if (aov_enabled(2u)) { textureStore(aov_depth, xy, vec4<f32>(depth, 0.0, 0.0, 0.0)); }
  if (aov_enabled(3u)) { textureStore(aov_direct, xy, vec4<f16>(vec4<f32>(direct, 1.0))); }
  if (aov_enabled(4u)) { textureStore(aov_indirect, xy, vec4<f16>(vec4<f32>(indirect, 1.0))); }
  if (aov_enabled(5u)) { textureStore(aov_emission, xy, vec4<f16>(f16(0.0), f16(0.0), f16(0.0), f16(1.0))); }
  if (aov_enabled(6u)) { textureStore(aov_visibility, xy, vec4<f32>(vis, 0.0, 0.0, 0.0)); }
}

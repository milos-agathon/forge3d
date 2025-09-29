// src/shaders/mesh_instanced.wgsl
// Instanced 3D mesh shader (per-instance transform matrix)

struct SceneUniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  color: vec4<f32>,
  light_dir_ws: vec4<f32>, // xyz: dir, w: intensity
}

@group(0) @binding(0) var<uniform> U : SceneUniforms;

struct VsIn {
  // Per-vertex attributes
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  // Per-instance transform (column-major as 4x vec4)
  @location(2) i_m0: vec4<f32>,
  @location(3) i_m1: vec4<f32>,
  @location(4) i_m2: vec4<f32>,
  @location(5) i_m3: vec4<f32>,
}

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) n_ws: vec3<f32>,
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
  var out: VsOut;
  let M = mat4x4<f32>(in.i_m0, in.i_m1, in.i_m2, in.i_m3);
  let pos_ws = M * vec4<f32>(in.position, 1.0);
  let n_ws = normalize((M * vec4<f32>(in.normal, 0.0)).xyz);
  out.pos = U.proj * U.view * pos_ws;
  out.n_ws = n_ws;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let n = normalize(in.n_ws);
  let l = normalize(U.light_dir_ws.xyz);
  let ndotl = max(dot(n, -l), 0.0);
  let intensity = max(U.light_dir_ws.w, 0.0);
  let base = U.color.rgb;
  let lit = base * (0.2 + 0.7 * ndotl * intensity);
  return vec4<f32>(lit, U.color.a);
}

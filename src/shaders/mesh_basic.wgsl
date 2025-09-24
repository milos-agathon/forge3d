// src/shaders/mesh_basic.wgsl
// Simple unlit/Lambert mesh shader for 3D text meshes

struct MeshUniforms {
  model: mat4x4<f32>,
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  color: vec4<f32>,
  light_dir_ws: vec4<f32>, // xyz direction, w=unused
  mr: vec2<f32>,           // x=metallic, y=roughness
  _pad_mr: vec2<f32>,
};

@group(0) @binding(0) var<uniform> U : MeshUniforms;

struct VsIn {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
};

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) n_ws: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
  var out: VsOut;
  let pos_ws = (U.model * vec4<f32>(in.position, 1.0));
  let n_ws = normalize((U.model * vec4<f32>(in.normal, 0.0)).xyz);
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
  // Simple metallic-roughness approximation
  let metallic = clamp(U.mr.x, 0.0, 1.0);
  let roughness = clamp(U.mr.y, 0.04, 1.0);
  let shininess = mix(8.0, 64.0, 1.0 - roughness);
  let spec = pow(ndotl, shininess) * metallic;
  let diffuse = base * (0.2 + 0.7 * ndotl * intensity);
  let spec_col = mix(vec3<f32>(0.04), base, metallic) * spec * intensity;
  let lit = diffuse + spec_col;
  return vec4<f32>(lit, U.color.a);
}

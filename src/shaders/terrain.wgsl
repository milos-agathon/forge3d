// A2-BEGIN:terrain-shader
struct UBO {
  mvp   : mat4x4<f32>,
  light : vec4<f32>,
};
@group(0) @binding(0) var<uniform> ubo : UBO;

struct VSIn {
  @location(0) pos    : vec3<f32>,
  @location(1) normal : vec3<f32>,
};
struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) n         : vec3<f32>,
};

@vertex
fn vs_main(in: VSIn) -> VSOut {
  var out: VSOut;
  out.pos = ubo.mvp * vec4<f32>(in.pos, 1.0);
  out.n   = normalize(in.normal);
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let L = normalize(ubo.light.xyz);
  let N = normalize(in.n);
  let ndotl = max(dot(N, L), 0.0);
  let base  = vec3<f32>(0.35, 0.55, 0.35);
  let color = base * (0.25 + 0.75 * ndotl);
  return vec4<f32>(color, 1.0);
}
// A2-END:terrain-shader

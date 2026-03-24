// src/shaders/mesh_instanced.wgsl
// Instanced 3D mesh shader (per-instance transform matrix)

struct ScatterBatchUniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  color: vec4<f32>,
  light_dir_ws: vec4<f32>, // xyz: dir, w: intensity
  wind_phase: vec4<f32>,
  wind_vec_bounds: vec4<f32>,
  wind_bend_fade: vec4<f32>,
}

@group(0) @binding(0) var<uniform> U : ScatterBatchUniforms;

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
  var pos_ws = M * vec4<f32>(in.position, 1.0);
  var n_ws = normalize((M * vec4<f32>(in.normal, 0.0)).xyz);

  let wind_local = U.wind_vec_bounds.xyz;
  let wind_amp = length(wind_local);

  if (wind_amp > 1e-6) {
    // Bend weight from mesh-local normalized Y height
    let norm_h = clamp(in.position.y / max(U.wind_vec_bounds.w, 1e-4), 0.0, 1.0);
    let bend_weight = smoothstep(
      U.wind_bend_fade.x,
      U.wind_bend_fade.x + U.wind_bend_fade.y,
      norm_h
    );

    // Wind direction in world space (for spatial phase variety)
    let wind_dir_ws = normalize((M * vec4<f32>(wind_local, 0.0)).xyz);

    // Deterministic sway + gust
    let spatial = dot(pos_ws.xyz, wind_dir_ws) * 0.1;
    let sway = sin(U.wind_phase.x + spatial) * (1.0 - U.wind_phase.w) * wind_amp;
    let gust = sin(U.wind_phase.y + spatial * 0.37) * U.wind_phase.z;  // 0.37: decorrelation

    // Displacement in local frame, transformed to world through M
    let wind_dir_local = wind_local / wind_amp;
    let disp_local = wind_dir_local * (sway + gust) * bend_weight;
    var disp_ws = (M * vec4<f32>(disp_local, 0.0)).xyz;

    // Distance fade (view-space distance)
    let fade_start = U.wind_bend_fade.z;
    let fade_end = U.wind_bend_fade.w;
    if (fade_end > fade_start) {
      let view_pos = U.view * pos_ws;
      let view_dist = length(view_pos.xyz);
      disp_ws *= 1.0 - smoothstep(fade_start, fade_end, view_dist);
    }

    pos_ws = vec4<f32>(pos_ws.xyz + disp_ws, 1.0);

    // Cheap normal tilt
    let tilt = length(disp_ws) * 0.3;
    let up_ws = normalize(in.i_m1.xyz);
    n_ws = normalize(n_ws + wind_dir_ws * tilt * max(dot(n_ws, up_ws), 0.0));
  }

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

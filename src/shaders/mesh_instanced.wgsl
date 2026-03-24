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
  terrain_blend: vec4<f32>, // x=enabled, y=bury_depth, z=fade_distance
  terrain_contact: vec4<f32>, // x=enabled, y=distance, z=strength, w=vertical_weight
}

@group(0) @binding(0) var<uniform> U : ScatterBatchUniforms;

struct TerrainContextUniforms {
  world_to_uv_scale_bias: vec4<f32>, // xy=scale, zw=bias
  height_to_world: vec4<f32>, // x=scale, y=bias
}

@group(1) @binding(0) var<uniform> T : TerrainContextUniforms;
@group(1) @binding(1) var height_tex : texture_2d<f32>;

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
  @location(1) world_pos: vec3<f32>,
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
  out.world_pos = pos_ws.xyz;
  return out;
}

fn saturate(value: f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn load_height(pixel: vec2<i32>) -> f32 {
  let dims_u = textureDimensions(height_tex, 0);
  let dims = vec2<i32>(i32(dims_u.x), i32(dims_u.y));
  let clamped = clamp(pixel, vec2<i32>(0, 0), max(dims - vec2<i32>(1, 1), vec2<i32>(0, 0)));
  return textureLoad(height_tex, clamped, 0).r;
}

fn sample_height_bilinear(uv: vec2<f32>) -> f32 {
  let dims_u = textureDimensions(height_tex, 0);
  let dims = vec2<f32>(f32(dims_u.x), f32(dims_u.y));
  let max_index = max(dims - vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 1.0));
  let coord = clamp(uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0)) * max_index;
  let base = vec2<i32>(i32(floor(coord.x)), i32(floor(coord.y)));
  let frac = fract(coord);
  let h00 = load_height(base);
  let h10 = load_height(base + vec2<i32>(1, 0));
  let h01 = load_height(base + vec2<i32>(0, 1));
  let h11 = load_height(base + vec2<i32>(1, 1));
  let hx0 = mix(h00, h10, frac.x);
  let hx1 = mix(h01, h11, frac.x);
  return mix(hx0, hx1, frac.y);
}

fn terrain_height_delta(world_pos: vec3<f32>) -> f32 {
  let uv = vec2<f32>(
    world_pos.x * T.world_to_uv_scale_bias.x + T.world_to_uv_scale_bias.z,
    world_pos.z * T.world_to_uv_scale_bias.y + T.world_to_uv_scale_bias.w
  );
  let terrain_height = sample_height_bilinear(uv) * T.height_to_world.x + T.height_to_world.y;
  return world_pos.y - terrain_height;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let n = normalize(in.n_ws);
  let l = normalize(U.light_dir_ws.xyz);
  let ndotl = max(dot(n, -l), 0.0);
  let intensity = max(U.light_dir_ws.w, 0.0);
  let base = U.color.rgb;
  let height_delta = terrain_height_delta(in.world_pos);

  var alpha = U.color.a;
  if (U.terrain_blend.x > 0.5) {
    let bury_depth = max(U.terrain_blend.y, 1e-4);
    let fade_distance = max(U.terrain_blend.z, 1e-4);
    alpha = alpha * smoothstep(-bury_depth, fade_distance, height_delta);
  }

  if (alpha <= 1e-3) {
    discard;
  }

  let lit = base * (0.2 + 0.7 * ndotl * intensity);
  var contact = 0.0;
  if (U.terrain_contact.x > 0.5) {
    let contact_distance = max(U.terrain_contact.y, 1e-4);
    let strength = saturate(U.terrain_contact.z);
    let vertical_weight = saturate(U.terrain_contact.w);
    let proximity = 1.0 - smoothstep(0.0, contact_distance, abs(height_delta));
    let side_factor = mix(1.0, saturate(1.0 - abs(n.y)), vertical_weight);
    contact = proximity * side_factor * strength;
  }
  let shaded = lit * (1.0 - contact);
  return vec4<f32>(shaded, alpha);
}

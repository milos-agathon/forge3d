// src/shaders/mesh_instanced.wgsl
// Instanced 3D mesh shader (per-instance transform matrix)

struct SceneUniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  color: vec4<f32>,
  light_dir_ws: vec4<f32>, // xyz: dir, w: intensity
  terrain_blend: vec4<f32>, // enabled, blend_distance, contact_strength, contact_distance
}

@group(0) @binding(0) var<uniform> U : SceneUniforms;
@group(1) @binding(0) var terrain_height_tex: texture_2d<f32>;

struct TerrainBlendContext {
  uv_scale_bias: vec4<f32>, // scale.xy, bias.xy
  height_decode: vec4<f32>, // min, max, z_scale, axis_mode
}

@group(1) @binding(1) var<uniform> T : TerrainBlendContext;

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
  @location(0) world_pos: vec3<f32>,
  @location(1) n_ws: vec3<f32>,
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
  var out: VsOut;
  let M = mat4x4<f32>(in.i_m0, in.i_m1, in.i_m2, in.i_m3);
  let pos_ws = M * vec4<f32>(in.position, 1.0);
  let n_ws = normalize((M * vec4<f32>(in.normal, 0.0)).xyz);
  out.pos = U.proj * U.view * pos_ws;
  out.world_pos = pos_ws.xyz;
  out.n_ws = n_ws;
  return out;
}

fn terrain_blend_enabled() -> bool {
  return U.terrain_blend.x > 0.5;
}

fn terrain_axis_mode() -> u32 {
  return u32(T.height_decode.w + 0.5);
}

fn terrain_up_axis() -> vec3<f32> {
  if (terrain_axis_mode() == 1u) {
    return vec3<f32>(0.0, 0.0, 1.0);
  }
  return vec3<f32>(0.0, 1.0, 0.0);
}

fn terrain_horizontal(world_pos: vec3<f32>) -> vec2<f32> {
  if (terrain_axis_mode() == 1u) {
    return world_pos.xy;
  }
  return vec2<f32>(world_pos.x, world_pos.z);
}

fn terrain_up_value(world_pos: vec3<f32>) -> f32 {
  if (terrain_axis_mode() == 1u) {
    return world_pos.z;
  }
  return world_pos.y;
}

fn sample_height_bilinear(uv: vec2<f32>) -> f32 {
  let dims_u = textureDimensions(terrain_height_tex);
  let dims = vec2<i32>(i32(dims_u.x), i32(dims_u.y));
  let max_texel = dims - vec2<i32>(1, 1);
  let dims_f = vec2<f32>(f32(dims_u.x), f32(dims_u.y));
  let max_texel_f = vec2<f32>(f32(max_texel.x), f32(max_texel.y));
  let pixel = clamp(uv * dims_f - vec2<f32>(0.5, 0.5), vec2<f32>(0.0), max_texel_f);

  let p0 = vec2<i32>(i32(floor(pixel.x)), i32(floor(pixel.y)));
  let p1 = min(p0 + vec2<i32>(1, 1), max_texel);
  let frac = fract(pixel);

  let h00 = textureLoad(terrain_height_tex, p0, 0).r;
  let h10 = textureLoad(terrain_height_tex, vec2<i32>(p1.x, p0.y), 0).r;
  let h01 = textureLoad(terrain_height_tex, vec2<i32>(p0.x, p1.y), 0).r;
  let h11 = textureLoad(terrain_height_tex, p1, 0).r;

  let top = mix(h00, h10, frac.x);
  let bottom = mix(h01, h11, frac.x);
  return mix(top, bottom, frac.y);
}

fn terrain_world_position(world_pos: vec3<f32>) -> vec4<f32> {
  let horiz = terrain_horizontal(world_pos);
  let uv = horiz * T.uv_scale_bias.xy + T.uv_scale_bias.zw;
  if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }

  let h_raw = sample_height_bilinear(uv);
  let h_min = T.height_decode.x;
  let h_max = T.height_decode.y;
  let z_scale = T.height_decode.z;
  if (terrain_axis_mode() == 1u) {
    let h_center = 0.5 * (h_min + h_max);
    return vec4<f32>(horiz.x, horiz.y, (h_raw - h_center) * z_scale, 1.0);
  }
  return vec4<f32>(horiz.x, (h_raw - h_min) * z_scale, horiz.y, 1.0);
}

fn compute_terrain_blend(world_pos: vec3<f32>, normal_ws: vec3<f32>) -> vec2<f32> {
  if (!terrain_blend_enabled()) {
    return vec2<f32>(1.0, 1.0);
  }

  let terrain_world = terrain_world_position(world_pos);
  if (terrain_world.w < 0.5) {
    return vec2<f32>(1.0, 1.0);
  }

  let blend_distance = max(U.terrain_blend.y, 1e-4);
  let mesh_up = terrain_up_value(world_pos);
  let terrain_up = terrain_up_value(terrain_world.xyz);
  let height_gap = max(mesh_up - terrain_up, 0.0);
  let height_factor = smoothstep(0.0, blend_distance, height_gap);

  let mesh_view = U.view * vec4<f32>(world_pos, 1.0);
  let terrain_view = U.view * terrain_world;
  let depth_gap = abs(mesh_view.z - terrain_view.z);
  let depth_factor = smoothstep(0.0, blend_distance, depth_gap);

  let up_axis = terrain_up_axis();
  let up_alignment = dot(normalize(normal_ws), up_axis);
  let side_gate = 1.0 - smoothstep(0.35, 0.85, up_alignment);
  let alpha_factor = mix(1.0, height_factor * depth_factor, side_gate);

  let contact_distance = max(U.terrain_blend.w, 1e-4);
  let contact_mask = 1.0 - smoothstep(0.0, contact_distance, height_gap);
  let contact_factor = 1.0 - clamp(U.terrain_blend.z, 0.0, 1.0) * contact_mask;
  return vec2<f32>(alpha_factor, contact_factor);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let n = normalize(in.n_ws);
  let l = normalize(U.light_dir_ws.xyz);
  let ndotl = max(dot(n, -l), 0.0);
  let intensity = max(U.light_dir_ws.w, 0.0);
  let base = U.color.rgb;
  let blend = compute_terrain_blend(in.world_pos, n);
  let lit = base * (0.2 + 0.7 * ndotl * intensity) * blend.y;
  let alpha = U.color.a * blend.x;
  if (alpha <= 1e-3) {
    discard;
  }
  return vec4<f32>(lit, alpha);
}

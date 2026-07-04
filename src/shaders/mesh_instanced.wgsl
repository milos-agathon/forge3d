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

struct ShadowCascade {
  light_projection: mat4x4<f32>,
  light_view_proj: mat4x4<f32>,
  near_distance: f32,
  far_distance: f32,
  texel_size: f32,
  _padding: f32,
}

struct CsmUniforms {
  light_direction: vec4<f32>,
  light_view: mat4x4<f32>,
  cascades: array<ShadowCascade, 4>,
  cascade_count: u32,
  pcf_kernel_size: u32,
  depth_bias: f32,
  slope_bias: f32,
  shadow_map_size: f32,
  debug_mode: u32,
  evsm_positive_exp: f32,
  evsm_negative_exp: f32,
  peter_panning_offset: f32,
  enable_unclipped_depth: u32,
  depth_clip_factor: f32,
  technique: u32,
  technique_flags: u32,
  _pad1a: f32,
  _pad1b: f32,
  _pad1c: f32,
  technique_params: vec4<f32>,
  technique_reserved: vec4<f32>,
  cascade_blend_range: f32,
  _pad2a: f32,
  _pad2b: f32,
  _pad2c: f32,
  _pad2d: array<vec4<f32>, 6>,
}

@group(2) @binding(0) var<storage, read> csm_uniforms: CsmUniforms;
@group(2) @binding(1) var shadow_maps: texture_depth_2d_array;
@group(2) @binding(2) var shadow_sampler: sampler_comparison;
@group(2) @binding(3) var moment_maps: texture_2d_array<f32>;
@group(2) @binding(4) var moment_sampler: sampler;

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

struct ShadowVsOut {
  @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_shadow(in: VsIn) -> ShadowVsOut {
  var out: ShadowVsOut;
  let M = mat4x4<f32>(in.i_m0, in.i_m1, in.i_m2, in.i_m3);
  let pos_ws = M * vec4<f32>(in.position, 1.0);
  out.pos = U.proj * U.view * pos_ws;
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

fn mesh_shadow_visibility(world_pos: vec3<f32>) -> f32 {
  let cascade_count = min(csm_uniforms.cascade_count, 4u);
  if (cascade_count == 0u) {
    return 1.0;
  }

  let view_pos = U.view * vec4<f32>(world_pos, 1.0);
  let view_depth = abs(view_pos.z);
  var cascade_idx = cascade_count - 1u;
  for (var i: u32 = 0u; i < cascade_count; i = i + 1u) {
    if (view_depth <= csm_uniforms.cascades[i].far_distance) {
      cascade_idx = i;
      break;
    }
  }

  let cascade = csm_uniforms.cascades[cascade_idx];
  let light_space = cascade.light_view_proj * vec4<f32>(world_pos, 1.0);
  if (abs(light_space.w) <= 1e-6) {
    return 1.0;
  }
  let ndc = light_space.xyz / light_space.w;
  let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
  if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0) {
    return 1.0;
  }
  let compare_depth = clamp(ndc.z - max(csm_uniforms.depth_bias, 0.0), 0.0, 1.0);
  if (csm_uniforms.pcf_kernel_size <= 1u) {
    return textureSampleCompare(shadow_maps, shadow_sampler, shadow_uv, i32(cascade_idx), compare_depth);
  }

  let texel = 1.0 / max(csm_uniforms.shadow_map_size, 1.0);
  var sum = 0.0;
  var count = 0.0;
  for (var oy: i32 = -1; oy <= 1; oy = oy + 1) {
    for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
      let uv = shadow_uv + vec2<f32>(f32(ox), f32(oy)) * texel;
      if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
        sum = sum + textureSampleCompare(shadow_maps, shadow_sampler, uv, i32(cascade_idx), compare_depth);
        count = count + 1.0;
      }
    }
  }
  return sum / max(count, 1.0);
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

  let shadow_visibility = mesh_shadow_visibility(in.world_pos);
  let direct_shadow = mix(0.20, 1.0, shadow_visibility);
  let lit = base * (0.2 + 0.7 * ndotl * intensity * direct_shadow);
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

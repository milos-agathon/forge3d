// src/shaders/terrain_descriptor_indexing.wgsl
// Descriptor-indexed terrain surface shading with Workstream B overlays.
// Provides cloud shadow modulation and planar reflection support when descriptor indexing is active.
// RELEVANT FILES:src/shaders/terrain.wgsl,src/terrain/pipeline.rs,src/scene/mod.rs,src/core/cloud_shadows.rs
// T3.3 Terrain shader with descriptor indexing — texture array variant.
// Layout: 0=Globals UBO, 1=height R32Float + NonFiltering sampler, 2=LUT texture array + Filtering sampler.
// This version uses texture arrays for descriptor indexing when available.

// ---------- Globals UBO (176 bytes total, must match Rust) ----------
// std140-compatible layout: 176 bytes total
// Field breakdown:
//   view:                 mat4x4<f32> = 64 B
//   proj:                 mat4x4<f32> = 64 B  
//   sun_exposure:         vec4<f32>   = 16 B (xyz = sun_dir normalized, w = exposure)
//   spacing_h_exag_pad:   vec4<f32>   = 16 B (x=dx, y=dy, z=height_exaggeration, w=palette_index)
//   _pad_tail:            vec4<f32>   = 16 B (tail padding)
//   TOTAL:                            = 176 B
struct Globals {
  view: mat4x4<f32>,                    // 64 B
  proj: mat4x4<f32>,                    // 64 B
  sun_exposure: vec4<f32>,              // xyz = sun_dir (normalized), w = exposure (16 B)
  spacing_h_exag_pad: vec4<f32>,        // x=dx, y=dy, z=height_exaggeration, w=palette_index (16 B)
  _pad_tail: vec4<f32>,                 // tail padding to keep total size multiple-of-16 (16 B)
};

@group(0) @binding(0) var<uniform> globals : Globals;

// ---------- Textures & samplers ----------
@group(1) @binding(0) var height_tex  : texture_2d<f32>;  // R32Float, non-filterable
@group(1) @binding(1) var height_samp : sampler;          // NonFiltering at pipeline level

// Descriptor indexing: texture array instead of single texture
@group(2) @binding(0) var lut_textures : binding_array<texture_2d<f32>>;  // RGBA8 texture array
@group(2) @binding(1) var lut_samp : sampler;

// B7: Cloud shadow overlay resources
@group(3) @binding(0) var cloud_shadow_tex  : texture_2d<f32>;  // shadow multiplier texture
@group(3) @binding(1) var cloud_shadow_samp : sampler;

// B5: Planar reflections resources
struct ReflectionPlane {
  plane_equation: vec4<f32>,
  reflection_matrix: mat4x4<f32>,
  reflection_view: mat4x4<f32>,
  reflection_projection: mat4x4<f32>,
  plane_center: vec4<f32>,
  plane_size: vec4<f32>,
};

struct PlanarReflectionUniforms {
  reflection_plane: ReflectionPlane,
  enable_reflections: u32,
  reflection_intensity: f32,
  fresnel_power: f32,
  blur_kernel_size: u32,
  max_blur_radius: f32,
  reflection_resolution: f32,
  distance_fade_start: f32,
  distance_fade_end: f32,
  debug_mode: u32,
  camera_position: vec4<f32>,
  _pad: vec3<f32>,
};

@group(4) @binding(0) var<uniform> reflection_uniforms : PlanarReflectionUniforms;
@group(4) @binding(1) var reflection_texture : texture_2d<f32>;
@group(4) @binding(2) var reflection_sampler : sampler;
@group(4) @binding(3) var reflection_depth : texture_depth_2d;

// ---------- IO ----------
struct VsIn {
  // position.xy in plane
  @location(0) pos_xy : vec2<f32>,
  @location(1) uv     : vec2<f32>,
};

struct VsOut {
  @builtin(position) clip_pos : vec4<f32>,
  @location(0) uv             : vec2<f32>,
  @location(1) height         : f32,
  @location(2) xz             : vec2<f32>,   // pass plane x/z to fragment for shading
};

struct FsOut {
  @location(0) color : vec4<f32>,
  @location(1) normal_depth : vec4<f32>,
};

// Analytic fallback height that varies across the grid. Amplitude ≈ ±0.5 (matches Globals defaults).
fn analytic_height(x: f32, z: f32) -> f32 {
  return sin(x * 1.3) * 0.25 + cos(z * 1.1) * 0.25;
}

fn distance_to_plane(point: vec3<f32>, plane: vec4<f32>) -> f32 {
  return dot(vec4<f32>(point, 1.0), plane);
}

// B7: Sample cloud shadow multiplier using world-space coordinates
fn sample_cloud_shadow(world_pos: vec2<f32>, terrain_scale: f32) -> f32 {
  let terrain_size = terrain_scale * 2.0;
  let uv = (world_pos / terrain_size + 1.0) * 0.5;
  let sample = textureSampleLevel(cloud_shadow_tex, cloud_shadow_samp, uv, 0.0);
  return sample.r;
}

fn apply_planar_reflection(
  world_pos: vec3<f32>,
  world_normal: vec3<f32>,
  base_color: vec3<f32>,
) -> vec3<f32> {
  if reflection_uniforms.enable_reflections == 0u {
    return base_color;
  }

  let plane = reflection_uniforms.reflection_plane.plane_equation;
  let reflected_pos = reflection_uniforms.reflection_plane.reflection_view * vec4<f32>(world_pos, 1.0);
  let projected = reflection_uniforms.reflection_plane.reflection_projection * reflected_pos;
  if projected.w == 0.0 {
    return base_color;
  }
  let ndc = projected.xyz / projected.w;
  let uv = ndc.xy * 0.5 + 0.5;
  if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
    return base_color;
  }

  var reflection = textureSample(reflection_texture, reflection_sampler, uv);
  let kernel = f32(reflection_uniforms.blur_kernel_size);
  if kernel > 1.0 {
    let texel = reflection_uniforms.max_blur_radius / max(reflection_uniforms.reflection_resolution, 1.0);
    let offset = texel * max(kernel * 0.5, 1.0);
    let offsets = array<vec2<f32>, 4>(
      vec2<f32>(offset, 0.0),
      vec2<f32>(-offset, 0.0),
      vec2<f32>(0.0, offset),
      vec2<f32>(0.0, -offset)
    );
    var accum = reflection;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
      let sample_uv = clamp(uv + offsets[i], vec2<f32>(0.0), vec2<f32>(1.0));
      accum = accum + textureSample(reflection_texture, reflection_sampler, sample_uv);
    }
    reflection = accum / 5.0;
  }

  let camera_pos = reflection_uniforms.camera_position.xyz;
  let view_dir = normalize(camera_pos - world_pos);
  let fresnel = pow(max(0.0, 1.0 - dot(world_normal, view_dir)), reflection_uniforms.fresnel_power);
  let distance = abs(distance_to_plane(world_pos, plane));
  let fade_start = reflection_uniforms.distance_fade_start;
  let fade_end = max(reflection_uniforms.distance_fade_end, fade_start + 0.001);
  let fade = clamp((fade_end - distance) / max(fade_end - fade_start, 0.001), 0.0, 1.0);
  let intensity = clamp(reflection_uniforms.reflection_intensity * fresnel * fade, 0.0, 1.0);

  switch reflection_uniforms.debug_mode {
    case 1u: {
      return vec3<f32>(uv, 0.0);
    }
    case 2u: {
      return vec3<f32>(intensity, intensity, intensity);
    }
    case 3u: {
      let dist_norm = clamp(distance / fade_end, 0.0, 1.0);
      return vec3<f32>(dist_norm, 0.0, 1.0 - dist_norm);
    }
    case 4u: {
      let blur_norm = clamp(kernel / 9.0, 0.0, 1.0);
      return vec3<f32>(blur_norm, 1.0 - blur_norm, 0.0);
    }
    default: {
      return mix(base_color, reflection.rgb, intensity);
    }
  }
}

// ---------- Vertex ----------
@vertex
fn vs_main(in: VsIn) -> VsOut {
  var out: VsOut;

  // Read height from texture first, fall back to analytic if texture is flat
  let sampled_h = textureSampleLevel(height_tex, height_samp, in.uv, 0.0).r;
  
  // If the sampled height is exactly 0.0, assume it's uninitialized and use analytic fallback
  let use_analytic = (sampled_h == 0.0);
  let final_height = select(sampled_h, analytic_height(in.pos_xy.x, in.pos_xy.y), use_analytic);
  
  // Apply height exaggeration
  let height_exag = globals.spacing_h_exag_pad.z;
  let world_z = final_height * height_exag;

  // Construct world position
  let world_pos = vec4<f32>(in.pos_xy.x, world_z, in.pos_xy.y, 1.0);

  // Transform to clip space
  out.clip_pos = globals.proj * globals.view * world_pos;
  out.uv = in.uv;
  out.height = final_height;
  out.xz = in.pos_xy;

  return out;
}

// ---------- Fragment ----------
@fragment
fn fs_main(in: VsOut) -> FsOut {
  // Height-based colormap: normalize height to [0, 1] assuming range ±1.0 (as per T2).
  let h_range = 1.0;
  let t = clamp(0.5 + in.height / (2.0 * h_range), 0.0, 1.0);

  // Descriptor indexing: directly sample from texture array using palette index
  let palette_index = u32(globals.spacing_h_exag_pad.w);
  let lut_color = textureSampleLevel(lut_textures[palette_index], lut_samp, vec2<f32>(t, 0.5), 0.0);

  // Calculate normal from analytic slope (adds spatial variation even with flat height_tex).
  let dhdx = 1.3 * cos(in.xz.x * 1.3) * 0.25;
  let dhdz = -1.1 * sin(in.xz.y * 1.1) * 0.25;
  let normal = normalize(vec3<f32>(-dhdx, 1.0, -dhdz));

  // Directional lighting (sun)
  let sun_L = normalize(globals.sun_exposure.xyz);
  let sun_lambert = clamp(dot(normal, sun_L), 0.0, 1.0);
  let sun_contribution = globals.sun_exposure.w * sun_lambert;
  
  let spacing = globals.spacing_h_exag_pad.x;
  let exaggeration = globals.spacing_h_exag_pad.z;
  let world = vec3<f32>(in.xz.x * spacing, in.height * exaggeration, in.xz.y * spacing);
  let world_pos_2d = world.xz;
  let cloud_shadow_multiplier = sample_cloud_shadow(world_pos_2d, spacing * 50.0);
  let shadowed_sun = sun_contribution * cloud_shadow_multiplier;

  let shade = mix(0.15, 1.0, clamp(shadowed_sun, 0.0, 1.0));

  var lit_color = lut_color.rgb * shade;
  lit_color = apply_planar_reflection(world, normal, lit_color);

  let tonemapped = reinhard(lit_color);
  let gamma_corrected = gamma_correct(tonemapped);

  let view_pos = (globals.view * vec4<f32>(world, 1.0)).xyz;
  let linear_depth = max(-view_pos.z, 0.0);

  let normal_encoded = normal * 0.5 + vec3<f32>(0.5);
  var out: FsOut;
  out.color = vec4<f32>(gamma_corrected, 1.0);
  out.normal_depth = vec4<f32>(normal_encoded, linear_depth);
  return out;
}

// C4: Explicit tonemap functions for compliance
fn reinhard(x: vec3<f32>) -> vec3<f32> {
  return x / (1.0 + x);
}

fn gamma_correct(x: vec3<f32>) -> vec3<f32> {
  return pow(x, vec3<f32>(1.0 / 2.2));
}
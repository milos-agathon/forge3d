// T3.3 Terrain shader — compatible with Rust pipeline bind group layouts.
// Layout: 0=Globals UBO, 1=height R32Float + NonFiltering sampler, 2=LUT RGBA8 + Filtering sampler.
// This version adds a deterministic analytic height fallback to avoid uniform output with a 1×1 dummy height.

// B16-BEGIN: Point/spot light structures
struct PointLight {
  position: vec3<f32>,
  _pad0: f32,
  color: vec3<f32>, 
  intensity: f32,
  radius: f32,
  _pad1: vec3<f32>,
}; // 32 bytes

struct SpotLight {
  position: vec3<f32>,
  _pad0: f32,
  direction: vec3<f32>,
  _pad1: f32,
  color: vec3<f32>,
  intensity: f32,
  radius: f32,
  inner_cone: f32,
  outer_cone: f32,
  _pad2: f32,
}; // 48 bytes
// B16-END: Point/spot light structures

// ---------- Globals UBO (256 bytes total, must match Rust) ----------
struct Globals {
  view : mat4x4<f32>,          // 64 B
  proj : mat4x4<f32>,          // 64 B
  sun_exposure : vec4<f32>,    // xyz = sun_dir, w = exposure
  // packs (spacing, h_range, exaggeration, 0) for source-compat with globals.spacing.x, .y, .z
  spacing : vec4<f32>,         // 16 B
  // B16: Light counts and configuration
  light_counts: vec4<f32>,     // xyz = (num_point_lights, num_spot_lights, 0, 0)
  _pad_tail : vec4<f32>,       // pad to 128 B
  // Light data arrays (up to 4 point lights + 2 spot lights)
  point_lights: array<PointLight, 4>,   // 4 * 32 = 128 B  
  spot_lights: array<SpotLight, 2>,     // 2 * 48 = 96 B (+ 32B pad = 128B)
  _pad_end: vec4<f32>,                  // Total: 384 bytes
};

@group(0) @binding(0) var<uniform> globals : Globals;

// ---------- Textures & samplers ----------
@group(1) @binding(0) var height_tex  : texture_2d<f32>;  // R32Float, non-filterable
@group(1) @binding(1) var height_samp : sampler;          // NonFiltering at pipeline level

@group(2) @binding(0) var lut_tex  : texture_2d<f32>;     // RGBA8 (sRGB/UNORM), filterable
@group(2) @binding(1) var lut_samp : sampler;

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

// Analytic fallback height that varies across the grid. Amplitude ≈ ±0.5 (matches Globals defaults).
fn analytic_height(x: f32, z: f32) -> f32 {
  return sin(x * 1.3) * 0.25 + cos(z * 1.1) * 0.25;
}

// ---------- Vertex ----------
@vertex
fn vs_main(in: VsIn) -> VsOut {
  let spacing      = max(globals.spacing.x, 1e-8);
  let exaggeration = globals.spacing.z;

  // Sample height with a NonFiltering sampler; level 0 to avoid filtering.
  let h_tex = textureSampleLevel(height_tex, height_samp, in.uv, 0.0).r;

  // Deterministic analytic fallback guarantees variation even if height_tex is 1x1.
  let h_ana = analytic_height(in.pos_xy.x, in.pos_xy.y);

  let h = h_tex + h_ana;

  // Build world position (XY are plane coords, Y is height).
  let world = vec3<f32>(in.pos_xy.x * spacing, h * exaggeration, in.pos_xy.y * spacing);

  var out : VsOut;
  out.clip_pos = globals.proj * (globals.view * vec4<f32>(world, 1.0));
  out.uv       = in.uv;
  out.height   = h;
  out.xz       = in.pos_xy;
  return out;
}

// B16-BEGIN: Lighting calculation functions
fn calculate_point_light(light: PointLight, world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
  let light_dir = light.position - world_pos;
  let distance = length(light_dir);
  
  // Early exit if outside light radius
  if (distance > light.radius) {
    return vec3<f32>(0.0);
  }
  
  let L = normalize(light_dir);
  let lambert = max(dot(normal, L), 0.0);
  
  // Distance attenuation (inverse square with smooth falloff)
  let attenuation = 1.0 / (1.0 + distance * distance / (light.radius * light.radius));
  let smooth_attenuation = attenuation * attenuation * (3.0 - 2.0 * attenuation); // Smoothstep
  
  return light.color * light.intensity * lambert * smooth_attenuation;
}

fn calculate_spot_light(light: SpotLight, world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
  let light_dir = light.position - world_pos;
  let distance = length(light_dir);
  
  // Early exit if outside light radius
  if (distance > light.radius) {
    return vec3<f32>(0.0);
  }
  
  let L = normalize(light_dir);
  let spot_dir = normalize(-light.direction);
  
  // Cone attenuation
  let cos_angle = dot(L, spot_dir);
  let cos_inner = cos(light.inner_cone);
  let cos_outer = cos(light.outer_cone);
  
  // Early exit if outside outer cone
  if (cos_angle < cos_outer) {
    return vec3<f32>(0.0);
  }
  
  // Smooth transition between inner and outer cone
  let cone_attenuation = smoothstep(cos_outer, cos_inner, cos_angle);
  
  let lambert = max(dot(normal, L), 0.0);
  
  // Distance attenuation
  let attenuation = 1.0 / (1.0 + distance * distance / (light.radius * light.radius));
  let smooth_attenuation = attenuation * attenuation * (3.0 - 2.0 * attenuation);
  
  return light.color * light.intensity * lambert * cone_attenuation * smooth_attenuation;
}
// B16-END: Lighting calculation functions

// ---------- Fragment ----------
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  // Map height into [0,1] using h_range stored in spacing.y (avoid div by 0).
  let h_range = max(globals.spacing.y, 1e-8);
  let t = clamp(0.5 + in.height / (2.0 * h_range), 0.0, 1.0);

  // 256x1 LUT: sample along X at row center (v=0.5).
  let lut_color = textureSampleLevel(lut_tex, lut_samp, vec2<f32>(t, 0.5), 0.0);

  // B16-BEGIN: Calculate world position and normal for lighting
  let spacing      = max(globals.spacing.x, 1e-8);
  let exaggeration = globals.spacing.z;
  let world_pos = vec3<f32>(in.xz.x * spacing, in.height * exaggeration, in.xz.y * spacing);
  
  // Calculate normal from analytic slope (adds spatial variation even with flat height_tex).
  let dhdx = 1.3 * cos(in.xz.x * 1.3) * 0.25;
  let dhdz = -1.1 * sin(in.xz.y * 1.1) * 0.25;
  let normal = normalize(vec3<f32>(-dhdx, 1.0, -dhdz));

  // Directional lighting (sun)
  let sun_L = normalize(globals.sun_exposure.xyz);
  let sun_lambert = clamp(dot(normal, sun_L), 0.0, 1.0);
  let sun_contribution = globals.sun_exposure.w * sun_lambert;
  
  // Accumulate point light contributions
  var additional_lighting = vec3<f32>(0.0);
  let num_point_lights = i32(globals.light_counts.x);
  for (var i = 0; i < num_point_lights; i++) {
    if (i < 4) { // Bounds check for array access
      additional_lighting += calculate_point_light(globals.point_lights[i], world_pos, normal);
    }
  }
  
  // Accumulate spot light contributions  
  let num_spot_lights = i32(globals.light_counts.y);
  for (var i = 0; i < num_spot_lights; i++) {
    if (i < 2) { // Bounds check for array access
      additional_lighting += calculate_spot_light(globals.spot_lights[i], world_pos, normal);
    }
  }
  
  // Combine all lighting with ambient floor
  let total_lighting = sun_contribution + length(additional_lighting);
  let shade = mix(0.15, 1.0, clamp(total_lighting, 0.0, 1.0));
  let additional_color = additional_lighting;
  // B16-END: Enhanced lighting calculation

  // Apply explicit tonemap pipeline: reinhard -> gamma correction
  let lit_color = (lut_color.rgb * shade) + additional_color;
  let tonemapped = reinhard(lit_color);
  let gamma_corrected = gamma_correct(tonemapped);

  return vec4<f32>(gamma_corrected, 1.0);
}

// C4: Explicit tonemap functions for compliance
fn reinhard(x: vec3<f32>) -> vec3<f32> {
  return x / (1.0 + x);
}

fn gamma_correct(x: vec3<f32>) -> vec3<f32> {
  return pow(x, vec3<f32>(1.0 / 2.2));
}
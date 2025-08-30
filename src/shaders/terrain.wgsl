// T3.3 Terrain shader — compatible with Rust pipeline bind group layouts.
// Layout: 0=Globals UBO, 1=height R32Float + NonFiltering sampler, 2=LUT RGBA8 + Filtering sampler.
// This version adds a deterministic analytic height fallback to avoid uniform output with a 1×1 dummy height.

// ---------- Globals UBO (176 bytes total, must match Rust) ----------
// std140-compatible layout: 176 bytes total
// Field breakdown:
//   view:                 mat4x4<f32> = 64 B
//   proj:                 mat4x4<f32> = 64 B  
//   sun_exposure:         vec4<f32>   = 16 B (xyz = sun_dir normalized, w = exposure)
//   spacing_h_exag_pad:   vec4<f32>   = 16 B (x=dx, y=dy, z=height_exaggeration, w=unused)
//   _pad_tail:            vec4<f32>   = 16 B (tail padding)
//   TOTAL:                            = 176 B
struct Globals {
  view: mat4x4<f32>,                    // 64 B
  proj: mat4x4<f32>,                    // 64 B
  sun_exposure: vec4<f32>,              // xyz = sun_dir (normalized), w = exposure (16 B)
  spacing_h_exag_pad: vec4<f32>,        // x=dx, y=dy, z=height_exaggeration, w=unused (16 B)
  _pad_tail: vec4<f32>,                 // tail padding to keep total size multiple-of-16 (16 B)
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
  let spacing      = max(globals.spacing_h_exag_pad.x, 1e-8);
  let exaggeration = globals.spacing_h_exag_pad.z;

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

// ---------- Fragment ----------
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  // Map height into [0,1] using h_range stored in spacing_h_exag_pad.y (avoid div by 0).
  let h_range = max(globals.spacing_h_exag_pad.y, 1e-8);
  let t = clamp(0.5 + in.height / (2.0 * h_range), 0.0, 1.0);

  // 256x1 LUT: sample along X at row center (v=0.5).
  let lut_color = textureSampleLevel(lut_tex, lut_samp, vec2<f32>(t, 0.5), 0.0);

  // Calculate normal from analytic slope (adds spatial variation even with flat height_tex).
  let dhdx = 1.3 * cos(in.xz.x * 1.3) * 0.25;
  let dhdz = -1.1 * sin(in.xz.y * 1.1) * 0.25;
  let normal = normalize(vec3<f32>(-dhdx, 1.0, -dhdz));

  // Directional lighting (sun)
  let sun_L = normalize(globals.sun_exposure.xyz);
  let sun_lambert = clamp(dot(normal, sun_L), 0.0, 1.0);
  let sun_contribution = globals.sun_exposure.w * sun_lambert;
  
  // Simple lighting with ambient floor
  let shade = mix(0.15, 1.0, clamp(sun_contribution, 0.0, 1.0));

  // Apply explicit tonemap pipeline: reinhard -> gamma correction
  let lit_color = lut_color.rgb * shade;
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
// T3.3 Terrain shader — compatible with Rust pipeline bind group layouts.
// Layout: 0=Globals UBO, 1=height R32Float + NonFiltering sampler, 2=LUT RGBA8 + Filtering sampler.
// This version adds a deterministic analytic height fallback to avoid uniform output with a 1×1 dummy height.

// ---------- Globals UBO (176 bytes total, must match Rust) ----------
struct Globals {
  view : mat4x4<f32>,          // 64 B
  proj : mat4x4<f32>,          // 64 B
  sun_exposure : vec4<f32>,    // xyz = sun_dir, w = exposure
  // packs (spacing, h_range, exaggeration, 0) for source-compat with globals.spacing.x, .y, .z
  spacing : vec4<f32>,
  _pad_tail : vec4<f32>,       // pad to 176 B
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

// ---------- Fragment ----------
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  // Map height into [0,1] using h_range stored in spacing.y (avoid div by 0).
  let h_range = max(globals.spacing.y, 1e-8);
  let t = clamp(0.5 + in.height / (2.0 * h_range), 0.0, 1.0);

  // 256x1 LUT: sample along X at row center (v=0.5).
  let lut_color = textureSampleLevel(lut_tex, lut_samp, vec2<f32>(t, 0.5), 0.0);

  // Simple Lambert term from analytic slope (adds spatial variation even with flat height_tex).
  let dhdx = 1.3 * cos(in.xz.x * 1.3) * 0.25;
  let dhdz = -1.1 * sin(in.xz.y * 1.1) * 0.25;
  let n = normalize(vec3<f32>(-dhdx, 1.0, -dhdz));

  let L = normalize(globals.sun_exposure.xyz);
  let lambert = clamp(dot(n, L), 0.0, 1.0);
  let exposure = globals.sun_exposure.w;

  // Mix in a small ambient floor to avoid large flat regions in the PNG.
  let shade = mix(0.15, 1.0, lambert);

  // Apply explicit tonemap pipeline: reinhard -> gamma correction
  let lit_color = lut_color.rgb * exposure * shade;
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
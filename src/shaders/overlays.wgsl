// src/shaders/overlays.wgsl
// Fullscreen overlay compositor: drape overlay (RGBA8 sRGB) and optional altitude ramp from height texture.

struct OverlayUniforms {
  view_proj: mat4x4<f32>,
  overlay_params: vec4<f32>, // x: overlay_enabled, y: overlay_alpha, z: altitude_enabled, w: altitude_alpha
  overlay_uv: vec4<f32>,     // x: uv_off_x, y: uv_off_y, z: uv_scale_x, w: uv_scale_y
  contour_params: vec4<f32>, // x: contour_enabled, y: interval, z: thickness_mul, w: pad
  contour_color: vec4<f32>,  // rgba for contour lines
};

@group(0) @binding(0) var<uniform> U : OverlayUniforms;
@group(0) @binding(1) var overlay_tex : texture_2d<f32>;
@group(0) @binding(2) var overlay_samp : sampler;
@group(0) @binding(3) var height_tex : texture_2d<f32>;
@group(0) @binding(4) var height_samp : sampler;

struct VsOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi : u32) -> VsOut {
  var out : VsOut;
  // Fullscreen triangle
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 3.0,  1.0)
  );
  let pos = p[vi];
  out.pos = vec4<f32>(pos, 0.0, 1.0);
  out.uv = pos * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
  return out;
}

fn ramp_color(h: f32) -> vec3<f32> {
  // Simple green->brown->white ramp
  let c0 = vec3<f32>(0.1, 0.5, 0.2);
  let c1 = vec3<f32>(0.5, 0.35, 0.2);
  let c2 = vec3<f32>(0.9, 0.9, 0.9);
  let mid = 0.6;
  let t = clamp(h, 0.0, 1.0);
  let low = mix(c0, c1, clamp(t / mid, 0.0, 1.0));
  let hi = mix(c1, c2, clamp((t - mid) / max(1e-3, 1.0 - mid), 0.0, 1.0));
  return mix(low, hi, select(0.0, 1.0, t > mid));
}

@fragment
fn fs_overlay(in: VsOut) -> @location(0) vec4<f32> {
  var col : vec3<f32> = vec3<f32>(0.0);
  var a   : f32 = 0.0;

  let ov_en = U.overlay_params.x > 0.5;
  let ov_a  = clamp(U.overlay_params.y, 0.0, 1.0);
  let alt_en = U.overlay_params.z > 0.5;

  if (ov_en) {
    let uv_ov = vec2<f32>(U.overlay_uv.x, U.overlay_uv.y) + in.uv * vec2<f32>(U.overlay_uv.z, U.overlay_uv.w);
    let ov = textureSample(overlay_tex, overlay_samp, uv_ov);
    // Assume overlay in linear; if sRGB, we rely on render target sRGB conversion.
    col = ov.rgb;
    a = ov_a;
  }

  if (alt_en) {
    // Sample height; assume height in 0..1 for now
    let h = textureSampleLevel(height_tex, height_samp, in.uv, 0.0).r;
    let alt_col = ramp_color(clamp(h, 0.0, 1.0));
    let alt_a = clamp(U.overlay_params.w, 0.0, 1.0);
    // Composite altitude under overlay in shader, then alpha blend with target
    col = mix(alt_col, col, a);
    a = clamp(a + alt_a * (1.0 - a), 0.0, 1.0);
  }

  // GPU contour overlay from height texture (line rendering in screen space)
  if (U.contour_params.x > 0.5) {
    let interval = max(1e-6, U.contour_params.y);
    let h = textureSampleLevel(height_tex, height_samp, in.uv, 0.0).r;
    let pos = h / interval;
    // Distance to nearest iso level in repeating unit space
    let g = abs(pos - floor(pos + 0.5));
    // Anti-aliased line width based on derivatives; thickness scales this width
    let aa = fwidth(pos) * max(1.0, U.contour_params.z);
    let line = 1.0 - smoothstep(0.0, aa, g);
    let c_col = U.contour_color.rgb;
    let c_a = clamp(U.contour_color.a, 0.0, 1.0) * line;
    col = mix(col, c_col, c_a);
    a = clamp(a + c_a * (1.0 - a), 0.0, 1.0);
  }

  return vec4<f32>(col, a);
}

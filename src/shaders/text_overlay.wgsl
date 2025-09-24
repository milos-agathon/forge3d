// src/shaders/text_overlay.wgsl
// MSDF-capable text overlay shader (also supports SDF). Renders instances as screen-space quads.

struct TextOverlayUniforms {
  resolution: vec2<f32>, // (width, height)
  alpha: f32,
  enabled: f32,
  channels: f32, // 1=SDF, 3=MSDF
  smoothing: f32, // px smoothing scale
};

@group(0) @binding(0) var<uniform> U : TextOverlayUniforms;
@group(0) @binding(1) var atlas_tex : texture_2d<f32>;
@group(0) @binding(2) var atlas_samp : sampler;

struct VsOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) color    : vec4<f32>,
  @location(1) uv       : vec2<f32>,
};

@vertex
fn vs_main(@location(0) quad_pos: vec2<f32>,
           @location(1) rect_min: vec2<f32>,
           @location(2) rect_max: vec2<f32>,
           @location(3) uv_min: vec2<f32>,
           @location(4) uv_max: vec2<f32>,
           @location(5) color: vec4<f32>) -> VsOut {
  var out: VsOut;
  // Scale unit quad to rect in pixel space
  let pos_px = mix(rect_min, rect_max, quad_pos);
  let ndc = vec2<f32>(
    (pos_px.x / U.resolution.x) * 2.0 - 1.0,
    1.0 - (pos_px.y / U.resolution.y) * 2.0);
  out.pos = vec4<f32>(ndc, 0.0, 1.0);
  out.color = color;
  out.uv = mix(uv_min, uv_max, quad_pos);
  return out;
}

fn median3(v: vec3<f32>) -> f32 {
  return max(min(v.x, v.y), min(max(v.x, v.y), v.z));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  if (U.enabled < 0.5) { discard; }
  let sample = textureSample(atlas_tex, atlas_samp, in.uv);
  var sdf: f32;
  // Decode SDF/MSDF
  if (U.channels >= 2.5) {
    // MSDF: use median of RGB
    sdf = median3(sample.rgb) - 0.5;
  } else {
    // SDF: use single channel (assume red)
    sdf = sample.r - 0.5;
  }
  // Compute width from derivatives in UV space -> scale with smoothing
  let w = fwidth(sdf) * U.smoothing;
  let a_edge = clamp(sdf / max(1e-6, w) + 0.5, 0.0, 1.0);
  let a = clamp(a_edge * in.color.a * U.alpha, 0.0, 1.0);
  let col = in.color.rgb;
  return vec4<f32>(col, a);
}
